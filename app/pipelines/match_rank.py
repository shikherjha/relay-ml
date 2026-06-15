"""Reverse-wishlist relevance reranker (Bedrock Nova Lite, text-only).

The second stage of the industry retrieve→rerank pattern. relay-api recalls a
candidate pool (vertical-gated cosine ANN) and asks this endpoint to score how
well each candidate ACTUALLY satisfies the shopper's wish. An LLM understands
intent far better than raw cosine, so a "macbook" wish scores a laptop ~1.0,
earphones ~0.2 and a tee ~0.0 — killing the cross-category noise cosine leaves.

Cheap: one text-only Nova Lite call over a short candidate list. Fails loud
(``RerankUnavailable``) so relay-api can fall back to deterministic taxonomy
scoring instead of returning garbage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


class RerankUnavailable(RuntimeError):
    """Bedrock not configured / call failed — caller should fall back."""


@dataclass(frozen=True)
class RankedMatch:
    unit_id: str
    score: float
    reason: str


def _build_prompt(wish: str, size: str | None, max_price: float | None, candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        price = c.get("price")
        price_str = f"₹{price:,.0f}" if isinstance(price, (int, float)) else "n/a"
        lines.append(
            f'{i}. title="{c.get("title") or "?"}", category="{c.get("category") or "?"}", price={price_str}'
        )
    constraints = []
    if size:
        constraints.append(f'size "{size}"')
    if max_price:
        constraints.append(f"budget ≤ ₹{max_price:,.0f}")
    cstr = (" Constraints: " + ", ".join(constraints) + ".") if constraints else ""
    catalogue = "\n".join(lines)
    return f"""You are a shopping assistant matching a shopper's wish to returned/second-hand items.

Shopper wants: "{wish}".{cstr}

Candidate items:
{catalogue}

For EACH candidate, judge how well it satisfies what the shopper actually wants.
Score 0.0–1.0:
- 1.0 = same kind of product the shopper asked for (e.g. wish "macbook" ↔ a laptop/MacBook)
- 0.3–0.6 = related but not the same kind
- 0.0 = a different kind of product (e.g. wish "macbook" ↔ earphones, or "hoodie" ↔ a t-shirt)
Judge by product TYPE/intent, not price. Be strict: different product types must score near 0.

Return ONLY a JSON array, one object per candidate:
[{{"i": 0, "score": 0.0, "reason": "..."}}]"""


def rank_matches(
    *,
    wish: str,
    size: str | None,
    max_price: float | None,
    candidates: list[dict],
    region: str,
    model_id: str,
) -> list[RankedMatch]:
    """Score candidates' relevance to the wish via Bedrock. Raises
    ``RerankUnavailable`` on any configuration/transport/parse failure."""
    if not candidates:
        return []
    try:
        import boto3

        client = boto3.client("bedrock-runtime", region_name=region)
    except Exception as exc:  # noqa: BLE001
        raise RerankUnavailable(f"bedrock unavailable: {exc}") from exc

    prompt = _build_prompt(wish, size, max_price, candidates)
    try:
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 1024, "temperature": 0.0},
        )
        text = resp["output"]["message"]["content"][0]["text"]
    except Exception as exc:  # noqa: BLE001
        raise RerankUnavailable(f"bedrock rerank call failed: {exc}") from exc

    return _parse(text, candidates)


def _parse(text: str, candidates: list[dict]) -> list[RankedMatch]:
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        t = t[nl + 1:] if nl != -1 else t[3:]
        if t.endswith("```"):
            t = t[:-3]
    t = t.strip()
    # Be lenient: grab the JSON array if the model added prose around it.
    if not t.startswith("["):
        lo, hi = t.find("["), t.rfind("]")
        if lo != -1 and hi != -1 and hi > lo:
            t = t[lo:hi + 1]
    try:
        data = json.loads(t)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RerankUnavailable(f"unparseable rerank response: {exc}") from exc

    out: list[RankedMatch] = []
    for item in data if isinstance(data, list) else []:
        try:
            idx = int(item.get("i"))
            cand = candidates[idx]
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        out.append(RankedMatch(
            unit_id=str(cand.get("unit_id")),
            score=max(0.0, min(1.0, score)),
            reason=str(item.get("reason") or "")[:200],
        ))
    if not out:
        raise RerankUnavailable("rerank produced no scored candidates")
    return out
