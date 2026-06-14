import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.schemas.passport import FitFlag, FitFlagsResponse


MIN_AGGREGATE_TOTAL = 25
BIAS_THRESHOLD = 0.2
BIAS_MARGIN = 0.03


@dataclass(frozen=True)
class FitAggregate:
    source: str
    category: str
    fit_count: int
    small_count: int
    large_count: int
    total: int
    runs_small_rate: float
    runs_large_rate: float


def _normalize_category(category: str | None) -> str:
    value = (category or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _category_keys(category: str | None) -> list[str]:
    normalized = _normalize_category(category)
    if not normalized:
        return []

    keys = [normalized]
    last_word = normalized.split()[-1]
    if last_word != normalized:
        keys.append(last_word)

    aliases = {
        "bottoms": "bottom",
        "dresses": "dress",
        "leggings": "legging",
        "pants": "pant",
        "skirts": "skirt",
        "tops": "top",
        "trousers": "trouser",
    }

    expanded: list[str] = []
    for key in keys:
        expanded.append(key)
        expanded.append(aliases.get(key, key))
        if key.endswith("s") and len(key) > 3:
            expanded.append(key[:-1])

    return list(dict.fromkeys(expanded))


def _parse_aggregate(row: dict[str, Any]) -> FitAggregate:
    return FitAggregate(
        source=str(row.get("source", "unknown")),
        category=_normalize_category(str(row["category"])),
        fit_count=int(row.get("fit_count", 0)),
        small_count=int(row.get("small_count", 0)),
        large_count=int(row.get("large_count", 0)),
        total=int(row.get("total", 0)),
        runs_small_rate=float(row.get("runs_small_rate", 0.0)),
        runs_large_rate=float(row.get("runs_large_rate", 0.0)),
    )


@lru_cache(maxsize=4)
def load_fit_aggregates(path: str | Path | None = None) -> tuple[FitAggregate, ...]:
    aggregate_path = Path(path or settings.fit_aggregates_path)
    if not aggregate_path.exists():
        return ()

    rows = json.loads(aggregate_path.read_text(encoding="utf-8"))
    return tuple(_parse_aggregate(row) for row in rows)


def _best_aggregate(category: str | None) -> FitAggregate | None:
    keys = set(_category_keys(category))
    if not keys:
        return None

    matches = [
        aggregate
        for aggregate in load_fit_aggregates()
        if aggregate.category in keys
    ]
    if not matches:
        return None

    return max(matches, key=lambda aggregate: aggregate.total)


def _confidence(aggregate: FitAggregate) -> float:
    support = min(0.25, aggregate.total / 100000)
    signal = abs(aggregate.runs_small_rate - aggregate.runs_large_rate)
    return round(min(0.95, 0.62 + support + signal), 2)


def _flag_from_aggregate(aggregate: FitAggregate) -> FitFlag:
    if aggregate.total < MIN_AGGREGATE_TOTAL:
        return FitFlag(
            type="critical_fit",
            message="Fit history is thin for this category; confirm measurements before checkout.",
            confidence=0.56,
        )

    small_rate = aggregate.runs_small_rate
    large_rate = aggregate.runs_large_rate
    confidence = _confidence(aggregate)

    if small_rate >= BIAS_THRESHOLD and small_rate >= large_rate + BIAS_MARGIN:
        return FitFlag(
            type="runs_small",
            message=f"Category fit history leans small ({small_rate:.0%} of reviews).",
            confidence=confidence,
        )

    if large_rate >= BIAS_THRESHOLD and large_rate >= small_rate + BIAS_MARGIN:
        return FitFlag(
            type="runs_large",
            message=f"Category fit history leans large ({large_rate:.0%} of reviews).",
            confidence=confidence,
        )

    return FitFlag(
        type="true_to_size",
        message="Most category fit history is true to size.",
        confidence=confidence,
    )


def _fallback_flag(category: str | None) -> FitFlag:
    normalized = _normalize_category(category)
    if "shoe" in normalized:
        return FitFlag(
            type="runs_small",
            message="Many buyers prefer half a size up for this category.",
            confidence=0.62,
        )
    if any(token in normalized for token in ("dress", "fashion", "clothing")):
        return FitFlag(
            type="true_to_size",
            message="Most recent fit signals indicate this item is true to size.",
            confidence=0.7,
        )
    return FitFlag(
        type="critical_fit",
        message="Fit history is limited; confirm size details before checkout.",
        confidence=0.55,
    )


def predict_fit_flags(sku_id: str, category: str | None) -> FitFlagsResponse:
    aggregate = _best_aggregate(category)
    if aggregate is None:
        return FitFlagsResponse(
            sku_id=sku_id,
            flags=[_fallback_flag(category)],
            source="rules_v1",
        )

    return FitFlagsResponse(
        sku_id=sku_id,
        flags=[_flag_from_aggregate(aggregate)],
        source=f"fit_aggregates_v1:{aggregate.source}",
    )
