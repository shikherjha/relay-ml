"""Bedrock escalation pipeline.

Supports three grading modes via GRADING_MODE env var:
- cnn (default): CNN T1 → Bedrock escalation on low confidence
- bedrock_only: Skip CNN, grade every request via Nova Lite (demo-safe fallback)
- mock: Deterministic stub passport (no AI, for relay-api local dev)

Tiered escalation (cnn mode):
- T1: CNN prediction (fast, cheap)
- T2: Bedrock Haiku/Nova Lite when CNN confidence < threshold_t2
- T3: Bedrock Nova Pro when T2 confidence < threshold_t3
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.passport import ConditionPassport, Defect, Grade, Verification
from app.utils.hash import passport_hash, sha256_hex


# Grade numeric values (shared with image_grade.py)
GRADE_NUMERIC = {
    "A+": 1.0,
    "A": 0.9,
    "B+": 0.78,
    "B": 0.68,
    "C": 0.45,
    "D": 0.2,
}


class BedrockGradingError(RuntimeError):
    """Raised when Bedrock grading fails."""


def grade_bedrock_only(
    *,
    image_bytes: bytes,
    unit_id: str,
    category: str,
    return_id: str | None,
    expected_size: str | None = None,
    expected_color: str | None = None,
    product_title: str | None = None,
) -> ConditionPassport:
    """Grade an image using Bedrock Nova Lite directly (no CNN).

    This is the demo-safe escape hatch: real grades without trained model weights.
    Slower and pricier per call, but always available if AWS credentials are configured.

    ``expected_*`` are ADDITIVE order-vs-item context (no extra image/call): when
    present, the prompt also reports colour/item match → a ``verification`` block.
    """
    from app.core.config import settings

    media_hash = sha256_hex(image_bytes)

    try:
        import boto3

        client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
        )

        # Build the prompt for structured grading
        prompt = _build_grading_prompt(
            category, expected_color=expected_color,
            product_title=product_title, expected_size=expected_size,
        )

        # Detect image format from bytes
        image_format = "jpeg"  # default
        if image_bytes[:8].startswith(b"\x89PNG\r\n\x1a\n"):
            image_format = "png"
        elif image_bytes[:2] in (b"\xff\xd8",):
            image_format = "jpeg"

        # Call Nova Lite with the image (raw bytes, not base64)
        response = client.converse(
            modelId=settings.bedrock_model_t1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": image_format,
                                "source": {"bytes": image_bytes},
                            }
                        },
                        {"text": prompt},
                    ],
                }
            ],
            inferenceConfig={
                "maxTokens": 1024,
                "temperature": 0.1,
            },
        )

        # Parse the structured response
        response_text = response["output"]["message"]["content"][0]["text"]
        parsed = _parse_grading_response(response_text, category)

        grade = parsed["grade"]
        defect_type = parsed["defect_type"]
        confidence = parsed["confidence"]
        no_damage = parsed.get("no_damage", False)
        description = parsed.get("description", "")

        # Build defects list — empty if no damage detected
        defects = []
        if not no_damage:
            defects = [
                Defect(
                    type=defect_type,
                    severity=_severity_for_grade(grade),
                    confidence=confidence,
                    description=description or f"Bedrock Nova Lite detected {defect_type.replace('_', ' ')}.",
                ).model_dump(mode="json")
            ]

        body = {
            "schema_version": "1.0.0",
            "unit_id": unit_id,
            "return_id": return_id,
            "grade": grade,
            "grade_numeric": GRADE_NUMERIC.get(grade, 0.5),
            "category": category.strip().lower().replace("-", "_").replace(" ", "_") or "unknown",
            "vertical": _infer_vertical(category),
            "disposition_hint": _disposition_for_grade(grade),
            "defects": defects,
            "packaging_state": "sealed" if grade in ("A+", "A") else "opened",
            "confidence": confidence,
            "media_hashes": [media_hash],
            "passport_hash": "",
            "graded_at": datetime.now(timezone.utc),
            "model_tier_used": "bedrock-only",
            "warranty_months_remaining": 0,
            "repair_events": [],
        }
        verification = _build_verification(
            parsed, expected_color=expected_color, product_title=product_title,
        )
        if verification is not None:
            body["verification"] = verification
        body["passport_hash"] = passport_hash(body)
        return ConditionPassport.model_validate(body)

    except ImportError as exc:
        raise BedrockGradingError(
            "boto3 is not installed. Install it to use bedrock_only grading mode."
        ) from exc
    except Exception as exc:
        raise BedrockGradingError(f"Bedrock grading failed: {exc}") from exc


def grade_mock(
    *,
    image_bytes: bytes,
    unit_id: str,
    category: str,
    return_id: str | None,
    expected_size: str | None = None,
    expected_color: str | None = None,
    product_title: str | None = None,
) -> ConditionPassport:
    """Return a deterministic stub passport (no AI). For relay-api local dev."""
    media_hash = sha256_hex(image_bytes)

    body = {
        "schema_version": "1.0.0",
        "unit_id": unit_id,
        "return_id": return_id,
        "grade": "B",
        "grade_numeric": 0.68,
        "category": category.strip().lower().replace("-", "_").replace(" ", "_") or "unknown",
        "vertical": _infer_vertical(category),
        "disposition_hint": "p2p_resale",
        "defects": [
            Defect(
                type="scuff",
                severity="minor",
                confidence=0.85,
                description="Mock grade — no AI inference performed.",
            ).model_dump(mode="json")
        ],
        "packaging_state": "opened",
        "confidence": 0.85,
        "media_hashes": [media_hash],
        "passport_hash": "",
        "graded_at": datetime.now(timezone.utc),
        "model_tier_used": "mock",
        "warranty_months_remaining": 0,
        "repair_events": [],
    }
    verification = _build_verification(
        {"observed_color": expected_color,
         "color_match": "match" if expected_color else "unknown",
         "item_match": "match" if product_title else "unknown"},
        expected_color=expected_color, product_title=product_title,
    )
    if verification is not None:
        body["verification"] = verification
    body["passport_hash"] = passport_hash(body)
    return ConditionPassport.model_validate(body)


def _verification_prompt_suffix(
    *,
    expected_color: str | None,
    product_title: str | None,
    expected_size: str | None,
) -> str:
    """ADDITIVE order-vs-item verification instructions (prompt-only, no extra
    image / Bedrock call). Returns "" when the caller sent no expected context,
    so the base prompt + existing callers are unaffected."""
    if not (expected_color or product_title):
        return ""
    expectations = []
    if product_title:
        expectations.append(f'the buyer ordered: "{product_title}"')
    if expected_color:
        expectations.append(f'expected colour: "{expected_color}"')
    if expected_size:
        expectations.append(f'expected size: "{expected_size}"')
    ctx = "; ".join(expectations)
    return f"""

ALSO verify the photographed item against the order ({ctx}). Add these JSON fields:
- "observed_color": the dominant colour you actually see (one or two words), or null
- "color_match": "match" if the observed colour matches the expected colour, "mismatch" if clearly different, "unknown" if you cannot tell or no colour was expected
- "item_match": "match" if the item looks like the ordered product, "mismatch" if it is clearly a different kind of product, "unknown" if unsure
Judge colour/item match ONLY from this same image — do not invent details."""


def _build_grading_prompt(
    category: str,
    *,
    expected_color: str | None = None,
    product_title: str | None = None,
    expected_size: str | None = None,
) -> str:
    """Build the structured extraction prompt for Bedrock."""
    return f"""You are an AI product condition grader for a circular commerce platform.

Analyze this product image (category: {category}) and assess its physical condition.

Return a JSON object with these fields:
- "grade": one of "A+", "A", "B+", "B", "C", "D"
  - A+ = pristine/sealed, looks brand new, no visible wear
  - A = like new, minimal signs of use
  - B+ = good condition, light cosmetic wear only
  - B = fair, visible wear but fully functional
  - C = poor, significant damage or defects
  - D = heavily damaged, may not be functional
- "defect_type": one of "none", "scuff", "crack", "stain", "tear", "dent", "discoloration", "missing_part", "screen_damage", "water_damage", "functional_fault", "other"
  - Use "none" if the product appears undamaged (grade A+ or A)
- "confidence": float 0.0-1.0 how confident you are in your assessment
- "description": brief description of the product condition (1-2 sentences)

Important: If the product looks new or undamaged, grade it A+ or A with defect_type "none".
Only report defects you can actually see in the image.{_verification_prompt_suffix(expected_color=expected_color, product_title=product_title, expected_size=expected_size)}

Return ONLY valid JSON, no other text."""


def _build_verification(
    parsed: dict,
    *,
    expected_color: str | None,
    product_title: str | None,
) -> dict | None:
    """Assemble the verification block from the parsed grade response. Returns
    None when the caller sent no expected context (additive / unaffected)."""
    if not (expected_color or product_title):
        return None
    states = {"match", "mismatch", "unknown"}
    observed = parsed.get("observed_color")
    observed = observed.strip() if isinstance(observed, str) and observed.strip() else None

    color_match = parsed.get("color_match")
    if not expected_color:
        # Nothing to compare against → never assert a (mis)match on colour.
        color_match = "unknown"
    elif color_match not in states:
        if observed:
            e, o = expected_color.strip().lower(), observed.lower()
            color_match = "match" if (e == o or e in o or o in e) else "mismatch"
        else:
            color_match = "unknown"

    item_match = parsed.get("item_match")
    if item_match not in states:
        item_match = "unknown"

    return Verification(
        color_match=color_match, item_match=item_match,
        observed_color=observed, expected_color=expected_color,
    ).model_dump(mode="json")


def _parse_grading_response(response_text: str, category: str) -> dict:
    """Parse the Bedrock response into structured fields."""
    try:
        # Strip markdown code fences if present (Bedrock often wraps in ```json ... ```)
        text = response_text.strip()
        if text.startswith("```"):
            first_newline = text.index("\n")
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
        grade = data.get("grade", "B")
        if grade not in GRADE_NUMERIC:
            grade = "B"
        defect_type = data.get("defect_type", "other")
        valid_defects = {
            "none", "scuff", "crack", "stain", "tear", "dent", "discoloration",
            "missing_part", "screen_damage", "water_damage", "functional_fault", "other"
        }
        if defect_type not in valid_defects:
            defect_type = "other"
        # Map "none" to indicate no damage found
        no_damage = defect_type == "none"
        if no_damage:
            defect_type = "other"
        confidence = float(data.get("confidence", 0.75))
        confidence = max(0.0, min(1.0, confidence))
        description = data.get("description", "")
        return {
            "grade": grade,
            "defect_type": defect_type,
            "confidence": confidence,
            "no_damage": no_damage,
            "description": description,
            # Additive order-vs-item verification fields (present only when the
            # prompt asked for them; ignored otherwise).
            "observed_color": data.get("observed_color"),
            "color_match": data.get("color_match"),
            "item_match": data.get("item_match"),
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        # Fallback: return conservative defaults
        return {"grade": "B", "defect_type": "other", "confidence": 0.6, "no_damage": False, "description": ""}


def _infer_vertical(category: str) -> str:
    category_lower = category.lower()
    electronics_terms = {
        "phone", "headphone", "laptop", "screen", "camera", "tablet", "electronics",
    }
    if any(term in category_lower for term in electronics_terms):
        return "electronics"
    return "fashion"


def _disposition_for_grade(grade: str) -> str:
    if grade in {"A+", "A"}:
        return "restock"
    if grade in {"B+", "B"}:
        return "p2p_resale"
    if grade == "C":
        return "rescue"
    return "recycle"


def _severity_for_grade(grade: str) -> str:
    if grade in {"A+", "A", "B+"}:
        return "minor"
    if grade in {"B", "C"}:
        return "moderate"
    return "major"


def escalate_if_needed(
    *,
    cnn_passport: ConditionPassport,
    image_bytes: bytes,
    unit_id: str,
    category: str,
    return_id: str | None,
) -> ConditionPassport:
    """Check CNN confidence and escalate to Bedrock T2/T3 if below thresholds.

    Returns the original CNN passport if confidence is high enough,
    or a Bedrock-graded passport if escalation is triggered.
    Falls back gracefully to the CNN result if Bedrock is unavailable.
    """
    from app.core.config import settings

    # T1 confidence sufficient — no escalation needed
    if cnn_passport.confidence >= settings.confidence_threshold_t2:
        return cnn_passport

    # T2 escalation: try Bedrock (Haiku / Nova Lite)
    if settings.bedrock_model_t2:
        try:
            t2_passport = _grade_with_bedrock(
                image_bytes=image_bytes,
                unit_id=unit_id,
                category=category,
                return_id=return_id,
                model_id=settings.bedrock_model_t2,
                tier_label="T2",
            )
            # If T2 confidence is high enough, return it
            if t2_passport.confidence >= settings.confidence_threshold_t3:
                return t2_passport

            # T3 escalation: try Nova Pro
            if settings.bedrock_model_t3:
                try:
                    return _grade_with_bedrock(
                        image_bytes=image_bytes,
                        unit_id=unit_id,
                        category=category,
                        return_id=return_id,
                        model_id=settings.bedrock_model_t3,
                        tier_label="T3",
                    )
                except BedrockGradingError:
                    return t2_passport  # T3 failed, use T2 result

            return t2_passport
        except BedrockGradingError:
            # T2 failed, fall back to CNN result
            return cnn_passport

    # No Bedrock models configured, fall back to CNN result
    return cnn_passport


def _grade_with_bedrock(
    *,
    image_bytes: bytes,
    unit_id: str,
    category: str,
    return_id: str | None,
    model_id: str,
    tier_label: str,
) -> ConditionPassport:
    """Grade an image using a specific Bedrock model."""
    from app.core.config import settings

    media_hash = sha256_hex(image_bytes)

    try:
        import boto3

        client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
        )

        prompt = _build_grading_prompt(category)

        # Detect image format from bytes
        image_format = "jpeg"
        if image_bytes[:8].startswith(b"\x89PNG\r\n\x1a\n"):
            image_format = "png"
        elif image_bytes[:2] in (b"\xff\xd8",):
            image_format = "jpeg"

        response = client.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": image_format,
                                "source": {"bytes": image_bytes},
                            }
                        },
                        {"text": prompt},
                    ],
                }
            ],
            inferenceConfig={
                "maxTokens": 1024,
                "temperature": 0.1,
            },
        )

        response_text = response["output"]["message"]["content"][0]["text"]
        parsed = _parse_grading_response(response_text, category)

        grade = parsed["grade"]
        defect_type = parsed["defect_type"]
        confidence = parsed["confidence"]

        body = {
            "schema_version": "1.0.0",
            "unit_id": unit_id,
            "return_id": return_id,
            "grade": grade,
            "grade_numeric": GRADE_NUMERIC.get(grade, 0.5),
            "category": category.strip().lower().replace("-", "_").replace(" ", "_") or "unknown",
            "vertical": _infer_vertical(category),
            "disposition_hint": _disposition_for_grade(grade),
            "defects": [
                Defect(
                    type=defect_type,
                    severity=_severity_for_grade(grade),
                    confidence=confidence,
                    description=f"Bedrock {tier_label} ({model_id}) detected {defect_type.replace('_', ' ')}.",
                ).model_dump(mode="json")
            ],
            "packaging_state": "opened",
            "confidence": confidence,
            "media_hashes": [media_hash],
            "passport_hash": "",
            "graded_at": datetime.now(timezone.utc),
            "model_tier_used": tier_label,
            "warranty_months_remaining": 0,
            "repair_events": [],
        }
        body["passport_hash"] = passport_hash(body)
        return ConditionPassport.model_validate(body)

    except ImportError as exc:
        raise BedrockGradingError(
            "boto3 is not installed. Install it for Bedrock escalation."
        ) from exc
    except BedrockGradingError:
        raise
    except Exception as exc:
        raise BedrockGradingError(f"Bedrock {tier_label} grading failed: {exc}") from exc


def grade_multi_image_bedrock(
    *,
    images: list[bytes],
    unit_id: str,
    category: str,
    return_id: str | None,
    expected_size: str | None = None,
    expected_color: str | None = None,
    product_title: str | None = None,
) -> ConditionPassport:
    """Grade a product from multiple angle images in a single Bedrock call.

    Sends all images together so the model can assess overall condition
    from multiple perspectives. Worst defect wins (like video aggregation).

    ``expected_*`` are ADDITIVE order-vs-item context → a ``verification`` block
    (prompt-only, no extra image/call).
    """
    from app.core.config import settings

    if not images:
        raise BedrockGradingError("No images provided.")

    # Compute media hashes for all images
    media_hashes = [sha256_hex(img) for img in images]

    try:
        import boto3

        client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
        )

        prompt = f"""You are an AI product condition grader for a circular commerce platform.

You are shown {len(images)} images of the SAME product from different angles (category: {category}).
Assess the overall condition considering ALL angles together.

Return a JSON object with:
- "grade": one of "A+", "A", "B+", "B", "C", "D"
  - A+ = pristine/sealed, looks brand new from all angles
  - A = like new, minimal signs of use across all views
  - B+ = good condition, light cosmetic wear only
  - B = fair, visible wear but fully functional
  - C = poor, significant damage or defects visible
  - D = heavily damaged, may not be functional
- "defect_type": one of "none", "scuff", "crack", "stain", "tear", "dent", "discoloration", "missing_part", "screen_damage", "water_damage", "functional_fault", "other"
  - Use "none" if the product appears undamaged from all angles
  - Report the WORST defect if multiple are visible across different angles
- "confidence": float 0.0-1.0 (higher with more angles confirming the assessment)
- "description": brief description covering what you see across all angles

Important: Multiple angles should INCREASE your confidence. If one angle shows damage not visible from others, report it.
Only report defects you can actually see. If the product looks new from all angles, grade A+ or A with "none".{_verification_prompt_suffix(expected_color=expected_color, product_title=product_title, expected_size=expected_size)}

Return ONLY valid JSON, no other text."""

        # Build content array with all images + prompt
        content = []
        for img_bytes in images:
            image_format = "jpeg"
            if img_bytes[:8].startswith(b"\x89PNG\r\n\x1a\n"):
                image_format = "png"
            content.append(
                {"image": {"format": image_format, "source": {"bytes": img_bytes}}}
            )
        content.append({"text": prompt})

        response = client.converse(
            modelId=settings.bedrock_model_t1,
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"maxTokens": 1024, "temperature": 0.1},
        )

        response_text = response["output"]["message"]["content"][0]["text"]
        parsed = _parse_grading_response(response_text, category)

        grade = parsed["grade"]
        defect_type = parsed["defect_type"]
        confidence = parsed["confidence"]
        no_damage = parsed.get("no_damage", False)
        description = parsed.get("description", "")

        defects = []
        if not no_damage:
            defects = [
                Defect(
                    type=defect_type,
                    severity=_severity_for_grade(grade),
                    confidence=confidence,
                    description=description or f"Assessed from {len(images)} angles.",
                ).model_dump(mode="json")
            ]

        body = {
            "schema_version": "1.0.0",
            "unit_id": unit_id,
            "return_id": return_id,
            "grade": grade,
            "grade_numeric": GRADE_NUMERIC.get(grade, 0.5),
            "category": category.strip().lower().replace("-", "_").replace(" ", "_") or "unknown",
            "vertical": _infer_vertical(category),
            "disposition_hint": _disposition_for_grade(grade),
            "defects": defects,
            "packaging_state": "sealed" if grade in ("A+", "A") else "opened",
            "confidence": confidence,
            "media_hashes": media_hashes,
            "passport_hash": "",
            "graded_at": datetime.now(timezone.utc),
            "model_tier_used": f"bedrock-only+{len(images)}angles",
            "warranty_months_remaining": 0,
            "repair_events": [],
        }
        verification = _build_verification(
            parsed, expected_color=expected_color, product_title=product_title,
        )
        if verification is not None:
            body["verification"] = verification
        body["passport_hash"] = passport_hash(body)
        return ConditionPassport.model_validate(body)

    except ImportError as exc:
        raise BedrockGradingError("boto3 is not installed.") from exc
    except BedrockGradingError:
        raise
    except Exception as exc:
        raise BedrockGradingError(f"Multi-image Bedrock grading failed: {exc}") from exc
