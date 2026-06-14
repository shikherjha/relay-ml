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

from app.schemas.passport import ConditionPassport, Defect, Grade
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
) -> ConditionPassport:
    """Grade an image using Bedrock Nova Lite directly (no CNN).

    This is the demo-safe escape hatch: real grades without trained model weights.
    Slower and pricier per call, but always available if AWS credentials are configured.
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
        prompt = _build_grading_prompt(category)

        # Encode image for Bedrock
        import base64

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Call Nova Lite with the image
        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": "png",
                                "source": {"bytes": image_b64},
                            }
                        },
                        {"text": prompt},
                    ],
                }
            ],
            "inferenceConfig": {
                "maxTokens": 1024,
                "temperature": 0.1,
            },
        }

        response = client.converse(
            modelId=settings.bedrock_model_t1,
            messages=request_body["messages"],
            inferenceConfig=request_body["inferenceConfig"],
        )

        # Parse the structured response
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
                    description=f"Bedrock Nova Lite detected {defect_type.replace('_', ' ')}.",
                ).model_dump(mode="json")
            ],
            "packaging_state": "opened",
            "confidence": confidence,
            "media_hashes": [media_hash],
            "passport_hash": "",
            "graded_at": datetime.now(timezone.utc),
            "model_tier_used": "bedrock-only",
            "warranty_months_remaining": 0,
            "repair_events": [],
        }
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
    body["passport_hash"] = passport_hash(body)
    return ConditionPassport.model_validate(body)


def _build_grading_prompt(category: str) -> str:
    """Build the structured extraction prompt for Bedrock."""
    return f"""You are an AI product condition grader for a circular commerce platform.

Analyze this product image (category: {category}) and return a JSON object with:
- "grade": one of "A+", "A", "B+", "B", "C", "D" (A+ = pristine, D = heavily damaged)
- "defect_type": one of "scuff", "crack", "stain", "tear", "dent", "discoloration", "missing_part", "screen_damage", "water_damage", "functional_fault", "other"
- "confidence": float 0.0-1.0 how confident you are in the grade
- "description": brief description of the product condition

Return ONLY valid JSON, no other text."""


def _parse_grading_response(response_text: str, category: str) -> dict:
    """Parse the Bedrock response into structured fields."""
    try:
        # Try to parse as JSON directly
        data = json.loads(response_text.strip())
        grade = data.get("grade", "B")
        if grade not in GRADE_NUMERIC:
            grade = "B"
        defect_type = data.get("defect_type", "other")
        valid_defects = {
            "scuff", "crack", "stain", "tear", "dent", "discoloration",
            "missing_part", "screen_damage", "water_damage", "functional_fault", "other"
        }
        if defect_type not in valid_defects:
            defect_type = "other"
        confidence = float(data.get("confidence", 0.75))
        confidence = max(0.0, min(1.0, confidence))
        return {"grade": grade, "defect_type": defect_type, "confidence": confidence}
    except (json.JSONDecodeError, TypeError, ValueError):
        # Fallback: return conservative defaults
        return {"grade": "B", "defect_type": "other", "confidence": 0.6}


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
        import base64
        import boto3

        client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
        )

        prompt = _build_grading_prompt(category)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        response = client.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": "png",
                                "source": {"bytes": image_b64},
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
