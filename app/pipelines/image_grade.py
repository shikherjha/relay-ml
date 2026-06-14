from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import re

from app.models.cnn import CnnPrediction, RelayGradeCnn
from app.schemas.passport import ConditionPassport, Defect, Grade
from app.utils.hash import passport_hash, sha256_hex


SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024

GRADE_NUMERIC = {
    "A+": 1.0,
    "A": 0.9,
    "B+": 0.78,
    "B": 0.68,
    "C": 0.45,
    "D": 0.2,
}

CONTRACT_DEFECT_TYPES = {
    "scuff",
    "crack",
    "stain",
    "tear",
    "dent",
    "discoloration",
    "missing_part",
    "screen_damage",
    "water_damage",
    "functional_fault",
    "other",
}


class ImageValidationError(ValueError):
    pass


class ImageGradingUnavailable(RuntimeError):
    pass


def grade_image_bytes(
    *,
    image_bytes: bytes,
    content_type: str | None,
    unit_id: str,
    category: str,
    return_id: str | None,
    model_path: Path,
) -> ConditionPassport:
    image = _load_image(image_bytes=image_bytes, content_type=content_type)
    prediction = _predict(image=image, model_path=model_path)
    media_hash = sha256_hex(image_bytes)

    defect_type = _normalize_defect_type(prediction.defect_type, category)
    defect_confidence = _clamp(prediction.defect_confidence)
    grade = _normalize_grade(prediction.grade)
    confidence = _clamp((prediction.grade_confidence + defect_confidence) / 2)

    body = {
        "schema_version": "1.0.0",
        "unit_id": unit_id,
        "return_id": return_id,
        "grade": grade.value,
        "grade_numeric": GRADE_NUMERIC[grade.value],
        "category": _normalize_category(category),
        "vertical": _infer_vertical(category),
        "disposition_hint": _disposition_for_grade(grade),
        "defects": [
            Defect(
                type=defect_type,
                severity=_severity_for_grade(grade),
                confidence=defect_confidence,
                description=f"Detected {defect_type.replace('_', ' ')} evidence.",
            ).model_dump(mode="json")
        ],
        "packaging_state": "opened",
        "confidence": confidence,
        "media_hashes": [media_hash],
        "passport_hash": "",
        "graded_at": datetime.now(timezone.utc),
        "model_tier_used": "cnn-v1",
        "warranty_months_remaining": 0,
        "repair_events": [],
    }
    body["passport_hash"] = passport_hash(body)
    return ConditionPassport.model_validate(body)


def _load_image(*, image_bytes: bytes, content_type: str | None):
    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise ImageValidationError("Only JPEG and PNG images are supported.")
    if not image_bytes:
        raise ImageValidationError("Image upload is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageValidationError("Image upload exceeds the 8 MB limit.")

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImageGradingUnavailable(
            "Pillow is not installed. Install requirements.txt before image grading."
        ) from exc

    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ImageValidationError("Upload is not a valid image file.") from exc


def _predict(*, image, model_path: Path) -> CnnPrediction:
    try:
        return RelayGradeCnn(model_path=model_path).predict(image)
    except RuntimeError as exc:
        raise ImageGradingUnavailable(str(exc)) from exc


def _normalize_defect_type(defect_type: str, category: str) -> str:
    normalized = defect_type.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in CONTRACT_DEFECT_TYPES and normalized != "damaged":
        if normalized == "crack" and _is_screen_like(category):
            return "screen_damage"
        return normalized

    if normalized == "damaged":
        if _is_screen_like(category):
            return "screen_damage"
        if _infer_vertical(category) == "electronics":
            return "functional_fault"
        if _is_fabric_like(category):
            return "tear"
        return "dent"

    return "other"


def _normalize_grade(grade: str) -> Grade:
    try:
        return Grade(grade)
    except ValueError:
        return Grade.b


def _normalize_category(category: str) -> str:
    normalized = category.strip().lower().replace("-", "_").replace(" ", "_")
    return "".join(char for char in normalized if char.isalnum() or char == "_") or "unknown"


def _infer_vertical(category: str) -> str:
    category_lower = category.lower()
    electronics_terms = {
        "phone",
        "headphone",
        "laptop",
        "screen",
        "camera",
        "tablet",
        "electronics",
    }
    if any(term in category_lower for term in electronics_terms):
        return "electronics"
    return "fashion"


def _is_screen_like(category: str) -> bool:
    tokens = _category_tokens(category)
    return bool(tokens & {"phone", "smartphone", "screen", "tablet", "laptop", "monitor", "display"})


def _is_fabric_like(category: str) -> bool:
    tokens = _category_tokens(category)
    return bool(
        tokens
        & {
            "apparel",
            "blouse",
            "clothing",
            "denim",
            "dress",
            "fashion",
            "jeans",
            "shirt",
            "skirt",
            "top",
            "trouser",
            "trousers",
        }
    )


def _category_tokens(category: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", category.lower()) if token}


def _disposition_for_grade(grade: Grade) -> str:
    if grade in {Grade.a_plus, Grade.a}:
        return "restock"
    if grade in {Grade.b_plus, Grade.b}:
        return "p2p_resale"
    if grade == Grade.c:
        return "rescue"
    return "recycle"


def _severity_for_grade(grade: Grade) -> str:
    if grade in {Grade.a_plus, Grade.a, Grade.b_plus}:
        return "minor"
    if grade in {Grade.b, Grade.c}:
        return "moderate"
    return "major"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
