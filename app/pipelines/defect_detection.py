"""Defect detection via Amazon Nova 2 Lite object detection.

Nova 2 Lite has built-in object detection: ask it to find specific defect types
and it returns precise bounding boxes in normalized coordinates. This gives us
visual defect localization that enriches the Condition Passport (where IS the
crack, not just "there's a crack somewhere").

Used as part of the multi-model ensemble:
1. Nova 2 Lite detects + locates defects (this module)
2. Nova Lite grades holistically (bedrock_tiers.py)
3. Agreement between the two boosts/lowers confidence
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DetectedDefect:
    """A localized defect found by object detection."""
    label: str  # crack, dent, stain, tear, scratch, screen_damage, etc.
    bbox: list[float]  # [x1, y1, x2, y2] normalized 0-1
    confidence: float = 0.0


@dataclass
class DetectionResult:
    """Result of defect detection on an image."""
    defects_found: bool
    detections: list[DetectedDefect] = field(default_factory=list)
    model_id: str = ""
    error: str | None = None


# Defect labels to search for (category-specific)
_FASHION_DEFECTS = ["stain", "tear", "hole", "pilling", "discoloration", "missing button", "loose thread"]
_ELECTRONICS_DEFECTS = ["crack", "dent", "scratch", "screen damage", "water damage", "missing part", "broken port"]
_GENERAL_DEFECTS = ["crack", "dent", "stain", "tear", "scratch", "damage"]


def _defect_labels_for(category: str, vertical: str | None) -> list[str]:
    """Select defect labels to detect based on product type."""
    if vertical == "electronics" or category in (
        "laptop", "phone", "smartphone", "headphones", "speaker", "smartwatch",
        "camera", "keyboard", "tablet", "monitor",
    ):
        return _ELECTRONICS_DEFECTS
    if vertical == "fashion" or category in (
        "dress", "jeans", "hoodie", "jacket", "shirt", "tshirt", "sneakers",
        "coat", "skirt", "top",
    ):
        return _FASHION_DEFECTS
    return _GENERAL_DEFECTS


def detect_defects(
    image_bytes: bytes,
    category: str,
    vertical: str | None = None,
    region: str = "us-east-1",
    model_id: str = "amazon.nova-lite-v1:0",
) -> DetectionResult:
    """Detect and localize defects in a product image using Nova.

    Asks the model to find specific defect types and return bounding boxes.
    Falls back gracefully if detection fails (non-fatal — the grade pipeline
    continues without bbox enrichment).
    """
    labels = _defect_labels_for(category, vertical)
    labels_str = ", ".join(f'"{l}"' for l in labels)

    prompt = f"""You are a product defect detection system. Examine this product image (category: {category}).

Look for these defect types: {labels_str}

For each defect you find, return its location as a bounding box.
If the product appears undamaged (no visible defects), return an empty array.

Return ONLY a JSON array:
[{{"label": "defect_type", "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}}]

Where x1,y1 is top-left and x2,y2 is bottom-right, all normalized 0.0-1.0.
If no defects found, return: []

Return ONLY valid JSON, no other text."""

    try:
        import boto3

        client = boto3.client("bedrock-runtime", region_name=region)

        # Detect image format
        image_format = "jpeg"
        if image_bytes[:8].startswith(b"\x89PNG\r\n\x1a\n"):
            image_format = "png"

        response = client.converse(
            modelId=model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"image": {"format": image_format, "source": {"bytes": image_bytes}}},
                    {"text": prompt},
                ],
            }],
            inferenceConfig={"maxTokens": 1024, "temperature": 0.0},
        )

        text = response["output"]["message"]["content"][0]["text"]
        return _parse_detection(text, model_id)

    except Exception as exc:
        return DetectionResult(defects_found=False, error=str(exc), model_id=model_id)


def _parse_detection(text: str, model_id: str) -> DetectionResult:
    """Parse the detection response JSON."""
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        t = t[nl + 1:] if nl != -1 else t[3:]
        if t.endswith("```"):
            t = t[:-3]
    t = t.strip()

    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        # Try to extract array from prose
        lo, hi = t.find("["), t.rfind("]")
        if lo != -1 and hi != -1:
            try:
                data = json.loads(t[lo:hi + 1])
            except json.JSONDecodeError:
                return DetectionResult(defects_found=False, model_id=model_id, error="parse_failed")
        else:
            return DetectionResult(defects_found=False, model_id=model_id, error="parse_failed")

    if not isinstance(data, list):
        return DetectionResult(defects_found=False, model_id=model_id)

    detections = []
    for item in data:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower().replace(" ", "_")
        bbox = item.get("bbox")
        conf = float(item.get("confidence", 0.7))
        if label and isinstance(bbox, list) and len(bbox) == 4:
            try:
                bbox_floats = [float(x) for x in bbox]
                detections.append(DetectedDefect(label=label, bbox=bbox_floats, confidence=conf))
            except (TypeError, ValueError):
                continue

    return DetectionResult(
        defects_found=len(detections) > 0,
        detections=detections,
        model_id=model_id,
    )
