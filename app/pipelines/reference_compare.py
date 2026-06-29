"""Reference-based condition comparison.

Sends both the RETURNED item photo AND a REFERENCE (catalogue/pristine) image
to Nova in one call, asking the model to compare and identify differences.

This is the AWS-recommended approach for visual inspection (see
aws-samples/sample-generative-visual-inspection). Having a baseline to compare
against produces more precise defect descriptions and fewer false positives.

Usage:
- relay-api stores product catalogue images in S3 (`products.image_url`)
- When grading, fetch the reference image and send both to this module
- The comparison result enriches the ConditionPassport with what CHANGED
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonResult:
    """Result of reference-based condition comparison."""
    differences_found: bool
    differences: list[str]  # human-readable difference descriptions
    overall_condition: str  # "pristine", "minor_wear", "moderate_damage", "heavy_damage"
    confidence: float
    model_id: str = ""
    error: str | None = None


def compare_with_reference(
    *,
    return_image_bytes: bytes,
    reference_image_bytes: bytes,
    category: str,
    region: str = "us-east-1",
    model_id: str = "amazon.nova-lite-v1:0",
) -> ComparisonResult:
    """Compare a returned item against its pristine reference image.

    Sends both images in one Bedrock call. The model identifies what's
    different between the returned item and the reference.
    """
    prompt = f"""You are comparing a RETURNED product against its PRISTINE reference.

Image 1 = the RETURNED item (may have wear, defects, or damage)
Image 2 = the PRISTINE reference (brand new, perfect condition)

Category: {category}

Compare the two images and identify any differences in condition:
- Scratches, dents, stains, tears, cracks not present in the reference
- Missing parts or accessories
- Color changes, fading, discoloration
- Packaging differences (sealed vs opened)

Return a JSON object:
{{
  "differences_found": true/false,
  "differences": ["brief description of each difference"],
  "overall_condition": "pristine" | "minor_wear" | "moderate_damage" | "heavy_damage",
  "confidence": 0.0-1.0
}}

If both images look identical (no visible differences): differences_found=false, overall_condition="pristine".

Return ONLY valid JSON, no other text."""

    try:
        import boto3

        client = boto3.client("bedrock-runtime", region_name=region)

        # Detect formats
        def _fmt(b: bytes) -> str:
            if b[:8].startswith(b"\x89PNG\r\n\x1a\n"):
                return "png"
            return "jpeg"

        response = client.converse(
            modelId=model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"image": {"format": _fmt(return_image_bytes), "source": {"bytes": return_image_bytes}}},
                    {"image": {"format": _fmt(reference_image_bytes), "source": {"bytes": reference_image_bytes}}},
                    {"text": prompt},
                ],
            }],
            inferenceConfig={"maxTokens": 1024, "temperature": 0.1},
        )

        text = response["output"]["message"]["content"][0]["text"]
        return _parse_comparison(text, model_id)

    except Exception as exc:
        return ComparisonResult(
            differences_found=False, differences=[], overall_condition="unknown",
            confidence=0.0, model_id=model_id, error=str(exc),
        )


def _parse_comparison(text: str, model_id: str) -> ComparisonResult:
    """Parse the comparison response."""
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
        return ComparisonResult(
            differences_found=False, differences=[], overall_condition="unknown",
            confidence=0.0, model_id=model_id, error="parse_failed",
        )

    diffs_found = bool(data.get("differences_found", False))
    diffs = data.get("differences", [])
    if not isinstance(diffs, list):
        diffs = [str(diffs)] if diffs else []
    diffs = [str(d)[:200] for d in diffs]  # cap length

    condition = str(data.get("overall_condition", "unknown"))
    valid_conditions = {"pristine", "minor_wear", "moderate_damage", "heavy_damage"}
    if condition not in valid_conditions:
        condition = "unknown"

    confidence = float(data.get("confidence", 0.7))
    confidence = max(0.0, min(1.0, confidence))

    return ComparisonResult(
        differences_found=diffs_found,
        differences=diffs,
        overall_condition=condition,
        confidence=confidence,
        model_id=model_id,
    )


# Condition → grade mapping for the reference-based approach
CONDITION_GRADE_MAP = {
    "pristine": ("A+", 1.0),
    "minor_wear": ("B+", 0.78),
    "moderate_damage": ("C", 0.45),
    "heavy_damage": ("D", 0.2),
    "unknown": ("B", 0.68),
}
