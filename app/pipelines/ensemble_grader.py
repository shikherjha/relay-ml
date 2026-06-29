"""Multi-model ensemble grader.

Combines three signals for more accurate and explainable grading:
1. Nova Lite holistic grade (existing bedrock_tiers.grade_bedrock_only)
2. Nova defect detection with bounding boxes (defect_detection.py)
3. Reference-based comparison when a catalogue image is available (reference_compare.py)

Agreement between models boosts confidence. Disagreement lowers it and
flags the passport for review. This is the "approach 1 + 3 + 4" combo.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.pipelines.defect_detection import DetectionResult
from app.pipelines.reference_compare import CONDITION_GRADE_MAP, ComparisonResult
from app.schemas.passport import ConditionPassport, GradingAudit


GRADE_NUMERIC = {
    "A+": 1.0, "A": 0.9, "B+": 0.78, "B": 0.68, "C": 0.45, "D": 0.2,
}


@dataclass
class EnsembleResult:
    """Result of ensemble aggregation."""
    passport: ConditionPassport
    detection_used: bool
    reference_used: bool
    agreement: str  # "full", "partial", "disagree"


def enrich_with_detection(passport: ConditionPassport, detection: DetectionResult) -> None:
    """Enrich a passport with defect detection results (bboxes + confidence adjustment).
    
    Rules:
    - Detection found defects + grading agrees → boost confidence, add bboxes
    - Detection found nothing + grading says undamaged → boost confidence
    - Detection found defects + grading says undamaged → lower confidence (disagreement)
    - Detection found nothing + grading says damaged → lower confidence (possible hallucination)
    """
    if detection.error:
        return  # detection failed, don't adjust

    grading_says_damaged = passport.grade.value in ("C", "D") or len(passport.defects) > 0
    detection_says_damaged = detection.defects_found

    if detection_says_damaged and grading_says_damaged:
        # Agreement: both say damaged → boost confidence + add bbox
        passport.confidence = min(0.98, passport.confidence + 0.08)
        if detection.detections and passport.defects:
            # Attach the best bbox to the first defect
            best = max(detection.detections, key=lambda d: d.confidence)
            passport.defects[0].bbox = best.bbox

    elif not detection_says_damaged and not grading_says_damaged:
        # Agreement: both say clean → boost confidence
        passport.confidence = min(0.99, passport.confidence + 0.05)

    elif detection_says_damaged and not grading_says_damaged:
        # Disagreement: detection sees damage but grading doesn't → lower confidence
        passport.confidence = max(0.4, passport.confidence - 0.15)
        if passport.grading_audit:
            passport.grading_audit.confidence_band = "needs_review"

    elif not detection_says_damaged and grading_says_damaged:
        # Disagreement: grading says damaged but detection sees nothing → lower confidence
        passport.confidence = max(0.5, passport.confidence - 0.10)
        if passport.grading_audit:
            passport.grading_audit.confidence_band = "needs_review"


def enrich_with_reference(passport: ConditionPassport, comparison: ComparisonResult) -> None:
    """Enrich a passport with reference-based comparison results.
    
    Rules:
    - If reference says "pristine" and grade is low → boost grade toward A
    - If reference says "heavy_damage" and grade is high → lower confidence
    - Add comparison differences to the passport description
    """
    if comparison.error or comparison.overall_condition == "unknown":
        return

    ref_grade, ref_numeric = CONDITION_GRADE_MAP.get(
        comparison.overall_condition, ("B", 0.68)
    )
    our_numeric = GRADE_NUMERIC.get(passport.grade.value, 0.68)

    # How far apart are the two assessments?
    diff = abs(our_numeric - ref_numeric)

    if diff <= 0.15:
        # Close agreement → boost confidence
        passport.confidence = min(0.97, passport.confidence + 0.05)
    elif diff > 0.3:
        # Strong disagreement → flag for review
        passport.confidence = max(0.45, passport.confidence - 0.15)
        if passport.grading_audit:
            passport.grading_audit.confidence_band = "needs_review"
            passport.grading_audit.fallback_reason = (
                f"reference_disagrees: ref={comparison.overall_condition} vs grade={passport.grade.value}"
            )

    # Enrich defect descriptions with what changed vs reference
    if comparison.differences and passport.defects:
        existing_desc = passport.defects[0].description or ""
        ref_note = " | Vs reference: " + "; ".join(comparison.differences[:2])
        passport.defects[0].description = (existing_desc + ref_note)[:280]


def compute_agreement(
    passport: ConditionPassport,
    detection: DetectionResult | None,
    comparison: ComparisonResult | None,
) -> str:
    """Determine the level of agreement between the ensemble models."""
    signals = []

    grading_damaged = passport.grade.value in ("B", "C", "D") or len(passport.defects) > 0

    if detection and not detection.error:
        signals.append(detection.defects_found == grading_damaged)

    if comparison and not comparison.error and comparison.overall_condition != "unknown":
        ref_damaged = comparison.overall_condition in ("moderate_damage", "heavy_damage")
        signals.append(ref_damaged == grading_damaged)

    if not signals:
        return "partial"  # only one signal available
    if all(signals):
        return "full"
    if any(signals):
        return "partial"
    return "disagree"
