"""Tests for the multi-model ensemble grader."""

from app.pipelines.defect_detection import DetectedDefect, DetectionResult
from app.pipelines.ensemble_grader import (
    compute_agreement,
    enrich_with_detection,
    enrich_with_reference,
)
from app.pipelines.reference_compare import ComparisonResult
from app.schemas.passport import ConditionPassport, Defect, GradingAudit


def _passport(grade="B", confidence=0.85, defects=None) -> ConditionPassport:
    return ConditionPassport(
        unit_id="test",
        grade=grade,
        grade_numeric={"A+": 1.0, "A": 0.9, "B+": 0.78, "B": 0.68, "C": 0.45, "D": 0.2}[grade],
        category="phone",
        vertical="electronics",
        disposition_hint="p2p_resale",
        defects=defects or [],
        packaging_state="opened",
        confidence=confidence,
        media_hashes=["abc"],
        passport_hash="hash",
        model_tier_used="bedrock-only",
        grading_audit=GradingAudit(confidence_band="auto_pass"),
    )


# --- Detection enrichment ---

def test_detection_agrees_damaged_boosts_confidence():
    p = _passport("C", 0.80, [Defect(type="crack", severity="moderate", confidence=0.8)])
    det = DetectionResult(defects_found=True, detections=[
        DetectedDefect(label="crack", bbox=[0.1, 0.2, 0.4, 0.5], confidence=0.9)
    ], model_id="nova")
    enrich_with_detection(p, det)
    assert p.confidence > 0.80
    assert p.defects[0].bbox == [0.1, 0.2, 0.4, 0.5]


def test_detection_agrees_clean_boosts_confidence():
    p = _passport("A", 0.90)
    det = DetectionResult(defects_found=False, detections=[], model_id="nova")
    enrich_with_detection(p, det)
    assert p.confidence > 0.90


def test_detection_disagrees_lowers_confidence():
    p = _passport("A", 0.90)  # says clean
    det = DetectionResult(defects_found=True, detections=[
        DetectedDefect(label="scratch", bbox=[0.5, 0.5, 0.8, 0.8], confidence=0.7)
    ], model_id="nova")
    enrich_with_detection(p, det)
    assert p.confidence < 0.90
    assert p.grading_audit.confidence_band == "needs_review"


def test_detection_error_no_change():
    p = _passport("B", 0.85)
    det = DetectionResult(defects_found=False, error="timeout", model_id="nova")
    enrich_with_detection(p, det)
    assert p.confidence == 0.85


# --- Reference comparison enrichment ---

def test_reference_agrees_boosts():
    p = _passport("B+", 0.82)
    comp = ComparisonResult(
        differences_found=True, differences=["light scratch on back"],
        overall_condition="minor_wear", confidence=0.88, model_id="nova",
    )
    enrich_with_reference(p, comp)
    assert p.confidence > 0.82


def test_reference_disagrees_lowers():
    p = _passport("A", 0.90)  # says pristine
    comp = ComparisonResult(
        differences_found=True, differences=["large crack on screen"],
        overall_condition="heavy_damage", confidence=0.85, model_id="nova",
    )
    enrich_with_reference(p, comp)
    assert p.confidence < 0.90
    assert p.grading_audit.confidence_band == "needs_review"


def test_reference_error_no_change():
    p = _passport("B", 0.85)
    comp = ComparisonResult(
        differences_found=False, differences=[], overall_condition="unknown",
        confidence=0.0, error="no_reference",
    )
    enrich_with_reference(p, comp)
    assert p.confidence == 0.85


# --- Agreement computation ---

def test_full_agreement():
    p = _passport("C", 0.80, [Defect(type="crack", severity="moderate", confidence=0.8)])
    det = DetectionResult(defects_found=True, detections=[], model_id="nova")
    comp = ComparisonResult(
        differences_found=True, differences=["crack"],
        overall_condition="moderate_damage", confidence=0.85,
    )
    assert compute_agreement(p, det, comp) == "full"


def test_partial_agreement():
    p = _passport("B", 0.80, [Defect(type="scuff", severity="minor", confidence=0.7)])
    det = DetectionResult(defects_found=True, detections=[], model_id="nova")
    comp = ComparisonResult(
        differences_found=False, differences=[],
        overall_condition="pristine", confidence=0.85,
    )
    assert compute_agreement(p, det, comp) == "partial"


def test_no_extra_signals():
    p = _passport("B", 0.80)
    assert compute_agreement(p, None, None) == "partial"
