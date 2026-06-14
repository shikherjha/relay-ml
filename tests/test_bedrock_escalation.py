"""Tests for Bedrock tiered escalation logic."""

from unittest.mock import patch, MagicMock

from app.pipelines.bedrock_tiers import escalate_if_needed, BedrockGradingError
from app.schemas.passport import ConditionPassport, Defect


def _make_passport(confidence: float, model_tier: str = "cnn-v1") -> ConditionPassport:
    """Create a test passport with the given confidence."""
    return ConditionPassport(
        schema_version="1.0.0",
        unit_id="unit-test",
        grade="B",
        grade_numeric=0.68,
        category="fashion",
        vertical="fashion",
        disposition_hint="p2p_resale",
        defects=[
            Defect(type="scuff", severity="minor", confidence=confidence)
        ],
        packaging_state="opened",
        confidence=confidence,
        media_hashes=["abc123"],
        passport_hash="test_hash",
        model_tier_used=model_tier,
    )


def test_no_escalation_when_confidence_high() -> None:
    """If CNN confidence >= threshold_t2, no escalation happens."""
    passport = _make_passport(confidence=0.92)

    with patch("app.core.config.settings") as mock_settings:
        mock_settings.confidence_threshold_t2 = 0.85
        mock_settings.confidence_threshold_t3 = 0.75
        mock_settings.bedrock_model_t2 = "some-model"
        mock_settings.bedrock_model_t3 = "some-model-pro"
        mock_settings.aws_region = "us-east-1"

        result = escalate_if_needed(
            cnn_passport=passport,
            image_bytes=b"fake",
            unit_id="unit-test",
            category="fashion",
            return_id=None,
        )

    assert result.model_tier_used == "cnn-v1"
    assert result.confidence == 0.92


def test_fallback_to_cnn_when_no_bedrock_configured() -> None:
    """If no Bedrock models are configured, returns CNN result even if low confidence."""
    passport = _make_passport(confidence=0.5)

    with patch("app.core.config.settings") as mock_settings:
        mock_settings.confidence_threshold_t2 = 0.85
        mock_settings.confidence_threshold_t3 = 0.75
        mock_settings.bedrock_model_t2 = ""
        mock_settings.bedrock_model_t3 = ""

        result = escalate_if_needed(
            cnn_passport=passport,
            image_bytes=b"fake",
            unit_id="unit-test",
            category="fashion",
            return_id=None,
        )

    assert result.model_tier_used == "cnn-v1"
    assert result.confidence == 0.5


def test_fallback_to_cnn_when_bedrock_fails() -> None:
    """If Bedrock call fails, returns the CNN result gracefully."""
    passport = _make_passport(confidence=0.6)

    with patch("app.pipelines.bedrock_tiers._grade_with_bedrock", side_effect=BedrockGradingError("no creds")):
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.confidence_threshold_t2 = 0.85
            mock_settings.confidence_threshold_t3 = 0.75
            mock_settings.bedrock_model_t2 = "test-model"
            mock_settings.bedrock_model_t3 = ""
            mock_settings.aws_region = "us-east-1"

            result = escalate_if_needed(
                cnn_passport=passport,
                image_bytes=b"fake",
                unit_id="unit-test",
                category="fashion",
                return_id=None,
            )

    assert result.model_tier_used == "cnn-v1"


def test_escalation_to_t2_when_confidence_low() -> None:
    """If CNN confidence < threshold_t2 and Bedrock T2 succeeds, returns T2 result."""
    passport = _make_passport(confidence=0.6)
    t2_passport = _make_passport(confidence=0.88, model_tier="T2")

    with patch("app.pipelines.bedrock_tiers._grade_with_bedrock", return_value=t2_passport):
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.confidence_threshold_t2 = 0.85
            mock_settings.confidence_threshold_t3 = 0.75
            mock_settings.bedrock_model_t2 = "test-model-t2"
            mock_settings.bedrock_model_t3 = ""
            mock_settings.aws_region = "us-east-1"

            result = escalate_if_needed(
                cnn_passport=passport,
                image_bytes=b"fake",
                unit_id="unit-test",
                category="fashion",
                return_id=None,
            )

    assert result.model_tier_used == "T2"
    assert result.confidence == 0.88


def test_escalation_to_t3_when_t2_confidence_low() -> None:
    """If T2 confidence < threshold_t3, escalates to T3."""
    passport = _make_passport(confidence=0.5)
    t2_passport = _make_passport(confidence=0.6, model_tier="T2")
    t3_passport = _make_passport(confidence=0.92, model_tier="T3")

    call_count = {"n": 0}

    def mock_grade(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return t2_passport
        return t3_passport

    with patch("app.pipelines.bedrock_tiers._grade_with_bedrock", side_effect=mock_grade):
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.confidence_threshold_t2 = 0.85
            mock_settings.confidence_threshold_t3 = 0.75
            mock_settings.bedrock_model_t2 = "test-model-t2"
            mock_settings.bedrock_model_t3 = "test-model-t3"
            mock_settings.aws_region = "us-east-1"

            result = escalate_if_needed(
                cnn_passport=passport,
                image_bytes=b"fake",
                unit_id="unit-test",
                category="fashion",
                return_id=None,
            )

    assert result.model_tier_used == "T3"
    assert result.confidence == 0.92
    assert call_count["n"] == 2
