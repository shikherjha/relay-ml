"""Tests for GRADING_MODE support (mock and bedrock_only paths)."""

from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.pipelines.bedrock_tiers import grade_mock

client = TestClient(app)


def _make_test_png() -> bytes:
    """Create a minimal valid PNG image for testing."""
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_mock_mode_returns_valid_passport() -> None:
    """GRADING_MODE=mock returns a deterministic stub passport."""
    image_bytes = _make_test_png()
    passport = grade_mock(
        image_bytes=image_bytes,
        unit_id="unit-mock-1",
        category="fashion",
        return_id=None,
    )
    assert passport.grade.value == "B"
    assert passport.model_tier_used == "mock"
    assert passport.confidence == 0.85
    assert passport.schema_version == "1.0.0"
    assert len(passport.media_hashes) == 1
    assert passport.passport_hash  # non-empty


def test_mock_mode_deterministic() -> None:
    """Same image produces same passport_hash in mock mode."""
    image_bytes = _make_test_png()
    p1 = grade_mock(
        image_bytes=image_bytes, unit_id="u1", category="shoes", return_id=None
    )
    p2 = grade_mock(
        image_bytes=image_bytes, unit_id="u1", category="shoes", return_id=None
    )
    assert p1.media_hashes == p2.media_hashes


def test_mock_mode_route_level() -> None:
    """POST /grade-image with GRADING_MODE=mock returns a valid passport via route."""
    image_bytes = _make_test_png()
    with patch("app.routers.grade.settings") as mock_settings:
        mock_settings.grading_mode = "mock"
        mock_settings.cnn_model_path = "models/grade_cnn_v1.pt"
        response = client.post(
            "/grade-image",
            files={"image": ("test.png", image_bytes, "image/png")},
            data={"unit_id": "unit-route-mock", "category": "dress"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["model_tier_used"] == "mock"
    assert data["grade"] == "B"


def test_mock_mode_rejects_unsupported_content_type() -> None:
    """Mock mode still validates content type."""
    with patch("app.routers.grade.settings") as mock_settings:
        mock_settings.grading_mode = "mock"
        response = client.post(
            "/grade-image",
            files={"image": ("test.gif", b"GIF89a", "image/gif")},
            data={"unit_id": "u1", "category": "shoes"},
        )
    assert response.status_code == 400


def test_bedrock_only_mode_rejects_unsupported_content_type() -> None:
    """bedrock_only mode still validates content type."""
    with patch("app.routers.grade.settings") as mock_settings:
        mock_settings.grading_mode = "bedrock_only"
        response = client.post(
            "/grade-image",
            files={"image": ("test.bmp", b"\x00" * 100, "image/bmp")},
            data={"unit_id": "u1", "category": "phone"},
        )
    assert response.status_code == 400
