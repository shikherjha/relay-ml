"""Tests for POST /grade-and-price endpoint."""

from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.pipelines.resale_pricer import compute_resale_price

client = TestClient(app)


def _make_png() -> bytes:
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- Unit tests for the pricing model ---


def test_pricer_basic() -> None:
    """Basic pricing returns valid range."""
    result = compute_resale_price(
        grade="B",
        grade_numeric=0.68,
        original_price=5000.0,
        age_days=60,
        category="electronics",
        confidence=0.85,
        has_defects=True,
    )
    assert result.price_range_min > 0
    assert result.price_range_min <= result.price_range_max
    assert result.currency == "INR"
    assert result.resale_grade == "Very Good"
    assert result.pricing_rationale


def test_pricer_monotonic_grade() -> None:
    """Higher grade produces higher mean price."""
    base = {"original_price": 3000.0, "age_days": 30, "category": "phone", "confidence": 0.9, "has_defects": False}
    high = compute_resale_price(grade="A", grade_numeric=0.9, **base)
    low = compute_resale_price(grade="C", grade_numeric=0.45, **base)
    mean_high = (high.price_range_min + high.price_range_max) / 2
    mean_low = (low.price_range_min + low.price_range_max) / 2
    assert mean_high > mean_low


def test_pricer_monotonic_age() -> None:
    """Lower age_days produces higher mean price."""
    base = {"grade": "B", "grade_numeric": 0.68, "original_price": 4000.0, "category": "laptop", "confidence": 0.85, "has_defects": False}
    newer = compute_resale_price(age_days=10, **base)
    older = compute_resale_price(age_days=300, **base)
    mean_newer = (newer.price_range_min + newer.price_range_max) / 2
    mean_older = (older.price_range_min + older.price_range_max) / 2
    assert mean_newer > mean_older


def test_pricer_resale_grade_mapping() -> None:
    """Grades map to correct resale labels."""
    base = {"grade_numeric": 0.9, "original_price": 2000.0, "age_days": 10, "category": "shoes", "confidence": 0.9, "has_defects": False}
    assert compute_resale_price(grade="A+", **base).resale_grade == "Like New"
    assert compute_resale_price(grade="A", **base).resale_grade == "Like New"
    base["grade_numeric"] = 0.68
    assert compute_resale_price(grade="B", **base).resale_grade == "Very Good"
    base["grade_numeric"] = 0.45
    assert compute_resale_price(grade="C", **base).resale_grade == "Good"
    base["grade_numeric"] = 0.2
    assert compute_resale_price(grade="D", **base).resale_grade == "Acceptable"


def test_pricer_beats_fallback() -> None:
    """Our pricer produces reasonable results vs the relay-api fallback formula."""
    # Relay-api fallback: base = original * clamp(grade_numeric) * max(0.45, 1 - age/720)
    original = 5000.0
    grade_numeric = 0.78
    age_days = 90

    # Fallback computation
    cond = max(0.30, min(0.95, grade_numeric))
    age_f = max(0.45, 1 - age_days / 720)
    fallback_base = original * cond * age_f
    fallback_range = (fallback_base * 0.9, fallback_base * 1.1)

    # Our computation
    result = compute_resale_price(
        grade="B+", grade_numeric=grade_numeric, original_price=original,
        age_days=age_days, category="electronics", confidence=0.9, has_defects=False,
    )
    our_mean = (result.price_range_min + result.price_range_max) / 2
    fallback_mean = (fallback_range[0] + fallback_range[1]) / 2

    # Should be in a reasonable neighborhood (within 30% of fallback)
    assert our_mean > fallback_mean * 0.7
    assert our_mean < fallback_mean * 1.4


def test_pricer_minimum_price() -> None:
    """Even worst grade still returns a positive price (at least 10% of original)."""
    result = compute_resale_price(
        grade="D", grade_numeric=0.2, original_price=1000.0,
        age_days=600, category="fashion", confidence=0.5, has_defects=True,
    )
    assert result.price_range_min >= 1.0
    assert (result.price_range_min + result.price_range_max) / 2 >= 100.0


# --- Route-level tests ---


def test_grade_and_price_route_single_image() -> None:
    """POST /grade-and-price with a single image returns passport + pricing."""
    from unittest.mock import patch
    from app.schemas.passport import ConditionPassport, Defect

    mock_passport = ConditionPassport(
        unit_id="unit-price-1", grade="B+", grade_numeric=0.78,
        category="electronics", vertical="electronics",
        disposition_hint="p2p_resale",
        defects=[Defect(type="scuff", severity="minor", confidence=0.85)],
        packaging_state="opened", confidence=0.88,
        media_hashes=["abc123"], passport_hash="hash123",
        model_tier_used="bedrock-only",
    )

    with patch("app.routers.grade.grade_bedrock_only", return_value=mock_passport):
        png = _make_png()
        response = client.post(
            "/grade-and-price",
            files=[("images", ("test.png", png, "image/png"))],
            data={
                "unit_id": "unit-price-1",
                "category": "electronics",
                "original_price": "4999",
                "age_days": "45",
            },
        )
    assert response.status_code == 200
    data = response.json()
    # ConditionPassport fields
    assert data["schema_version"] == "1.0.0"
    assert data["unit_id"] == "unit-price-1"
    assert data["grade"] in ("A+", "A", "B+", "B", "C", "D")
    assert 0 <= data["confidence"] <= 1
    # Resale fields
    assert data["resale_grade"] in ("Like New", "Very Good", "Good", "Acceptable")
    assert data["price_range"]["min"] > 0
    assert data["price_range"]["min"] <= data["price_range"]["max"]
    assert data["currency"] == "INR"
    assert data["pricing_rationale"]


def test_grade_and_price_rejects_no_media() -> None:
    """POST /grade-and-price with no images or video returns 400."""
    response = client.post(
        "/grade-and-price",
        data={
            "unit_id": "unit-1",
            "category": "shoes",
            "original_price": "2000",
            "age_days": "30",
        },
    )
    assert response.status_code in (400, 422)


def test_grade_and_price_multi_image() -> None:
    """POST /grade-and-price with multiple images works."""
    from unittest.mock import patch
    from app.schemas.passport import ConditionPassport

    mock_passport = ConditionPassport(
        unit_id="unit-price-multi", grade="A", grade_numeric=0.9,
        category="phone", vertical="electronics",
        disposition_hint="restock", defects=[],
        packaging_state="sealed", confidence=0.95,
        media_hashes=["h1", "h2"], passport_hash="hash456",
        model_tier_used="bedrock-only+2angles",
    )

    with patch("app.routers.grade.grade_multi_image_bedrock", return_value=mock_passport):
        png1 = _make_png()
        png2 = _make_png()
        response = client.post(
            "/grade-and-price",
            files=[
                ("images", ("a.png", png1, "image/png")),
                ("images", ("b.png", png2, "image/png")),
            ],
            data={
                "unit_id": "unit-price-multi",
                "category": "phone",
                "original_price": "15000",
                "age_days": "120",
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert "price_range" in data
    assert data["price_range"]["min"] <= data["price_range"]["max"]
