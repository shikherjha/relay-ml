from fastapi.testclient import TestClient

from app.main import app
from app.schemas.passport import ConditionPassport, Defect


client = TestClient(app)


def test_condition_passport_contract_fields() -> None:
    passport = ConditionPassport(
        unit_id="unit-1",
        return_id=None,
        grade="B",
        grade_numeric=0.68,
        category="jeans",
        vertical="fashion",
        disposition_hint="p2p_resale",
        defects=[
            Defect(
                type="stain",
                severity="minor",
                confidence=0.81,
                description="Small faint mark.",
            )
        ],
        packaging_state="missing",
        confidence=0.88,
        media_hashes=[
            "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        ],
        passport_hash="0" * 64,
        model_tier_used="cnn-v1",
    )

    assert passport.schema_version == "1.0.0"
    assert passport.packaging_state == "missing"
    assert passport.defects[0].confidence == 0.81


def test_grade_image_rejects_unsupported_content_type() -> None:
    response = client.post(
        "/grade-image",
        data={"unit_id": "unit-1", "category": "jeans"},
        files={"image": ("image.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only JPEG and PNG images are supported."
