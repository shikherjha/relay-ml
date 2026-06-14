from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.passport import ConditionPassport, Defect


client = TestClient(app)


def _passport(
    *,
    unit_id: str,
    grade: str,
    defect_type: str,
    confidence: float,
    media_hash: str,
) -> ConditionPassport:
    return ConditionPassport(
        unit_id=unit_id,
        return_id=None,
        grade=grade,
        grade_numeric={"A": 0.9, "C": 0.45, "D": 0.2}[grade],
        category="electronics",
        vertical="electronics",
        disposition_hint={"A": "restock", "C": "rescue", "D": "recycle"}[grade],
        defects=[
            Defect(
                type=defect_type,
                severity={"A": "minor", "C": "moderate", "D": "major"}[grade],
                confidence=confidence,
                description=f"{defect_type} evidence.",
            )
        ],
        packaging_state="opened",
        confidence=confidence,
        media_hashes=[media_hash],
        passport_hash="0" * 64,
        graded_at=datetime.now(timezone.utc),
        model_tier_used="cnn-v1",
    )


def test_grade_video_aggregates_keyframe_results(monkeypatch) -> None:
    from app.pipelines import video_keyframes
    from app.core import config

    # Force CNN mode for this test (monkeypatches grade_image_bytes)
    monkeypatch.setattr(config.settings, "grading_mode", "cnn")

    monkeypatch.setattr(
        video_keyframes,
        "extract_keyframe_images",
        lambda **_: [b"frame-1", b"frame-2", b"frame-3"],
    )

    passports = [
        _passport(
            unit_id="unit-1:frame:1",
            grade="A",
            defect_type="other",
            confidence=0.9,
            media_hash="1" * 64,
        ),
        _passport(
            unit_id="unit-1:frame:2",
            grade="D",
            defect_type="crack",
            confidence=0.82,
            media_hash="2" * 64,
        ),
        _passport(
            unit_id="unit-1:frame:3",
            grade="C",
            defect_type="crack",
            confidence=0.91,
            media_hash="3" * 64,
        ),
    ]

    def fake_grade_image_bytes(**kwargs):
        return passports.pop(0)

    monkeypatch.setattr(video_keyframes, "grade_image_bytes", fake_grade_image_bytes)

    response = client.post(
        "/grade-video",
        data={"unit_id": "unit-1", "category": "electronics"},
        files={"video": ("return.mp4", b"video-bytes", "video/mp4")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["unit_id"] == "unit-1"
    assert data["grade"] == "D"
    assert data["model_tier_used"] == "cnn-v1+keyframes"
    assert data["defects"][0]["type"] == "crack"
    assert data["defects"][0]["severity"] == "major"
    assert data["defects"][0]["confidence"] == 0.91
    assert data["media_hashes"][0] != "1" * 64


def test_grade_video_rejects_unsupported_content_type() -> None:
    response = client.post(
        "/grade-video",
        data={"unit_id": "unit-1", "category": "electronics"},
        files={"video": ("return.txt", b"not a video", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only MP4, MOV, WEBM, and AVI videos are supported."
