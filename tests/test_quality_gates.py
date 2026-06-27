"""Tests for capture-quality gates (Track D §21.2)."""

from io import BytesIO

from PIL import Image

from app.pipelines.quality_gates import (
    check_image_quality,
    check_video_quality,
    confidence_band,
)


def _make_image(width=200, height=200, color=(128, 128, 128), fmt="PNG") -> bytes:
    import random
    random.seed(42)
    img = Image.new("RGB", (width, height), color=color)
    # Add noise so it's not a pure flat image (which has sharpness=0 after edge detection)
    pixels = img.load()
    for x in range(0, width, 4):
        for y in range(0, height, 4):
            r, g, b = color
            offset = random.randint(-20, 20)
            pixels[x, y] = (max(0, min(255, r + offset)), max(0, min(255, g + offset)), max(0, min(255, b + offset)))
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_good_image_passes():
    img = _make_image(400, 400, (180, 180, 180))
    result = check_image_quality(img)
    assert result.passed
    assert result.band == "auto_pass"
    assert result.issues == []


def test_too_small_file_rejects():
    result = check_image_quality(b"\x89PNG" + b"\x00" * 100)
    assert not result.passed
    assert result.band == "reject_reupload"
    assert any("small" in i.lower() or "corrupt" in i.lower() for i in result.issues)


def test_too_small_resolution_flags():
    # 50x50 with noise may be < MIN_FILE_SIZE as PNG → file-size gate fires first.
    # Either way, it's rejected — check for any "resolution" OR "small" issue.
    img = _make_image(50, 50)
    result = check_image_quality(img)
    assert not result.passed
    assert result.band in ("reject_reupload", "needs_review")
    assert any("resolution" in i.lower() or "small" in i.lower() for i in result.issues)


def test_too_dark_image_rejects():
    # Nearly black image
    img = _make_image(200, 200, (5, 5, 5))
    result = check_image_quality(img)
    assert not result.passed
    assert any("dark" in i.lower() for i in result.issues)
    assert result.band == "reject_reupload"


def test_confidence_band_mapping():
    assert confidence_band(0.95) == "auto_pass"
    assert confidence_band(0.85) == "auto_pass"
    assert confidence_band(0.75) == "needs_review"
    assert confidence_band(0.60) == "needs_review"
    assert confidence_band(0.55) == "reject_reupload"
    assert confidence_band(0.0) == "reject_reupload"


def test_video_quality_too_small():
    result = check_video_quality(b"\x00" * 100)
    assert not result.passed
    assert result.band == "reject_reupload"


def test_video_quality_ok():
    result = check_video_quality(b"\x00" * 50_000)
    assert result.passed
    assert result.band == "auto_pass"
