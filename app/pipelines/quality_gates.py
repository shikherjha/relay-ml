"""Capture-quality gates for uploaded media.

Before grading, check if the image is usable. Reject early with a clear
message rather than producing an overconfident passport from bad input.

Gates:
- Too small (< 100×100): likely a thumbnail, not a real photo
- Too dark (mean luminance < 30): can't see defects
- Too blurry (Laplacian variance < threshold): camera shake
- No object detected (optional, future): empty background
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Literal

QualityBand = Literal["auto_pass", "needs_review", "reject_reupload"]

# Thresholds (tuned conservatively — better to let through than over-reject)
MIN_WIDTH = 100
MIN_HEIGHT = 100
MIN_LUMINANCE = 25  # 0-255 scale
MIN_LAPLACIAN_VAR = 50.0  # variance of Laplacian (sharpness)
MIN_FILE_SIZE = 1_000  # 1KB — smaller is likely corrupt or a 1px placeholder


@dataclass(frozen=True)
class QualityResult:
    """Result of capture-quality assessment."""
    passed: bool
    band: QualityBand
    issues: list[str]
    luminance: float | None = None
    sharpness: float | None = None
    width: int | None = None
    height: int | None = None


def confidence_band(confidence: float) -> QualityBand:
    """Map a numeric confidence to a named band."""
    if confidence >= 0.85:
        return "auto_pass"
    elif confidence >= 0.60:
        return "needs_review"
    return "reject_reupload"


def check_image_quality(image_bytes: bytes) -> QualityResult:
    """Run quality gates on image bytes. Returns pass/fail + issues.
    
    Uses only Pillow (no OpenCV dependency) for portability.
    Blurriness is estimated via high-frequency energy in the image.
    """
    issues: list[str] = []
    luminance: float | None = None
    sharpness: float | None = None
    width: int | None = None
    height: int | None = None

    # Gate 0: file size
    if len(image_bytes) < MIN_FILE_SIZE:
        issues.append(f"Image too small ({len(image_bytes)} bytes) — likely corrupt or placeholder")
        return QualityResult(passed=False, band="reject_reupload", issues=issues)

    try:
        from PIL import Image, ImageFilter, ImageStat

        img = Image.open(BytesIO(image_bytes))
        width, height = img.size

        # Gate 1: resolution
        if width < MIN_WIDTH or height < MIN_HEIGHT:
            issues.append(f"Resolution too low ({width}×{height}) — need at least {MIN_WIDTH}×{MIN_HEIGHT}")

        # Convert to grayscale for luminance + sharpness
        gray = img.convert("L")
        stat = ImageStat.Stat(gray)
        luminance = stat.mean[0]  # 0-255

        # Gate 2: too dark
        if luminance < MIN_LUMINANCE:
            issues.append(f"Image too dark (luminance {luminance:.0f}/255) — product not visible")

        # Gate 3: blurriness (Laplacian-like via edge detection variance)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)
        sharpness = edge_stat.var[0] if edge_stat.var else 0.0

        if sharpness < MIN_LAPLACIAN_VAR:
            issues.append(f"Image appears blurry (sharpness {sharpness:.1f}) — please retake with steady hand")

    except Exception as exc:
        issues.append(f"Cannot decode image: {exc}")
        return QualityResult(passed=False, band="reject_reupload", issues=issues)

    if issues:
        # Soft issues (low res alone) → needs_review; hard issues (dark+blurry) → reject
        hard_issues = [i for i in issues if "dark" in i or "blurry" in i or "corrupt" in i or "Cannot" in i]
        band: QualityBand = "reject_reupload" if hard_issues else "needs_review"
        return QualityResult(
            passed=False, band=band, issues=issues,
            luminance=luminance, sharpness=sharpness, width=width, height=height,
        )

    return QualityResult(
        passed=True, band="auto_pass", issues=[],
        luminance=luminance, sharpness=sharpness, width=width, height=height,
    )


def check_video_quality(video_bytes: bytes) -> QualityResult:
    """Basic video quality gate — just checks file size for now.
    
    Full frame-level analysis would require extracting keyframes first (done
    in the video pipeline), so we gate on obvious issues only.
    """
    issues: list[str] = []
    if len(video_bytes) < 10_000:
        issues.append("Video file too small — likely corrupt or incomplete")
        return QualityResult(passed=False, band="reject_reupload", issues=issues)
    return QualityResult(passed=True, band="auto_pass", issues=[])
