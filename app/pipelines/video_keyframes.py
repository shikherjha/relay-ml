from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from app.pipelines.image_grade import (
    ImageGradingUnavailable,
    ImageValidationError,
    grade_image_bytes,
)
from app.schemas.passport import ConditionPassport, Defect, Grade
from app.utils.hash import passport_hash, sha256_hex


SUPPORTED_VIDEO_CONTENT_TYPES = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-msvideo": ".avi",
}
MAX_VIDEO_BYTES = 64 * 1024 * 1024
KEYFRAME_COUNT = 5
MAX_SCANNED_FRAMES = 180

GRADE_NUMERIC = {
    "A+": 1.0,
    "A": 0.9,
    "B+": 0.78,
    "B": 0.68,
    "C": 0.45,
    "D": 0.2,
}
GRADE_RANK = {"A+": 0, "A": 1, "B+": 2, "B": 3, "C": 4, "D": 5}
SEVERITY_RANK = {"minor": 0, "moderate": 1, "major": 2}


class VideoValidationError(ValueError):
    pass


class VideoGradingUnavailable(RuntimeError):
    pass


def grade_video_bytes(
    *,
    video_bytes: bytes,
    content_type: str | None,
    unit_id: str,
    category: str,
    return_id: str | None,
    model_path: Path,
) -> ConditionPassport:
    _validate_video(video_bytes=video_bytes, content_type=content_type)
    frame_images = extract_keyframe_images(
        video_bytes=video_bytes,
        content_type=content_type,
        count=KEYFRAME_COUNT,
    )
    if not frame_images:
        raise VideoValidationError("No readable video frames were found.")

    # Route through Bedrock or CNN based on GRADING_MODE
    from app.core.config import settings

    if settings.grading_mode == "bedrock_only":
        frame_passports = _grade_frames_bedrock(
            frame_images=frame_images,
            unit_id=unit_id,
            category=category,
            return_id=return_id,
        )
    else:
        frame_passports = [
            grade_image_bytes(
                image_bytes=frame_bytes,
                content_type="image/png",
                unit_id=f"{unit_id}:frame:{index}",
                category=category,
                return_id=return_id,
                model_path=model_path,
            )
            for index, frame_bytes in enumerate(frame_images, start=1)
        ]

    return aggregate_frame_passports(
        frame_passports=frame_passports,
        video_bytes=video_bytes,
        unit_id=unit_id,
        return_id=return_id,
        category=category,
    )


def _grade_frames_bedrock(
    *,
    frame_images: list[bytes],
    unit_id: str,
    category: str,
    return_id: str | None,
) -> list[ConditionPassport]:
    """Grade keyframes using Bedrock multi-image (all frames in one call)."""
    from app.pipelines.bedrock_tiers import (
        BedrockGradingError,
        grade_bedrock_only,
        grade_multi_image_bedrock,
    )

    try:
        # Send all keyframes in one Bedrock call for holistic video assessment
        passport = grade_multi_image_bedrock(
            images=frame_images,
            unit_id=unit_id,
            category=category,
            return_id=return_id,
        )
        # Return as a single-element list for aggregation compatibility
        return [passport]
    except BedrockGradingError:
        # Fallback: grade each frame individually via Bedrock
        passports = []
        for i, frame_bytes in enumerate(frame_images, start=1):
            try:
                passport = grade_bedrock_only(
                    image_bytes=frame_bytes,
                    unit_id=f"{unit_id}:frame:{i}",
                    category=category,
                    return_id=return_id,
                )
                passports.append(passport)
            except BedrockGradingError:
                continue
        if not passports:
            raise VideoGradingUnavailable("Bedrock grading failed for all keyframes.")
        return passports


def extract_keyframe_images(
    *,
    video_bytes: bytes,
    content_type: str | None,
    count: int = KEYFRAME_COUNT,
) -> list[bytes]:
    try:
        import imageio.v3 as iio
        from PIL import Image
    except ImportError as exc:
        raise VideoGradingUnavailable(
            "Video decoding requires imageio, imageio-ffmpeg, and Pillow."
        ) from exc

    suffix = SUPPORTED_VIDEO_CONTENT_TYPES.get(content_type or "")
    if suffix is None:
        raise VideoValidationError("Only MP4, MOV, WEBM, and AVI videos are supported.")

    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(video_bytes)
            temp_path = Path(handle.name)

        frames = []
        for index, frame in enumerate(iio.imiter(temp_path)):
            if index >= MAX_SCANNED_FRAMES:
                break
            frames.append(frame)
    except VideoGradingUnavailable:
        raise
    except Exception as exc:
        raise VideoValidationError("Upload is not a readable video file.") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    selected = _select_evenly(frames, count)
    keyframes: list[bytes] = []
    for frame in selected:
        image = Image.fromarray(frame).convert("RGB")
        output = BytesIO()
        image.save(output, format="PNG")
        keyframes.append(output.getvalue())
    return keyframes


def aggregate_frame_passports(
    *,
    frame_passports: Iterable[ConditionPassport],
    video_bytes: bytes,
    unit_id: str,
    return_id: str | None,
    category: str,
) -> ConditionPassport:
    passports = list(frame_passports)
    if not passports:
        raise VideoValidationError("No frame grading results were produced.")

    worst = max(passports, key=lambda passport: GRADE_RANK[passport.grade.value])
    defects = _merge_defects(passports)
    confidence = min(passport.confidence for passport in passports)
    media_hashes = [sha256_hex(video_bytes)]
    for passport in passports:
        media_hashes.extend(passport.media_hashes)

    body = {
        "schema_version": "1.0.0",
        "unit_id": unit_id,
        "return_id": return_id,
        "grade": worst.grade.value,
        "grade_numeric": GRADE_NUMERIC[worst.grade.value],
        "category": worst.category or category,
        "vertical": worst.vertical,
        "disposition_hint": worst.disposition_hint,
        "defects": [defect.model_dump(mode="json") for defect in defects],
        "packaging_state": worst.packaging_state,
        "confidence": confidence,
        "media_hashes": list(dict.fromkeys(media_hashes)),
        "passport_hash": "",
        "graded_at": datetime.now(timezone.utc),
        "model_tier_used": f"{worst.model_tier_used}+keyframes",
        "warranty_months_remaining": worst.warranty_months_remaining,
        "repair_events": [],
    }
    body["passport_hash"] = passport_hash(body)
    return ConditionPassport.model_validate(body)


def _validate_video(*, video_bytes: bytes, content_type: str | None) -> None:
    if content_type not in SUPPORTED_VIDEO_CONTENT_TYPES:
        raise VideoValidationError("Only MP4, MOV, WEBM, and AVI videos are supported.")
    if not video_bytes:
        raise VideoValidationError("Video upload is empty.")
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise VideoValidationError("Video upload exceeds the 64 MB limit.")


def _select_evenly(frames: list, count: int) -> list:
    if len(frames) <= count:
        return frames
    if count <= 1:
        return [frames[0]]

    last = len(frames) - 1
    indexes = [round(index * last / (count - 1)) for index in range(count)]
    return [frames[index] for index in indexes]


def _merge_defects(passports: list[ConditionPassport]) -> list[Defect]:
    by_type: dict[str, Defect] = {}
    for passport in passports:
        for defect in passport.defects:
            existing = by_type.get(defect.type)
            if existing is None:
                by_type[defect.type] = defect
                continue

            severity = max(
                existing.severity,
                defect.severity,
                key=lambda value: SEVERITY_RANK[value],
            )
            confidence = max(existing.confidence, defect.confidence)
            by_type[defect.type] = Defect(
                type=defect.type,
                severity=severity,
                bbox=existing.bbox or defect.bbox,
                confidence=confidence,
                description=existing.description or defect.description,
            )

    return sorted(
        by_type.values(),
        key=lambda defect: (SEVERITY_RANK[defect.severity], defect.confidence),
        reverse=True,
    )
