from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.pipelines.image_grade import (
    ImageGradingUnavailable,
    ImageValidationError,
    grade_image_bytes,
)
from app.pipelines.video_keyframes import (
    VideoGradingUnavailable,
    VideoValidationError,
    grade_video_bytes,
)
from app.schemas.passport import ConditionPassport

router = APIRouter(tags=["grading"])


@router.post("/grade-image", response_model=ConditionPassport)
async def grade_image(
    image: UploadFile = File(...),
    unit_id: str = Form(...),
    category: str = Form(...),
    return_id: str | None = Form(default=None),
) -> ConditionPassport:
    image_bytes = await image.read()
    try:
        return grade_image_bytes(
            image_bytes=image_bytes,
            content_type=image.content_type,
            unit_id=unit_id,
            category=category,
            return_id=return_id,
            model_path=settings.cnn_model_path,
        )
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ImageGradingUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.post("/grade-video", response_model=ConditionPassport)
async def grade_video(
    video: UploadFile = File(...),
    unit_id: str = Form(...),
    category: str = Form(default="unknown"),
    return_id: str | None = Form(default=None),
) -> ConditionPassport:
    video_bytes = await video.read()
    try:
        return grade_video_bytes(
            video_bytes=video_bytes,
            content_type=video.content_type,
            unit_id=unit_id,
            category=category,
            return_id=return_id,
            model_path=settings.cnn_model_path,
        )
    except VideoValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except (VideoGradingUnavailable, ImageGradingUnavailable, ImageValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
