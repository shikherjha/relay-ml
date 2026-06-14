from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from typing import Annotated

from app.core.config import settings
from app.pipelines.bedrock_tiers import (
    BedrockGradingError,
    grade_bedrock_only,
    grade_mock,
    grade_multi_image_bedrock,
)
from app.pipelines.image_grade import (
    ImageGradingUnavailable,
    ImageValidationError,
    grade_image_bytes,
    SUPPORTED_CONTENT_TYPES,
    MAX_IMAGE_BYTES,
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

    # Route based on GRADING_MODE
    if settings.grading_mode == "mock":
        # Validate basics even in mock mode
        if image.content_type not in SUPPORTED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only JPEG and PNG images are supported.",
            )
        return grade_mock(
            image_bytes=image_bytes,
            unit_id=unit_id,
            category=category,
            return_id=return_id,
        )

    if settings.grading_mode == "bedrock_only":
        # Validate basics before calling Bedrock
        if image.content_type not in SUPPORTED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only JPEG and PNG images are supported.",
            )
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image upload exceeds the 8 MB limit.",
            )
        try:
            return grade_bedrock_only(
                image_bytes=image_bytes,
                unit_id=unit_id,
                category=category,
                return_id=return_id,
            )
        except BedrockGradingError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

    # Default: CNN path
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


@router.post("/grade-images", response_model=ConditionPassport)
async def grade_images(
    images: list[UploadFile] = File(...),
    unit_id: str = Form(...),
    category: str = Form(...),
    return_id: str | None = Form(default=None),
) -> ConditionPassport:
    """Grade a product from multiple angle images (1-8 images).

    All images are sent to Bedrock in a single call so the model
    assesses overall condition from multiple perspectives.
    """
    if len(images) > 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 8 images allowed per grading request.",
        )

    image_bytes_list: list[bytes] = []
    for img in images:
        if img.content_type not in SUPPORTED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Only JPEG and PNG images are supported. Got: {img.content_type}",
            )
        data = await img.read()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image '{img.filename}' exceeds the 8 MB limit.",
            )
        image_bytes_list.append(data)

    try:
        return grade_multi_image_bedrock(
            images=image_bytes_list,
            unit_id=unit_id,
            category=category,
            return_id=return_id,
        )
    except BedrockGradingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
