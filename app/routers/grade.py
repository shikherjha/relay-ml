from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.schemas.passport import ConditionPassport

router = APIRouter(tags=["grading"])


@router.post("/grade-image", response_model=ConditionPassport)
async def grade_image(
    image: UploadFile = File(...),
    unit_id: str = Form(...),
    category: str = Form(...),
) -> ConditionPassport:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "message": "Phase 1 placeholder. Implement image grading in Phase 3.",
            "unit_id": unit_id,
            "category": category,
            "filename": image.filename,
        },
    )


@router.post("/grade-video", response_model=ConditionPassport)
async def grade_video(
    video: UploadFile = File(...),
    unit_id: str = Form(...),
) -> ConditionPassport:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "message": "Phase 1 placeholder. Implement video keyframe grading later.",
            "unit_id": unit_id,
            "filename": video.filename,
        },
    )
