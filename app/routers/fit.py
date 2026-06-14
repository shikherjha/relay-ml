from fastapi import APIRouter

from app.pipelines.fit_flags import predict_fit_flags
from app.schemas.passport import FitFlagsRequest, FitFlagsResponse

router = APIRouter(tags=["fit"])


@router.post("/fit-flags", response_model=FitFlagsResponse)
def fit_flags(payload: FitFlagsRequest) -> FitFlagsResponse:
    return predict_fit_flags(sku_id=payload.sku_id, category=payload.category)
