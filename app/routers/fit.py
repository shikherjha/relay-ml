from fastapi import APIRouter

from app.schemas.passport import FitFlag, FitFlagsRequest, FitFlagsResponse

router = APIRouter(tags=["fit"])


@router.post("/fit-flags", response_model=FitFlagsResponse)
def fit_flags(payload: FitFlagsRequest) -> FitFlagsResponse:
    category = (payload.category or "").lower()

    if "shoe" in category:
        flag = FitFlag(
            type="runs_small",
            message="Many buyers prefer half a size up for this category.",
            confidence=0.62,
        )
    elif "dress" in category or "fashion" in category or "clothing" in category:
        flag = FitFlag(
            type="true_to_size",
            message="Most recent fit signals indicate this item is true to size.",
            confidence=0.7,
        )
    else:
        flag = FitFlag(
            type="critical_fit",
            message="Fit history is limited; confirm size details before checkout.",
            confidence=0.55,
        )

    return FitFlagsResponse(sku_id=payload.sku_id, flags=[flag])
