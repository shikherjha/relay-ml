from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.pipelines.fit_flags import predict_fit_flags
from app.pipelines.multiflags import ArticleFitData, compute_multiflags
from app.schemas.passport import FitFlagsRequest, FitFlagsResponse

router = APIRouter(tags=["fit"])


class MultiFlagsRequest(BaseModel):
    """Request with article-level return data for MultiFlags computation."""

    sku_id: str
    category: str
    total_returns: int = Field(..., ge=0)
    too_small_count: int = Field(..., ge=0)
    too_large_count: int = Field(..., ge=0)
    fit_count: int = Field(..., ge=0)
    is_article_level: bool = Field(
        default=False,
        description="True if data is for this specific article, not category-level.",
    )


@router.post("/fit-flags", response_model=FitFlagsResponse)
def fit_flags(payload: FitFlagsRequest) -> FitFlagsResponse:
    return predict_fit_flags(sku_id=payload.sku_id, category=payload.category)


@router.post("/fit-flags/multi", response_model=FitFlagsResponse)
def fit_flags_multi(payload: MultiFlagsRequest) -> FitFlagsResponse:
    """Compute Bayesian MultiFlags from article/category return data.

    Unlike /fit-flags (which uses pre-built aggregates), this endpoint
    accepts raw return counts and computes Bayesian posteriors for
    multi-directional fit flags.
    """
    data = ArticleFitData(
        sku_id=payload.sku_id,
        category=payload.category,
        total_returns=payload.total_returns,
        too_small_count=payload.too_small_count,
        too_large_count=payload.too_large_count,
        fit_count=payload.fit_count,
        is_article_level=payload.is_article_level,
    )
    result = compute_multiflags(data)
    return FitFlagsResponse(
        sku_id=payload.sku_id,
        flags=result.flags,
        source=result.source,
    )
