"""Router for embedding and wish-score endpoints."""

from fastapi import APIRouter, HTTPException, status

from app.pipelines.embed import EmbeddingUnavailable, compute_embedding
from app.pipelines.wish_score import compute_wish_score
from app.schemas.passport import (
    EmbedRequest,
    EmbedResponse,
    WishScoreRequest,
    WishScoreResponse,
)

router = APIRouter(tags=["matching"])


@router.post("/embed", response_model=EmbedResponse)
def embed(payload: EmbedRequest) -> EmbedResponse:
    """Return a 384-d embedding vector from text or structured attributes."""
    try:
        vector, model_name = compute_embedding(
            text=payload.text,
            category=payload.category,
            grade=payload.grade,
            size=payload.size,
            vertical=payload.vertical,
        )
    except EmbeddingUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return EmbedResponse(vector=vector, model=model_name)


@router.post("/wish-score", response_model=WishScoreResponse)
def wish_score(payload: WishScoreRequest) -> WishScoreResponse:
    """Return buyer-intent confidence score (0-1) for a wish."""
    score = compute_wish_score(
        wish_age_days=payload.wish_age_days,
        user_purchase_count=payload.user_purchase_count,
        category_affinity=payload.category_affinity,
        has_fit_profile=payload.has_fit_profile,
    )
    return WishScoreResponse(score=score)
