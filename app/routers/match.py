"""Router for reverse-wishlist relevance reranking (Bedrock).

Second stage of retrieve→rerank: relay-api sends a wish + a recalled candidate
pool, this scores each candidate's relevance via Nova Lite so cross-category
cosine noise (macbook↔earphones, hoodie↔tee) is filtered out.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.pipelines.match_rank import RerankUnavailable, rank_matches

router = APIRouter(tags=["matching"])


class MatchCandidate(BaseModel):
    unit_id: str
    title: str | None = None
    category: str | None = None
    price: float | None = None


class MatchRankRequest(BaseModel):
    wish: str = Field(..., description="The shopper's free-text wish (category/intent).")
    size: str | None = None
    max_price: float | None = None
    candidates: list[MatchCandidate] = Field(..., min_length=1)


class RankedMatchOut(BaseModel):
    unit_id: str
    score: float
    reason: str | None = None


class MatchRankResponse(BaseModel):
    ranked: list[RankedMatchOut]
    model: str


@router.post("/match-rank", response_model=MatchRankResponse)
def match_rank(payload: MatchRankRequest) -> MatchRankResponse:
    try:
        ranked = rank_matches(
            wish=payload.wish,
            size=payload.size,
            max_price=payload.max_price,
            candidates=[c.model_dump() for c in payload.candidates],
            region=settings.aws_region,
            model_id=settings.bedrock_model_t1,
        )
    except RerankUnavailable as exc:
        # Loud failure → relay-api falls back to deterministic taxonomy scoring.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    return MatchRankResponse(
        ranked=[RankedMatchOut(unit_id=r.unit_id, score=r.score, reason=r.reason) for r in ranked],
        model=settings.bedrock_model_t1,
    )
