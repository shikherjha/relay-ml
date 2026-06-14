"""Router for return-reason clustering (T3 stretch)."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.pipelines.return_cluster import (
    ClusteringUnavailable,
    ReasonCluster,
    cluster_return_reasons,
)

router = APIRouter(tags=["clustering"])


class ClusterRequest(BaseModel):
    reasons: list[str] = Field(..., min_length=1, description="Free-text return reasons to cluster.")
    n_clusters: int | None = Field(default=None, ge=2, le=20, description="Number of clusters (auto if omitted).")
    min_cluster_size: int = Field(default=2, ge=1, description="Minimum cluster size to report.")


class ClusterItem(BaseModel):
    cluster_id: int
    label: str
    reasons: list[str]
    count: int


class ClusterResponse(BaseModel):
    clusters: list[ClusterItem]
    total_reasons: int
    num_clusters: int


@router.post("/return-clusters", response_model=ClusterResponse)
def return_clusters(payload: ClusterRequest) -> ClusterResponse:
    """Cluster free-text return reasons into semantic groups.

    Enriches seller signals beyond fixed reason codes by collapsing
    variants like "color not as shown" / "looks different" / "shade mismatch"
    into a single signal.
    """
    try:
        clusters = cluster_return_reasons(
            reasons=payload.reasons,
            n_clusters=payload.n_clusters,
            min_cluster_size=payload.min_cluster_size,
        )
    except ClusteringUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return ClusterResponse(
        clusters=[
            ClusterItem(
                cluster_id=c.cluster_id,
                label=c.label,
                reasons=c.reasons,
                count=c.count,
            )
            for c in clusters
        ],
        total_reasons=len(payload.reasons),
        num_clusters=len(clusters),
    )
