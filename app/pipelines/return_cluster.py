"""Return reason clustering pipeline (T3 stretch).

Clusters free-text return reasons using embeddings + KMeans so seller signals
go beyond fixed reason codes. For example, multiple variants of "color not as
shown" (e.g. "looks different in person", "not the same shade", "color
mismatch") collapse into a single signal for the ops dashboard.

Used by relay-api's seller-signal aggregation to enrich beyond fixed reason codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence


@dataclass(frozen=True)
class ReasonCluster:
    """A cluster of similar return reasons."""

    cluster_id: int
    label: str  # Representative label for this cluster
    reasons: list[str]  # All reasons in this cluster
    count: int  # Number of reasons in this cluster


class ClusteringUnavailable(RuntimeError):
    """Raised when clustering dependencies are not available."""


def cluster_return_reasons(
    reasons: Sequence[str],
    n_clusters: int | None = None,
    min_cluster_size: int = 2,
) -> list[ReasonCluster]:
    """Cluster a list of free-text return reasons into semantic groups.

    Args:
        reasons: List of free-text return reason strings.
        n_clusters: Number of clusters to create. If None, auto-determined
                    based on the number of reasons (sqrt(n) heuristic).
        min_cluster_size: Minimum cluster size to report (smaller clusters
                         are merged into an "other" bucket).

    Returns:
        List of ReasonCluster objects sorted by count descending.
    """
    if not reasons:
        return []

    # Deduplicate while preserving order for embedding
    unique_reasons = list(dict.fromkeys(reasons))

    if len(unique_reasons) < 3:
        # Too few reasons to cluster meaningfully
        return [
            ReasonCluster(
                cluster_id=0,
                label=unique_reasons[0],
                reasons=list(reasons),
                count=len(reasons),
            )
        ]

    try:
        from sklearn.cluster import KMeans
        import numpy as np
    except ImportError as exc:
        raise ClusteringUnavailable(
            "scikit-learn is required for return-reason clustering. "
            "Install with: pip install scikit-learn"
        ) from exc

    # Use the embedding pipeline to vectorize reasons
    from app.pipelines.embed import compute_embedding

    embeddings = []
    for reason in unique_reasons:
        vector, _ = compute_embedding(text=reason)
        embeddings.append(vector)

    X = np.array(embeddings)

    # Auto-determine cluster count if not specified
    if n_clusters is None:
        import math

        n_clusters = max(2, min(int(math.sqrt(len(unique_reasons))), 10))

    n_clusters = min(n_clusters, len(unique_reasons))

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Build reason-to-cluster mapping
    reason_to_cluster: dict[str, int] = {}
    for reason, label in zip(unique_reasons, labels):
        reason_to_cluster[reason] = int(label)

    # Group all original reasons (including duplicates) by cluster
    cluster_reasons: dict[int, list[str]] = {}
    for reason in reasons:
        # For duplicate reasons, use the cluster of the unique version
        cluster_id = reason_to_cluster.get(reason, 0)
        cluster_reasons.setdefault(cluster_id, []).append(reason)

    # Find representative label for each cluster (closest to centroid)
    clusters: list[ReasonCluster] = []
    for cluster_id in sorted(cluster_reasons.keys()):
        members = cluster_reasons[cluster_id]
        # Find the reason in this cluster closest to the centroid
        cluster_mask = labels == cluster_id
        cluster_embeddings = X[cluster_mask]
        centroid = kmeans.cluster_centers_[cluster_id]

        # Get the unique reasons in this cluster
        cluster_unique = [r for r, c in reason_to_cluster.items() if c == cluster_id]
        cluster_unique_embeddings = np.array(
            [embeddings[unique_reasons.index(r)] for r in cluster_unique]
        )

        # Closest to centroid = representative
        distances = np.linalg.norm(cluster_unique_embeddings - centroid, axis=1)
        representative_idx = int(np.argmin(distances))
        representative_label = cluster_unique[representative_idx]

        clusters.append(
            ReasonCluster(
                cluster_id=cluster_id,
                label=representative_label,
                reasons=members,
                count=len(members),
            )
        )

    # Merge small clusters into "other"
    result: list[ReasonCluster] = []
    other_reasons: list[str] = []
    for cluster in clusters:
        if cluster.count >= min_cluster_size:
            result.append(cluster)
        else:
            other_reasons.extend(cluster.reasons)

    if other_reasons:
        result.append(
            ReasonCluster(
                cluster_id=-1,
                label="other",
                reasons=other_reasons,
                count=len(other_reasons),
            )
        )

    # Sort by count descending
    result.sort(key=lambda c: c.count, reverse=True)
    return result
