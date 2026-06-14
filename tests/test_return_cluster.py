"""Tests for POST /return-clusters endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_cluster_groups_similar_reasons() -> None:
    """Similar reasons get grouped into the same cluster."""
    reasons = [
        "color not as shown in photo",
        "color looks different in person",
        "shade is not what I expected",
        "color mismatch with listing",
        "too small for me",
        "size runs small",
        "doesn't fit, too tight",
        "smaller than expected",
    ]
    response = client.post(
        "/return-clusters",
        json={"reasons": reasons, "n_clusters": 2},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_reasons"] == 8
    assert data["num_clusters"] >= 1
    # Each cluster should have multiple related reasons
    for cluster in data["clusters"]:
        assert cluster["count"] >= 1
        assert cluster["label"]  # non-empty label


def test_cluster_few_reasons() -> None:
    """With fewer than 3 unique reasons, returns a single cluster."""
    reasons = ["too small", "too small"]
    response = client.post(
        "/return-clusters",
        json={"reasons": reasons},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["num_clusters"] == 1
    assert data["clusters"][0]["count"] == 2


def test_cluster_preserves_duplicates() -> None:
    """Duplicate reasons are counted in total and assigned to clusters."""
    reasons = [
        "color mismatch",
        "color mismatch",
        "color mismatch",
        "wrong size",
        "wrong size",
        "defective item",
    ]
    response = client.post(
        "/return-clusters",
        json={"reasons": reasons, "n_clusters": 3, "min_cluster_size": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_reasons"] == 6
    # Sum of all cluster counts should equal total
    total_in_clusters = sum(c["count"] for c in data["clusters"])
    assert total_in_clusters == 6


def test_cluster_empty_reasons_rejected() -> None:
    """Empty reasons list is rejected by validation."""
    response = client.post(
        "/return-clusters",
        json={"reasons": []},
    )
    assert response.status_code == 422


def test_cluster_sorted_by_count() -> None:
    """Clusters are sorted by count descending."""
    reasons = [
        "too big", "too big", "too big", "too big", "too big",
        "color wrong", "color wrong",
        "damaged in shipping",
    ]
    response = client.post(
        "/return-clusters",
        json={"reasons": reasons, "n_clusters": 3, "min_cluster_size": 1},
    )
    assert response.status_code == 200
    data = response.json()
    counts = [c["count"] for c in data["clusters"]]
    assert counts == sorted(counts, reverse=True)
