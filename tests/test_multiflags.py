"""Tests for MultiFlags Bayesian fit-flag computation."""

from fastapi.testclient import TestClient

from app.main import app
from app.pipelines.multiflags import ArticleFitData, compute_multiflags

client = TestClient(app)


def test_multiflags_runs_small_signal() -> None:
    """High too_small_count produces a runs_small flag."""
    data = ArticleFitData(
        sku_id="SKU-001",
        category="dress",
        total_returns=100,
        too_small_count=40,
        too_large_count=5,
        fit_count=55,
    )
    result = compute_multiflags(data)
    flag_types = [f.type for f in result.flags]
    assert "runs_small" in flag_types
    assert result.posterior_small > 0.3


def test_multiflags_runs_large_signal() -> None:
    """High too_large_count produces a runs_large flag."""
    data = ArticleFitData(
        sku_id="SKU-002",
        category="coat",
        total_returns=80,
        too_small_count=3,
        too_large_count=35,
        fit_count=42,
    )
    result = compute_multiflags(data)
    flag_types = [f.type for f in result.flags]
    assert "runs_large" in flag_types


def test_multiflags_true_to_size() -> None:
    """Balanced returns produce true_to_size flag."""
    data = ArticleFitData(
        sku_id="SKU-003",
        category="jeans",
        total_returns=200,
        too_small_count=15,
        too_large_count=10,
        fit_count=175,
    )
    result = compute_multiflags(data)
    flag_types = [f.type for f in result.flags]
    assert "true_to_size" in flag_types


def test_multiflags_critical_fit() -> None:
    """Both directions high produces critical_fit flag."""
    data = ArticleFitData(
        sku_id="SKU-004",
        category="sneakers",
        total_returns=50,
        too_small_count=20,
        too_large_count=20,
        fit_count=10,
    )
    result = compute_multiflags(data)
    flag_types = [f.type for f in result.flags]
    assert "critical_fit" in flag_types


def test_multiflags_low_data() -> None:
    """Very few returns produces a low-confidence critical_fit."""
    data = ArticleFitData(
        sku_id="SKU-005",
        category="hat",
        total_returns=3,
        too_small_count=1,
        too_large_count=1,
        fit_count=1,
    )
    result = compute_multiflags(data)
    assert len(result.flags) == 1
    assert result.flags[0].type == "critical_fit"
    assert result.flags[0].confidence <= 0.5


def test_multiflags_confidence_increases_with_data() -> None:
    """More observations increases confidence."""
    small_data = ArticleFitData(
        sku_id="SKU-A",
        category="dress",
        total_returns=10,
        too_small_count=5,
        too_large_count=1,
        fit_count=4,
    )
    large_data = ArticleFitData(
        sku_id="SKU-B",
        category="dress",
        total_returns=500,
        too_small_count=250,
        too_large_count=25,
        fit_count=225,
    )
    result_small = compute_multiflags(small_data)
    result_large = compute_multiflags(large_data)
    # Both should emit runs_small, but larger sample should have higher confidence
    small_conf = max(f.confidence for f in result_small.flags)
    large_conf = max(f.confidence for f in result_large.flags)
    assert large_conf > small_conf


def test_multiflags_endpoint_basic() -> None:
    """POST /fit-flags/multi returns valid response."""
    response = client.post(
        "/fit-flags/multi",
        json={
            "sku_id": "SKU-ROUTE-1",
            "category": "shirt",
            "total_returns": 60,
            "too_small_count": 25,
            "too_large_count": 5,
            "fit_count": 30,
            "is_article_level": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sku_id"] == "SKU-ROUTE-1"
    assert "multiflags_v1" in data["source"]
    assert len(data["flags"]) >= 1
    for flag in data["flags"]:
        assert flag["type"] in {"runs_small", "runs_large", "true_to_size", "critical_fit"}
        assert 0 <= flag["confidence"] <= 1


def test_multiflags_endpoint_article_level() -> None:
    """Article-level flag reports correct source."""
    response = client.post(
        "/fit-flags/multi",
        json={
            "sku_id": "SKU-ART-1",
            "category": "dress",
            "total_returns": 100,
            "too_small_count": 40,
            "too_large_count": 5,
            "fit_count": 55,
            "is_article_level": True,
        },
    )
    assert response.status_code == 200
    assert "article" in response.json()["source"]
