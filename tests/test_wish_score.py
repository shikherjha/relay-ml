"""Tests for POST /wish-score endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_wish_score_basic() -> None:
    """Returns a score between 0 and 1."""
    response = client.post(
        "/wish-score",
        json={
            "wish_age_days": 5,
            "user_purchase_count": 3,
            "category_affinity": 0.7,
            "has_fit_profile": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["score"] <= 1.0
    assert data["model"] == "logreg_v1"


def test_wish_score_monotonic_recency() -> None:
    """Newer wish (lower age) scores higher than older wish."""
    payload_new = {
        "wish_age_days": 1,
        "user_purchase_count": 3,
        "category_affinity": 0.5,
        "has_fit_profile": False,
    }
    payload_old = {
        "wish_age_days": 25,
        "user_purchase_count": 3,
        "category_affinity": 0.5,
        "has_fit_profile": False,
    }
    r_new = client.post("/wish-score", json=payload_new)
    r_old = client.post("/wish-score", json=payload_old)
    assert r_new.json()["score"] > r_old.json()["score"]


def test_wish_score_monotonic_purchases() -> None:
    """More purchases in category scores higher."""
    base = {
        "wish_age_days": 5,
        "category_affinity": 0.5,
        "has_fit_profile": False,
    }
    r_few = client.post("/wish-score", json={**base, "user_purchase_count": 1})
    r_many = client.post("/wish-score", json={**base, "user_purchase_count": 10})
    assert r_many.json()["score"] > r_few.json()["score"]


def test_wish_score_monotonic_affinity() -> None:
    """Higher category affinity scores higher."""
    base = {
        "wish_age_days": 5,
        "user_purchase_count": 3,
        "has_fit_profile": False,
    }
    r_low = client.post("/wish-score", json={**base, "category_affinity": 0.1})
    r_high = client.post("/wish-score", json={**base, "category_affinity": 0.9})
    assert r_high.json()["score"] > r_low.json()["score"]


def test_wish_score_monotonic_fit_profile() -> None:
    """Having a fit profile scores higher."""
    base = {
        "wish_age_days": 5,
        "user_purchase_count": 3,
        "category_affinity": 0.5,
    }
    r_no = client.post("/wish-score", json={**base, "has_fit_profile": False})
    r_yes = client.post("/wish-score", json={**base, "has_fit_profile": True})
    assert r_yes.json()["score"] > r_no.json()["score"]


def test_wish_score_high_intent() -> None:
    """Ideal buyer: recent wish, many purchases, high affinity, fit profile -> high score."""
    response = client.post(
        "/wish-score",
        json={
            "wish_age_days": 1,
            "user_purchase_count": 15,
            "category_affinity": 0.95,
            "has_fit_profile": True,
        },
    )
    assert response.json()["score"] > 0.85


def test_wish_score_low_intent() -> None:
    """Old wish, no purchases, low affinity, no profile -> low score."""
    response = client.post(
        "/wish-score",
        json={
            "wish_age_days": 29,
            "user_purchase_count": 0,
            "category_affinity": 0.1,
            "has_fit_profile": False,
        },
    )
    assert response.json()["score"] < 0.5
