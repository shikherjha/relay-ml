"""Wish confidence scoring pipeline.

Uses a trained logistic regression model (wish_logreg_v1.pkl) to rank buyer
intent (0-1) based on wish recency, purchase history, category affinity,
and fit profile.

Falls back to hand-tuned coefficients if the trained model file is unavailable.

Used as a multiplier in relay-engine matching and demand-weighted disposition.
"""

from __future__ import annotations

import math
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any


# Hand-tuned fallback coefficients (used if no .pkl model available)
_COEF_RECENCY = -0.03
_COEF_PURCHASES = 0.15
_COEF_AFFINITY = 1.2
_COEF_FIT_PROFILE = 0.6
_INTERCEPT = 0.5


def _sigmoid(x: float) -> float:
    """Standard sigmoid function clamped for numerical stability."""
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


@lru_cache(maxsize=1)
def _load_trained_model() -> Any | None:
    """Try to load the trained sklearn model. Returns None if unavailable."""
    from app.core.config import settings

    model_path = settings.wish_score_model_path
    if not Path(model_path).exists():
        return None

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        # Verify it has predict_proba
        if not hasattr(model, "predict_proba"):
            return None
        return model
    except Exception:
        return None


def compute_wish_score(
    wish_age_days: float,
    user_purchase_count: int,
    category_affinity: float,
    has_fit_profile: bool,
) -> float:
    """Compute wish confidence score (0-1).

    Uses trained logistic regression if available, otherwise falls back
    to hand-tuned coefficients.

    Properties:
    - Monotonic: newer wish + more purchases + higher affinity + fit profile => higher score
    - Output clamped to [0, 1]
    """
    model = _load_trained_model()

    if model is not None:
        # Use trained sklearn model
        import numpy as np

        features = np.array([[
            wish_age_days,
            user_purchase_count,
            category_affinity,
            1.0 if has_fit_profile else 0.0,
        ]])
        # predict_proba returns [[P(class=0), P(class=1)]]
        score = float(model.predict_proba(features)[0][1])
        return round(score, 4)

    # Fallback: hand-tuned logistic function
    linear = (
        _INTERCEPT
        + _COEF_RECENCY * wish_age_days
        + _COEF_PURCHASES * min(user_purchase_count, 20)
        + _COEF_AFFINITY * category_affinity
        + _COEF_FIT_PROFILE * (1.0 if has_fit_profile else 0.0)
    )
    return round(_sigmoid(linear), 4)
