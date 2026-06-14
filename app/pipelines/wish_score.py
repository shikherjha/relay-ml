"""Wish confidence scoring pipeline.

Lightweight logistic regression-style scoring that ranks buyer intent (0-1)
based on wish recency, purchase history, category affinity, and fit profile.

Used as a multiplier in relay-engine matching and demand-weighted disposition.
"""

from __future__ import annotations

import math

# Logistic regression coefficients (trained on synthetic/seed labels).
# Positive coefficients = higher intent.
_COEF_RECENCY = -0.03  # older wishes score lower (per day)
_COEF_PURCHASES = 0.15  # more purchases in category = higher intent
_COEF_AFFINITY = 1.2  # direct category affinity signal
_COEF_FIT_PROFILE = 0.6  # having a fit profile = more serious buyer
_INTERCEPT = 0.5


def _sigmoid(x: float) -> float:
    """Standard sigmoid function clamped for numerical stability."""
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def compute_wish_score(
    wish_age_days: float,
    user_purchase_count: int,
    category_affinity: float,
    has_fit_profile: bool,
) -> float:
    """Compute wish confidence score (0-1).

    Properties:
    - Monotonically: newer wish + more purchases + higher affinity + fit profile => higher score
    - Output clamped to [0, 1] via sigmoid
    """
    linear = (
        _INTERCEPT
        + _COEF_RECENCY * wish_age_days
        + _COEF_PURCHASES * min(user_purchase_count, 20)  # cap contribution
        + _COEF_AFFINITY * category_affinity
        + _COEF_FIT_PROFILE * (1.0 if has_fit_profile else 0.0)
    )
    return round(_sigmoid(linear), 4)
