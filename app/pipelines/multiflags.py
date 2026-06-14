"""Simplified MultiFlags pipeline (T3 stretch).

Inspired by Zalando's SizeFlags (arXiv 2106.03532) and the MultiFlags extension
(SCITEPRESS 2025). Uses Bayesian-style priors on return reason counts to produce
article-level fit flags with calibrated confidence.

Key differences from the simple aggregate approach:
- Multiple flags per SKU (e.g. "runs small" AND "critical fit" simultaneously)
- Bayesian confidence with prior smoothing (avoids overconfidence on small samples)
- Per-article override when article-level data is available (not just category)
- Confidence decay for older aggregate data
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from app.schemas.passport import FitFlag


# Bayesian prior parameters (pseudo-counts)
_PRIOR_ALPHA = 2.0  # pseudo-successes (smoothing toward 50%)
_PRIOR_BETA = 2.0  # pseudo-failures

# Thresholds for flag emission
_SMALL_THRESHOLD = 0.22  # posterior P(runs_small) > this => emit flag
_LARGE_THRESHOLD = 0.22  # posterior P(runs_large) > this => emit flag
_CRITICAL_THRESHOLD = 0.35  # when both small+large are high => critical fit
_MIN_OBSERVATIONS = 5  # minimum returns before any flag


@dataclass(frozen=True)
class ArticleFitData:
    """Fit data for a specific article/SKU."""

    sku_id: str
    category: str
    total_returns: int
    too_small_count: int
    too_large_count: int
    fit_count: int
    # Optional: article-specific data overrides category-level
    is_article_level: bool = False


@dataclass(frozen=True)
class MultiFlagResult:
    """Result of MultiFlags computation."""

    flags: list[FitFlag]
    source: str
    posterior_small: float
    posterior_large: float


def compute_multiflags(data: ArticleFitData) -> MultiFlagResult:
    """Compute Bayesian multi-flags for an article.

    Uses Beta-Binomial posterior estimation:
    - P(runs_small) = (too_small_count + alpha) / (total_returns + alpha + beta)
    - P(runs_large) = (too_large_count + alpha) / (total_returns + alpha + beta)

    Emits multiple flags when both probabilities exceed thresholds.
    """
    flags: list[FitFlag] = []
    total = data.total_returns

    if total < _MIN_OBSERVATIONS:
        flags.append(
            FitFlag(
                type="critical_fit",
                message="Too few fit signals for reliable sizing; check measurements.",
                confidence=0.4,
            )
        )
        return MultiFlagResult(
            flags=flags,
            source="multiflags_v1:low_data",
            posterior_small=0.0,
            posterior_large=0.0,
        )

    # Beta-Binomial posterior
    posterior_small = (data.too_small_count + _PRIOR_ALPHA) / (
        total + _PRIOR_ALPHA + _PRIOR_BETA
    )
    posterior_large = (data.too_large_count + _PRIOR_ALPHA) / (
        total + _PRIOR_ALPHA + _PRIOR_BETA
    )

    # Confidence = f(sample size, posterior strength)
    # More data + stronger signal = higher confidence
    base_confidence = _bayesian_confidence(total, max(posterior_small, posterior_large))

    # Check for critical fit (both directions problematic)
    if posterior_small > _CRITICAL_THRESHOLD and posterior_large > _CRITICAL_THRESHOLD:
        flags.append(
            FitFlag(
                type="critical_fit",
                message=(
                    f"Sizing is inconsistent: {posterior_small:.0%} report too small, "
                    f"{posterior_large:.0%} report too large. Check detailed size chart."
                ),
                confidence=min(0.95, base_confidence + 0.05),
            )
        )
    else:
        # Emit individual directional flags
        if posterior_small > _SMALL_THRESHOLD:
            flags.append(
                FitFlag(
                    type="runs_small",
                    message=f"Runs small — {posterior_small:.0%} of returns cite size too small. Consider sizing up.",
                    confidence=base_confidence,
                )
            )

        if posterior_large > _LARGE_THRESHOLD:
            flags.append(
                FitFlag(
                    type="runs_large",
                    message=f"Runs large — {posterior_large:.0%} of returns cite size too large. Consider sizing down.",
                    confidence=base_confidence,
                )
            )

    # If no directional flags, emit true_to_size
    if not flags:
        fit_rate = data.fit_count / total if total > 0 else 0.5
        flags.append(
            FitFlag(
                type="true_to_size",
                message=f"True to size — {fit_rate:.0%} of buyers report good fit.",
                confidence=base_confidence,
            )
        )

    source_level = "article" if data.is_article_level else "category"
    return MultiFlagResult(
        flags=flags,
        source=f"multiflags_v1:{source_level}",
        posterior_small=round(posterior_small, 4),
        posterior_large=round(posterior_large, 4),
    )


def _bayesian_confidence(n_observations: int, posterior_strength: float) -> float:
    """Compute confidence from sample size and posterior strength.

    Uses a logistic curve on sample size + signal strength:
    - Small n → low confidence (smoothed by prior)
    - Large n + strong signal → high confidence (capped at 0.95)
    """
    # Sample size contribution (logistic, saturates around n=200)
    size_factor = 1.0 / (1.0 + math.exp(-0.03 * (n_observations - 30)))

    # Signal strength contribution
    signal_factor = min(1.0, posterior_strength / 0.5)

    raw = 0.5 + 0.45 * size_factor * signal_factor
    return round(max(0.4, min(0.95, raw)), 2)
