"""Resale pricing pipeline for POST /grade-and-price.

Combines visual condition (grade_numeric from Bedrock grading) with age
depreciation and category demand signals to produce a price range.

The contract: return {min, max} — relay-api lists the mean.
Must match or beat the deterministic fallback pricer in relay-api.
"""

from __future__ import annotations

from dataclasses import dataclass

# Category demand multipliers — higher demand categories retain more value.
_CATEGORY_DEMAND = {
    "phone": 1.05,
    "smartphone": 1.05,
    "laptop": 1.02,
    "tablet": 1.00,
    "headphones": 0.95,
    "electronics": 1.00,
    "dress": 0.92,
    "shoes": 0.97,
    "sneakers": 0.98,
    "jacket": 0.95,
    "coat": 0.93,
    "jeans": 0.90,
    "fashion": 0.92,
}

# Resale grade labels mapped from letter grades
_RESALE_GRADE_MAP = {
    "A+": "Like New",
    "A": "Like New",
    "B+": "Very Good",
    "B": "Very Good",
    "C": "Good",
    "D": "Acceptable",
}

# Spread widths per grade — higher confidence (better grade) = tighter range
_SPREAD = {
    "A+": 0.06,
    "A": 0.08,
    "B+": 0.10,
    "B": 0.12,
    "C": 0.15,
    "D": 0.20,
}


@dataclass(frozen=True)
class ResalePricing:
    """Result of resale pricing computation."""

    resale_grade: str
    price_range_min: float
    price_range_max: float
    currency: str
    pricing_rationale: str


def compute_resale_price(
    *,
    grade: str,
    grade_numeric: float,
    original_price: float,
    age_days: float,
    category: str,
    confidence: float,
    has_defects: bool,
) -> ResalePricing:
    """Compute a resale price range based on condition, age, and category.

    Price model (beats the relay-api deterministic fallback):
    - condition_factor: grade_numeric clamped [0.30, 0.95]
    - age_factor: exponential decay over 2 years, floored at 0.40
    - demand_factor: category-specific multiplier (electronics retain more)
    - confidence_boost: high-confidence grades get +3% (trust premium)

    Returns a price range — relay-api lists the mean.
    """
    # Condition factor from grade (primary driver)
    condition_factor = max(0.30, min(0.95, grade_numeric))

    # Age depreciation — exponential decay, more realistic than linear
    # Items lose ~50% value in first year, slower after that
    age_factor = max(0.40, 1.0 / (1.0 + age_days / 365.0))

    # Category demand multiplier
    cat_lower = category.lower().strip()
    demand_factor = _CATEGORY_DEMAND.get(cat_lower, 0.95)

    # Confidence premium — high-confidence grading = buyer trusts the passport more
    confidence_boost = 1.03 if confidence >= 0.90 else 1.0

    # Defect penalty — explicit defects reduce price slightly beyond grade
    defect_penalty = 0.95 if has_defects else 1.0

    # Base price computation
    base = original_price * condition_factor * age_factor * demand_factor * confidence_boost * defect_penalty

    # Ensure minimum viable price (at least 10% of original for any item)
    base = max(base, original_price * 0.10)

    # Spread: tighter for better grades (more confident pricing)
    spread = _SPREAD.get(grade, 0.12)
    price_min = round(base * (1 - spread), 2)
    price_max = round(base * (1 + spread), 2)

    # Ensure min > 0
    price_min = max(1.0, price_min)

    # Resale grade label
    resale_grade = _RESALE_GRADE_MAP.get(grade, "Good")

    # Build rationale
    rationale_parts = []
    rationale_parts.append(f"Grade {grade} ({resale_grade})")
    if age_days <= 30:
        rationale_parts.append("nearly new")
    elif age_days <= 90:
        rationale_parts.append(f"{int(age_days)} days old")
    elif age_days <= 365:
        rationale_parts.append(f"~{int(age_days / 30)} months old")
    else:
        rationale_parts.append(f"~{age_days / 365:.1f} years old")

    if demand_factor >= 1.02:
        rationale_parts.append("high category demand")
    elif demand_factor <= 0.92:
        rationale_parts.append("moderate category demand")
    else:
        rationale_parts.append("steady category demand")

    if confidence >= 0.90:
        rationale_parts.append("high-confidence assessment")

    pricing_rationale = " · ".join(rationale_parts)

    return ResalePricing(
        resale_grade=resale_grade,
        price_range_min=price_min,
        price_range_max=price_max,
        currency="INR",
        pricing_rationale=pricing_rationale,
    )
