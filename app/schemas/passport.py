from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Grade(str, Enum):
    a_plus = "A+"
    a = "A"
    b_plus = "B+"
    b = "B"
    c = "C"
    d = "D"


class Defect(BaseModel):
    type: Literal[
        "scuff",
        "crack",
        "stain",
        "tear",
        "dent",
        "discoloration",
        "missing_part",
        "screen_damage",
        "water_damage",
        "functional_fault",
        "other",
    ] = Field(..., examples=["scuff", "crack", "stain", "missing_part"])
    severity: Literal["minor", "moderate", "major"]
    bbox: list[float] | None = Field(
        default=None,
        description="Optional [x, y, w, h] bounding box in image coordinates.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    description: str | None = None


class ConditionPassport(BaseModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    unit_id: str
    return_id: str | None = None
    grade: Grade
    grade_numeric: float = Field(..., ge=0.0, le=1.0)
    category: str
    vertical: Literal["fashion", "electronics"]
    disposition_hint: Literal[
        "exchange",
        "rescue",
        "p2p_resale",
        "refurb",
        "donate",
        "recycle",
        "restock",
    ]
    defects: list[Defect] = Field(default_factory=list)
    packaging_state: Literal["sealed", "opened", "damaged", "missing"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    media_hashes: list[str] = Field(default_factory=list)
    passport_hash: str
    graded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_tier_used: str
    warranty_months_remaining: int = Field(default=0, ge=0)
    repair_events: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    cnn_version: str
    model_path: str
    model_bytes: int
    notes: list[str] = Field(default_factory=list)


class FitFlag(BaseModel):
    type: Literal["runs_large", "runs_small", "true_to_size", "critical_fit"]
    message: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class FitFlagsRequest(BaseModel):
    sku_id: str
    brand: str | None = None
    category: str | None = None


class FitFlagsResponse(BaseModel):
    sku_id: str
    flags: list[FitFlag]
    source: str = "rules_v1"


# --- Embed schemas ---


class EmbedRequest(BaseModel):
    text: str | None = None
    category: str | None = None
    grade: str | None = None
    size: str | None = None
    vertical: str | None = None


class EmbedResponse(BaseModel):
    vector: list[float] = Field(..., min_length=384, max_length=384)
    model: str


# --- Wish score schemas ---


class WishScoreRequest(BaseModel):
    wish_age_days: float = Field(..., ge=0)
    user_purchase_count: int = Field(..., ge=0)
    category_affinity: float = Field(..., ge=0.0, le=1.0)
    has_fit_profile: bool = False


class WishScoreResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    model: str = "logreg_v1"


# --- Grade-and-price schemas (Track B: resale grading) ---


class PriceRange(BaseModel):
    min: float = Field(..., gt=0)
    max: float = Field(..., gt=0)


class GradeAndPriceResponse(BaseModel):
    """ConditionPassport + resale pricing fields."""

    # Core ConditionPassport fields
    schema_version: Literal["1.0.0"] = "1.0.0"
    unit_id: str
    return_id: str | None = None
    grade: Grade
    grade_numeric: float = Field(..., ge=0.0, le=1.0)
    category: str
    vertical: Literal["fashion", "electronics"]
    disposition_hint: Literal[
        "exchange", "rescue", "p2p_resale", "refurb", "donate", "recycle", "restock",
    ]
    defects: list[Defect] = Field(default_factory=list)
    packaging_state: Literal["sealed", "opened", "damaged", "missing"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    media_hashes: list[str] = Field(default_factory=list)
    passport_hash: str = ""
    graded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_tier_used: str

    # Resale pricing fields (Track B)
    resale_grade: Literal["Like New", "Very Good", "Good", "Acceptable"]
    price_range: PriceRange
    currency: str = "INR"
    pricing_rationale: str
