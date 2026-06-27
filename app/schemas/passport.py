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


class Verification(BaseModel):
    """ADDITIVE (return-grading decisions): cheap, prompt-only order-vs-item check.

    relay-api passes the order's ``expected_color`` / ``product_title`` (+ size)
    as Form fields on the grade endpoints. The grade prompt is enriched to ALSO
    report whether the photographed item's colour and identity match — no second
    image, no extra Bedrock call. Defaults to "unknown" when the caller sends no
    expected context (so existing callers are unaffected).
    """

    color_match: Literal["match", "mismatch", "unknown"] = "unknown"
    item_match: Literal["match", "mismatch", "unknown"] = "unknown"
    observed_color: str | None = None
    expected_color: str | None = None


class GradingAudit(BaseModel):
    """Audit metadata for every grade response (Track D §21.2).
    
    Enables prompt versioning, confidence bands, and production monitoring.
    Old clients can safely ignore this — it's additive.
    """
    bedrock_model_id: str | None = None
    prompt_version: str | None = None
    confidence_band: Literal["auto_pass", "needs_review", "reject_reupload"] | None = None
    fallback_reason: str | None = None
    expected_context_used: bool = False
    quality_issues: list[str] = Field(default_factory=list)


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
    # ADDITIVE: order-vs-item verification (None unless expected context sent).
    verification: Verification | None = None
    # ADDITIVE (Track D §21.2): grading audit metadata for production monitoring.
    grading_audit: GradingAudit | None = None


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
    # ADDITIVE: order-vs-item verification (None unless expected context sent).
    verification: Verification | None = None

    # Resale pricing fields (Track B)
    resale_grade: Literal["Like New", "Very Good", "Good", "Acceptable"]
    price_range: PriceRange
    currency: str = "INR"
    pricing_rationale: str
