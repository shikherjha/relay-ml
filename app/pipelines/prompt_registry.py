"""Prompt registry for Bedrock grading.

Named, versioned prompts so every grade response can report which prompt
produced it. This enables A/B testing, audit trails, and roll-forward on
regressions.

Each prompt is a callable that takes the category + optional context and returns
the full prompt string. The registry maps a version tag to its builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PromptSpec:
    """A named, versioned prompt specification."""
    name: str
    version: str
    vertical: str  # "fashion" | "electronics" | "any"
    description: str
    builder: Callable[..., str]

    @property
    def full_id(self) -> str:
        return f"{self.name}_v{self.version}"


def _grading_fashion_v1(category: str, **kwargs) -> str:
    verification_suffix = kwargs.get("verification_suffix", "")
    return f"""You are an AI product condition grader for a circular commerce platform.

Analyze this FASHION product image (category: {category}) and assess its physical condition.

Look for: stains, tears, pilling, loose threads, discoloration, missing buttons/zippers, stretching, fading, odor stains, seam damage.

Return a JSON object with these fields:
- "grade": one of "A+", "A", "B+", "B", "C", "D"
  - A+ = pristine/sealed, looks brand new, tags still on
  - A = like new, no visible wear, could pass as new
  - B+ = good condition, light cosmetic wear only (faint pilling, minor fade)
  - B = fair, visible wear but fully wearable (light stains, minor holes)
  - C = poor, significant damage or defects (tears, heavy staining, missing parts)
  - D = heavily damaged, not wearable without repair
- "defect_type": one of "none", "scuff", "stain", "tear", "dent", "discoloration", "missing_part", "other"
  - Use "none" if the product appears undamaged (grade A+ or A)
- "confidence": float 0.0-1.0 how confident you are in your assessment
- "description": brief description of the product condition (1-2 sentences)

Important: If the product looks new or undamaged, grade it A+ or A with defect_type "none".
Only report defects you can actually see in the image.{verification_suffix}

Return ONLY valid JSON, no other text."""


def _grading_electronics_v1(category: str, **kwargs) -> str:
    verification_suffix = kwargs.get("verification_suffix", "")
    return f"""You are an AI product condition grader for a circular commerce platform.

Analyze this ELECTRONICS product image (category: {category}) and assess its physical condition.

Look for: screen damage (cracks, dead pixels, scratches), dents, scuffs on casing, port damage, missing accessories, water damage indicators, functional indicators (power button, screen display).

Return a JSON object with these fields:
- "grade": one of "A+", "A", "B+", "B", "C", "D"
  - A+ = pristine/sealed, factory packaging, zero signs of use
  - A = like new, no visible wear, all accessories present
  - B+ = good condition, minor cosmetic scratches only
  - B = fair, visible wear marks but fully functional
  - C = poor, significant damage (cracked screen, dented casing, missing parts)
  - D = heavily damaged, may not be functional
- "defect_type": one of "none", "crack", "scuff", "dent", "screen_damage", "water_damage", "functional_fault", "missing_part", "other"
  - Use "none" if the product appears undamaged (grade A+ or A)
- "confidence": float 0.0-1.0 how confident you are in your assessment
- "description": brief description of the product condition (1-2 sentences)
- "bbox": optional [x, y, w, h] bounding box around the primary defect if clearly localizable (normalized 0-1 coordinates), else null

Important: If the product looks new or undamaged, grade it A+ or A with defect_type "none".
Only report defects you can actually see in the image.{verification_suffix}

Return ONLY valid JSON, no other text."""


def _resale_pricing_v1(category: str, **kwargs) -> str:
    original_price = kwargs.get("original_price", "unknown")
    age_days = kwargs.get("age_days", "unknown")
    return f"""You are an AI product condition grader AND resale price estimator for a circular commerce platform.

Analyze this product image (category: {category}, original price: {original_price}, age: {age_days} days) and assess:
1. Physical condition (grade A+ through D)
2. A fair resale price range

Return a JSON object with:
- "grade": one of "A+", "A", "B+", "B", "C", "D"
- "defect_type": one of "none", "scuff", "crack", "stain", "tear", "dent", "discoloration", "missing_part", "screen_damage", "water_damage", "functional_fault", "other"
- "confidence": float 0.0-1.0
- "description": brief condition description
- "resale_grade": one of "Like New", "Very Good", "Good", "Acceptable"
- "price_range": {{"min": float, "max": float}} in INR
- "pricing_rationale": brief explanation of the price drivers

Return ONLY valid JSON, no other text."""


def _match_rank_v1(wish: str, candidates: list, **kwargs) -> str:
    size = kwargs.get("size")
    max_price = kwargs.get("max_price")
    cand_text = "\n".join(
        f"  - id={c.get('unit_id','?')}, title={c.get('title','?')}, category={c.get('category','?')}, price={c.get('price','?')}"
        for c in candidates[:20]
    )
    constraints = []
    if size:
        constraints.append(f"size: {size}")
    if max_price:
        constraints.append(f"budget: ≤{max_price} INR")
    constraint_str = f" ({', '.join(constraints)})" if constraints else ""

    return f"""You are a product matching AI for a second-hand marketplace.

A buyer wants: "{wish}"{constraint_str}

Rate each candidate 0.0-1.0 on how well it matches the buyer's intent:

Candidates:
{cand_text}

Rules:
- 1.0 = exact match (same product type, right category)
- 0.5 = related but different (same vertical, different subcategory)
- 0.0 = completely unrelated or wrong vertical
- A "jeans" wish should NEVER match a "jacket" — different garments
- Category match is more important than brand or price

Return a JSON array: [{{"unit_id": "...", "score": 0.0-1.0}}]
Return ONLY valid JSON, no other text."""


# --- Registry ---

PROMPTS: dict[str, PromptSpec] = {
    "grading_fashion_v1": PromptSpec(
        name="grading_fashion", version="1", vertical="fashion",
        description="Fashion product condition grading with fabric-specific defects",
        builder=_grading_fashion_v1,
    ),
    "grading_electronics_v1": PromptSpec(
        name="grading_electronics", version="1", vertical="electronics",
        description="Electronics product condition grading with hardware-specific defects",
        builder=_grading_electronics_v1,
    ),
    "resale_pricing_v1": PromptSpec(
        name="resale_pricing", version="1", vertical="any",
        description="Combined condition grading + resale price estimation",
        builder=_resale_pricing_v1,
    ),
    "match_rank_v1": PromptSpec(
        name="match_rank", version="1", vertical="any",
        description="LLM reranker for wishlist matching candidates",
        builder=_match_rank_v1,
    ),
}


def get_prompt(prompt_id: str) -> PromptSpec:
    """Get a prompt spec by its full ID (e.g. 'grading_fashion_v1')."""
    if prompt_id not in PROMPTS:
        raise ValueError(f"Unknown prompt: {prompt_id}. Available: {list(PROMPTS.keys())}")
    return PROMPTS[prompt_id]


def select_grading_prompt(category: str, vertical: str | None = None) -> PromptSpec:
    """Select the best grading prompt for a given category/vertical."""
    if vertical == "electronics" or category in (
        "laptop", "phone", "smartphone", "headphones", "speaker", "smartwatch",
        "camera", "keyboard", "tablet", "monitor", "mouse",
    ):
        return PROMPTS["grading_electronics_v1"]
    return PROMPTS["grading_fashion_v1"]


def build_grading_prompt(category: str, vertical: str | None = None, **kwargs) -> tuple[str, str]:
    """Build the prompt text and return (prompt_text, prompt_version).
    
    Returns the prompt string and the prompt_version identifier for audit.
    """
    spec = select_grading_prompt(category, vertical)
    prompt_text = spec.builder(category, **kwargs)
    return prompt_text, spec.full_id
