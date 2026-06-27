"""Tests for the prompt registry (Track D §21.2)."""

from app.pipelines.prompt_registry import (
    PROMPTS,
    build_grading_prompt,
    get_prompt,
    select_grading_prompt,
)


def test_registry_has_all_required_prompts():
    required = {"grading_fashion_v1", "grading_electronics_v1", "resale_pricing_v1", "match_rank_v1"}
    assert required.issubset(set(PROMPTS.keys()))


def test_select_fashion_for_dress():
    spec = select_grading_prompt("dress", "fashion")
    assert spec.full_id == "grading_fashion_v1"


def test_select_electronics_for_laptop():
    spec = select_grading_prompt("laptop", "electronics")
    assert spec.full_id == "grading_electronics_v1"


def test_select_electronics_for_phone_without_vertical():
    spec = select_grading_prompt("phone")
    assert spec.full_id == "grading_electronics_v1"


def test_select_fashion_default():
    spec = select_grading_prompt("unknown_category")
    assert spec.full_id == "grading_fashion_v1"


def test_build_returns_prompt_and_version():
    prompt, version = build_grading_prompt("jeans", "fashion")
    assert "FASHION" in prompt
    assert version == "grading_fashion_v1"
    assert "jeans" in prompt


def test_build_electronics_mentions_screen():
    prompt, version = build_grading_prompt("smartphone", "electronics")
    assert "screen" in prompt.lower()
    assert version == "grading_electronics_v1"


def test_get_prompt_raises_on_unknown():
    import pytest
    with pytest.raises(ValueError, match="Unknown prompt"):
        get_prompt("nonexistent_v99")


def test_all_prompts_produce_nonempty_strings():
    for spec in PROMPTS.values():
        if spec.name == "match_rank":
            text = spec.builder("laptop", candidates=[{"unit_id": "u1", "title": "MacBook", "category": "laptop", "price": 50000}])
        elif spec.name == "resale_pricing":
            text = spec.builder("hoodie", original_price=2000, age_days=30)
        else:
            text = spec.builder("hoodie")
        assert len(text) > 100
        assert "JSON" in text
