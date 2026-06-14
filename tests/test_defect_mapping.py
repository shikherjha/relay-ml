from app.pipelines.image_grade import _normalize_defect_type


def test_damaged_screen_category_maps_to_screen_damage() -> None:
    assert _normalize_defect_type("damaged", "phone screen") == "screen_damage"


def test_damaged_electronics_category_maps_to_functional_fault() -> None:
    assert _normalize_defect_type("damaged", "headphones") == "functional_fault"


def test_damaged_fashion_category_maps_to_tear() -> None:
    assert _normalize_defect_type("damaged", "denim jeans") == "tear"


def test_crack_screen_category_maps_to_screen_damage() -> None:
    assert _normalize_defect_type("crack", "tablet display") == "screen_damage"


def test_unknown_defect_label_maps_to_other() -> None:
    assert _normalize_defect_type("weird artifact", "books") == "other"
