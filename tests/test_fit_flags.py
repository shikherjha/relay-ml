import json

from fastapi.testclient import TestClient

from app.main import app
from app.pipelines import fit_flags
from scripts.build_fit_aggregates import build_aggregates


client = TestClient(app)


def test_fit_flags_uses_aggregate_for_known_category() -> None:
    response = client.post(
        "/fit-flags",
        json={"sku_id": "sku-1", "category": "coat"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["sku_id"] == "sku-1"
    assert data["source"].startswith("fit_aggregates_v1:")
    assert data["flags"][0]["type"] == "runs_large"


def test_fit_flags_falls_back_when_category_unknown(monkeypatch) -> None:
    fit_flags.load_fit_aggregates.cache_clear()
    monkeypatch.setattr(fit_flags.settings, "fit_aggregates_path", "missing.json")

    try:
        response = client.post(
            "/fit-flags",
            json={"sku_id": "sku-2", "category": "unknown hardgood"},
        )
    finally:
        fit_flags.load_fit_aggregates.cache_clear()

    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "rules_v1"
    assert data["flags"][0]["type"] == "critical_fit"


def test_build_fit_aggregates_from_jsonl(tmp_path) -> None:
    source = tmp_path / "renttherunway_final_data.json"
    rows = [
        {"category": "coat", "fit": "large"},
        {"category": "coat", "fit": "large"},
        {"category": "coat", "fit": "fit"},
        {"category": "dress", "fit": "small"},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    records = build_aggregates([source])

    coat = next(record for record in records if record["category"] == "coat")
    assert coat["source"] == "renttherunway"
    assert coat["total"] == 3
    assert coat["large_count"] == 2
    assert coat["runs_large_rate"] == 2 / 3
