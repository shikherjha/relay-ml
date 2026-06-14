import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from app.main import app
from app.schemas.passport import ConditionPassport, Defect, FitFlag, FitFlagsResponse


ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "relay-contracts"
SCHEMAS = CONTRACTS / "schemas"
EXAMPLES = CONTRACTS / "examples"

client = TestClient(app)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validator(schema_name: str) -> Draft202012Validator:
    schema = _load_json(SCHEMAS / schema_name)
    defect_schema = _load_json(SCHEMAS / "defect.schema.json")
    registry = Registry().with_resource(
        "https://relay.dev/contracts/v1/defect.schema.json",
        Resource.from_contents(defect_schema),
    )
    return Draft202012Validator(
        schema,
        registry=registry,
        format_checker=FormatChecker(),
    )


def _assert_valid(schema_name: str, payload: dict) -> None:
    _validator(schema_name).validate(payload)


def test_relay_contract_examples_validate_against_shared_schemas() -> None:
    _assert_valid(
        "condition-passport.schema.json",
        _load_json(EXAMPLES / "condition-passport.electronics.json"),
    )
    _assert_valid(
        "condition-passport.schema.json",
        _load_json(EXAMPLES / "condition-passport.fashion.json"),
    )
    _assert_valid("fit-flags.schema.json", _load_json(EXAMPLES / "fit-flags.json"))


def test_local_condition_passport_validates_against_shared_schema() -> None:
    passport = ConditionPassport(
        unit_id="3f1c9b2a-7d4e-4a1b-9c3f-1a2b3c4d5e6f",
        return_id="a1b2c3d4-e5f6-4789-90ab-cdef12345678",
        grade="B+",
        grade_numeric=0.78,
        category="jeans",
        vertical="fashion",
        disposition_hint="p2p_resale",
        defects=[
            Defect(
                type="stain",
                severity="minor",
                bbox=[120, 340, 60, 45],
                confidence=0.81,
                description="Small faint mark near left knee",
            )
        ],
        packaging_state="missing",
        confidence=0.88,
        media_hashes=[
            "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        ],
        passport_hash="0" * 64,
        graded_at=datetime.now(timezone.utc),
        model_tier_used="cnn-v1",
    )

    _assert_valid("condition-passport.schema.json", passport.model_dump(mode="json"))


def test_local_fit_flags_response_validates_against_shared_schema() -> None:
    response = FitFlagsResponse(
        sku_id="SKU-123",
        flags=[
            FitFlag(
                type="runs_large",
                message="Category fit history leans large.",
                confidence=0.87,
            )
        ],
        source="fit_aggregates_v1:renttherunway",
    )

    _assert_valid("fit-flags.schema.json", response.model_dump(mode="json"))


def test_fit_flags_endpoint_output_validates_against_shared_schema() -> None:
    response = client.post(
        "/fit-flags",
        json={"sku_id": "SKU-123", "category": "coat"},
    )

    assert response.status_code == 200
    _assert_valid("fit-flags.schema.json", response.json())
