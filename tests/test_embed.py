"""Tests for POST /embed endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_embed_structured_attrs() -> None:
    """Structured category/grade/size/vertical produces a 384-d vector."""
    response = client.post(
        "/embed",
        json={"category": "dress", "grade": "B", "size": "M", "vertical": "fashion"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["vector"]) == 384
    assert data["model"] == "all-MiniLM-L6-v2"
    # All floats
    assert all(isinstance(v, float) for v in data["vector"])


def test_embed_free_text() -> None:
    """Free text input produces a 384-d vector."""
    response = client.post("/embed", json={"text": "red leather jacket size M"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["vector"]) == 384
    assert data["model"] == "all-MiniLM-L6-v2"


def test_embed_deterministic() -> None:
    """Same input produces the same vector."""
    payload = {"text": "blue denim jeans size 32"}
    r1 = client.post("/embed", json=payload)
    r2 = client.post("/embed", json=payload)
    assert r1.json()["vector"] == r2.json()["vector"]


def test_embed_different_inputs_differ() -> None:
    """Different inputs produce different vectors."""
    r1 = client.post("/embed", json={"text": "red shoes"})
    r2 = client.post("/embed", json={"text": "blue laptop"})
    assert r1.json()["vector"] != r2.json()["vector"]


def test_embed_empty_body_fallback() -> None:
    """Empty body still returns a valid vector (uses 'unknown item' fallback)."""
    response = client.post("/embed", json={})
    assert response.status_code == 200
    data = response.json()
    assert len(data["vector"]) == 384
