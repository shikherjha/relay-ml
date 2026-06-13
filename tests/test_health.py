from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_returns_model_status() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["cnn_version"] == "v1"
    assert "model_loaded" in data
    assert "model_bytes" in data
