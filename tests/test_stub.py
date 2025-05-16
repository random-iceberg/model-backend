from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_stub():
    response = client.post("/inference", json={"data": {}})
    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code}"
    )
