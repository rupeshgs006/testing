from fastapi.testclient import TestClient

from app.main import app
import pytest
client = TestClient(app)

# def test_health():
#     response = client.get("/health")
#     assert response.status_code==200
#     assert response.json() =={"status":"ok"}

# def test_predict():
#     response = client.post(
#         "/predict",
#         json={"x":5}
#     )

#     assert response.status_code==200
#     assert "y" in response.json()
@pytest.fixture
def client():
    return TestClient(app)

def test_predict_batch(client):
    response = client.post(
        "/predict_batch",
        json={"xs": [1, 2, 3]}
    )

    assert response.status_code == 200
    data = response.json()
    assert "ys" in data
    assert data["ys"] 
