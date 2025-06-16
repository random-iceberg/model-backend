import json
from os import environ
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app
from schemas import (
    AlgoDt,
    DatasetFeature,
    InferenceRequest,
    InferenceResponse,
    Model,
    ModelParams,
)


@pytest.fixture(scope="module")
def data_path(tmp_path_factory):
    # single tmp_path; tests will share this for data files
    # module-wide dataset directory
    tmp_path = tmp_path_factory.mktemp("dataset")
    environ["DATASET_PATH"] = str(tmp_path)
    return tmp_path


@pytest.fixture
def models_path(tmp_path):
    pass
    # fresh models directory for each test
    environ["MODELS_PATH"] = str(tmp_path)
    return tmp_path


@pytest.fixture
def client(data_path: Path, models_path: Path):
    _ = data_path
    _ = models_path
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def default_data_path(tmp_path_factory: pytest.TempPathFactory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture
def default_client(default_data_path: Path):
    environ["DATA_PREFIX"] = str(default_data_path)
    with TestClient(app) as client:
        yield client


def test_list_models(default_client: TestClient):
    client = default_client
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    # Should return a non-empty list of models
    assert isinstance(data, list)
    assert data, "Expected non-empty list of models"
    # Validate each entry against the Model schema
    for item in data:
        model = Model.model_validate(item)
        assert model.id, "Model id should not be empty"
        assert model.params.features, "Features set should not be empty"
        assert isinstance(model.info.accuracy, float)


def train_some_model(client: TestClient):
    # Prepare payload using the ModelParams schema
    params = ModelParams(
        algo=AlgoDt(),
        random_state=42,
        features={"pclass", "sex", "abc"},
    )
    payload = json.loads(params.model_dump_json())
    return client.post("/models/train", json=payload)


def test_train_model(client: TestClient):
    # Prepare payload using the ModelParams schema
    response = train_some_model(client)
    assert response.status_code == 200
    data = response.json()
    model = Model.model_validate(data)
    assert model.id.startswith("trained-"), "Expected model id to start with 'trained-'"
    assert model.params.random_state == 42
    assert model.params.features == {DatasetFeature.pclass, DatasetFeature.sex}


def test_run_inference(default_client: TestClient):
    client = default_client

    # Prepare inference payload using InferenceRequest schema
    infer_request = InferenceRequest(
        pclass=1,
        sex="male",
        age=30.0,
        fare=100.0,
        travelled_alone=False,
        embarked="cherbourg",
        title="mr",
        cabin_known=True,
    )
    model_id = "svm"
    response = client.post(
        f"/models/{model_id}/predict", json=infer_request.model_dump()
    )
    assert response.status_code == 200
    data = response.json()
    result = InferenceResponse.model_validate(data)
    # Ensure 'survived' key exists and is a boolean
    assert hasattr(result, "survived")
    assert isinstance(result.survived, bool)
    assert result.probability > 0.5


def test_delete_model(client: TestClient):
    model = train_some_model(client).json()
    model_id = model["id"]
    response = client.delete(f"/models/{model_id}")
    assert response.status_code == 200
    data = response.json()
    # Should return True indicating successful deletion
    assert data is True


def test_delete_protected_model(default_client: TestClient):
    client = default_client
    response = client.delete("/models/knn")
    assert response.status_code == 403


def test_delete_nonexistent_model(default_client: TestClient):
    client = default_client
    response = client.delete("/models/nonexistent")
    assert response.status_code == 404
