import logging
from textwrap import dedent
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError

from schemas import (
    DatasetFeature,
    InferenceRequest,
    InferenceResponse,
    Model,
    ModelParams,
)
from utils.data import filter_features, prepare_passenger_data
from utils.models import LoadedModels

# Configure module-level logger
logger = logging.getLogger(__name__)

models_router = APIRouter()


@models_router.get("/")
async def list_models(req: Request) -> list[Model]:
    """List trained models"""
    models: LoadedModels = req.state.models

    return [model.desc for model in models.models.values()]


@models_router.post("/train")
async def train_model(params: ModelParams[str], req: Request) -> Model:
    """Train a new model"""
    filtered_features = filter_features(params.features)
    if not filtered_features:
        raise RequestValidationError(
            dedent(
                f"""\
                    None of the requests features are available for training.
                    Use some of those: {[x.value for x in DatasetFeature]}
                """
            )
        )
    filtered_params = ModelParams[DatasetFeature](
        algo=params.algo,
        random_state=params.random_state,
        features=filtered_features,
    )

    models: LoadedModels = req.state.models
    model_id = "trained-" + uuid4().hex
    model = await models.train_model(model_id, filtered_params)

    return model


@models_router.post("/{model_id}/predict")
async def run_inference(
    model_id: str, input: InferenceRequest, req: Request
) -> InferenceResponse:
    """
    Endpoint to run machine learning inference.
    """
    models: LoadedModels = req.state.models
    model = models[model_id]

    prepared_data = prepare_passenger_data(input, model.desc.params.features)
    print(prepared_data)
    prediction_probs = model.impl.predict_proba(prepared_data)[0]
    prediction = prediction_probs.argmax().item()

    return InferenceResponse(
        survived=prediction, probability=prediction_probs[prediction]
    )


@models_router.delete("/{model_id}")
async def delete_model(model_id: str, req: Request) -> bool:
    models: LoadedModels = req.state.models
    model = models.models.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if not model.desc.removable:
        raise HTTPException(status_code=403, detail="Model marked as non-removable")

    await models.delete_model(model_id)
    return True
