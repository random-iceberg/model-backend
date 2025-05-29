import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from schemas import (
    InferenceRequest,
    InferenceResponse,
    Model,
    ModelParams,
)
from utils.data import prepare_passenger_data
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
async def train_model(params: ModelParams, req: Request) -> Model:
    """Train a new model"""
    models: LoadedModels = req.state.models
    model_id = "trained-" + uuid4().hex
    model = await models.train_model(model_id, params)

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
    if not await models.delete_model(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    return True
