import logging
from uuid import uuid4

from fastapi import APIRouter, Request

from utils.models import LoadedModels

from schemas import (
    InferenceRequest,
    InferenceResponse,
    Model,
    ModelParams,
)

# Configure module-level logger
logger = logging.getLogger(__name__)

models_router = APIRouter()


@models_router.get("/")
async def list_models(req: Request) -> list[Model]:
    """List trained models"""

    return []


@models_router.post("/train")
async def train_model(params: ModelParams, req: Request) -> Model:
    """Train a new model"""
    models: LoadedModels = req.state.models
    model_id = "trained-" + uuid4().hex
    model = await models.train_model(model_id, params)

    return model


@models_router.post("/{model_id}/predict")
async def run_inference(model_id: str, input: InferenceRequest, req: Request) -> InferenceResponse:
    """
    Endpoint to run machine learning inference.
    """
    return InferenceResponse(survived=False)


@models_router.delete("/{model_id}")
async def delete_model(model_id: str, req: Request) -> bool:
    return True
