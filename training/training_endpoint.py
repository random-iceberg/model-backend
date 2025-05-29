# !!! TODO: should not be used !!!
import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel

from schemas import DEFAULT_FEATURE_SET, AlgoSvm, ModelParams
from utils.models import LoadedModels

# Configure module-level logger
logger = logging.getLogger(__name__)


class TrainingRequest(BaseModel):
    """
    Data model for training request.
    TODO:
      - Define specific training parameters (e.g., selected algorithms, hyperparameters, training data subset)
    """

    parameters: dict


class TrainingResponse(BaseModel):
    """
    Data model for training response.
    TODO:
      - Include additional details such as training metrics, expected completion time, etc.
    """

    status: str
    accuracy: float


training_endpoint = APIRouter()


@training_endpoint.post(
    "/", response_model=TrainingResponse, summary="Initiate ML Model Training"
)
async def run_training(request: TrainingRequest, req: Request) -> TrainingResponse:
    """
    Endpoint to initiate model training.

    !!! TODO: remove. web-backend should use /models/train !!!
    """
    models: LoadedModels = req.state.models
    model_id = request.parameters["model_id"]
    params = ModelParams(  # Ignore incoming parameters. They are of a wrong format.
        algo=AlgoSvm(), random_state=None, features=DEFAULT_FEATURE_SET
    )

    model = await models.train_model(model_id, params)

    return TrainingResponse(accuracy=model.info.accuracy, status="Finished")
