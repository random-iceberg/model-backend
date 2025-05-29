import logging
from contextlib import asynccontextmanager
from os import environ
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from models_router import models_router
from utils.data import load_data
from utils.models import LoadedModels

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_prefix = Path(environ.get("DATA_PREFIX", "tmp/data"))
    # Dataset and models dirs
    dataset_path = Path(environ.get("DATASET_PATH", str(data_prefix / "dataset")))
    models_path = Path(environ.get("MODELS_PATH", str(data_prefix / "models")))

    # Load the Titanic data and existing models
    _ = load_data(dataset_path)
    models = LoadedModels(models_path, dataset_path)
    await models.load_existing()

    yield {"models": models}

    # Clean up


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance for the Model API.
    Includes health check endpoints and mounts the inference and training routers.

    TODO:
      - Load environment-specific configurations (e.g., via pydantic BaseSettings)
      - Integrate metrics/monitoring endpoints (e.g., Prometheus exporter)
    """
    app = FastAPI(
        title="Titanic Model Service",
        description="Microservice for ML model training and inference.",
        docs_url="/docs",
        redoc_url=None,
        swagger_ui_parameters={
            "syntaxHighlight": True,
            "docExpansion": "none",
        },
        version="1.0.0",
        lifespan=lifespan,
    )

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """
        Simple health check endpoint to verify the service is running.
        """
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/docs")

    # Include routers from submodules

    # Mount inference and training endpoints
    app.include_router(models_router, prefix="/models", tags=["Models"])

    return app


app = create_app()
