from fastapi import FastAPI, APIRouter
from fastapi.responses import RedirectResponse
import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
)
logger = logging.getLogger(__name__)

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
        redoc_url="/redoc",
        swagger_ui_parameters={
            "syntaxHighlight": True,
            "docExpansion": "none"
        },
        version="1.0.0"
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
    from inference.inference_endpoint import inference_endpoint
    from training.training_endpoint import training_endpoint

    # Mount inference and training endpoints
    app.include_router(inference_endpoint, prefix="/inference", tags=["Inference"])
    app.include_router(training_endpoint, prefix="/training", tags=["Training"])

    return app

app = create_app()

