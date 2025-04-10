from fastapi import FastAPI, APIRouter
import uvicorn
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
        version="1.0.0"
    )

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """
        Simple health check endpoint to verify the service is running.
        """
        return {"status": "ok"}

    # Include routers from submodules
    from inference.inference_endpoint import inference_endpoint
    from training.training_endpoint import training_endpoint

    # Mount inference and training endpoints
    app.include_router(inference_endpoint, prefix="/inference", tags=["Inference"])
    app.include_router(training_endpoint, prefix="/training", tags=["Training"])

    return app

app = create_app()

if __name__ == "__main__":
    # TODO:
    #   - Update production server settings (e.g., disable reload)
    #   - Consider using Gunicorn with Uvicorn workers for production deployment.
    uvicorn.run("model.main:app", host="0.0.0.0", port=5000, reload=True)
