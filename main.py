from fastapi import FastAPI
from inference import inference_endpoint
from training import training_endpoint

def create_app() -> FastAPI:
    # Create the main FastAPI app for model endpoints.
    app = FastAPI(
        title="Titanic Model Service",
        description="Microservice for ML model training and inference.",
        version="1.0.0"
    )
    # TODO: Add health check endpoints and monitoring tools.
    app.include_router(inference_endpoint, prefix="/inference")
    app.include_router(training_endpoint, prefix="/training")
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    # TODO: Adjust settings for production deployment.
    uvicorn.run("model.main:app", host="0.0.0.0", port=5000, reload=True)
