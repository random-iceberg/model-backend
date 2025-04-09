# Model Service for Titanic Survivor Prediction Application

The Model Service is a dedicated microservice responsible for handling machine learning model training and inference for the Titanic Survivor Prediction Application. Developed using FastAPI and scikit-learn, this service is containerized and fully integrated with the overall Docker Compose orchestration—ensuring zero manual configuration and a robust production environment.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)

## Overview

The Model Service performs the following key functions:
- **Machine Learning Inference:**  
  Processes prediction requests using robust ML algorithms such as Random Forest, SVM, Logistic Regression, and Decision Trees.
- **Model Training and Management:**  
  Allows the training of new models and the management (listing, reloading, and deletion) of existing models through secure administrative endpoints.
- **High Performance:**  
  Designed for asynchronous processing and optimized for real-time predictions, ensuring low latency communication with the backend.
- **Containerized Deployment:**  
  Fully containerized and orchestrated via Docker Compose for seamless integration within the application stack.

## Features

- **RESTful API with OpenAPI Documentation:**  
  Exposes secure endpoints for both inference and model management accessible via Swagger UI.
- **Integrated ML Algorithms:**  
  Leverages scikit-learn to deliver accurate and efficient predictions.
- **Administrative Controls:**  
  Provides endpoints for model training, listing, and deletion, following the Project Charter’s administrative requirements.
- **Zero Local Configuration:**  
  Pre-configured to run within Docker Compose—eliminating the need for manual environment variable setups.
- **Scalability and Performance:**  
  Optimized for high throughput and low latency to support real-time inference at scale.

## Getting Started

1. **Clone the Repository:**

   ```bash
   git clone https://your.git.repo/model.git
   cd model
   ```

2. **Create a Virtual Environment and Install Dependencies:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Development

- **Run Locally:**  
  Start the model service with the following command. This launches FastAPI with auto-reload enabled for development convenience:
  
  ```bash
  uvicorn main:app --reload --host 0.0.0.0 --port 5000
  ```

- **Code Quality:**  
  - **Linting:** Use `flake8` to check for code style issues.
  - **Formatting:** Use `black` to maintain consistent code formatting.

## Testing

- **Unit and Integration Tests:**  
  The Model Service includes a comprehensive suite of tests to ensure functionality and performance. Run tests using:
  
  ```bash
  pytest
  ```

## Deployment

This service is pre-configured for production deployment using Docker. No additional configuration or environment variable setup is required.

- **Using Docker Compose:**  
  From the root of the overall repository, deploy the full application stack (including the Model Service) with:
  
  ```bash
  docker-compose up --build -d
  ```

  This command builds and launches all services (Frontend, Backend, Model Service, and Supabase).

## Troubleshooting

- **Common Issues:**
  - **Service Logs:**  
    Check the Model Service logs for errors:
    
    ```bash
    docker-compose logs model
    ```
    
  - **Container Health:**  
    Verify that the container is running correctly:
    
    ```bash
    docker-compose ps
    ```
    
  - **API Endpoints:**  
    Access the integrated Swagger UI at [http://localhost:5000/docs](http://localhost:5000/docs) to verify that API endpoints are operational and correctly documented.

## Documentation

For further details on API endpoints, ML algorithm integration, and administrative controls, please refer to the comprehensive documentation maintained within the `/docs` submodule. All documentation is kept up-to-date with the latest modifications as outlined in the Project Charter.

---

*This Model Service is a key component of the Titanic Survivor Prediction Application, designed to deliver robust, real-time ML predictions while ensuring smooth integration and scalable performance across the entire system.*