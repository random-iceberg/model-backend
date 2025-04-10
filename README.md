# Model Service for Titanic Survivor Prediction Application

The Model Service is a dedicated microservice responsible for machine learning model training, inference, and management for the Titanic Survivor Prediction Application. Developed with FastAPI and scikit-learn, it is containerized and integrated with Docker Compose for seamless deployment.

## Project Structure

The Model Service handles machine learning model training, inference, and management using FastAPI and scikit-learn. Its modular structure ensures quick updates and scalability:

```
model/
├── README.md             # Model service documentation
├── main.py               # Entry point of the model service
├── requirements.txt      # Python dependencies for ML functionality
├── training/             # Scripts and configurations for model training
├── inference/            # Modules for real-time inference and API endpoints
└── tests/                # Unit and integration tests for model functionality
```

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

The Model Service performs the following functions:
- **Machine Learning Inference**: Processes prediction requests using ML algorithms (Random Forest, SVM, etc.).
- **Model Training and Management**: Supports training new models and managing existing ones through secure endpoints.
- **High Performance**: Designed for asynchronous processing and real-time predictions.
- **Containerized Deployment**: Fully integrated with Docker Compose for seamless production environments.

## Features

- **RESTful API with OpenAPI Documentation**: Easily accessible through Swagger UI.
- **Integrated ML Algorithms**: Leverages scikit-learn for accurate predictions.
- **Administrative Controls**: Endpoints for model training, listing, and deletion.
- **Zero Local Configuration**: Pre-configured to run using Docker Compose.
- **Optimized for Scalability**: Supports high throughput and low latency inference.
- **Dataset Download via kagglehub:**  
  To download the Titanic dataset:
  ```python
  import kagglehub
  kagglehub.competition_download('titanic')
  ```

## Getting Started

1. **Clone the Repository:**  
   Navigate to the `model/` directory.
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the Titanic Dataset:**
   ```bash
   python -m model.data_downloader
   ```
4. **Start the Service:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 5000
   ```
5. **Access API Docs:**  
   Open [http://localhost:5000/docs](http://localhost:5000/docs) in your browser.

## Development

- **Run Locally:**
  ```bash
  uvicorn main:app --reload --host 0.0.0.0 --port 5000
  ```
  This command starts the model service with auto-reload for development convenience.

- **Code Quality Tools:**
  - Run `flake8` for linting.
  - Run `black` for consistent code formatting.

## Testing

Execute tests using:
```bash
pytest
```
This suite covers both unit and integration tests to ensure functionality and performance.

## Deployment

This service is production-ready and can be deployed using Docker Compose. From the repository root, run:
```bash
docker-compose up --build -d
```

## Troubleshooting

- **Service Logs:**
  ```bash
  docker-compose logs model
  ```
- **Container Health:**
  ```bash
  docker-compose ps
  ```
- **API Verification:**
  Access the Swagger UI at [http://localhost:5000/docs](http://localhost:5000/docs) to verify endpoint functionality.

## Documentation

For additional details on API endpoints, ML algorithm integration, and administrative controls, refer to the comprehensive documentation maintained in the [docs submodule](https://mygit.th-deg.de/schober-teaching/student-projects/ain-23-software-engineering/ss-25/Random_Iceberg/docker-compose/-/wikis/home).
