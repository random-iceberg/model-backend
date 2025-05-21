# Model API for Titanic Survivor Prediction Application

## Overview

The Model API is a dedicated FastAPI microservice that handles machine learning model training and real-time inference for the Titanic Survivor Prediction Application. It leverages scikit-learn for reliable predictions and includes endpoints for both initiating training and running inference. Fully containerized and integrated via Docker Compose, the service is designed to meet production standards with robust error handling, logging, and scalability.

## Features

- **Real-Time Inference:**  
  Offers fast and accurate predictions via a RESTful API using ML models (e.g., Random Forest, SVM).
- **Model Training & Management:**  
  Provides endpoints for initiating training, monitoring progress, and managing model versions.
- **Interactive API Documentation:**  
  Automatically generated documentation available at `/docs`.
- **Containerized & Scalable:**  
  Easily deployed with Docker Compose, ensuring consistency across environments.
- **Robust Error Handling and Logging:**  
  Detailed logging for debugging and operational insights.
- **Dataset Integration:**  
  Includes utilities (via `kagglehub`) to download and manage the Titanic dataset for model training.

## Project Structure

```plaintext
model/
├── README.md              # Model API documentation (this file)
├── main.py                # Entry point for the Model API service
├── requirements.txt       # Python dependencies for machine learning functionality
├── data_downloader.py     # Script to download the Titanic dataset via Kaggle
├── inference/             # Modules for real-time inference endpoints
│   └── inference_endpoint.py  # Inference API endpoint definitions
├── training/             # Modules for model training endpoints
│   └── training_endpoint.py   # Training API endpoint definitions
└── tests/                 # Unit and integration tests for the Model API
```

## Getting Started

### Prerequisites

- Python 3.9.x
- Virtual environment (recommended)
- Docker & Docker Compose (for containerized deployment)

### Setup Instructions

Follow these steps to set up your development environment:

1. **Clone the Repository (with Submodules)**  
   Clone the repository together with all its submodules:
   ```bash
   git clone --recurse-submodules https://mygit.th-deg.de/schober-teaching/student-projects/ain-23-software-engineering/ss-25/Random_Iceberg/web-backend.git
   ```

2. **Enter the Project Directory**  
   Change directory into the Docker Compose folder:
   ```bash
   cd docker-compose
   ```

3. **Checkout the Development Branch**  
   Create and switch to a local branch named `dev` that tracks the remote development branch:
   ```bash
   git checkout -b dev origin/dev
   ```

4. **Update All Submodules**  
   Initialize and update every submodule recursively:
   ```bash
   git submodule update --init --recursive
   ```

5. **Create and Activate a Virtual Environment:**
   (From the appropriate directory, e.g., `model/`) activate Python 3.9.x virtual environment:
   ```bash
   py -3.9 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

6. **Install Dependencies:**
   ```bash
   pip install -e '.[dev]'
   ```

7. **Download the Titanic Dataset (Optional):**
   To fetch the dataset via Kaggle, run:
   ```bash
   python -m model.data_downloader
   ```

8. **Run the Service Locally:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 5000
   ```
   Access the API and its documentation at [http://localhost:5000/docs](http://localhost:5000/docs).

## Development & Testing

- **Hot Reloading:**  
  Use uvicorn’s `--reload` option during development for immediate feedback.
- **Code Quality:**  
  Follow best practices with linters (flake8) and formatters (black); perform thorough code reviews.
- **Testing:**  
  Execute tests with:
  ```bash
  pytest
  ```

## Deployment

Deploy the Model API as part of the complete application stack using Docker Compose:
```bash
docker-compose up --build -d
```
This command builds and launches the Model API along with other interconnected services.

## Troubleshooting

- **View Docker Logs:**
  ```bash
  docker-compose logs model
  ```
- **Check Container Status:**
  ```bash
  docker-compose ps
  ```
- **API Verification:**  
  Validate functionality by visiting [http://localhost:5000/docs](http://localhost:5000/docs).

## Documentation & References

For detailed API endpoints, training parameters, and inference logic, please refer to the [Project Charter](#) and the extensive documentation provided within the `docs/` submodule.

---

Maintained by **team/random_iceberg**.
