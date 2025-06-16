# Model Microservice

Machine learning service for training and inference using scikit-learn models.

## 🚀 Quick Start (Zero Configuration)

```bash
# From the project root directory
docker compose -f 'compose/compose.dev.yaml' up -d --build

# Access Swagger UI
open http://localhost:8001/docs  # Development mode
```

No setup needed! The service starts with pre-trained models ready for inference.

## 📋 Features

- **5 ML Algorithms**: Random Forest, SVM, Decision Tree, KNN, Logistic Regression
- **Real-time Inference**: Fast predictions with model caching
- **Model Training**: Train custom models with configurable features
- **Model Persistence**: Automatic saving and loading of trained models
- **RESTful API**: Full Swagger/OpenAPI documentation
- **Hot Reload**: Automatic restart on code changes (dev mode)

## 🏗️ API Documentation

### Interactive API Explorer
Access the Swagger UI at: **http://localhost:8001/docs** (development mode)

### Main Endpoints

- `GET /health` - Service health check
- `GET /models` - List all trained models with metadata
- `POST /models/train` - Train a new model
- `POST /models/{id}/predict` - Get prediction from specific model
- `DELETE /models/{id}` - Delete a trained model

## 🤖 Available Algorithms

| Algorithm | ID | Key Parameters | Best For |
|-----------|-----|----------------|----------|
| Random Forest | `rf` | `n_estimators=100` | General purpose, robust |
| Support Vector Machine | `svm` | kernel, C, gamma | Non-linear patterns |
| Decision Tree | `dt` | max_depth, min_samples | Interpretability |
| K-Nearest Neighbors | `knn` | `n_neighbors=5` | Local patterns |
| Logistic Regression | `lr` | solver, regularization | Linear relationships |

## 📊 Available Features

All features from the Titanic dataset:
- `pclass` - Passenger class (1, 2, 3)
- `sex` - Gender (male/female)
- `age` - Age in years
- `fare` - Ticket fare
- `embarked` - Port of embarkation
- `title` - Extracted from name (Mr, Mrs, etc.)
- `is_alone` - Traveling alone flag
- `age_class` - Age × Class interaction

## 🛠️ Development Workflow

### Using Docker Compose (Recommended)

### Testing the API with Swagger

1. Go to http://localhost:8001/docs
2. Try `/models` to see pre-loaded models
3. Test prediction with `/models/{model_id}/predict`:
   ```json
   {
     "pclass": 1,
     "sex": "female",
     "age": 30,
     "fare": 100,
     "travelled_alone": false,
     "embarked": "cherbourg",
     "title": "mrs"
   }
   ```
4. Train a custom model with `/models/train`

## 🧪 Testing

```bash
cd model

# Install dependencies (if not already done)
uv sync --no-cache --extra dev

# Run tests
uv run pytest

# Linting and formatting check
uv run ruff check
uv run ruff format --check

# Auto-fix formatting
uv run ruff format
```

## 📁 Project Structure

```
model/
├── main.py              # FastAPI application
├── models_router.py     # Model management endpoints
├── schemas.py           # Pydantic data models
├── train.py            # Training script
├── utils/
│   ├── data.py         # Data preprocessing
│   ├── models.py       # Model loading/saving
│   └── model_factory.py # Algorithm factory
├── data/               # Included dataset
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
└── tests/              # Test suite
```

## 🔧 Model Training

### Using Swagger UI

1. Go to http://localhost:8001/docs
2. Expand `POST /models/train`
3. Click "Try it out"
4. Use this example request:
   ```json
   {
     "algo": {
       "name": "rf",
       "n_estimators": 150
     },
     "features": ["pclass", "sex", "age", "fare"],
     "random_state": 42
   }
   ```
5. Click "Execute"

### Training Response
```json
{
  "id": "trained-abc123",
  "params": { ... },
  "info": {
    "accuracy": 0.85
  }
}
```

## 📈 Making Predictions

### Using Swagger UI

1. Get model ID from `/models` endpoint
2. Use `POST /models/{model_id}/predict`
3. Provide passenger data
4. Receive survival prediction with probability

### Prediction Request
```json
{
  "pclass": 3,
  "sex": "male",
  "age": 25,
  "fare": 15.5,
  "travelled_alone": true,
  "embarked": "southampton",
  "title": "mr"
}
```

### Prediction Response
```json
{
  "survived": false,
  "probability": 0.78
}
```

## 💾 Data Management

### Model Storage
- Models automatically saved to `/data/models/`
- Persisted across container restarts
- Each model includes:
  - `model.pkl` - Serialized scikit-learn model
  - `params.json` - Training parameters
  - `info.json` - Model metadata and accuracy

### Pre-loaded Models
On startup, the service loads:
- `rf` - Random Forest (default)
- `svm` - Support Vector Machine
- `knn` - K-Nearest Neighbors
- `lr` - Logistic Regression

## 🐳 Production Deployment

The service is production-ready when deployed via:
```bash
docker compose -f compose/compose.prod-local.yaml up
```

Features in production:
- Optimized model loading
- Cached predictions
- Health monitoring
- Structured logging

## 🔍 Troubleshooting

### Model Not Found
- Check model ID with `GET /models`
- Verify model files in container: `docker compose exec model ls /data/models`

### Slow Predictions
- First prediction loads model into memory
- Subsequent predictions are cached
- Use `random_state` for reproducible results

### Training Failures
- Check feature names match schema
- Verify algorithm parameters are valid
- Review logs: `docker compose logs model`

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Project Requirements](../../docs/Project-Requirements.md)
