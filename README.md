# Juris AI - ML Microservice

Machine Learning microservice for **Juris AI** - Provides AI-powered predictions for judicial case adjournment risk, delay probability, and resolution estimates.

## 🎯 Overview

This service uses machine learning models to predict:
- **Adjournment Risk** - Probability that a case will be adjourned
- **Delay Probability** - Likelihood of case delays
- **Resolution Estimate** - Expected time to case resolution
- **Top Contributing Factors** - Key factors influencing predictions

## 🏗️ Architecture

```
juris-ai-ml/
├── app/
│   ├── api/
│   │   └── routes.py              # FastAPI endpoints
│   ├── models/
│   │   └── predictor.py           # Model wrapper
│   ├── training/
│   │   ├── train_model.py         # Model training
│   │   ├── feature_engineering.py # Feature engineering
│   │   └── dataset_loader.py      # Data loading & generation
│   ├── services/
│   │   └── prediction_service.py  # Prediction service
│   ├── schemas/
│   │   └── prediction_schema.py   # Pydantic schemas
│   └── utils/
│       ├── logger.py              # Logging utility
│       └── helpers.py             # Helper functions
├── data/
│   ├── raw/                       # Raw datasets
│   └── processed/                 # Processed datasets
├── models/
│   └── adjournment_model.joblib   # Trained model
├── scripts/
│   ├── prepare_dataset.py         # Dataset preparation
│   └── retrain_model.py           # Model retraining
├── main.py                        # FastAPI application
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/manikkDev/juris-ai-ml.git
cd juris-ai-ml
```

2. **Create virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Prepare dataset**

```bash
python scripts/prepare_dataset.py
```

This generates a synthetic dataset with 2,000 samples for training.

5. **Train the model**

```bash
python app/training/train_model.py
```

This trains the model and saves it to `models/adjournment_model.joblib`.

6. **Start the server**

```bash
uvicorn main:app --reload --port 8000
```

The service will be available at `http://localhost:8000`

## 📡 API Endpoints

### Health Check

```bash
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "timestamp": "2026-03-05T10:00:00"
}
```

### Predict Case

```bash
POST /api/predict
```

**Request Body:**
```json
{
  "case_age_days": 420,
  "adjournment_history": 3,
  "hearings_count": 8,
  "case_type": "civil",
  "days_since_last_hearing": 40,
  "judge_workload": 120
}
```

**Response:**
```json
{
  "adjournmentRisk": 0.74,
  "delayProbability": 0.61,
  "resolutionEstimate": "6-12 months",
  "topFactors": [
    {
      "factor": "High adjournment history",
      "importance": 0.35,
      "impact": "High"
    },
    {
      "factor": "Old case age",
      "importance": 0.28,
      "impact": "High"
    },
    {
      "factor": "High judge workload",
      "importance": 0.18,
      "impact": "Medium"
    }
  ],
  "confidence": 0.82,
  "modelVersion": "1.0.0"
}
```

### Get Model Info

```bash
GET /api/model/info
```

**Response:**
```json
{
  "model_name": "Case Adjournment Predictor",
  "version": "1.0.0",
  "trained_date": "2026-03-05T10:00:00",
  "accuracy": 0.87,
  "features": [
    "case_age_days",
    "adjournment_history",
    "hearings_count",
    "days_since_last_hearing",
    "judge_workload",
    "adjournment_rate",
    "hearing_frequency",
    "case_type_encoded"
  ],
  "status": "loaded"
}
```

### Retrain Model

```bash
POST /api/retrain
```

**Request Body:**
```json
{
  "dataset_path": "data/raw/synthetic_cases.csv",
  "test_size": 0.2,
  "random_state": 42
}
```

**Response:**
```json
{
  "status": "started",
  "message": "Model retraining started in background. This may take a few minutes.",
  "model_path": "models/adjournment_model.joblib"
}
```

### Reload Model

```bash
POST /api/model/reload
```

Reloads the model from disk (useful after retraining).

## 🧪 Testing the API

### Using cURL

```bash
# Health check
curl http://localhost:8000/api/health

# Make prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "case_age_days": 420,
    "adjournment_history": 3,
    "hearings_count": 8,
    "case_type": "civil",
    "days_since_last_hearing": 40,
    "judge_workload": 120
  }'

# Get model info
curl http://localhost:8000/api/model/info
```

### Using Python

```python
import requests

# Make prediction
response = requests.post(
    "http://localhost:8000/api/predict",
    json={
        "case_age_days": 420,
        "adjournment_history": 3,
        "hearings_count": 8,
        "case_type": "civil",
        "days_since_last_hearing": 40,
        "judge_workload": 120
    }
)

prediction = response.json()
print(f"Adjournment Risk: {prediction['adjournmentRisk']:.2%}")
print(f"Delay Probability: {prediction['delayProbability']:.2%}")
print(f"Resolution Estimate: {prediction['resolutionEstimate']}")
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t juris-ai-ml:latest .
```

### Run Container

```bash
docker run -d \
  --name juris-ai-ml \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  juris-ai-ml:latest
```

### Docker Compose

```yaml
version: '3.8'

services:
  ml-service:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - ENVIRONMENT=production
```

## 🔄 Integration with Backend

The Node.js backend calls this service for predictions:

```javascript
// Backend integration example
const axios = require('axios');

async function getPrediction(caseData) {
  const response = await axios.post('http://localhost:8000/api/predict', {
    case_age_days: caseData.caseAgeDays,
    adjournment_history: caseData.adjournments,
    hearings_count: caseData.hearingsCount,
    case_type: caseData.caseType,
    days_since_last_hearing: caseData.daysSinceLastHearing,
    judge_workload: caseData.judgeWorkload
  });
  
  return response.data;
}
```

## 📊 Model Details

### Features

The model uses the following features:

1. **case_age_days** - Age of the case in days
2. **adjournment_history** - Number of previous adjournments
3. **hearings_count** - Total number of hearings
4. **days_since_last_hearing** - Days since last hearing
5. **judge_workload** - Number of cases assigned to judge
6. **adjournment_rate** - Calculated: adjournments / hearings
7. **hearing_frequency** - Calculated: hearings per month
8. **case_type_encoded** - Encoded case type

### Algorithms

- **RandomForestClassifier** - Primary model
- **GradientBoostingClassifier** - Alternative model

### Performance Metrics

Typical model performance:
- **Accuracy**: ~87%
- **Precision**: ~85%
- **Recall**: ~82%
- **ROC AUC**: ~0.90

## 🔧 Configuration

Create a `.env` file:

```env
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development
MODEL_PATH=models/adjournment_model.joblib
MODEL_VERSION=1.0.0
ALLOWED_ORIGINS=http://localhost:5000,http://localhost:5173
```

## 📝 Scripts

### Prepare Dataset

```bash
python scripts/prepare_dataset.py
```

Generates synthetic dataset for training.

### Retrain Model

```bash
python scripts/retrain_model.py
```

Retrains the model with current dataset.

## 🧪 Development

### Run in Development Mode

```bash
uvicorn main:app --reload --port 8000
```

### Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📈 Future Enhancements

- [ ] Integration with real Indian High Court datasets
- [ ] NLP-based case text analysis using sentence-transformers
- [ ] Deep learning models (PyTorch)
- [ ] Model versioning and A/B testing
- [ ] Real-time model monitoring
- [ ] Automated retraining pipeline
- [ ] Feature importance visualization
- [ ] Explainable AI (SHAP values)

## 🔒 Security

- Input validation using Pydantic
- CORS configuration
- Type hints throughout codebase
- Error handling and logging

## 🐛 Troubleshooting

### Model Not Loaded

If you get "Model not loaded" error:

1. Train the model first:
   ```bash
   python app/training/train_model.py
   ```

2. Verify model file exists:
   ```bash
   ls models/adjournment_model.joblib
   ```

### Port Already in Use

Change the port in `.env` or use:
```bash
uvicorn main:app --port 8001
```

### Dependencies Issues

Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

## 📚 API Documentation

Full interactive API documentation available at:
- http://localhost:8000/docs (Swagger)
- http://localhost:8000/redoc (ReDoc)

## 🤝 Integration

This ML service integrates with:
- **Backend**: https://github.com/manikkDev/juris-ai-backend
- **Frontend**: https://github.com/manikkDev/juris-ai-frontend

## 📄 License

MIT License

## 👥 Team

Juris AI Team

---

**Built with Python, FastAPI, and scikit-learn for judicial efficiency**
