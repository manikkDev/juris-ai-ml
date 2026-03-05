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

## � Data Pipeline

### Overview

Complete data ingestion and preprocessing pipeline for Indian High Court judgments:

1. **Download** - Fetch PDFs from AWS Open Data registry
2. **Extract** - Extract text from PDFs (with OCR fallback)
3. **Parse** - Extract case metadata and hearing events
4. **Generate** - Create ML features from parsed data
5. **Build** - Construct structured training dataset
6. **Store** - Version and manage datasets
7. **Retrain** - Automatically trigger model retraining

### Pipeline Components

#### AWS Downloader
```python
from app.pipeline.download.aws_downloader import IndianHighCourtDownloader

downloader = IndianHighCourtDownloader()
files = downloader.download_batch(year=2023, max_files=100)
```

#### Text Extraction
```python
from app.pipeline.extract.pdf_text_extractor import PDFTextExtractor

extractor = PDFTextExtractor()
result = extractor.extract_and_save(pdf_path)
```

#### Metadata Parsing
```python
from app.pipeline.parsers.metadata_parser import MetadataParser

parser = MetadataParser()
metadata = parser.parse_complete_metadata(text)
```

#### Feature Generation
```python
from app.pipeline.dataset.feature_generator import FeatureGenerator

generator = FeatureGenerator()
features = generator.generate_features(metadata)
```

### Running the Pipeline

**Complete Pipeline:**
```bash
python app/pipeline/jobs/pipeline_runner.py --use-sample --max-files 10
```

**With Options:**
```bash
python app/pipeline/jobs/pipeline_runner.py \
  --year 2023 \
  --court "Delhi" \
  --max-files 50 \
  --use-ocr \
  --output dataset_2023.csv
```

**Auto-Retrain:**
```bash
python app/pipeline/jobs/retrain_trigger.py --force
```

### Pipeline Features

✅ **Parallel Processing** - Process multiple PDFs concurrently
✅ **Resume Support** - Resume interrupted pipeline runs
✅ **OCR Fallback** - Automatic OCR for scanned documents
✅ **Data Versioning** - Track dataset versions
✅ **Quality Validation** - Validate extracted data quality
✅ **Progress Tracking** - Real-time progress bars
✅ **Structured Logging** - Comprehensive logging at each stage

### Extracted Features

The pipeline generates these ML features:

- `case_age_days` - Age of case in days
- `adjournment_history` - Number of adjournments
- `hearings_count` - Total hearings
- `case_type` - Type of case (Civil, Criminal, etc.)
- `court` - Court name
- `days_since_last_hearing` - Days since last hearing
- `judge_workload` - Estimated judge workload
- `adjournment_rate` - Calculated adjournment rate
- `hearing_frequency` - Hearings per month

### Dataset Output

Pipeline produces `data/processed/dataset.csv`:

```csv
case_age_days,adjournment_history,hearings_count,case_type,court,...
420,3,8,Civil,Delhi High Court,...
```

## � Semantic Search Engine

### Overview

Vector-based semantic search for legal judgments using sentence transformers and FAISS:

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Search Type**: Cosine similarity with inner product index
- **Features**: Text chunking, metadata filtering, similar case discovery

### Building the Search Index

**From Pipeline Output:**
```bash
python scripts/build_vector_index.py --text-dir data/intermediate/text
```

**With Dataset Metadata:**
```bash
python scripts/build_vector_index.py --use-dataset --dataset data/processed/dataset.csv
```

**Limit Files (for testing):**
```bash
python scripts/build_vector_index.py --max-files 50
```

### Search API Endpoints

#### Search Judgments
```bash
POST /api/search
```

**Request:**
```json
{
  "query": "land dispute ownership evidence",
  "top_k": 5,
  "court": "Delhi High Court",
  "case_type": "Civil"
}
```

**Response:**
```json
{
  "query": "land dispute ownership evidence",
  "total_results": 5,
  "results": [
    {
      "case_id": "case_123",
      "chunk_id": "case_123_chunk_0",
      "court": "Delhi High Court",
      "judge": "Justice Sharma",
      "date": "2023-05-15",
      "case_type": "Civil",
      "score": 0.89,
      "excerpt": "In the matter of land ownership dispute...",
      "chunk_index": 0
    }
  ]
}
```

#### Find Similar Cases
```bash
POST /api/search/similar
```

**Request:**
```json
{
  "case_id": "case_123",
  "top_k": 5
}
```

#### Get Index Statistics
```bash
GET /api/search/stats
```

**Response:**
```json
{
  "total_vectors": 1500,
  "unique_cases": 50,
  "embedding_dim": 384,
  "index_type": "IP",
  "is_loaded": true
}
```

#### Reload Index
```bash
POST /api/search/reload
```

### Backend Integration

**Node.js Backend Example:**
```javascript
const axios = require('axios');

async function searchSimilarCases(query) {
  const response = await axios.post('http://localhost:8000/api/search', {
    query: query,
    top_k: 5,
    court: 'Delhi High Court'
  });
  
  return response.data.results;
}

// Usage
const results = await searchSimilarCases('property dispute inheritance');
console.log(`Found ${results.length} similar cases`);
```

### Python Usage

**Direct Search:**
```python
from app.search.search_engine.semantic_search import SemanticSearchEngine

engine = SemanticSearchEngine()
results = engine.search("contract breach damages", top_k=5)

for result in results:
    print(f"Case: {result['case_id']}")
    print(f"Score: {result['score']:.2f}")
    print(f"Excerpt: {result['excerpt']}\n")
```

**Build Index Programmatically:**
```python
from app.search.embedding.embedding_generator import EmbeddingGenerator
from app.search.vector_store.faiss_index import create_faiss_index

# Generate embeddings
generator = EmbeddingGenerator()
embeddings, metadata = generator.process_dataset(text_files)

# Create index
index = create_faiss_index(embeddings, metadata)
```

### Search Features

✅ **Semantic Understanding** - Finds conceptually similar cases, not just keyword matches
✅ **Text Chunking** - Handles long judgments by splitting into chunks
✅ **Metadata Filtering** - Filter by court, case type, date
✅ **Similar Case Discovery** - Find cases similar to a given case
✅ **Relevance Scoring** - Cosine similarity scores (0-1)
✅ **Excerpt Generation** - Returns relevant text excerpts
✅ **Incremental Updates** - Add/remove cases from index
✅ **Fast Retrieval** - FAISS optimized for speed

### Performance

- **Index Build Time**: ~2-3 seconds per 100 cases
- **Search Latency**: <100ms for top-5 results
- **Scalability**: Handles 10,000+ cases efficiently
- **Memory**: ~1.5MB per 1000 vectors (384-dim)

## �📈 Future Enhancements

- [x] Integration with real Indian High Court datasets
- [x] Automated data pipeline
- [x] Dataset versioning and management
- [x] Semantic search with sentence transformers
- [x] Vector database with FAISS
- [ ] Advanced NLP features (named entity recognition, summarization)
- [ ] Deep learning models (PyTorch)
- [ ] Model versioning and A/B testing
- [ ] Real-time model monitoring
- [ ] Feature importance visualization
- [ ] Explainable AI (SHAP values)
- [ ] Distributed processing for large-scale datasets
- [ ] Multi-language support (Hindi, regional languages)

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
