# Vercel Deployment - Lightweight ML Service

This is the lightweight version optimized for Vercel deployment.

## What's Included

- **FastAPI** - Web framework
- **Gemini AI** - For legal analysis
- **Basic ML** - Simple prediction logic
- **Total Size**: ~50MB (fits in Vercel's 500MB limit)

## What's Excluded

Heavy dependencies removed to fit Vercel limits:
- PyTorch (2GB+)
- Transformers (1GB+)
- Sentence-transformers (500MB+)
- Full ML pipeline

## Files

- `main.py` - Lightweight FastAPI app (production)
- `requirements.txt` - Minimal dependencies (~50MB)
- `main-full.py` - Full ML pipeline (local dev only)
- `requirements-full.txt` - Full dependencies (local dev only)

## Environment Variables

Required for Vercel:
```
GEMINI_API_KEY=your_gemini_api_key
ALLOWED_ORIGINS=https://juris-ai-backend.vercel.app,https://your-frontend.vercel.app
```

## Endpoints

- `GET /` - Service info
- `GET /api/health` - Health check
- `POST /api/analyze-case` - Gemini AI case analysis
- `POST /api/predict` - Basic case prediction

## Local Development

For full ML features locally:
```bash
pip install -r requirements-full.txt
python main-full.py
```

## Vercel Deployment

Vercel automatically uses:
- `requirements.txt` (lightweight)
- `main.py` (lightweight)

Deploy size: ~50MB ✅
