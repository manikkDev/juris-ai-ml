"""
Minimal FastAPI application for Vercel deployment
Only includes essential endpoints for Gemini AI functionality
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Get configuration
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://juris-ai-backend.vercel.app,https://your-frontend-url.vercel.app"
).split(",")

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Create FastAPI app
app = FastAPI(
    title="Juris AI ML Service (Vercel)",
    description="Lightweight ML service for judicial case analysis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class CaseAnalysisRequest(BaseModel):
    case_description: str
    similar_cases: List[Dict[str, Any]]

class PredictionRequest(BaseModel):
    case_data: Dict[str, Any]

# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "Juris AI ML Service (Vercel)",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "analyze_case": "/api/analyze-case",
            "predict": "/api/predict"
        }
    }

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "service": "juris-ai-ml",
        "gemini_configured": bool(GEMINI_API_KEY)
    }

@app.post("/api/analyze-case")
async def analyze_case(request: CaseAnalysisRequest):
    """
    Generate legal analysis using Gemini AI
    """
    if not GEMINI_API_KEY:
        return {
            "error": "Gemini API key not configured",
            "analysis": "AI analysis not available. Please configure GEMINI_API_KEY."
        }

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""You are a legal assistant helping judges analyze case similarity.

User case:
{request.case_description}

Similar cases:
{chr(10).join([f"{i+1}. {case.get('case_title', 'N/A')} - {case.get('summary', 'N/A')}" for i, case in enumerate(request.similar_cases)])}

Provide a structured legal analysis:
1. Key legal themes identified
2. Relevance of each similar case
3. Judicial insight and precedent considerations"""

        response = model.generate_content(prompt)
        
        return {
            "analysis": response.text,
            "model": "gemini-1.5-flash",
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "analysis": "Failed to generate analysis. Please try again.",
            "status": "error"
        }

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """
    Simple prediction based on case characteristics
    """
    try:
        # Simple rule-based prediction for demo
        case_data = request.case_data
        adjournments = case_data.get("adjournments", 0)
        status = case_data.get("status", "").lower()
        
        # Simple risk calculation
        risk_score = min(0.9, adjournments * 0.1)
        if status == "pending":
            risk_score += 0.2
            
        delay_probability = min(0.95, risk_score + 0.1)
        
        return {
            "risk_score": round(risk_score, 2),
            "delay_probability": round(delay_probability, 2),
            "resolution_estimate": "6-12 months" if risk_score > 0.5 else "3-6 months",
            "confidence": 0.75,
            "model": "rule-based-lightweight"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "prediction": None
        }

# Vercel serverless handler
handler = app
