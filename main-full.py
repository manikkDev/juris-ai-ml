"""
Main FastAPI application for Juris AI ML Service
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from app.api.routes import router
from app.utils.logger import logger

# Load environment variables
load_dotenv()

# Get configuration from environment
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5000,http://localhost:5173,http://localhost:8081"
).split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info("Starting Juris AI ML Service...")
    logger.info(f"Allowed CORS origins: {ALLOWED_ORIGINS}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Juris AI ML Service...")


# Create FastAPI application
app = FastAPI(
    title="Juris AI ML Service",
    description="Machine Learning microservice for judicial case prediction",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api", tags=["predictions"])

# Include search routes
from app.search.api.search_routes import router as search_router
app.include_router(search_router, prefix="/api", tags=["search"])

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with service information
    """
    return {
        "service": "Juris AI ML Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict",
            "model_info": "/api/model/info",
            "retrain": "/api/retrain",
            "reload": "/api/model/reload"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
