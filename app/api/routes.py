"""
FastAPI routes for ML service
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime
from typing import Dict

from app.schemas.prediction_schema import (
    CaseInput,
    PredictionOutput,
    ModelInfo,
    HealthResponse,
    TrainingRequest,
    TrainingResponse
)
from app.schemas.explanation_schema import ExplanationRequest, ExplanationResponse
from app.services.prediction_service import prediction_service
from app.training.train_model import CasePredictor
from app.llm.reasoning_engine import reasoning_engine
from app.utils.logger import logger

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Health status of the service
    """
    return HealthResponse(
        status="healthy",
        model_loaded=prediction_service.is_model_ready(),
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@router.post("/predict", response_model=PredictionOutput)
async def predict_case(case_input: CaseInput):
    """
    Predict adjournment risk and delay for a case
    
    Args:
        case_input: Case features for prediction
    
    Returns:
        Prediction results with risk scores and contributing factors
    """
    try:
        if not prediction_service.is_model_ready():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train the model first or check model file."
            )
        
        prediction = prediction_service.predict_case(case_input)
        return prediction
    
    except ValueError as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@router.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the loaded model
    
    Returns:
        Model information including version, metrics, and features
    """
    try:
        info = prediction_service.get_model_info()
        
        return ModelInfo(
            model_name=info['model_name'],
            version=info['version'],
            trained_date=info.get('trained_date'),
            accuracy=info.get('accuracy'),
            features=info.get('features', []),
            status=info['status']
        )
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving model information")


async def train_model_background(dataset_path: str = None, test_size: float = 0.2, random_state: int = 42):
    """
    Background task for model training
    
    Args:
        dataset_path: Path to training dataset
        test_size: Test set proportion
        random_state: Random seed
    """
    try:
        logger.info("Starting background model training...")
        
        # Initialize and train model
        predictor = CasePredictor(model_type='random_forest')
        metrics = predictor.train(
            dataset_path=dataset_path,
            test_size=test_size,
            random_state=random_state
        )
        
        # Save the trained model
        predictor.save_model("models/adjournment_model.joblib")
        
        # Reload model in prediction service
        prediction_service.reload_model()
        
        logger.info("Background model training completed successfully")
        logger.log_training(metrics)
        
    except Exception as e:
        logger.error(f"Error during background training: {str(e)}")


@router.post("/retrain", response_model=TrainingResponse)
async def retrain_model(
    training_request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Retrain the model with new data
    
    Args:
        training_request: Training configuration
        background_tasks: FastAPI background tasks
    
    Returns:
        Training status response
    """
    try:
        # Add training task to background
        background_tasks.add_task(
            train_model_background,
            dataset_path=training_request.dataset_path,
            test_size=training_request.test_size,
            random_state=training_request.random_state
        )
        
        return TrainingResponse(
            status="started",
            message="Model retraining started in background. This may take a few minutes.",
            metrics=None,
            model_path="models/adjournment_model.joblib"
        )
    
    except Exception as e:
        logger.error(f"Error starting model retraining: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting model retraining")


@router.post("/model/reload")
async def reload_model():
    """
    Reload the model from disk
    
    Returns:
        Status of model reload
    """
    try:
        prediction_service.reload_model()
        
        if prediction_service.is_model_ready():
            return {
                "status": "success",
                "message": "Model reloaded successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "warning",
                "message": "Model file not found. Please train the model first.",
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Error reloading model")


@router.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(request: ExplanationRequest):
    """
    Generate human-readable explanation for AI prediction using Gemini LLM
    
    This endpoint uses the Gemini API to generate natural language explanations
    that help judges understand why the model predicts case delays or adjournment risks.
    
    Args:
        request: Explanation request containing case data, prediction, and similar cases
    
    Returns:
        Structured explanation with key factors, historical patterns, and recommendations
    """
    try:
        if not reasoning_engine.is_available():
            raise HTTPException(
                status_code=503,
                detail="Reasoning engine not available. Please configure GEMINI_API_KEY environment variable."
            )
        
        logger.info("Generating explanation for case prediction")
        
        explanation_result = reasoning_engine.generate_explanation(
            case_data=request.case_data,
            prediction=request.prediction,
            similar_cases=request.similar_cases
        )
        
        return ExplanationResponse(
            explanation=explanation_result.get("explanation", ""),
            key_factors=explanation_result.get("key_factors", []),
            historical_patterns=explanation_result.get("historical_patterns", ""),
            recommendation=explanation_result.get("recommendation", "")
        )
    
    except ValueError as e:
        logger.error(f"Explanation generation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error during explanation generation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during explanation generation")
