"""
Prediction service for handling prediction requests
"""
from typing import Dict
from app.models.predictor import CasePredictorWrapper
from app.schemas.prediction_schema import CaseInput, PredictionOutput, TopFactor
from app.utils.logger import logger


class PredictionService:
    """Service for handling case predictions"""
    
    def __init__(self, model_path: str = "models/adjournment_model.joblib"):
        """
        Initialize prediction service
        
        Args:
            model_path: Path to the trained model
        """
        self.predictor = CasePredictorWrapper(model_path)
    
    def predict_case(self, case_input: CaseInput) -> PredictionOutput:
        """
        Predict adjournment risk and delay for a case
        
        Args:
            case_input: Case input data
        
        Returns:
            Prediction output with risk scores and factors
        """
        # Convert input to dictionary
        case_data = case_input.model_dump()
        
        # Get prediction
        prediction = self.predictor.predict(case_data)
        
        # Convert top factors to schema format
        top_factors = [
            TopFactor(
                factor=f['factor'],
                importance=f['importance'],
                impact=f['impact']
            )
            for f in prediction['topFactors']
        ]
        
        # Create output
        output = PredictionOutput(
            adjournmentRisk=prediction['adjournmentRisk'],
            delayProbability=prediction['delayProbability'],
            resolutionEstimate=prediction['resolutionEstimate'],
            topFactors=top_factors,
            confidence=prediction['confidence'],
            modelVersion=prediction['modelVersion']
        )
        
        return output
    
    def reload_model(self):
        """Reload the model (useful after retraining)"""
        logger.info("Reloading model in prediction service...")
        self.predictor.reload_model()
    
    def is_model_ready(self) -> bool:
        """Check if model is loaded and ready"""
        return self.predictor.is_model_loaded()
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return self.predictor.get_model_info()


# Global prediction service instance
prediction_service = PredictionService()
