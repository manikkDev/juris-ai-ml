"""
Predictor wrapper for case prediction
"""
import numpy as np
from typing import Dict, List
from pathlib import Path

from app.training.train_model import CasePredictor
from app.utils.helpers import (
    calculate_resolution_estimate,
    get_impact_level,
    calculate_confidence,
    format_factor_name
)
from app.utils.logger import logger


class CasePredictorWrapper:
    """Wrapper for case prediction model with additional logic"""
    
    def __init__(self, model_path: str = "models/adjournment_model.joblib"):
        """
        Initialize predictor wrapper
        
        Args:
            model_path: Path to the trained model
        """
        self.model_path = model_path
        self.predictor = None
        self.model_version = "1.0.0"
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if Path(self.model_path).exists():
                self.predictor = CasePredictor.load_model(self.model_path)
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.warning(f"Model file not found at {self.model_path}. Please train the model first.")
                self.predictor = None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.predictor = None
    
    def reload_model(self):
        """Reload the model (useful after retraining)"""
        logger.info("Reloading model...")
        self.load_model()
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.predictor is not None
    
    def predict(self, case_data: Dict) -> Dict:
        """
        Make prediction for a case
        
        Args:
            case_data: Dictionary with case features
        
        Returns:
            Dictionary with prediction results
        """
        if not self.is_model_loaded():
            raise ValueError("Model not loaded. Please train the model first.")
        
        # Prepare features
        X = self.predictor.feature_engineer.prepare_single_case(case_data)
        
        # Get predictions
        adj_proba = self.predictor.adjournment_model.predict_proba(X)[0]
        delay_proba = self.predictor.delay_model.predict_proba(X)[0]
        
        adjournment_risk = float(adj_proba[1])  # Probability of class 1 (adjournment)
        delay_probability = float(delay_proba[1])  # Probability of class 1 (delay)
        
        # Calculate resolution estimate
        resolution_estimate = calculate_resolution_estimate(
            adjournment_risk,
            delay_probability,
            case_data['case_age_days']
        )
        
        # Get top contributing factors
        top_factors = self._get_top_factors(X[0], adjournment_risk)
        
        # Calculate confidence
        confidence = calculate_confidence(adj_proba)
        
        result = {
            'adjournmentRisk': adjournment_risk,
            'delayProbability': delay_probability,
            'resolutionEstimate': resolution_estimate,
            'topFactors': top_factors,
            'confidence': confidence,
            'modelVersion': self.model_version
        }
        
        logger.log_prediction(case_data, result)
        
        return result
    
    def _get_top_factors(self, features: np.ndarray, risk_score: float) -> List[Dict]:
        """
        Identify top contributing factors
        
        Args:
            features: Feature vector
            risk_score: Overall risk score
        
        Returns:
            List of top factors with importance
        """
        if self.predictor.feature_importance is None:
            return []
        
        feature_names = self.predictor.feature_engineer.feature_names
        importances = self.predictor.feature_importance
        
        # Get feature values (unscaled for interpretation)
        feature_values = features
        
        # Calculate contribution scores
        contributions = []
        for i, (name, importance, value) in enumerate(zip(feature_names, importances, feature_values)):
            contributions.append({
                'name': name,
                'importance': float(importance),
                'value': float(value),
                'contribution': float(importance * abs(value))
            })
        
        # Sort by contribution
        contributions.sort(key=lambda x: x['contribution'], reverse=True)
        
        # Format top 5 factors
        top_factors = []
        for contrib in contributions[:5]:
            factor_name = format_factor_name(contrib['name'])
            importance = contrib['importance']
            impact = get_impact_level(importance)
            
            # Create descriptive factor text
            if contrib['name'] == 'adjournment_history' and contrib['value'] > 0.5:
                description = "High adjournment history"
            elif contrib['name'] == 'case_age_days' and contrib['value'] > 0.5:
                description = "Old case age"
            elif contrib['name'] == 'judge_workload' and contrib['value'] > 0.5:
                description = "High judge workload"
            elif contrib['name'] == 'days_since_last_hearing' and contrib['value'] > 0.5:
                description = "Long hearing inactivity"
            else:
                description = factor_name
            
            top_factors.append({
                'factor': description,
                'importance': importance,
                'impact': impact
            })
        
        return top_factors
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if not self.is_model_loaded():
            return {
                'model_name': 'Case Adjournment Predictor',
                'version': self.model_version,
                'status': 'not_loaded',
                'trained_date': None,
                'accuracy': None,
                'features': []
            }
        
        metrics = self.predictor.metrics.get('adjournment_model', {})
        
        return {
            'model_name': 'Case Adjournment Predictor',
            'version': self.model_version,
            'status': 'loaded',
            'trained_date': self.predictor.training_date,
            'accuracy': metrics.get('accuracy'),
            'features': self.predictor.feature_engineer.feature_names,
            'model_type': self.predictor.model_type,
            'metrics': self.predictor.metrics
        }
