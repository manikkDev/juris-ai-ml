"""
Model training script for judicial case prediction
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from typing import Tuple, Dict

from app.training.dataset_loader import load_or_generate_dataset, get_dataset_statistics
from app.training.feature_engineering import FeatureEngineer, split_features_labels
from app.utils.logger import logger


class CasePredictor:
    """Machine learning model for case prediction"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the predictor
        
        Args:
            model_type: Type of model ('random_forest' or 'gradient_boosting')
        """
        self.model_type = model_type
        self.adjournment_model = None
        self.delay_model = None
        self.feature_engineer = FeatureEngineer()
        self.feature_importance = None
        self.training_date = None
        self.metrics = {}
        
        if model_type == 'random_forest':
            self.adjournment_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.delay_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.adjournment_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.delay_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
    
    def train(
        self,
        dataset_path: str = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Train the model on dataset
        
        Args:
            dataset_path: Path to dataset CSV file
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting model training...")
        
        # Load or generate dataset
        df = load_or_generate_dataset(dataset_path, n_samples=2000, random_state=random_state)
        
        # Get dataset statistics
        stats = get_dataset_statistics(df)
        logger.info(f"Dataset statistics: {stats}")
        
        # Engineer features
        logger.info("Engineering features...")
        X = self.feature_engineer.create_features(df, fit_scaler=True)
        
        # Get labels
        y_adjournment = df['adjournment_label']
        y_delay_binary = (df['delay_probability'] > 0.5).astype(int)  # Convert to binary for classification
        
        # Split data
        X_train, X_test, y_adj_train, y_adj_test = train_test_split(
            X, y_adjournment, test_size=test_size, random_state=random_state, stratify=y_adjournment
        )
        
        _, _, y_delay_train, y_delay_test = train_test_split(
            X, y_delay_binary, test_size=test_size, random_state=random_state, stratify=y_delay_binary
        )
        
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Train adjournment model
        logger.info("Training adjournment prediction model...")
        self.adjournment_model.fit(X_train, y_adj_train)
        
        # Train delay model
        logger.info("Training delay prediction model...")
        self.delay_model.fit(X_train, y_delay_train)
        
        # Evaluate models
        logger.info("Evaluating models...")
        metrics = self._evaluate_models(X_test, y_adj_test, y_delay_test)
        
        # Store feature importance
        if hasattr(self.adjournment_model, 'feature_importances_'):
            self.feature_importance = self.adjournment_model.feature_importances_
        
        self.training_date = datetime.now().isoformat()
        self.metrics = metrics
        
        logger.info(f"Training completed. Metrics: {metrics}")
        
        return metrics
    
    def _evaluate_models(
        self,
        X_test: pd.DataFrame,
        y_adj_test: pd.Series,
        y_delay_test: pd.Series
    ) -> Dict:
        """
        Evaluate trained models
        
        Args:
            X_test: Test features
            y_adj_test: Test labels for adjournment
            y_delay_test: Test labels for delay
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Adjournment model predictions
        y_adj_pred = self.adjournment_model.predict(X_test)
        y_adj_proba = self.adjournment_model.predict_proba(X_test)[:, 1]
        
        # Delay model predictions
        y_delay_pred = self.delay_model.predict(X_test)
        y_delay_proba = self.delay_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'adjournment_model': {
                'accuracy': float(accuracy_score(y_adj_test, y_adj_pred)),
                'precision': float(precision_score(y_adj_test, y_adj_pred, zero_division=0)),
                'recall': float(recall_score(y_adj_test, y_adj_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_adj_test, y_adj_proba))
            },
            'delay_model': {
                'accuracy': float(accuracy_score(y_delay_test, y_delay_pred)),
                'precision': float(precision_score(y_delay_test, y_delay_pred, zero_division=0)),
                'recall': float(recall_score(y_delay_test, y_delay_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_delay_test, y_delay_proba))
            }
        }
        
        return metrics
    
    def save_model(self, filepath: str = "models/adjournment_model.joblib"):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'adjournment_model': self.adjournment_model,
            'delay_model': self.delay_model,
            'feature_engineer': self.feature_engineer,
            'feature_importance': self.feature_importance,
            'training_date': self.training_date,
            'metrics': self.metrics,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str = "models/adjournment_model.joblib"):
        """
        Load trained model from disk
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            Loaded CasePredictor instance
        """
        model_data = joblib.load(filepath)
        
        predictor = cls(model_type=model_data.get('model_type', 'random_forest'))
        predictor.adjournment_model = model_data['adjournment_model']
        predictor.delay_model = model_data['delay_model']
        predictor.feature_engineer = model_data['feature_engineer']
        predictor.feature_importance = model_data.get('feature_importance')
        predictor.training_date = model_data.get('training_date')
        predictor.metrics = model_data.get('metrics', {})
        
        logger.info(f"Model loaded from {filepath}")
        return predictor


def main():
    """Main training function"""
    print("=" * 60)
    print("Juris AI - Model Training")
    print("=" * 60)
    
    # Initialize and train model
    predictor = CasePredictor(model_type='random_forest')
    
    # Train the model
    metrics = predictor.train(
        dataset_path="data/raw/synthetic_cases.csv",
        test_size=0.2,
        random_state=42
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    print("\nAdjournment Model:")
    for metric, value in metrics['adjournment_model'].items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    print("\nDelay Model:")
    for metric, value in metrics['delay_model'].items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Save the model
    predictor.save_model("models/adjournment_model.joblib")
    
    print("\n" + "=" * 60)
    print("Model training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
