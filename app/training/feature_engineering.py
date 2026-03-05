"""
Feature engineering for case prediction model
"""
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Feature engineering for judicial case data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def create_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Create engineered features from raw data
        
        Args:
            df: Input dataframe with raw features
            fit_scaler: Whether to fit the scaler (True for training, False for inference)
        
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Calculate adjournment rate
        df['adjournment_rate'] = df['adjournment_history'] / (df['hearings_count'] + 1)
        
        # Calculate hearing frequency (hearings per month)
        df['hearing_frequency'] = df['hearings_count'] / (df['case_age_days'] / 30 + 1)
        
        # Encode case type
        case_type_encoding = {
            'civil': 0,
            'criminal': 1,
            'family': 2,
            'commercial': 3,
            'constitutional': 4,
            'labor': 5,
            'tax': 6,
            'property': 7
        }
        df['case_type_encoded'] = df['case_type'].str.lower().map(case_type_encoding).fillna(8)
        
        # Select features for modeling
        feature_cols = [
            'case_age_days',
            'adjournment_history',
            'hearings_count',
            'days_since_last_hearing',
            'judge_workload',
            'adjournment_rate',
            'hearing_frequency',
            'case_type_encoded'
        ]
        
        self.feature_names = feature_cols
        X = df[feature_cols].copy()
        
        # Normalize numeric features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Create DataFrame with scaled features
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        return X_scaled_df
    
    def prepare_single_case(self, case_data: dict) -> np.ndarray:
        """
        Prepare a single case for prediction
        
        Args:
            case_data: Dictionary with case features
        
        Returns:
            Numpy array ready for prediction
        """
        # Create DataFrame from single case
        df = pd.DataFrame([case_data])
        
        # Calculate derived features
        df['adjournment_rate'] = df['adjournment_history'] / (df['hearings_count'] + 1)
        df['hearing_frequency'] = df['hearings_count'] / (df['case_age_days'] / 30 + 1)
        
        # Encode case type
        case_type_encoding = {
            'civil': 0,
            'criminal': 1,
            'family': 2,
            'commercial': 3,
            'constitutional': 4,
            'labor': 5,
            'tax': 6,
            'property': 7
        }
        df['case_type_encoded'] = df['case_type'].str.lower().map(case_type_encoding).fillna(8)
        
        # Select features in correct order
        feature_cols = [
            'case_age_days',
            'adjournment_history',
            'hearings_count',
            'days_since_last_hearing',
            'judge_workload',
            'adjournment_rate',
            'hearing_frequency',
            'case_type_encoded'
        ]
        
        X = df[feature_cols].values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def get_feature_importance_mapping(self) -> dict:
        """
        Get mapping of feature indices to names
        
        Returns:
            Dictionary mapping indices to feature names
        """
        return {i: name for i, name in enumerate(self.feature_names)}


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataframe into features and labels
    
    Args:
        df: Input dataframe with features and labels
    
    Returns:
        Tuple of (features, adjournment_label, delay_probability)
    """
    # Separate features from labels
    label_cols = ['adjournment_label', 'delay_probability']
    feature_cols = [col for col in df.columns if col not in label_cols]
    
    X = df[feature_cols]
    y_adjournment = df['adjournment_label']
    y_delay = df['delay_probability']
    
    return X, y_adjournment, y_delay
