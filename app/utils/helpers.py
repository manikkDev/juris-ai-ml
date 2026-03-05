"""
Helper utilities for ML service
"""
import numpy as np
from typing import Dict, List, Tuple


def calculate_resolution_estimate(adjournment_risk: float, delay_prob: float, case_age: int) -> str:
    """
    Calculate estimated resolution time based on predictions
    
    Args:
        adjournment_risk: Probability of adjournment (0-1)
        delay_prob: Probability of delay (0-1)
        case_age: Current age of case in days
    
    Returns:
        String estimate of resolution time
    """
    combined_risk = (adjournment_risk + delay_prob) / 2
    
    if combined_risk > 0.7 or case_age > 730:
        return "12-18 months"
    elif combined_risk > 0.5 or case_age > 365:
        return "6-12 months"
    elif combined_risk > 0.3 or case_age > 180:
        return "3-6 months"
    else:
        return "1-3 months"


def get_impact_level(importance: float) -> str:
    """
    Convert feature importance to impact level
    
    Args:
        importance: Feature importance score (0-1)
    
    Returns:
        Impact level string
    """
    if importance > 0.25:
        return "High"
    elif importance > 0.15:
        return "Medium"
    else:
        return "Low"


def normalize_case_type(case_type: str) -> str:
    """
    Normalize case type string
    
    Args:
        case_type: Raw case type string
    
    Returns:
        Normalized case type
    """
    case_type_map = {
        'civil': 'Civil',
        'criminal': 'Criminal',
        'family': 'Family',
        'commercial': 'Commercial',
        'constitutional': 'Constitutional',
        'labor': 'Labor',
        'tax': 'Tax',
        'property': 'Property'
    }
    
    return case_type_map.get(case_type.lower(), 'Other')


def calculate_confidence(model_proba: np.ndarray) -> float:
    """
    Calculate prediction confidence based on probability distribution
    
    Args:
        model_proba: Probability array from model
    
    Returns:
        Confidence score (0-1)
    """
    # Higher confidence when probabilities are more extreme (closer to 0 or 1)
    max_proba = np.max(model_proba)
    confidence = abs(max_proba - 0.5) * 2  # Scale to 0-1
    
    # Add base confidence
    base_confidence = 0.6
    final_confidence = base_confidence + (confidence * 0.4)
    
    return min(0.95, max(0.5, final_confidence))


def get_feature_names() -> List[str]:
    """
    Get list of feature names used in the model
    
    Returns:
        List of feature names
    """
    return [
        'case_age_days',
        'adjournment_history',
        'hearings_count',
        'days_since_last_hearing',
        'judge_workload',
        'adjournment_rate',
        'hearing_frequency',
        'case_type_encoded'
    ]


def encode_case_type(case_type: str) -> int:
    """
    Encode case type to numeric value
    
    Args:
        case_type: Case type string
    
    Returns:
        Encoded integer value
    """
    encoding = {
        'Civil': 0,
        'Criminal': 1,
        'Family': 2,
        'Commercial': 3,
        'Constitutional': 4,
        'Labor': 5,
        'Tax': 6,
        'Property': 7,
        'Other': 8
    }
    
    normalized = normalize_case_type(case_type)
    return encoding.get(normalized, 8)


def format_factor_name(feature_name: str) -> str:
    """
    Convert feature name to human-readable factor name
    
    Args:
        feature_name: Technical feature name
    
    Returns:
        Human-readable factor name
    """
    factor_map = {
        'case_age_days': 'Case Age',
        'adjournment_history': 'Adjournment History',
        'hearings_count': 'Number of Hearings',
        'days_since_last_hearing': 'Hearing Inactivity',
        'judge_workload': 'Judge Workload',
        'adjournment_rate': 'Adjournment Rate',
        'hearing_frequency': 'Hearing Frequency',
        'case_type_encoded': 'Case Type'
    }
    
    return factor_map.get(feature_name, feature_name.replace('_', ' ').title())
