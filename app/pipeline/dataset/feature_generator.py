"""
Feature Generator Module
Generates ML features from parsed case metadata
"""
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from app.utils.logger import logger


class FeatureGenerator:
    """Generate ML features from case metadata"""
    
    def __init__(self):
        """Initialize feature generator"""
        logger.info("Initialized feature generator")
    
    def generate_features(self, metadata: Dict) -> Dict:
        """
        Generate ML features from metadata
        
        Args:
            metadata: Parsed case metadata
        
        Returns:
            Dictionary with ML features
        """
        features = {}
        
        # Basic features
        features['case_age_days'] = self._get_case_age(metadata)
        features['adjournment_history'] = metadata.get('adjournment_count', 0)
        features['hearings_count'] = metadata.get('hearing_count', 0)
        features['case_type'] = self._normalize_case_type(metadata.get('case_type'))
        features['court'] = self._normalize_court(metadata.get('court'))
        
        # Derived features
        features['days_since_last_hearing'] = self._calculate_days_since_last_hearing(metadata)
        features['judge_workload'] = self._estimate_judge_workload(metadata)
        
        # Additional calculated features
        features['adjournment_rate'] = self._calculate_adjournment_rate(
            features['adjournment_history'],
            features['hearings_count']
        )
        
        features['hearing_frequency'] = self._calculate_hearing_frequency(
            features['hearings_count'],
            features['case_age_days']
        )
        
        # Labels (for training)
        features['adjournment_label'] = self._generate_adjournment_label(metadata)
        features['delay_probability'] = self._estimate_delay_probability(metadata)
        
        # Quality indicators
        features['data_quality'] = metadata.get('completeness_score', 0.0)
        
        logger.info(f"Generated features for case: {metadata.get('case_number', 'unknown')}")
        
        return features
    
    def _get_case_age(self, metadata: Dict) -> int:
        """Calculate case age in days"""
        case_age = metadata.get('case_age_days')
        
        if case_age is not None:
            return max(0, case_age)
        
        # Fallback: estimate from judgment date
        judgment_date = metadata.get('judgment_date')
        if judgment_date:
            try:
                jd = datetime.strptime(judgment_date, '%Y-%m-%d')
                # Estimate filing date as 2 years before judgment (average)
                estimated_age = 730  # 2 years
                return estimated_age
            except:
                pass
        
        # Default estimate
        return 365  # 1 year default
    
    def _normalize_case_type(self, case_type: Optional[str]) -> str:
        """Normalize case type"""
        if not case_type:
            return 'Other'
        
        case_type_lower = case_type.lower()
        
        if 'criminal' in case_type_lower:
            return 'Criminal'
        elif 'civil' in case_type_lower:
            return 'Civil'
        elif 'family' in case_type_lower:
            return 'Family'
        elif 'commercial' in case_type_lower:
            return 'Commercial'
        elif 'constitutional' in case_type_lower:
            return 'Constitutional'
        elif 'tax' in case_type_lower:
            return 'Tax'
        elif 'labor' in case_type_lower or 'labour' in case_type_lower:
            return 'Labor'
        elif 'property' in case_type_lower:
            return 'Property'
        
        return 'Other'
    
    def _normalize_court(self, court: Optional[str]) -> str:
        """Normalize court name"""
        if not court:
            return 'Unknown'
        
        court_lower = court.lower()
        
        if 'supreme' in court_lower:
            return 'Supreme Court'
        elif 'delhi' in court_lower:
            return 'Delhi High Court'
        elif 'bombay' in court_lower or 'mumbai' in court_lower:
            return 'Bombay High Court'
        elif 'calcutta' in court_lower or 'kolkata' in court_lower:
            return 'Calcutta High Court'
        elif 'madras' in court_lower or 'chennai' in court_lower:
            return 'Madras High Court'
        elif 'karnataka' in court_lower or 'bangalore' in court_lower:
            return 'Karnataka High Court'
        elif 'high court' in court_lower:
            return 'High Court'
        
        return court.title()
    
    def _calculate_days_since_last_hearing(self, metadata: Dict) -> int:
        """Calculate days since last hearing"""
        events = metadata.get('events', [])
        
        if not events:
            return 0
        
        # Find last hearing event
        hearing_types = {'hearing', 'listed', 'next_hearing'}
        hearing_events = [e for e in events if e['type'] in hearing_types and e.get('date')]
        
        if not hearing_events:
            return 0
        
        # Sort by date
        hearing_events.sort(key=lambda x: x['date'], reverse=True)
        last_hearing_date = hearing_events[0]['date']
        
        # Calculate days from last hearing to judgment date
        judgment_date = metadata.get('judgment_date')
        
        if judgment_date:
            try:
                last_h = datetime.strptime(last_hearing_date, '%Y-%m-%d')
                judgment = datetime.strptime(judgment_date, '%Y-%m-%d')
                
                days = (judgment - last_h).days
                return max(0, days)
            except:
                pass
        
        # Default estimate
        return 30  # 1 month default
    
    def _estimate_judge_workload(self, metadata: Dict) -> int:
        """Estimate judge workload (number of cases)"""
        # This is a placeholder - in production, this would come from a database
        # For now, we'll use a heuristic based on court
        
        court = metadata.get('court', '')
        
        # Estimate based on court level
        if 'supreme' in court.lower():
            return np.random.randint(80, 150)
        elif 'high court' in court.lower():
            return np.random.randint(100, 200)
        else:
            return np.random.randint(50, 120)
    
    def _calculate_adjournment_rate(self, adjournments: int, hearings: int) -> float:
        """Calculate adjournment rate"""
        if hearings == 0:
            return 0.0
        
        return adjournments / (hearings + 1)
    
    def _calculate_hearing_frequency(self, hearings: int, case_age_days: int) -> float:
        """Calculate hearing frequency (hearings per month)"""
        if case_age_days == 0:
            return 0.0
        
        months = case_age_days / 30
        return hearings / max(1, months)
    
    def _generate_adjournment_label(self, metadata: Dict) -> int:
        """
        Generate adjournment label (1 if high risk, 0 otherwise)
        
        Args:
            metadata: Case metadata
        
        Returns:
            Binary label (0 or 1)
        """
        adjournments = metadata.get('adjournment_count', 0)
        hearings = metadata.get('hearing_count', 1)
        
        # High risk if adjournment rate > 0.5 or adjournments > 3
        adjournment_rate = adjournments / max(1, hearings)
        
        if adjournment_rate > 0.5 or adjournments > 3:
            return 1
        
        return 0
    
    def _estimate_delay_probability(self, metadata: Dict) -> float:
        """
        Estimate delay probability (0-1)
        
        Args:
            metadata: Case metadata
        
        Returns:
            Delay probability
        """
        # Factors contributing to delay
        adjournments = metadata.get('adjournment_count', 0)
        case_age = metadata.get('case_age_days', 0)
        hearings = metadata.get('hearing_count', 1)
        avg_gap = metadata.get('avg_hearing_gap', 30)
        
        # Calculate delay score
        delay_score = 0.0
        
        # Adjournment contribution
        delay_score += min(0.4, adjournments / 10 * 0.4)
        
        # Case age contribution
        delay_score += min(0.3, case_age / 1095 * 0.3)  # 3 years max
        
        # Hearing gap contribution
        if avg_gap:
            delay_score += min(0.2, avg_gap / 180 * 0.2)  # 6 months max
        
        # Hearing frequency contribution (inverse)
        hearing_freq = hearings / max(1, case_age / 30)
        if hearing_freq < 1:  # Less than 1 hearing per month
            delay_score += 0.1
        
        return min(1.0, delay_score)
    
    def generate_batch_features(self, metadata_list: List[Dict]) -> List[Dict]:
        """
        Generate features for multiple cases
        
        Args:
            metadata_list: List of metadata dictionaries
        
        Returns:
            List of feature dictionaries
        """
        logger.info(f"Generating features for {len(metadata_list)} cases")
        
        features_list = []
        
        for metadata in metadata_list:
            try:
                features = self.generate_features(metadata)
                features_list.append(features)
            except Exception as e:
                logger.error(f"Error generating features for case {metadata.get('case_number')}: {str(e)}")
                continue
        
        logger.info(f"Successfully generated features for {len(features_list)} cases")
        
        return features_list


def generate_ml_features(metadata: Dict) -> Dict:
    """
    Convenience function to generate ML features
    
    Args:
        metadata: Case metadata
    
    Returns:
        Feature dictionary
    """
    generator = FeatureGenerator()
    return generator.generate_features(metadata)
