"""
Dataset loader and synthetic data generator for training
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class SyntheticDataGenerator:
    """Generate synthetic judicial case data for training"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic dataset for model training
        
        Args:
            n_samples: Number of samples to generate
        
        Returns:
            DataFrame with synthetic case data
        """
        data = []
        
        case_types = ['civil', 'criminal', 'family', 'commercial', 'constitutional', 
                      'labor', 'tax', 'property']
        
        for _ in range(n_samples):
            # Generate base features
            case_age_days = np.random.randint(30, 1095)  # 1 month to 3 years
            adjournment_history = np.random.poisson(3)  # Poisson distribution for adjournments
            hearings_count = max(1, adjournment_history + np.random.randint(1, 8))
            case_type = np.random.choice(case_types)
            days_since_last_hearing = np.random.randint(0, 180)
            judge_workload = np.random.randint(20, 200)
            
            # Generate labels based on features (with some noise)
            # Adjournment label: 1 if likely to be adjourned, 0 otherwise
            adjournment_score = (
                (adjournment_history * 0.3) +
                (case_age_days / 365 * 0.2) +
                (days_since_last_hearing / 90 * 0.2) +
                (judge_workload / 150 * 0.15) +
                np.random.normal(0, 0.15)
            )
            adjournment_label = 1 if adjournment_score > 0.5 else 0
            
            # Delay probability: continuous value between 0 and 1
            delay_base = (
                (adjournment_history / 10 * 0.35) +
                (case_age_days / 1095 * 0.25) +
                (days_since_last_hearing / 180 * 0.2) +
                (judge_workload / 200 * 0.2)
            )
            delay_probability = np.clip(delay_base + np.random.normal(0, 0.1), 0, 1)
            
            data.append({
                'case_age_days': case_age_days,
                'adjournment_history': adjournment_history,
                'hearings_count': hearings_count,
                'case_type': case_type,
                'days_since_last_hearing': days_since_last_hearing,
                'judge_workload': judge_workload,
                'adjournment_label': adjournment_label,
                'delay_probability': delay_probability
            })
        
        df = pd.DataFrame(data)
        return df
    
    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """
        Save dataset to CSV file
        
        Args:
            df: DataFrame to save
            filepath: Path to save the file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from CSV file
        
        Args:
            filepath: Path to the CSV file
        
        Returns:
            Loaded DataFrame
        """
        return pd.read_csv(filepath)


def load_or_generate_dataset(
    dataset_path: Optional[str] = None,
    n_samples: int = 1000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load existing dataset or generate new synthetic data
    
    Args:
        dataset_path: Path to existing dataset (if None, generates new data)
        n_samples: Number of samples to generate if creating new dataset
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with case data
    """
    generator = SyntheticDataGenerator(random_state=random_state)
    
    if dataset_path and Path(dataset_path).exists():
        print(f"Loading dataset from {dataset_path}")
        return generator.load_dataset(dataset_path)
    else:
        print(f"Generating synthetic dataset with {n_samples} samples")
        df = generator.generate_dataset(n_samples)
        
        # Save generated dataset
        save_path = dataset_path or "data/raw/synthetic_cases.csv"
        generator.save_dataset(df, save_path)
        
        return df


def get_dataset_statistics(df: pd.DataFrame) -> dict:
    """
    Get statistics about the dataset
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_samples': len(df),
        'adjournment_rate': df['adjournment_label'].mean(),
        'avg_delay_probability': df['delay_probability'].mean(),
        'case_type_distribution': df['case_type'].value_counts().to_dict(),
        'avg_case_age': df['case_age_days'].mean(),
        'avg_adjournments': df['adjournment_history'].mean(),
        'avg_judge_workload': df['judge_workload'].mean()
    }
    
    return stats
