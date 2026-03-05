"""
Dataset Builder Module
Combines parsed metadata and features into structured ML dataset
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from app.pipeline.parsers.metadata_parser import MetadataParser
from app.pipeline.dataset.feature_generator import FeatureGenerator
from app.utils.logger import logger


class DatasetBuilder:
    """Build structured ML dataset from parsed judgments"""
    
    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize dataset builder
        
        Args:
            output_dir: Directory to save processed datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_parser = MetadataParser()
        self.feature_generator = FeatureGenerator()
        
        logger.info(f"Initialized dataset builder. Output: {self.output_dir}")
    
    def process_text_file(self, text_file: Path) -> Optional[Dict]:
        """
        Process a single text file into features
        
        Args:
            text_file: Path to extracted text file
        
        Returns:
            Feature dictionary or None if processing failed
        """
        try:
            logger.info(f"Processing: {text_file.name}")
            
            # Read text
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Parse metadata
            metadata = self.metadata_parser.parse_complete_metadata(
                text,
                source_file=text_file.name
            )
            
            # Validate metadata
            metadata = self.metadata_parser.validate_metadata(metadata)
            
            # Check if valid
            if not metadata.get('validation', {}).get('is_valid', True):
                logger.warning(f"Invalid metadata for {text_file.name}")
                return None
            
            # Generate features
            features = self.feature_generator.generate_features(metadata)
            
            # Add source information
            features['source_file'] = text_file.name
            features['processed_date'] = datetime.now().isoformat()
            
            return features
        
        except Exception as e:
            logger.error(f"Error processing {text_file.name}: {str(e)}")
            return None
    
    def build_dataset_from_texts(
        self,
        text_files: List[Path],
        output_filename: str = "dataset.csv"
    ) -> pd.DataFrame:
        """
        Build dataset from multiple text files
        
        Args:
            text_files: List of text file paths
            output_filename: Output CSV filename
        
        Returns:
            DataFrame with processed dataset
        """
        logger.info(f"Building dataset from {len(text_files)} text files")
        
        # Process all files
        features_list = []
        
        for text_file in text_files:
            features = self.process_text_file(text_file)
            
            if features:
                features_list.append(features)
        
        if not features_list:
            logger.error("No valid features extracted from any file")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(features_list)
        
        # Select and order columns
        df = self._organize_columns(df)
        
        # Save dataset
        output_path = self.output_dir / output_filename
        df.to_csv(output_path, index=False)
        
        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Dataset shape: {df.shape}")
        
        return df
    
    def _organize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Organize DataFrame columns in logical order
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with organized columns
        """
        # Define column order
        primary_cols = [
            'case_age_days',
            'adjournment_history',
            'hearings_count',
            'case_type',
            'court',
            'days_since_last_hearing',
            'judge_workload',
        ]
        
        derived_cols = [
            'adjournment_rate',
            'hearing_frequency',
        ]
        
        label_cols = [
            'adjournment_label',
            'delay_probability',
        ]
        
        meta_cols = [
            'source_file',
            'data_quality',
            'processed_date',
        ]
        
        # Select available columns in order
        ordered_cols = []
        
        for col_list in [primary_cols, derived_cols, label_cols, meta_cols]:
            for col in col_list:
                if col in df.columns:
                    ordered_cols.append(col)
        
        # Add any remaining columns
        for col in df.columns:
            if col not in ordered_cols:
                ordered_cols.append(col)
        
        return df[ordered_cols]
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about the dataset
        
        Args:
            df: Dataset DataFrame
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_cases': len(df),
            'avg_case_age': df['case_age_days'].mean() if 'case_age_days' in df else None,
            'avg_adjournments': df['adjournment_history'].mean() if 'adjournment_history' in df else None,
            'avg_hearings': df['hearings_count'].mean() if 'hearings_count' in df else None,
            'adjournment_rate': df['adjournment_label'].mean() if 'adjournment_label' in df else None,
            'avg_delay_prob': df['delay_probability'].mean() if 'delay_probability' in df else None,
        }
        
        # Case type distribution
        if 'case_type' in df:
            stats['case_type_distribution'] = df['case_type'].value_counts().to_dict()
        
        # Court distribution
        if 'court' in df:
            stats['court_distribution'] = df['court'].value_counts().to_dict()
        
        # Data quality
        if 'data_quality' in df:
            stats['avg_data_quality'] = df['data_quality'].mean()
        
        return stats
    
    def merge_with_existing(
        self,
        new_df: pd.DataFrame,
        existing_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Merge new dataset with existing dataset
        
        Args:
            new_df: New DataFrame
            existing_file: Path to existing dataset file
        
        Returns:
            Merged DataFrame
        """
        if existing_file and Path(existing_file).exists():
            logger.info(f"Merging with existing dataset: {existing_file}")
            
            existing_df = pd.read_csv(existing_file)
            
            # Combine datasets
            merged_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Remove duplicates based on source_file
            if 'source_file' in merged_df.columns:
                merged_df = merged_df.drop_duplicates(subset=['source_file'], keep='last')
            
            logger.info(f"Merged dataset size: {len(merged_df)} (was {len(existing_df)})")
            
            return merged_df
        
        return new_df
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict:
        """
        Validate dataset quality
        
        Args:
            df: Dataset DataFrame
        
        Returns:
            Validation report
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # Check minimum size
        if len(df) < 10:
            validation['warnings'].append(f"Small dataset size: {len(df)} samples")
        
        # Check for required columns
        required_cols = ['case_age_days', 'adjournment_history', 'hearings_count', 'case_type']
        
        for col in required_cols:
            if col not in df.columns:
                validation['errors'].append(f"Missing required column: {col}")
                validation['is_valid'] = False
        
        # Check for missing values
        if validation['is_valid']:
            missing_counts = df[required_cols].isnull().sum()
            
            for col, count in missing_counts.items():
                if count > 0:
                    pct = (count / len(df)) * 100
                    validation['warnings'].append(f"{col}: {count} missing values ({pct:.1f}%)")
        
        # Check data ranges
        if 'case_age_days' in df.columns:
            if (df['case_age_days'] < 0).any():
                validation['errors'].append("Negative case age values found")
                validation['is_valid'] = False
        
        if 'delay_probability' in df.columns:
            if ((df['delay_probability'] < 0) | (df['delay_probability'] > 1)).any():
                validation['errors'].append("Delay probability out of range [0, 1]")
                validation['is_valid'] = False
        
        # Add statistics
        validation['statistics'] = self.get_dataset_statistics(df)
        
        return validation


def build_ml_dataset(
    text_files: List[Path],
    output_file: str = "data/processed/dataset.csv"
) -> pd.DataFrame:
    """
    Convenience function to build ML dataset
    
    Args:
        text_files: List of text file paths
        output_file: Output CSV file path
    
    Returns:
        Dataset DataFrame
    """
    builder = DatasetBuilder()
    df = builder.build_dataset_from_texts(text_files, Path(output_file).name)
    
    # Validate
    validation = builder.validate_dataset(df)
    
    if not validation['is_valid']:
        logger.error(f"Dataset validation failed: {validation['errors']}")
    else:
        logger.info("Dataset validation passed")
    
    return df
