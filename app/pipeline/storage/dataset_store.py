"""
Dataset Storage Module
Manages dataset storage, versioning, and retrieval
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import json
import shutil

from app.utils.logger import logger


class DatasetStore:
    """Manage dataset storage and versioning"""
    
    def __init__(self, storage_dir: str = "data/processed"):
        """
        Initialize dataset store
        
        Args:
            storage_dir: Directory for dataset storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Versioning directory
        self.versions_dir = self.storage_dir / "versions"
        self.versions_dir.mkdir(exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.storage_dir / "dataset_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"Initialized dataset store at {self.storage_dir}")
    
    def _load_metadata(self) -> Dict:
        """Load dataset metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        
        return {
            "versions": [],
            "current_version": None,
            "last_updated": None
        }
    
    def _save_metadata(self):
        """Save dataset metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_dataset(
        self,
        df: pd.DataFrame,
        filename: str = "dataset.csv",
        version: Optional[str] = None,
        description: Optional[str] = None
    ) -> Path:
        """
        Save dataset with versioning
        
        Args:
            df: Dataset DataFrame
            filename: Filename for dataset
            version: Version identifier (auto-generated if None)
            description: Version description
        
        Returns:
            Path to saved dataset
        """
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main dataset
        main_path = self.storage_dir / filename
        df.to_csv(main_path, index=False)
        
        # Save versioned copy
        version_filename = f"dataset_v{version}.csv"
        version_path = self.versions_dir / version_filename
        df.to_csv(version_path, index=False)
        
        # Update metadata
        version_info = {
            "version": version,
            "filename": version_filename,
            "path": str(version_path),
            "created_at": datetime.now().isoformat(),
            "description": description or "Dataset version",
            "num_samples": len(df),
            "num_features": len(df.columns),
            "columns": list(df.columns)
        }
        
        self.metadata["versions"].append(version_info)
        self.metadata["current_version"] = version
        self.metadata["last_updated"] = datetime.now().isoformat()
        self._save_metadata()
        
        logger.info(f"Saved dataset version {version}: {len(df)} samples, {len(df.columns)} features")
        
        return main_path
    
    def load_dataset(
        self,
        version: Optional[str] = None,
        filename: str = "dataset.csv"
    ) -> Optional[pd.DataFrame]:
        """
        Load dataset
        
        Args:
            version: Version to load (None for current)
            filename: Filename if loading current version
        
        Returns:
            Dataset DataFrame or None
        """
        try:
            if version:
                # Load specific version
                version_info = self._get_version_info(version)
                
                if version_info:
                    path = Path(version_info["path"])
                    logger.info(f"Loading dataset version {version}")
                    return pd.read_csv(path)
                else:
                    logger.error(f"Version {version} not found")
                    return None
            else:
                # Load current version
                path = self.storage_dir / filename
                
                if path.exists():
                    logger.info(f"Loading current dataset from {filename}")
                    return pd.read_csv(path)
                else:
                    logger.warning(f"Dataset file not found: {filename}")
                    return None
        
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return None
    
    def _get_version_info(self, version: str) -> Optional[Dict]:
        """Get version information"""
        for v in self.metadata.get("versions", []):
            if v["version"] == version:
                return v
        
        return None
    
    def list_versions(self) -> List[Dict]:
        """
        List all dataset versions
        
        Returns:
            List of version information dictionaries
        """
        return self.metadata.get("versions", [])
    
    def get_current_version(self) -> Optional[str]:
        """
        Get current version identifier
        
        Returns:
            Current version string or None
        """
        return self.metadata.get("current_version")
    
    def delete_version(self, version: str) -> bool:
        """
        Delete a specific version
        
        Args:
            version: Version to delete
        
        Returns:
            True if successful, False otherwise
        """
        version_info = self._get_version_info(version)
        
        if not version_info:
            logger.error(f"Version {version} not found")
            return False
        
        try:
            # Delete file
            path = Path(version_info["path"])
            if path.exists():
                path.unlink()
            
            # Remove from metadata
            self.metadata["versions"] = [
                v for v in self.metadata["versions"]
                if v["version"] != version
            ]
            
            # Update current version if deleted
            if self.metadata.get("current_version") == version:
                if self.metadata["versions"]:
                    self.metadata["current_version"] = self.metadata["versions"][-1]["version"]
                else:
                    self.metadata["current_version"] = None
            
            self._save_metadata()
            
            logger.info(f"Deleted version {version}")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting version {version}: {str(e)}")
            return False
    
    def get_dataset_stats(self) -> Dict:
        """
        Get statistics about stored datasets
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_versions": len(self.metadata.get("versions", [])),
            "current_version": self.metadata.get("current_version"),
            "last_updated": self.metadata.get("last_updated"),
            "storage_dir": str(self.storage_dir),
        }
        
        # Get current dataset stats
        current_df = self.load_dataset()
        
        if current_df is not None:
            stats["current_samples"] = len(current_df)
            stats["current_features"] = len(current_df.columns)
        
        return stats
    
    def export_dataset(
        self,
        output_path: str,
        version: Optional[str] = None,
        format: str = "csv"
    ) -> bool:
        """
        Export dataset to external location
        
        Args:
            output_path: Output file path
            version: Version to export (None for current)
            format: Export format ('csv', 'json', 'parquet')
        
        Returns:
            True if successful, False otherwise
        """
        try:
            df = self.load_dataset(version=version)
            
            if df is None:
                return False
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "csv":
                df.to_csv(output_path, index=False)
            elif format == "json":
                df.to_json(output_path, orient="records", indent=2)
            elif format == "parquet":
                df.to_parquet(output_path, index=False)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Exported dataset to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting dataset: {str(e)}")
            return False


def save_dataset(df: pd.DataFrame, filename: str = "dataset.csv") -> Path:
    """
    Convenience function to save dataset
    
    Args:
        df: Dataset DataFrame
        filename: Output filename
    
    Returns:
        Path to saved file
    """
    store = DatasetStore()
    return store.save_dataset(df, filename)


def load_dataset(filename: str = "dataset.csv") -> Optional[pd.DataFrame]:
    """
    Convenience function to load dataset
    
    Args:
        filename: Dataset filename
    
    Returns:
        Dataset DataFrame or None
    """
    store = DatasetStore()
    return store.load_dataset(filename=filename)
