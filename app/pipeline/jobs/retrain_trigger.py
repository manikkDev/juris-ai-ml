"""
Retrain Trigger
Automatically triggers model retraining when new data is available
"""
import sys
from pathlib import Path
from typing import Optional, Dict
import requests

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.pipeline.storage.dataset_store import DatasetStore
from app.training.train_model import CasePredictor
from app.utils.logger import logger


class RetrainTrigger:
    """Trigger model retraining based on new data"""
    
    def __init__(
        self,
        dataset_dir: str = "data/processed",
        model_path: str = "models/adjournment_model.joblib",
        ml_service_url: Optional[str] = None
    ):
        """
        Initialize retrain trigger
        
        Args:
            dataset_dir: Directory containing datasets
            model_path: Path to save trained model
            ml_service_url: URL of ML service API (for triggering via API)
        """
        self.dataset_store = DatasetStore(dataset_dir)
        self.model_path = model_path
        self.ml_service_url = ml_service_url or "http://localhost:8000"
        
        logger.info("Initialized retrain trigger")
    
    def check_new_data(self) -> bool:
        """
        Check if new data is available for retraining
        
        Returns:
            True if new data is available
        """
        versions = self.dataset_store.list_versions()
        
        if not versions:
            logger.info("No dataset versions found")
            return False
        
        # Check if latest version is newer than model
        latest_version = versions[-1]
        model_path = Path(self.model_path)
        
        if not model_path.exists():
            logger.info("Model file not found - retraining needed")
            return True
        
        # Compare timestamps
        from datetime import datetime
        
        latest_data_time = datetime.fromisoformat(latest_version['created_at'])
        model_time = datetime.fromtimestamp(model_path.stat().st_mtime)
        
        if latest_data_time > model_time:
            logger.info("New data available - retraining recommended")
            return True
        
        logger.info("No new data since last training")
        return False
    
    def trigger_local_training(
        self,
        dataset_filename: str = "dataset.csv",
        test_size: float = 0.2
    ) -> Dict:
        """
        Trigger local model training
        
        Args:
            dataset_filename: Dataset file to use
            test_size: Test set proportion
        
        Returns:
            Training results dictionary
        """
        logger.info("=" * 60)
        logger.info("TRIGGERING LOCAL MODEL RETRAINING")
        logger.info("=" * 60)
        
        try:
            # Load dataset
            df = self.dataset_store.load_dataset(filename=dataset_filename)
            
            if df is None or df.empty:
                logger.error("Failed to load dataset")
                return {"success": False, "error": "Dataset not found"}
            
            logger.info(f"Loaded dataset: {len(df)} samples")
            
            # Initialize and train model
            predictor = CasePredictor(model_type='random_forest')
            
            # Get dataset path
            dataset_path = self.dataset_store.storage_dir / dataset_filename
            
            # Train model
            metrics = predictor.train(
                dataset_path=str(dataset_path),
                test_size=test_size,
                random_state=42
            )
            
            # Save model
            predictor.save_model(self.model_path)
            
            logger.info("=" * 60)
            logger.info("MODEL RETRAINING COMPLETED")
            logger.info("=" * 60)
            
            return {
                "success": True,
                "metrics": metrics,
                "model_path": self.model_path,
                "samples_used": len(df)
            }
        
        except Exception as e:
            logger.error(f"Error during retraining: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def trigger_api_training(self) -> Dict:
        """
        Trigger model retraining via ML service API
        
        Returns:
            API response dictionary
        """
        logger.info("Triggering retraining via ML service API")
        
        try:
            url = f"{self.ml_service_url}/api/retrain"
            
            response = requests.post(
                url,
                json={
                    "dataset_path": "data/processed/dataset.csv",
                    "test_size": 0.2,
                    "random_state": 42
                },
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Retraining triggered successfully via API")
                return response.json()
            else:
                logger.error(f"API error: {response.status_code}")
                return {"success": False, "error": f"API returned {response.status_code}"}
        
        except requests.exceptions.ConnectionError:
            logger.warning("ML service not available - falling back to local training")
            return self.trigger_local_training()
        
        except Exception as e:
            logger.error(f"Error triggering API retraining: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def auto_retrain_if_needed(self, use_api: bool = False) -> Dict:
        """
        Automatically retrain if new data is available
        
        Args:
            use_api: Use API for retraining (otherwise local)
        
        Returns:
            Retraining results
        """
        logger.info("Checking if retraining is needed...")
        
        if not self.check_new_data():
            return {
                "success": True,
                "retrained": False,
                "message": "No retraining needed - model is up to date"
            }
        
        logger.info("New data detected - starting retraining")
        
        if use_api:
            result = self.trigger_api_training()
        else:
            result = self.trigger_local_training()
        
        if result.get("success"):
            result["retrained"] = True
        
        return result
    
    def get_training_status(self) -> Dict:
        """
        Get current training status
        
        Returns:
            Status dictionary
        """
        model_path = Path(self.model_path)
        
        status = {
            "model_exists": model_path.exists(),
            "model_path": str(model_path),
            "dataset_versions": len(self.dataset_store.list_versions()),
            "current_dataset_version": self.dataset_store.get_current_version(),
        }
        
        if model_path.exists():
            from datetime import datetime
            model_time = datetime.fromtimestamp(model_path.stat().st_mtime)
            status["model_last_updated"] = model_time.isoformat()
        
        return status


def main():
    """Main function for command-line execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Juris AI Model Retraining Trigger")
    
    parser.add_argument('--check-only', action='store_true', help='Only check if retraining is needed')
    parser.add_argument('--force', action='store_true', help='Force retraining even if not needed')
    parser.add_argument('--use-api', action='store_true', help='Use ML service API for retraining')
    parser.add_argument('--dataset', type=str, default='dataset.csv', help='Dataset filename')
    
    args = parser.parse_args()
    
    # Initialize trigger
    trigger = RetrainTrigger()
    
    if args.check_only:
        # Just check status
        status = trigger.get_training_status()
        needs_retrain = trigger.check_new_data()
        
        print("\n" + "=" * 60)
        print("TRAINING STATUS")
        print("=" * 60)
        print(f"Model exists: {status['model_exists']}")
        print(f"Model path: {status['model_path']}")
        print(f"Dataset versions: {status['dataset_versions']}")
        print(f"Retraining needed: {needs_retrain}")
        print("=" * 60)
        
        return 0
    
    # Trigger retraining
    if args.force:
        logger.info("Forcing retraining...")
        if args.use_api:
            result = trigger.trigger_api_training()
        else:
            result = trigger.trigger_local_training(dataset_filename=args.dataset)
    else:
        result = trigger.auto_retrain_if_needed(use_api=args.use_api)
    
    # Print results
    print("\n" + "=" * 60)
    print("RETRAINING RESULTS")
    print("=" * 60)
    print(f"Success: {result.get('success')}")
    print(f"Retrained: {result.get('retrained', False)}")
    
    if result.get('error'):
        print(f"Error: {result['error']}")
    
    if result.get('metrics'):
        print("\nMetrics:")
        for model_name, metrics in result['metrics'].items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    print("=" * 60)
    
    return 0 if result.get('success') else 1


if __name__ == "__main__":
    sys.exit(main())
