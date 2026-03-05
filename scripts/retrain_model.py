"""
Script to retrain the model
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.training.train_model import CasePredictor


def main():
    """Retrain the model"""
    print("=" * 60)
    print("Juris AI - Model Retraining")
    print("=" * 60)
    
    # Initialize predictor
    predictor = CasePredictor(model_type='random_forest')
    
    # Train the model
    print("\nStarting model training...")
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
    print("Model retraining completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
