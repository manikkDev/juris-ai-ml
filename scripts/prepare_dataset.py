"""
Script to prepare and generate synthetic dataset
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.training.dataset_loader import SyntheticDataGenerator, get_dataset_statistics


def main():
    """Generate and save synthetic dataset"""
    print("=" * 60)
    print("Juris AI - Dataset Preparation")
    print("=" * 60)
    
    # Initialize generator
    generator = SyntheticDataGenerator(random_state=42)
    
    # Generate dataset
    print("\nGenerating synthetic dataset...")
    n_samples = 2000
    df = generator.generate_dataset(n_samples=n_samples)
    
    # Get statistics
    stats = get_dataset_statistics(df)
    
    print(f"\nDataset Statistics:")
    print(f"  Total Samples: {stats['total_samples']}")
    print(f"  Adjournment Rate: {stats['adjournment_rate']:.2%}")
    print(f"  Avg Delay Probability: {stats['avg_delay_probability']:.2f}")
    print(f"  Avg Case Age: {stats['avg_case_age']:.0f} days")
    print(f"  Avg Adjournments: {stats['avg_adjournments']:.1f}")
    print(f"  Avg Judge Workload: {stats['avg_judge_workload']:.0f} cases")
    
    print("\nCase Type Distribution:")
    for case_type, count in stats['case_type_distribution'].items():
        print(f"  {case_type}: {count} ({count/stats['total_samples']*100:.1f}%)")
    
    # Save dataset
    output_path = "data/raw/synthetic_cases.csv"
    generator.save_dataset(df, output_path)
    
    print("\n" + "=" * 60)
    print("Dataset preparation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
