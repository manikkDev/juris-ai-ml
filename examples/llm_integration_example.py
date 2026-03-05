"""
Example: Integrating LLM Reasoning Layer with Prediction Workflow

This example demonstrates how to combine ML predictions with LLM-generated explanations
to create a complete judicial AI assistant workflow.
"""
import asyncio
import os
from typing import Dict, Any

from app.services.prediction_service import prediction_service
from app.llm.reasoning_engine import reasoning_engine
from app.schemas.prediction_schema import CaseInput


async def complete_prediction_workflow(case_input: CaseInput) -> Dict[str, Any]:
    """
    Complete workflow: Prediction + Similar Cases + Explanation
    
    Args:
        case_input: Case data for prediction
    
    Returns:
        Combined results with prediction and explanation
    """
    
    # Step 1: Generate ML prediction
    print("Step 1: Generating ML prediction...")
    prediction = prediction_service.predict_case(case_input)
    
    print(f"  - Adjournment Risk: {prediction.adjournmentRisk:.2%}")
    print(f"  - Delay Probability: {prediction.delayProbability:.2%}")
    print(f"  - Resolution Estimate: {prediction.resolutionEstimate}")
    
    # Step 2: Retrieve similar cases (mock data for example)
    print("\nStep 2: Retrieving similar cases...")
    similar_cases = [
        {
            "case_type": "civil",
            "case_age_days": 380,
            "adjournment_history": 4,
            "hearings_count": 7,
            "outcome": "delayed",
            "similarity_score": 0.89
        },
        {
            "case_type": "civil",
            "case_age_days": 450,
            "adjournment_history": 3,
            "hearings_count": 9,
            "outcome": "resolved",
            "similarity_score": 0.85
        }
    ]
    print(f"  - Found {len(similar_cases)} similar cases")
    
    # Step 3: Generate LLM explanation
    print("\nStep 3: Generating AI explanation...")
    
    if not reasoning_engine.is_available():
        print("  ⚠️  Warning: Gemini API not configured. Skipping explanation.")
        explanation = None
    else:
        explanation = reasoning_engine.generate_explanation(
            case_data=case_input.dict(),
            prediction=prediction.dict(),
            similar_cases=similar_cases
        )
        
        print(f"\n{'='*80}")
        print("EXPLANATION:")
        print(f"{'='*80}")
        print(explanation["explanation"])
        
        print(f"\n{'='*80}")
        print("KEY FACTORS:")
        print(f"{'='*80}")
        for i, factor in enumerate(explanation["key_factors"], 1):
            print(f"{i}. {factor}")
        
        if explanation.get("historical_patterns"):
            print(f"\n{'='*80}")
            print("HISTORICAL PATTERNS:")
            print(f"{'='*80}")
            print(explanation["historical_patterns"])
        
        print(f"\n{'='*80}")
        print("RECOMMENDATION:")
        print(f"{'='*80}")
        print(explanation["recommendation"])
        print(f"{'='*80}\n")
    
    # Step 4: Combine results
    return {
        "prediction": prediction.dict(),
        "similar_cases": similar_cases,
        "explanation": explanation
    }


async def main():
    """
    Main example function
    """
    print("="*80)
    print("JURIS AI - COMPLETE PREDICTION WORKFLOW WITH LLM REASONING")
    print("="*80)
    
    # Check if Gemini API is configured
    if not os.getenv("GEMINI_API_KEY"):
        print("\n⚠️  GEMINI_API_KEY not set in environment variables")
        print("Set it in .env file to enable AI explanations\n")
    
    # Example case with high delay risk
    case_input = CaseInput(
        case_age_days=420,
        adjournment_history=3,
        hearings_count=8,
        case_type="civil",
        days_since_last_hearing=40,
        judge_workload=120
    )
    
    print("\nCase Details:")
    print(f"  - Case Age: {case_input.case_age_days} days")
    print(f"  - Case Type: {case_input.case_type}")
    print(f"  - Previous Adjournments: {case_input.adjournment_history}")
    print(f"  - Total Hearings: {case_input.hearings_count}")
    print(f"  - Days Since Last Hearing: {case_input.days_since_last_hearing}")
    print(f"  - Judge Workload: {case_input.judge_workload} cases")
    print()
    
    # Run complete workflow
    result = await complete_prediction_workflow(case_input)
    
    print("\n✅ Workflow completed successfully!")
    print(f"   - Prediction generated: ✓")
    print(f"   - Similar cases retrieved: ✓")
    print(f"   - Explanation generated: {'✓' if result['explanation'] else '✗ (API not configured)'}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
