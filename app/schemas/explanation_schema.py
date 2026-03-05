"""
Pydantic schemas for explanation/reasoning endpoints
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SimilarCase(BaseModel):
    """Schema for similar case information"""
    case_type: Optional[str] = None
    case_age_days: Optional[int] = None
    adjournment_history: Optional[int] = None
    hearings_count: Optional[int] = None
    outcome: Optional[str] = None
    similarity_score: Optional[float] = None


class ExplanationRequest(BaseModel):
    """Input schema for explanation generation"""
    case_data: Dict[str, Any] = Field(..., description="Case metadata and features")
    prediction: Dict[str, Any] = Field(..., description="Prediction results from ML model")
    similar_cases: Optional[List[Dict[str, Any]]] = Field(
        default=[],
        description="List of similar cases from semantic search"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "case_data": {
                    "case_age_days": 420,
                    "adjournment_history": 3,
                    "hearings_count": 8,
                    "case_type": "civil",
                    "days_since_last_hearing": 40,
                    "judge_workload": 120
                },
                "prediction": {
                    "adjournmentRisk": 0.74,
                    "delayProbability": 0.61,
                    "resolutionEstimate": "4 months",
                    "topFactors": [
                        {
                            "factor": "High adjournment history",
                            "importance": 0.35,
                            "impact": "High"
                        }
                    ],
                    "confidence": 0.82,
                    "modelVersion": "1.0.0"
                },
                "similar_cases": [
                    {
                        "case_type": "civil",
                        "case_age_days": 380,
                        "adjournment_history": 4,
                        "outcome": "delayed",
                        "similarity_score": 0.89
                    }
                ]
            }
        }


class ExplanationResponse(BaseModel):
    """Output schema for explanation results"""
    explanation: str = Field(..., description="Human-readable explanation of the prediction")
    key_factors: List[str] = Field(..., description="Key factors contributing to the prediction")
    historical_patterns: Optional[str] = Field(
        default="",
        description="Patterns observed in similar historical cases"
    )
    recommendation: str = Field(..., description="Actionable recommendations for the judge")
    
    class Config:
        json_schema_extra = {
            "example": {
                "explanation": "This civil case shows a high risk of delay based on several critical factors. The case has been active for 420 days with 3 previous adjournments, indicating a pattern of procedural delays. The judge's current workload of 120 cases may contribute to scheduling challenges.",
                "key_factors": [
                    "High adjournment history (3 previous adjournments)",
                    "Extended case age (420 days)",
                    "Heavy judge workload (120 active cases)",
                    "Significant gap since last hearing (40 days)"
                ],
                "historical_patterns": "Similar civil cases with 3+ adjournments typically experience 60-70% longer resolution times. Cases with comparable characteristics show an average of 2-3 additional months before resolution.",
                "recommendation": "Consider prioritizing this case for the next available hearing slot. Implement strict time limits for proceedings to prevent further adjournments. Review if case consolidation or alternative dispute resolution could expedite resolution."
            }
        }
