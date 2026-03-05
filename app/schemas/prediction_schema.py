"""
Pydantic schemas for request/response validation
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class CaseInput(BaseModel):
    """Input schema for case prediction"""
    case_age_days: int = Field(..., ge=0, description="Age of the case in days")
    adjournment_history: int = Field(..., ge=0, description="Number of previous adjournments")
    hearings_count: int = Field(..., ge=0, description="Total number of hearings")
    case_type: str = Field(..., description="Type of case (civil, criminal, family, etc.)")
    days_since_last_hearing: int = Field(..., ge=0, description="Days since last hearing")
    judge_workload: int = Field(..., ge=0, description="Number of cases assigned to judge")
    
    class Config:
        json_schema_extra = {
            "example": {
                "case_age_days": 420,
                "adjournment_history": 3,
                "hearings_count": 8,
                "case_type": "civil",
                "days_since_last_hearing": 40,
                "judge_workload": 120
            }
        }


class TopFactor(BaseModel):
    """Schema for top contributing factors"""
    factor: str
    importance: float
    impact: str


class PredictionOutput(BaseModel):
    """Output schema for prediction results"""
    adjournmentRisk: float = Field(..., ge=0, le=1, description="Probability of adjournment")
    delayProbability: float = Field(..., ge=0, le=1, description="Probability of delay")
    resolutionEstimate: str = Field(..., description="Estimated time to resolution")
    topFactors: List[TopFactor] = Field(..., description="Top contributing factors")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")
    modelVersion: str = Field(..., description="Version of the model used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "adjournmentRisk": 0.74,
                "delayProbability": 0.61,
                "resolutionEstimate": "4 months",
                "topFactors": [
                    {
                        "factor": "High adjournment history",
                        "importance": 0.35,
                        "impact": "High"
                    },
                    {
                        "factor": "Old case age",
                        "importance": 0.28,
                        "impact": "High"
                    }
                ],
                "confidence": 0.82,
                "modelVersion": "1.0.0"
            }
        }


class ModelInfo(BaseModel):
    """Schema for model information"""
    model_name: str
    version: str
    trained_date: Optional[str]
    accuracy: Optional[float]
    features: List[str]
    status: str


class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str
    model_loaded: bool
    version: str
    timestamp: str


class TrainingRequest(BaseModel):
    """Schema for model retraining request"""
    dataset_path: Optional[str] = None
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = Field(default=42)


class TrainingResponse(BaseModel):
    """Schema for training response"""
    status: str
    message: str
    metrics: Optional[dict] = None
    model_path: Optional[str] = None
