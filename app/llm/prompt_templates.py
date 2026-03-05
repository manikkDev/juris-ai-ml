"""
Prompt templates for legal reasoning explanations
"""
from typing import Dict, List, Any


class PromptTemplates:
    """
    Templates for generating structured prompts for legal AI explanations
    """
    
    @staticmethod
    def format_case_details(case_data: Dict[str, Any]) -> str:
        """
        Format case metadata into readable text
        
        Args:
            case_data: Dictionary containing case information
        
        Returns:
            Formatted case details string
        """
        return f"""
- Case Age: {case_data.get('case_age_days', 'N/A')} days
- Case Type: {case_data.get('case_type', 'N/A')}
- Previous Adjournments: {case_data.get('adjournment_history', 'N/A')}
- Total Hearings: {case_data.get('hearings_count', 'N/A')}
- Days Since Last Hearing: {case_data.get('days_since_last_hearing', 'N/A')}
- Judge Workload: {case_data.get('judge_workload', 'N/A')} cases
"""
    
    @staticmethod
    def format_prediction_results(prediction: Dict[str, Any]) -> str:
        """
        Format prediction results into readable text
        
        Args:
            prediction: Dictionary containing prediction results
        
        Returns:
            Formatted prediction results string
        """
        adjournment_risk = prediction.get('adjournmentRisk', 0) * 100
        delay_prob = prediction.get('delayProbability', 0) * 100
        resolution = prediction.get('resolutionEstimate', 'N/A')
        confidence = prediction.get('confidence', 0) * 100
        
        factors_text = ""
        if 'topFactors' in prediction:
            factors_text = "\nTop Contributing Factors:\n"
            for factor in prediction['topFactors']:
                factors_text += f"  - {factor.get('factor', 'N/A')} (Importance: {factor.get('importance', 0):.2f}, Impact: {factor.get('impact', 'N/A')})\n"
        
        return f"""
- Adjournment Risk: {adjournment_risk:.1f}%
- Delay Probability: {delay_prob:.1f}%
- Estimated Resolution Time: {resolution}
- Model Confidence: {confidence:.1f}%{factors_text}
"""
    
    @staticmethod
    def format_similar_cases(similar_cases: List[Dict[str, Any]]) -> str:
        """
        Format similar cases into readable text
        
        Args:
            similar_cases: List of similar case dictionaries
        
        Returns:
            Formatted similar cases string
        """
        if not similar_cases:
            return "No similar cases found in the database."
        
        cases_text = ""
        for idx, case in enumerate(similar_cases[:5], 1):
            cases_text += f"\n{idx}. "
            cases_text += f"Case Type: {case.get('case_type', 'N/A')}, "
            cases_text += f"Age: {case.get('case_age_days', 'N/A')} days, "
            cases_text += f"Adjournments: {case.get('adjournment_history', 'N/A')}, "
            cases_text += f"Outcome: {case.get('outcome', 'N/A')}"
            
            if 'similarity_score' in case:
                cases_text += f" (Similarity: {case['similarity_score']:.2f})"
        
        return cases_text
    
    @staticmethod
    def generate_explanation_prompt(
        case_data: Dict[str, Any],
        prediction: Dict[str, Any],
        similar_cases: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a comprehensive prompt for legal reasoning explanation
        
        Args:
            case_data: Case metadata
            prediction: Prediction results
            similar_cases: List of similar cases
        
        Returns:
            Complete prompt for Gemini API
        """
        case_details = PromptTemplates.format_case_details(case_data)
        prediction_results = PromptTemplates.format_prediction_results(prediction)
        similar_cases_text = PromptTemplates.format_similar_cases(similar_cases)
        
        prompt = f"""You are a legal assistant helping judges understand AI predictions for case management.

**Case Information:**
{case_details}

**AI Prediction Results:**
{prediction_results}

**Similar Historical Cases:**
{similar_cases_text}

**Task:**
Based on the above information, explain why this case is likely to face delays or adjournments. Provide a clear, professional explanation that helps judges make informed decisions.

**Please provide:**

1. **Key Factors**: List the 3-4 most important factors contributing to the prediction
2. **Historical Patterns**: Explain what similar cases reveal about this situation
3. **Recommendation**: Provide actionable recommendations for the judge to minimize delays

**Format your response as follows:**

EXPLANATION:
[Provide a 2-3 paragraph explanation in clear, professional language]

KEY FACTORS:
- [Factor 1]
- [Factor 2]
- [Factor 3]

HISTORICAL PATTERNS:
[Explain patterns from similar cases]

RECOMMENDATION:
[Provide specific, actionable recommendations]
"""
        return prompt
    
    @staticmethod
    def generate_delay_explanation_prompt(
        case_data: Dict[str, Any],
        prediction: Dict[str, Any],
        similar_cases: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a prompt specifically focused on delay explanation
        
        Args:
            case_data: Case metadata
            prediction: Prediction results
            similar_cases: List of similar cases
        
        Returns:
            Delay-focused prompt for Gemini API
        """
        case_details = PromptTemplates.format_case_details(case_data)
        prediction_results = PromptTemplates.format_prediction_results(prediction)
        similar_cases_text = PromptTemplates.format_similar_cases(similar_cases)
        
        delay_prob = prediction.get('delayProbability', 0) * 100
        
        prompt = f"""You are a legal assistant analyzing case delay risks.

**Case Information:**
{case_details}

**Delay Prediction:**
The AI model predicts a {delay_prob:.1f}% probability of significant delay.

{prediction_results}

**Similar Cases:**
{similar_cases_text}

**Explain why this case is at risk of delay:**

Provide a concise, evidence-based explanation focusing on:
1. Primary delay risk factors
2. Patterns from similar cases
3. Specific steps to mitigate delay

Keep the explanation professional and actionable for judicial decision-making.
"""
        return prompt
