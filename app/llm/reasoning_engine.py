"""
Reasoning Engine for generating legal explanations
"""
import re
from typing import Dict, List, Any, Optional
from app.llm.gemini_client import GeminiClient
from app.llm.prompt_templates import PromptTemplates
from app.utils.logger import logger


class ReasoningEngine:
    """
    Engine for generating human-readable explanations of AI predictions
    """
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """
        Initialize the reasoning engine
        
        Args:
            gemini_client: Optional pre-configured Gemini client
        """
        self.gemini_client = gemini_client
        self.prompt_templates = PromptTemplates()
        
        if self.gemini_client is None:
            try:
                self.gemini_client = GeminiClient()
            except ValueError as e:
                logger.warning(f"Failed to initialize Gemini client: {str(e)}")
                self.gemini_client = None
    
    def is_available(self) -> bool:
        """
        Check if the reasoning engine is available
        
        Returns:
            True if Gemini client is configured, False otherwise
        """
        return self.gemini_client is not None and self.gemini_client.is_configured()
    
    def _parse_explanation(self, raw_text: str) -> Dict[str, Any]:
        """
        Parse the structured explanation from Gemini response
        
        Args:
            raw_text: Raw text response from Gemini
        
        Returns:
            Structured explanation dictionary
        """
        try:
            explanation = ""
            key_factors = []
            historical_patterns = ""
            recommendation = ""
            
            explanation_match = re.search(r'EXPLANATION:\s*(.*?)(?=KEY FACTORS:|$)', raw_text, re.DOTALL | re.IGNORECASE)
            if explanation_match:
                explanation = explanation_match.group(1).strip()
            
            factors_match = re.search(r'KEY FACTORS:\s*(.*?)(?=HISTORICAL PATTERNS:|$)', raw_text, re.DOTALL | re.IGNORECASE)
            if factors_match:
                factors_text = factors_match.group(1).strip()
                key_factors = [
                    line.strip().lstrip('-').lstrip('*').strip() 
                    for line in factors_text.split('\n') 
                    if line.strip() and (line.strip().startswith('-') or line.strip().startswith('*') or line.strip()[0].isdigit())
                ]
            
            patterns_match = re.search(r'HISTORICAL PATTERNS:\s*(.*?)(?=RECOMMENDATION:|$)', raw_text, re.DOTALL | re.IGNORECASE)
            if patterns_match:
                historical_patterns = patterns_match.group(1).strip()
            
            recommendation_match = re.search(r'RECOMMENDATION:\s*(.*?)$', raw_text, re.DOTALL | re.IGNORECASE)
            if recommendation_match:
                recommendation = recommendation_match.group(1).strip()
            
            if not explanation:
                explanation = raw_text[:500] if len(raw_text) > 500 else raw_text
            
            return {
                "explanation": explanation,
                "key_factors": key_factors,
                "historical_patterns": historical_patterns,
                "recommendation": recommendation,
                "raw_response": raw_text
            }
        
        except Exception as e:
            logger.error(f"Error parsing explanation: {str(e)}")
            return {
                "explanation": raw_text,
                "key_factors": [],
                "historical_patterns": "",
                "recommendation": "",
                "raw_response": raw_text
            }
    
    def generate_explanation(
        self,
        case_data: Dict[str, Any],
        prediction: Dict[str, Any],
        similar_cases: List[Dict[str, Any]] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation for the prediction
        
        Args:
            case_data: Case metadata
            prediction: Prediction results from ML model
            similar_cases: List of similar cases (optional)
            temperature: Sampling temperature for generation
        
        Returns:
            Structured explanation with key factors and recommendations
        """
        if not self.is_available():
            logger.error("Reasoning engine not available - Gemini client not configured")
            raise ValueError("Gemini API is not configured. Please set GEMINI_API_KEY environment variable.")
        
        try:
            similar_cases = similar_cases or []
            
            prompt = self.prompt_templates.generate_explanation_prompt(
                case_data=case_data,
                prediction=prediction,
                similar_cases=similar_cases
            )
            
            logger.info("Generating explanation with Gemini API")
            raw_response = self.gemini_client.generate_text(
                prompt=prompt,
                temperature=temperature,
                max_tokens=1024
            )
            
            parsed_explanation = self._parse_explanation(raw_response)
            
            logger.info("Successfully generated and parsed explanation")
            return parsed_explanation
        
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            raise
    
    def generate_delay_explanation(
        self,
        case_data: Dict[str, Any],
        prediction: Dict[str, Any],
        similar_cases: List[Dict[str, Any]] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate a delay-focused explanation
        
        Args:
            case_data: Case metadata
            prediction: Prediction results from ML model
            similar_cases: List of similar cases (optional)
            temperature: Sampling temperature for generation
        
        Returns:
            Structured delay explanation
        """
        if not self.is_available():
            logger.error("Reasoning engine not available - Gemini client not configured")
            raise ValueError("Gemini API is not configured. Please set GEMINI_API_KEY environment variable.")
        
        try:
            similar_cases = similar_cases or []
            
            prompt = self.prompt_templates.generate_delay_explanation_prompt(
                case_data=case_data,
                prediction=prediction,
                similar_cases=similar_cases
            )
            
            logger.info("Generating delay explanation with Gemini API")
            raw_response = self.gemini_client.generate_text(
                prompt=prompt,
                temperature=temperature,
                max_tokens=800
            )
            
            return {
                "explanation": raw_response,
                "key_factors": [],
                "recommendation": "",
                "raw_response": raw_response
            }
        
        except Exception as e:
            logger.error(f"Error generating delay explanation: {str(e)}")
            raise


reasoning_engine = ReasoningEngine()
