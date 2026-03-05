"""
Gemini API Client for generating legal explanations
"""
import os
from typing import Optional
import google.generativeai as genai
from app.utils.logger import logger


class GeminiClient:
    """
    Client for interacting with Google Gemini API
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        """
        Initialize Gemini client
        
        Args:
            api_key: Gemini API key (defaults to environment variable)
            model_name: Model to use for generation
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables")
            raise ValueError("GEMINI_API_KEY must be set in environment variables")
        
        self.model_name = model_name
        
        genai.configure(api_key=self.api_key)
        
        self.model = genai.GenerativeModel(self.model_name)
        
        logger.info(f"Gemini client initialized with model: {self.model_name}")
    
    def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """
        Generate text using Gemini API
        
        Args:
            prompt: Input prompt for generation
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated text response
        """
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if not response.text:
                logger.error("Empty response from Gemini API")
                raise ValueError("Empty response from Gemini API")
            
            logger.info(f"Successfully generated text with {len(response.text)} characters")
            return response.text
        
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {str(e)}")
            raise
    
    def is_configured(self) -> bool:
        """
        Check if the client is properly configured
        
        Returns:
            True if API key is set, False otherwise
        """
        return self.api_key is not None
