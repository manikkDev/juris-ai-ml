# Legal Reasoning LLM Layer

## Overview

The Legal Reasoning LLM Layer extends the Juris AI ML service with natural language explanation capabilities using Google's Gemini API. This component generates human-readable explanations for AI predictions, helping judges understand why the model predicts case delays or adjournment risks.

## Architecture

### Directory Structure

```
app/
└── llm/
    ├── __init__.py
    ├── gemini_client.py       # Gemini API client
    ├── reasoning_engine.py    # Main reasoning engine
    └── prompt_templates.py    # Structured prompt templates
```

### Components

#### 1. Gemini Client (`gemini_client.py`)

**Purpose**: Interface with Google Gemini API for text generation.

**Key Features**:
- Initializes Gemini API with `gemini-1.5-flash` model
- Secure API key management from environment variables
- Configurable temperature and token limits
- Error handling and logging

**Main Function**:
```python
generate_text(prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str
```

#### 2. Prompt Templates (`prompt_templates.py`)

**Purpose**: Create structured, consistent prompts for legal explanations.

**Templates Available**:
- `generate_explanation_prompt()`: Comprehensive explanation with all sections
- `generate_delay_explanation_prompt()`: Focused on delay analysis
- Helper formatters for case details, predictions, and similar cases

**Prompt Structure**:
- Case information (age, type, history, workload)
- AI prediction results (risk scores, factors, confidence)
- Similar historical cases
- Structured output format (explanation, key factors, patterns, recommendations)

#### 3. Reasoning Engine (`reasoning_engine.py`)

**Purpose**: Orchestrate explanation generation and parse responses.

**Key Features**:
- Integrates Gemini client with prompt templates
- Parses structured responses from LLM
- Extracts key factors, patterns, and recommendations
- Fallback handling for parsing errors

**Main Functions**:
```python
generate_explanation(case_data, prediction, similar_cases) -> Dict
generate_delay_explanation(case_data, prediction, similar_cases) -> Dict
```

## API Endpoint

### POST `/explain`

Generate human-readable explanations for AI predictions.

**Request Body**:
```json
{
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
```

**Response**:
```json
{
  "explanation": "This civil case shows a high risk of delay based on several critical factors...",
  "key_factors": [
    "High adjournment history (3 previous adjournments)",
    "Extended case age (420 days)",
    "Heavy judge workload (120 active cases)"
  ],
  "historical_patterns": "Similar civil cases with 3+ adjournments typically experience 60-70% longer resolution times...",
  "recommendation": "Consider prioritizing this case for the next available hearing slot..."
}
```

**Status Codes**:
- `200`: Success
- `400`: Invalid request data
- `503`: Gemini API not configured
- `500`: Internal server error

## Configuration

### Environment Variables

Add to `.env` file:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

**Security Notes**:
- Never commit API keys to version control
- Use environment variables for all sensitive credentials
- Rotate API keys regularly
- Monitor API usage and set quotas

### Dependencies

Add to `requirements.txt`:

```
google-generativeai==0.3.2
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Integration Workflow

The complete prediction workflow with explanations:

```
1. Receive case data from frontend
2. Generate prediction using ML model (/predict endpoint)
3. Retrieve similar cases using semantic search
4. Generate explanation using LLM (/explain endpoint)
5. Return combined results to dashboard
```

### Example Integration

```python
# Step 1: Get prediction
prediction_response = await predict_case(case_input)

# Step 2: Get similar cases (from semantic search)
similar_cases = await search_similar_cases(case_input)

# Step 3: Generate explanation
explanation_request = {
    "case_data": case_input.dict(),
    "prediction": prediction_response.dict(),
    "similar_cases": similar_cases
}
explanation_response = await explain_prediction(explanation_request)

# Step 4: Combine and return
return {
    "prediction": prediction_response,
    "explanation": explanation_response
}
```

## Usage Examples

### Basic Usage

```python
from app.llm.reasoning_engine import reasoning_engine

# Generate explanation
result = reasoning_engine.generate_explanation(
    case_data={
        "case_age_days": 420,
        "case_type": "civil",
        "adjournment_history": 3,
        # ... other fields
    },
    prediction={
        "adjournmentRisk": 0.74,
        "delayProbability": 0.61,
        # ... other fields
    },
    similar_cases=[
        # ... similar case data
    ]
)

print(result["explanation"])
print(result["key_factors"])
print(result["recommendation"])
```

### Custom Prompts

```python
from app.llm.gemini_client import GeminiClient
from app.llm.prompt_templates import PromptTemplates

client = GeminiClient()
templates = PromptTemplates()

# Create custom prompt
prompt = templates.generate_delay_explanation_prompt(
    case_data=case_data,
    prediction=prediction,
    similar_cases=similar_cases
)

# Generate response
response = client.generate_text(prompt, temperature=0.5)
```

## Error Handling

### Common Errors

1. **Missing API Key**
   - Error: `ValueError: GEMINI_API_KEY must be set in environment variables`
   - Solution: Set `GEMINI_API_KEY` in `.env` file

2. **API Rate Limits**
   - Error: API quota exceeded
   - Solution: Implement request throttling or upgrade API plan

3. **Empty Responses**
   - Error: `ValueError: Empty response from Gemini API`
   - Solution: Check prompt format and API status

### Logging

All operations are logged with appropriate levels:
- `INFO`: Successful operations
- `WARNING`: Configuration issues
- `ERROR`: API failures and exceptions

Check logs at: `logs/ml_service.log`

## Performance Considerations

### Response Times
- Gemini API call: ~1-3 seconds
- Prompt generation: <100ms
- Response parsing: <50ms
- **Total**: ~1-4 seconds per explanation

### Optimization Tips
1. Cache explanations for identical inputs
2. Use async/await for parallel requests
3. Implement request batching for multiple cases
4. Set appropriate token limits to reduce latency

### Cost Management
- Monitor API usage through Google Cloud Console
- Set daily/monthly quotas
- Use temperature and token limits to control costs
- Consider caching frequently requested explanations

## Testing

### Unit Tests

```python
import pytest
from app.llm.reasoning_engine import ReasoningEngine

def test_explanation_generation():
    engine = ReasoningEngine()
    
    result = engine.generate_explanation(
        case_data=test_case_data,
        prediction=test_prediction,
        similar_cases=[]
    )
    
    assert "explanation" in result
    assert len(result["key_factors"]) > 0
    assert result["recommendation"] != ""
```

### Integration Tests

Test the `/explain` endpoint:

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d @test_data/explanation_request.json
```

## Security Best Practices

1. **API Key Management**
   - Store in environment variables only
   - Use secrets management service in production
   - Never log API keys

2. **Input Validation**
   - Validate all input data with Pydantic schemas
   - Sanitize user inputs before including in prompts
   - Limit input sizes to prevent abuse

3. **Rate Limiting**
   - Implement per-user rate limits
   - Monitor for unusual usage patterns
   - Set up alerts for quota thresholds

4. **Data Privacy**
   - Don't include sensitive case details in prompts
   - Anonymize data where possible
   - Comply with data retention policies

## Troubleshooting

### Issue: Reasoning engine not available

**Symptoms**: 503 error when calling `/explain`

**Solutions**:
1. Check `GEMINI_API_KEY` is set in environment
2. Verify API key is valid
3. Check internet connectivity
4. Review logs for initialization errors

### Issue: Poor quality explanations

**Solutions**:
1. Adjust temperature (lower = more focused, higher = more creative)
2. Improve prompt templates with more context
3. Include more similar cases for better patterns
4. Increase max_tokens for longer explanations

### Issue: Slow response times

**Solutions**:
1. Reduce max_tokens limit
2. Implement caching layer
3. Use async processing
4. Consider using faster Gemini model variants

## Future Enhancements

1. **Multi-language Support**: Generate explanations in multiple languages
2. **Custom Reasoning Styles**: Different explanation styles for different user roles
3. **Explanation Confidence Scores**: Add confidence metrics to explanations
4. **Interactive Explanations**: Allow follow-up questions about predictions
5. **Batch Processing**: Generate explanations for multiple cases simultaneously
6. **Fine-tuning**: Train custom models on legal domain data

## References

- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## Support

For issues or questions:
1. Check logs at `logs/ml_service.log`
2. Review this documentation
3. Contact the development team
