# LLM Module - Legal Reasoning Engine

## Overview

This module provides AI-powered legal reasoning capabilities using Google's Gemini API. It generates human-readable explanations for case predictions, helping judges understand the factors behind AI-predicted delays and adjournment risks.

## Components

### 1. `gemini_client.py`
- **Purpose**: Interface with Google Gemini API
- **Model**: `gemini-1.5-flash`
- **Key Method**: `generate_text(prompt, temperature, max_tokens)`

### 2. `prompt_templates.py`
- **Purpose**: Structured prompt generation
- **Templates**:
  - Comprehensive explanation prompts
  - Delay-focused prompts
  - Case detail formatters

### 3. `reasoning_engine.py`
- **Purpose**: Main orchestration engine
- **Features**:
  - Combines prompts with Gemini client
  - Parses structured responses
  - Extracts key factors and recommendations

## Quick Start

### Installation

```bash
# Install dependencies
pip install google-generativeai==0.3.2

# Set environment variable
export GEMINI_API_KEY="your_api_key_here"
```

### Basic Usage

```python
from app.llm.reasoning_engine import reasoning_engine

# Check if available
if reasoning_engine.is_available():
    # Generate explanation
    result = reasoning_engine.generate_explanation(
        case_data={
            "case_age_days": 420,
            "case_type": "civil",
            "adjournment_history": 3,
            "hearings_count": 8,
            "days_since_last_hearing": 40,
            "judge_workload": 120
        },
        prediction={
            "adjournmentRisk": 0.74,
            "delayProbability": 0.61,
            "resolutionEstimate": "4 months",
            "confidence": 0.82
        },
        similar_cases=[]
    )
    
    print(result["explanation"])
    print(result["key_factors"])
    print(result["recommendation"])
```

## API Integration

The module is integrated into FastAPI via the `/explain` endpoint:

```bash
POST /explain
Content-Type: application/json

{
  "case_data": {...},
  "prediction": {...},
  "similar_cases": [...]
}
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |

### Model Configuration

Default settings in `gemini_client.py`:
- Model: `gemini-1.5-flash`
- Temperature: `0.7`
- Max Tokens: `1024`

## Output Format

```python
{
    "explanation": str,          # 2-3 paragraph explanation
    "key_factors": List[str],    # 3-4 key contributing factors
    "historical_patterns": str,  # Patterns from similar cases
    "recommendation": str        # Actionable recommendations
}
```

## Error Handling

- **Missing API Key**: Raises `ValueError` with clear message
- **API Failures**: Logged and re-raised with context
- **Parsing Errors**: Fallback to raw response

## Security

- API keys loaded from environment only
- No hardcoded credentials
- Input validation via Pydantic schemas
- Secure prompt construction

## Performance

- Average response time: 1-4 seconds
- Configurable token limits for cost control
- Async-ready for parallel processing

## Testing

```python
# Check availability
assert reasoning_engine.is_available()

# Test explanation generation
result = reasoning_engine.generate_explanation(
    case_data=test_data,
    prediction=test_prediction,
    similar_cases=[]
)

assert "explanation" in result
assert len(result["key_factors"]) > 0
```

## Troubleshooting

**Issue**: `ValueError: GEMINI_API_KEY must be set`
- **Solution**: Set environment variable in `.env` file

**Issue**: Empty or poor quality responses
- **Solution**: Adjust temperature, improve prompts, add more context

**Issue**: Slow responses
- **Solution**: Reduce max_tokens, implement caching

## Documentation

For detailed documentation, see: `docs/LLM_REASONING_LAYER.md`
