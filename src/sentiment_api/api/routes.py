import time
from functools import wraps
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..core.config import get_settings
from ..core.exceptions import (
    ModelLoadError,
    PredictionError,
    TokenizationError,
    SentimentAPIError,
)
from ..services.sentiment import SentimentService, get_sentiment_service


router = APIRouter(prefix="/sentiment", tags=["sentiment"])


class SentimentRequest(BaseModel):
    text: str | list[str] = Field(..., description="Text or list of texts to analyze")
    return_scores: bool | None = Field(
        None, description="Include all class scores in response"
    )


class SentimentResponse(BaseModel):
    label: str
    confidence: float
    rule_override: bool = False
    scores: dict[str, float] | None = None


class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool


def with_retry(max_attempts: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """Decorator to retry a function with exponential backoff."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except (ModelLoadError, PredictionError, TokenizationError) as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        raise

            raise last_exception

        return wrapper

    return decorator


@router.post("", response_model=SentimentResponse | list[SentimentResponse])
@with_retry(max_attempts=3, delay=1.0, backoff_factor=2.0)
async def analyze_sentiment(
    request: SentimentRequest,
    service: SentimentService = Depends(get_sentiment_service),
) -> Any:
    """
    Analyze sentiment of text(s).

    Returns sentiment label (negative/neutral/positive) with confidence score.
    """
    try:
        results = service.analyze(request.text, return_scores=request.return_scores)

        if isinstance(request.text, str):
            result = results[0]
            return SentimentResponse(
                label=result.label,
                confidence=result.confidence,
                rule_override=result.rule_override,
                scores=result.scores,
            )

        return [
            SentimentResponse(
                label=r.label,
                confidence=r.confidence,
                rule_override=r.rule_override,
                scores=r.scores,
            )
            for r in results
        ]

    except ModelLoadError as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    except TokenizationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except PredictionError as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    except SentimentAPIError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check(
    service: SentimentService = Depends(get_sentiment_service),
) -> dict:
    """Health check endpoint."""
    model_loaded = service._session is not None
    return {
        "status": "ok" if model_loaded else "loading",
        "model": "onnx",
        "model_loaded": model_loaded,
    }
