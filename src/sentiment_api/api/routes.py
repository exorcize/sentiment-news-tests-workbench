import time
from functools import wraps
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.config import get_settings
from ..core.exceptions import (
    ModelLoadError,
    PredictionError,
    SentimentAPIError,
    TokenizationError,
)
from ..services.router import SentimentRouter, get_router
from ..services.sentiment import SentimentService, get_sentiment_service


router = APIRouter(prefix="/sentiment", tags=["sentiment"])


class SentimentRequest(BaseModel):
    text: str | list[str] = Field(..., description="Text or list of texts to analyze")
    return_scores: bool | None = Field(
        None, description="Include all class scores in response"
    )
    symbol: str | None = Field(
        None, description="Target ticker for target-aware routing"
    )
    company_name: str | None = Field(
        None, description="Company name for disambiguation (e.g. 'Skillz Platform Inc.')"
    )


class SentimentResponse(BaseModel):
    label: str
    confidence: float
    rule_override: bool = False
    scores: dict[str, float] | None = None
    route: str | None = None
    reasoning: str | None = None
    finbert_label: str | None = None
    finbert_confidence: float | None = None
    gemini_latency_ms: float | None = None


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


def _get_router(
    service: SentimentService = Depends(get_sentiment_service),
) -> SentimentRouter:
    return get_router(service)


@router.post("", response_model=SentimentResponse | list[SentimentResponse])
@with_retry(max_attempts=3, delay=1.0, backoff_factor=2.0)
async def analyze_sentiment(
    request: SentimentRequest,
    router_svc: SentimentRouter = Depends(_get_router),
) -> Any:
    """Analyze sentiment of text(s), optionally target-aware for a given symbol."""
    try:
        texts = request.text if isinstance(request.text, list) else [request.text]
        results = []
        for t in texts:
            r = await router_svc.classify(
                text=t,
                symbol=request.symbol,
                company_name=request.company_name,
                return_scores=request.return_scores,
            )
            results.append(
                SentimentResponse(
                    label=r.label,
                    confidence=r.confidence,
                    rule_override=r.rule_override,
                    scores=r.scores,
                    route=r.route,
                    reasoning=r.reasoning,
                    finbert_label=r.finbert_label,
                    finbert_confidence=r.finbert_confidence,
                    gemini_latency_ms=r.gemini_latency_ms,
                )
            )

        if isinstance(request.text, str):
            return results[0]
        return results

    except ModelLoadError as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    except TokenizationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except PredictionError as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    except SentimentAPIError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=list[SentimentResponse])
async def analyze_sentiment_batch(
    request: SentimentRequest,
    router_svc: SentimentRouter = Depends(_get_router),
) -> Any:
    """Batch endpoint scaffold (enable via SENTIMENT_BATCH_ENABLED=1)."""
    settings = get_settings()
    if not settings.sentiment_batch_enabled:
        raise HTTPException(status_code=404, detail="batch endpoint disabled")

    texts = request.text if isinstance(request.text, list) else [request.text]
    out: list[SentimentResponse] = []
    for t in texts:
        r = await router_svc.classify(
            text=t,
            symbol=request.symbol,
            company_name=request.company_name,
            return_scores=request.return_scores,
        )
        out.append(
            SentimentResponse(
                label=r.label,
                confidence=r.confidence,
                rule_override=r.rule_override,
                scores=r.scores,
                route=r.route,
                reasoning=r.reasoning,
                finbert_label=r.finbert_label,
                finbert_confidence=r.finbert_confidence,
                gemini_latency_ms=r.gemini_latency_ms,
            )
        )
    return out


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
