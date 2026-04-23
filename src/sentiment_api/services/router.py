"""Orchestrates target-aware sentiment classification.

Pipeline when symbol is provided:
    cache → detector → finbert | gemini → low-confidence escalation → cache write

When symbol is absent: legacy FinBERT path (backward compat).
"""
import logging
import time
from dataclasses import asdict, dataclass

from ..core.config import Settings, get_settings
from ..core.metrics import (
    record_cache,
    record_gemini_latency,
    record_gemini_tokens,
    record_route,
)
from .cache import SentimentCache, get_cache
from .detector import is_ambiguous
from .gemini import GeminiResult, GeminiService, get_gemini_service
from .normalize import cache_key
from .sentiment import SentimentResult, SentimentService

logger = logging.getLogger(__name__)


@dataclass
class RouterResult:
    label: str
    confidence: float
    route: str  # cache | finbert | gemini | gemini_escalated | rule
    rule_override: bool = False
    reasoning: str | None = None
    finbert_label: str | None = None
    finbert_confidence: float | None = None
    gemini_latency_ms: float | None = None
    gemini_input_tokens: int | None = None
    gemini_output_tokens: int | None = None
    scores: dict[str, float] | None = None


class SentimentRouter:
    def __init__(
        self,
        sentiment: SentimentService,
        gemini: GeminiService | None = None,
        cache: SentimentCache | None = None,
        settings: Settings | None = None,
    ):
        self.sentiment = sentiment
        self.gemini = gemini or get_gemini_service()
        self.cache = cache or get_cache()
        self.settings = settings or get_settings()

    async def classify(
        self,
        text: str,
        symbol: str | None,
        company_name: str | None,
        return_scores: bool | None = None,
    ) -> RouterResult:
        # Legacy path: no symbol means we can't do target-aware routing.
        if not symbol:
            fb = self._finbert(text, return_scores)
            record_route("finbert")
            return RouterResult(
                label=fb.label,
                confidence=fb.confidence,
                route="rule" if fb.rule_override else "finbert",
                rule_override=fb.rule_override,
                finbert_label=fb.label,
                finbert_confidence=fb.confidence,
                scores=fb.scores,
            )

        key = cache_key(text, symbol)
        cached = await self.cache.get(key)
        if cached is not None:
            record_cache(True)
            record_route("cache")
            cached["route"] = "cache"
            return RouterResult(**_filter_kwargs(cached))
        record_cache(False)

        # Run FinBERT first — cheap baseline, gives rule_override shortcut and
        # lets us escalate on low confidence.
        fb = self._finbert(text, return_scores)

        # Rule overrides (reverse split, buyback, etc.) are high-signal — trust them.
        if fb.rule_override:
            result = RouterResult(
                label=fb.label,
                confidence=fb.confidence,
                route="rule",
                rule_override=True,
                finbert_label=fb.label,
                finbert_confidence=fb.confidence,
                scores=fb.scores,
            )
            record_route("rule")
            await self.cache.set(key, asdict(result))
            return result

        ambiguous = is_ambiguous(text, symbol)
        low_conf = fb.confidence < self.settings.sentiment_low_confidence_threshold
        use_gemini = self.gemini.available and (ambiguous or low_conf)
        escalated = use_gemini and not ambiguous and low_conf

        if use_gemini:
            try:
                gem = await self.gemini.classify(text, symbol, company_name)
                record_gemini_latency(gem.latency_ms)
                record_gemini_tokens("input", gem.input_tokens)
                record_gemini_tokens("output", gem.output_tokens)
                route = "gemini_escalated" if escalated else "gemini"
                result = RouterResult(
                    label=gem.label,
                    confidence=gem.confidence,
                    route=route,
                    rule_override=False,
                    reasoning=gem.reasoning,
                    finbert_label=fb.label,
                    finbert_confidence=fb.confidence,
                    gemini_latency_ms=gem.latency_ms,
                    gemini_input_tokens=gem.input_tokens,
                    gemini_output_tokens=gem.output_tokens,
                    scores=fb.scores,
                )
                record_route(route)
                await self.cache.set(key, asdict(result))
                return result
            except Exception as e:
                logger.warning("gemini classify failed, falling back to finbert: %s", e)
                record_route("gemini_error")

        result = RouterResult(
            label=fb.label,
            confidence=fb.confidence,
            route="finbert",
            rule_override=False,
            finbert_label=fb.label,
            finbert_confidence=fb.confidence,
            scores=fb.scores,
        )
        record_route("finbert")
        await self.cache.set(key, asdict(result))
        return result

    def _finbert(
        self, text: str, return_scores: bool | None
    ) -> SentimentResult:
        results = self.sentiment.analyze(text, return_scores=return_scores)
        return results[0]


def _filter_kwargs(data: dict) -> dict:
    allowed = {
        "label", "confidence", "route", "rule_override", "reasoning",
        "finbert_label", "finbert_confidence", "gemini_latency_ms",
        "gemini_input_tokens", "gemini_output_tokens", "scores",
    }
    return {k: v for k, v in data.items() if k in allowed}


_router_instance: SentimentRouter | None = None


def get_router(sentiment: SentimentService) -> SentimentRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = SentimentRouter(sentiment=sentiment)
    return _router_instance
