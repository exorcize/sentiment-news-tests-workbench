import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from ..core.config import Settings, get_settings

logger = logging.getLogger(__name__)


_PROMPT_TEMPLATE = """You classify financial news sentiment for a specific target company.

Target ticker: {symbol}
Target company: {company}
Headline: {headline}

Is this headline bullish, bearish, or neutral for {symbol}? Consider who benefits
(plaintiff vs defendant, acquirer vs target, winner vs loser, beneficiary vs harmed
party) — not overall tone. Respond only with JSON.
"""

_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "label": {"type": "STRING", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "NUMBER"},
        "reasoning": {"type": "STRING"},
    },
    "required": ["label", "confidence", "reasoning"],
}

_OVERLOAD_HINTS = ("503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "429", "overloaded")


def _is_overload_error(exc: BaseException) -> bool:
    """Server-side overload (503/429). Try the next fallback model."""
    msg = str(exc)
    return any(hint in msg for hint in _OVERLOAD_HINTS)


def _is_unhealthy(exc: BaseException) -> bool:
    """Anything that should count toward the circuit breaker (overload OR timeout)."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    return _is_overload_error(exc)


@dataclass
class GeminiResult:
    label: str
    confidence: float
    reasoning: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class GeminiOverloadError(RuntimeError):
    """Raised when every model in the fallback chain returned overload."""


class GeminiService:
    """Target-aware sentiment classifier via Gemini with structured output."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client: genai.Client | None = None
        self._models: list[str] = self._build_model_chain()

    def _build_model_chain(self) -> list[str]:
        primary = (self.settings.gemini_model or "").strip()
        raw = self.settings.gemini_fallback_models or ""
        fallbacks = [m.strip() for m in raw.split(",") if m.strip()]
        chain: list[str] = []
        if primary:
            chain.append(primary)
        for m in fallbacks:
            if m not in chain:
                chain.append(m)
        return chain

    def _ensure_client(self) -> genai.Client:
        if self._client is None:
            if not self.settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY not configured")
            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    @property
    def available(self) -> bool:
        return bool(self.settings.gemini_api_key) and bool(self._models)

    async def classify(
        self, text: str, symbol: str, company_name: str | None = None
    ) -> GeminiResult:
        client = self._ensure_client()
        company = (company_name or symbol).strip()
        prompt = _PROMPT_TEMPLATE.format(
            symbol=symbol.upper(), company=company, headline=text
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
            temperature=0.0,
        )

        last_exc: BaseException | None = None
        for idx, model in enumerate(self._models):
            try:
                return await self._invoke(client, model, prompt, config)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_overload_error(exc) and idx < len(self._models) - 1:
                    logger.warning(
                        "gemini model %s overloaded (%s), trying fallback", model, exc
                    )
                    continue
                raise
        # Fallback chain exhausted on overload — raise dedicated error so the
        # router can mark the circuit breaker.
        raise GeminiOverloadError(str(last_exc) if last_exc else "all models overloaded")

    async def _invoke(
        self,
        client: genai.Client,
        model: str,
        prompt: str,
        config: types.GenerateContentConfig,
    ) -> GeminiResult:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        resp = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            ),
            timeout=self.settings.gemini_timeout_seconds,
        )
        elapsed_ms = (loop.time() - t0) * 1000

        parsed = _parse_response(resp.text)
        usage = getattr(resp, "usage_metadata", None)
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)

        return GeminiResult(
            label=parsed["label"],
            confidence=float(parsed.get("confidence", 0.0)),
            reasoning=str(parsed.get("reasoning", "")),
            latency_ms=elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )


def _parse_response(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    label = data.get("label", "neutral").lower()
    if label not in {"positive", "negative", "neutral"}:
        label = "neutral"
    data["label"] = label
    return data


_gemini_instance: GeminiService | None = None


def get_gemini_service() -> GeminiService:
    global _gemini_instance
    if _gemini_instance is None:
        _gemini_instance = GeminiService()
    return _gemini_instance
