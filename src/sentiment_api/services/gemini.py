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


@dataclass
class GeminiResult:
    label: str
    confidence: float
    reasoning: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0


class GeminiService:
    """Target-aware sentiment classifier via Gemini with structured output."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client: genai.Client | None = None

    def _ensure_client(self) -> genai.Client:
        if self._client is None:
            if not self.settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY not configured")
            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    @property
    def available(self) -> bool:
        return bool(self.settings.gemini_api_key)

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

        loop = asyncio.get_running_loop()
        t0 = loop.time()
        resp = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=self.settings.gemini_model,
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
