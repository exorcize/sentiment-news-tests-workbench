"""OpenTelemetry metrics for sentiment routing.

Silently becomes a no-op when opentelemetry is not installed or OTEL_ENABLED != 1.
"""
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_initialized = False
_route_counter: Any = None
_cache_counter: Any = None
_gemini_latency: Any = None
_gemini_tokens: Any = None


class _Noop:
    def add(self, *args, **kwargs) -> None:
        pass

    def record(self, *args, **kwargs) -> None:
        pass


def _make_noops() -> None:
    global _route_counter, _cache_counter, _gemini_latency, _gemini_tokens
    noop = _Noop()
    _route_counter = noop
    _cache_counter = noop
    _gemini_latency = noop
    _gemini_tokens = noop


def setup_metrics() -> None:
    """Initialize OTEL meters. Safe to call multiple times."""
    global _initialized, _route_counter, _cache_counter, _gemini_latency, _gemini_tokens
    if _initialized:
        return
    _initialized = True

    if os.getenv("OTEL_ENABLED") != "1":
        _make_noops()
        return

    try:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource

        endpoint = os.getenv("OTEL_OTLP_ENDPOINT") or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT"
        )
        if not endpoint:
            _make_noops()
            return

        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )

        exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=15000)
        resource = Resource.create({"service.name": "sentiment-api"})
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)
        meter = metrics.get_meter("sentiment-api")

        _route_counter = meter.create_counter(
            "sentiment.route.total",
            description="Sentiment requests classified by route",
        )
        _cache_counter = meter.create_counter(
            "sentiment.cache.total",
            description="Sentiment cache hits/misses",
        )
        _gemini_latency = meter.create_histogram(
            "sentiment.gemini.latency_ms",
            unit="ms",
            description="Gemini classification latency",
        )
        _gemini_tokens = meter.create_histogram(
            "sentiment.gemini.tokens",
            unit="1",
            description="Gemini tokens per request",
        )
        logger.info("OTEL metrics initialized (endpoint=%s)", endpoint)
    except Exception as e:
        logger.warning("OTEL metrics setup failed, using no-op: %s", e)
        _make_noops()


def record_route(route: str) -> None:
    if _route_counter is None:
        _make_noops()
    _route_counter.add(1, {"route": route})


def record_cache(hit: bool) -> None:
    if _cache_counter is None:
        _make_noops()
    _cache_counter.add(1, {"outcome": "hit" if hit else "miss"})


def record_gemini_latency(ms: float) -> None:
    if _gemini_latency is None:
        _make_noops()
    _gemini_latency.record(ms)


def record_gemini_tokens(direction: str, count: int) -> None:
    if _gemini_tokens is None:
        _make_noops()
    _gemini_tokens.record(count, {"direction": direction})
