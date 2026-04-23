import json
import logging
from typing import Any

import redis.asyncio as redis

from ..core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class SentimentCache:
    """Async Redis cache for target-aware sentiment results."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client: redis.Redis | None = None

    def _connect(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                db=self.settings.redis_db,
                password=self.settings.redis_password,
                decode_responses=True,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
            )
        return self._client

    async def get(self, key: str) -> dict[str, Any] | None:
        try:
            client = self._connect()
            raw = await client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.warning("cache get failed: %s", e)
            return None

    async def set(self, key: str, value: dict[str, Any]) -> None:
        try:
            client = self._connect()
            await client.set(
                key, json.dumps(value), ex=self.settings.sentiment_cache_ttl_seconds
            )
        except Exception as e:
            logger.warning("cache set failed: %s", e)

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None


_cache_instance: SentimentCache | None = None


def get_cache() -> SentimentCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SentimentCache()
    return _cache_instance
