import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    app_name: str = "sentiment-api"
    app_version: str = "1.0.0"
    debug: bool = False

    model_onnx_path: Path = (
        Path(__file__).resolve().parent.parent.parent / "models" / "sentiment-onnx"
    )
    max_length: int = 64
    return_scores: bool = False
    log_timings: bool = False

    model_type: Literal["onnx", "pytorch"] = "onnx"

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    # ONNX Runtime CPU tuning (0 intra = ORT default, usually all logical cores)
    ort_intra_op_num_threads: int = 0
    ort_inter_op_num_threads: int = 1
    # If set, applies OMP/MKL/OPENBLAS thread caps before InferenceSession (setdefault)
    omp_num_threads: int | None = None

    retry_max_attempts: int = 3
    retry_delay: float = 1.0
    retry_backoff_factor: float = 2.0

    # Redis cache for target-aware sentiment results
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    sentiment_cache_ttl_seconds: int = 7 * 24 * 3600  # 7 days

    # Target-aware routing
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_timeout_seconds: float = 5.0
    sentiment_low_confidence_threshold: float = 0.6
    sentiment_batch_enabled: bool = False

    def model_post_init(self, __context) -> None:
        if val := os.getenv("MODEL_ONNX_PATH"):
            self.model_onnx_path = Path(val)
        if val := os.getenv("MAX_LENGTH"):
            self.max_length = int(val)
        if val := os.getenv("RETURN_SCORES"):
            self.return_scores = val == "1"
        if val := os.getenv("LOG_TIMINGS"):
            self.log_timings = val == "1"
        if val := os.getenv("WORKERS"):
            self.workers = max(1, int(val))
        if val := os.getenv("ORT_INTRA_OP_NUM_THREADS"):
            self.ort_intra_op_num_threads = max(0, int(val))
        if val := os.getenv("ORT_INTER_OP_NUM_THREADS"):
            self.ort_inter_op_num_threads = max(1, int(val))
        if (val := os.getenv("OMP_NUM_THREADS")) is not None and val.strip() != "":
            self.omp_num_threads = int(val)
        elif os.getenv("OMP_NUM_THREADS") == "":
            self.omp_num_threads = None

        if val := os.getenv("REDIS_HOST"):
            self.redis_host = val
        if val := os.getenv("REDIS_PORT"):
            self.redis_port = int(val)
        if val := os.getenv("REDIS_DB"):
            self.redis_db = int(val)
        if val := os.getenv("REDIS_PASSWORD"):
            self.redis_password = val
        if val := os.getenv("SENTIMENT_CACHE_TTL_SECONDS"):
            self.sentiment_cache_ttl_seconds = int(val)
        if val := os.getenv("GEMINI_API_KEY"):
            self.gemini_api_key = val
        if val := os.getenv("GEMINI_MODEL"):
            self.gemini_model = val
        if val := os.getenv("GEMINI_TIMEOUT_SECONDS"):
            self.gemini_timeout_seconds = float(val)
        if val := os.getenv("SENTIMENT_LOW_CONFIDENCE_THRESHOLD"):
            self.sentiment_low_confidence_threshold = float(val)
        if val := os.getenv("SENTIMENT_BATCH_ENABLED"):
            self.sentiment_batch_enabled = val == "1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
