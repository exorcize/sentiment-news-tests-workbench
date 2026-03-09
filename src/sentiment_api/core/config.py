import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseModel as BaseSettings


class Settings(BaseSettings):
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

    retry_max_attempts: int = 3
    retry_delay: float = 1.0
    retry_backoff_factor: float = 2.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def model_post_init(self, __context) -> None:
        if val := os.getenv("MODEL_ONNX_PATH"):
            self.model_onnx_path = Path(val)
        if val := os.getenv("MAX_LENGTH"):
            self.max_length = int(val)
        if val := os.getenv("RETURN_SCORES"):
            self.return_scores = val == "1"
        if val := os.getenv("LOG_TIMINGS"):
            self.log_timings = val == "1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
