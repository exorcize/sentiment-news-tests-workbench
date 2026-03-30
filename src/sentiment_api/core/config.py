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


@lru_cache
def get_settings() -> Settings:
    return Settings()
