import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from ..core.config import Settings, get_settings
from ..core.exceptions import ModelLoadError, PredictionError, TokenizationError
from ..core.utils import softmax, normalize_label, apply_sentiment_rules


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""

    label: str
    confidence: float
    rule_override: bool = False
    scores: dict[str, float] | None = None


DEFAULT_OVERRIDE_RULES = [
    (["reverse split", "reverse stock split"], "negative"),
    (["stock buyback", "share repurchase"], "positive"),
    (["layoffs", "job cuts", "cutting jobs"], "negative"),
    (["beat estimates", "exceeded expectations", "strong earnings"], "positive"),
]


class SentimentService:
    """Service for sentiment analysis using ONNX model."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._tokenizer = None
        self._session = None
        self._id2label: dict[str, str] = {}
        self._input_names: list[str] = []
        self._override_rules = DEFAULT_OVERRIDE_RULES

    def load_model(self) -> None:
        """Load ONNX model and tokenizer."""
        try:
            model_dir = self.settings.model_onnx_path
            if not model_dir.exists():
                raise ModelLoadError(f"Model directory not found: {model_dir}")

            config_path = model_dir / "config.json"
            if not config_path.exists():
                raise ModelLoadError(f"Config not found: {config_path}")

            with open(config_path, encoding="utf-8") as f:
                model_config = json.load(f)
            self._id2label = model_config.get("id2label", {})

            print(f"Loading tokenizer from: {model_dir}")
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

            print(f"Loading ONNX model from: {model_dir / 'model.onnx'}")
            sess_opts = self._make_session_options()
            self._session = ort.InferenceSession(
                str(model_dir / "model.onnx"),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_names = [inp.name for inp in self._session.get_inputs()]

            print(
                f"ONNX model ready (CPU) | max_length={self.settings.max_length} | "
                f"return_scores={self.settings.return_scores} | inputs={self._input_names}"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}") from e

    def _make_session_options(self) -> ort.SessionOptions:
        """Configure ONNX Runtime session options for optimal CPU performance."""
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 0
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
        return opts

    def analyze(
        self, texts: str | list[str], return_scores: bool | None = None
    ) -> list[SentimentResult]:
        """
        Analyze sentiment for given texts.

        Args:
            texts: Single text or list of texts to analyze
            return_scores: Whether to return all class scores (overrides config)

        Returns:
            List of SentimentResult objects
        """
        if self._session is None or self._tokenizer is None:
            raise PredictionError("Model not loaded. Call load_model() first.")

        single_text = isinstance(texts, str)
        texts = [texts] if single_text else texts

        return_scores = (
            return_scores if return_scores is not None else self.settings.return_scores
        )

        try:
            inputs = self._tokenizer(
                texts,
                return_tensors="np",
                truncation=True,
                padding=True,
                max_length=self.settings.max_length,
            )
        except Exception as e:
            raise TokenizationError(f"Tokenization failed: {e}") from e

        feed = {k: inputs[k].astype(np.int64) for k in self._input_names if k in inputs}

        try:
            logits = self._session.run(None, feed)[0]
            logits = np.asarray(logits, dtype=np.float32)
        except Exception as e:
            raise PredictionError(f"Inference failed: {e}") from e

        probs = softmax(logits, axis=1)
        preds = np.argmax(probs, axis=1).tolist()

        results = []
        for i, pred in enumerate(preds):
            raw_label = self._id2label.get(str(pred), f"LABEL_{pred}")
            label = normalize_label(raw_label)
            confidence = round(float(probs[i, pred]), 4)

            rule_sentiment, rule_override = apply_sentiment_rules(
                texts[i], self._override_rules
            )
            if rule_override:
                label = rule_sentiment
                confidence = 1.0

            result = SentimentResult(
                label=label,
                confidence=confidence,
                rule_override=rule_override,
            )

            if return_scores:
                result.scores = {
                    normalize_label(self._id2label.get(str(j), f"LABEL_{j}")): round(
                        float(probs[i, j]), 4
                    )
                    for j in range(len(self._id2label))
                }

            results.append(result)

        if single_text:
            return [results[0]] if results else []
        return results

    def analyze_with_timing(
        self, texts: str | list[str], log_timing: bool | None = None
    ) -> tuple[list[SentimentResult], float]:
        """Analyze sentiment and return results with timing info in milliseconds."""
        log_timing = log_timing if log_timing is not None else self.settings.log_timings
        t_start = time.perf_counter()

        results = self.analyze(texts)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        if log_timing:
            print(
                f"Processed {len(texts) if isinstance(texts, list) else 1} text(s) in {elapsed_ms:.2f}ms"
            )

        return results, elapsed_ms


_service_instance: SentimentService | None = None


def get_sentiment_service() -> SentimentService:
    """Get singleton sentiment service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = SentimentService()
        _service_instance.load_model()
    return _service_instance
