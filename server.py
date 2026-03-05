import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

MODEL_ONNX_PATH = os.getenv("MODEL_ONNX_PATH", str(Path(__file__).resolve().parent / "models" / "sentiment-onnx"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "64"))
RETURN_SCORES = os.getenv("RETURN_SCORES", "0") == "1"
LOG_TIMINGS = os.getenv("LOG_TIMINGS", "0") == "1"

SENTIMENT_OVERRIDE_RULES = [
    (["reverse split", "reverse stock split"], "negative"),
]


def apply_sentiment_rules(text: str) -> str | None:
    lower = text.lower()
    for phrases, sentiment_label in SENTIMENT_OVERRIDE_RULES:
        if any(p in lower for p in phrases):
            return sentiment_label
    return None


def _make_session_options() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 0
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
    return opts


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _normalize_label(raw_label: str) -> str:
    # Keep compatibility across different model label styles.
    normalized = raw_label.lower()
    if normalized in {"label_0", "negative"}:
        return "negative"
    if normalized in {"label_1", "neutral"}:
        return "neutral"
    if normalized in {"label_2", "positive"}:
        return "positive"
    return raw_label


model_dir = Path(MODEL_ONNX_PATH)

with open(model_dir / "config.json", encoding="utf-8") as f:
    model_config = json.load(f)
id2label = model_config.get("id2label", {})

print(f"Loading ONNX model from: {MODEL_ONNX_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ONNX_PATH)
session = ort.InferenceSession(
    str(model_dir / "model.onnx"),
    sess_options=_make_session_options(),
    providers=["CPUExecutionProvider"],
)
input_names = [inp.name for inp in session.get_inputs()]
print(f"ONNX model ready (CPU) | max_length={MAX_LENGTH} | return_scores={RETURN_SCORES} | inputs={input_names}")

app = FastAPI()


class TextRequest(BaseModel):
    text: str | list[str]


@app.post("/sentiment")
def sentiment(req: TextRequest):
    t_start = time.perf_counter()
    texts = req.text if isinstance(req.text, list) else [req.text]

    inputs = tokenizer(
        texts,
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    feed = {k: inputs[k].astype(np.int64) for k in input_names if k in inputs}

    logits = session.run(None, feed)[0]
    logits = np.asarray(logits, dtype=np.float32)

    probs = _softmax(logits, axis=1)
    preds = np.argmax(probs, axis=1).tolist()

    results = []
    for i, pred in enumerate(preds):
        raw_label = id2label.get(str(pred), f"LABEL_{pred}")
        label = _normalize_label(raw_label)
        confidence = round(float(probs[i, pred]), 4)

        rule_sentiment = apply_sentiment_rules(texts[i])
        rule_override = rule_sentiment is not None
        if rule_override:
            label = rule_sentiment
            confidence = 1.0

        result = {
            "label": label,
            "confidence": confidence,
            "rule_override": rule_override,
        }
        if RETURN_SCORES:
            result["scores"] = {
                _normalize_label(id2label.get(str(j), f"LABEL_{j}")): round(float(probs[i, j]), 4)
                for j in range(len(id2label))
            }

        results.append(result)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    if LOG_TIMINGS:
        print(f"Processed {len(texts)} text(s) in {elapsed_ms:.2f}ms")

    return results[0] if isinstance(req.text, str) else results


@app.get("/health")
def health():
    return {"status": "ok", "model": "onnx", "path": MODEL_ONNX_PATH}
