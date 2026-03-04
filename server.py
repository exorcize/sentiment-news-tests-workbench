import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

MODEL_ONNX_PATH = os.getenv("MODEL_ONNX_PATH", str(Path(__file__).resolve().parent / "models" / "sentiment-onnx"))

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}

SENTIMENT_OVERRIDE_RULES = [
    (["reverse split", "reverse stock split"], "negative"),
]


def apply_sentiment_rules(text: str) -> str | None:
    lower = text.lower()
    for phrases, sentiment in SENTIMENT_OVERRIDE_RULES:
        if any(p in lower for p in phrases):
            return sentiment
    return None


def _make_cpu_session_options() -> ort.SessionOptions:
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


print(f"Loading ONNX model from: {MODEL_ONNX_PATH} ...")
session_options = _make_cpu_session_options()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ONNX_PATH)
model = ORTModelForSequenceClassification.from_pretrained(
    MODEL_ONNX_PATH,
    providers=["CPUExecutionProvider"],
    session_options=session_options,
)
print("ONNX model ready (CPU).")

app = FastAPI()


class TextRequest(BaseModel):
    text: str | list[str]


@app.post("/sentiment")
def sentiment(req: TextRequest):
    t_start = time.perf_counter()
    texts = req.text if isinstance(req.text, list) else [req.text]

    print(f"[{MODEL_ONNX_PATH}] Received {len(texts)} text(s):")
    for i, t in enumerate(texts):
        preview = t[:100] + "..." if len(t) > 100 else t
        print(f"  [{i}] {preview}")

    inputs = tokenizer(
        texts,
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=512,
    )
    inputs = {k: v.astype(np.int64) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits
    if hasattr(logits, "numpy"):
        logits = logits.numpy()
    logits = np.asarray(logits, dtype=np.float32)

    probs = _softmax(logits, axis=1)
    preds = np.argmax(probs, axis=1).tolist()
    id2label = model.config.id2label

    def get_label(idx: int) -> str:
        return id2label.get(idx, id2label.get(str(idx), f"LABEL_{idx}"))

    results = []
    for i, pred in enumerate(preds):
        raw_label = get_label(pred)
        label = LABEL_MAP.get(raw_label, raw_label)
        confidence = round(float(probs[i, pred]), 4)
        scores = {
            LABEL_MAP.get(get_label(j), get_label(j)): round(float(probs[i, j]), 4)
            for j in range(len(id2label))
        }

        rule_sentiment = apply_sentiment_rules(texts[i])
        rule_override = rule_sentiment is not None
        if rule_override:
            label = rule_sentiment
            confidence = 1.0

        results.append({
            "label": label,
            "confidence": confidence,
            "scores": scores,
            "rule_override": rule_override,
        })

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    print(f"[{MODEL_ONNX_PATH}] Processed in {elapsed_ms:.2f}ms")

    return results[0] if isinstance(req.text, str) else results


@app.get("/health")
def health():
    return {"status": "ok", "model": "onnx", "path": MODEL_ONNX_PATH}
