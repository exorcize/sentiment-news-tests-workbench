#!/usr/bin/env python3
import os
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

MODEL = os.getenv("MODEL", "cardiffnlp/twitter-roberta-base-sentiment")
OUTPUT_DIR = Path(__file__).resolve().parent / "models" / "sentiment-onnx"

print(f"Converting model: {MODEL}")
print(f"Output directory: {OUTPUT_DIR}")

model = ORTModelForSequenceClassification.from_pretrained(MODEL, export=True)
model.save_pretrained(OUTPUT_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Done! Model saved to: {OUTPUT_DIR}")
