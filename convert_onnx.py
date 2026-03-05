#!/usr/bin/env python3
"""Export and quantize a HuggingFace model for fastest CPU inference."""
import os
import shutil
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer

MODEL = os.getenv("MODEL", "ProsusAI/finbert")
BASE_DIR = Path(__file__).resolve().parent / "models"
TMP_DIR = BASE_DIR / "_conversion_tmp"
OUTPUT_DIR = BASE_DIR / "sentiment-onnx"


def export_onnx():
    print(f"[1/3] Exporting: {MODEL} → ONNX")
    model = ORTModelForSequenceClassification.from_pretrained(MODEL, export=True)
    model.save_pretrained(TMP_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.save_pretrained(TMP_DIR)


def quantize_int8():
    print("[2/3] Quantizing to INT8 (dynamic)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(TMP_DIR / "model.onnx"),
        model_output=str(OUTPUT_DIR / "model.onnx"),
        weight_type=QuantType.QInt8,
    )


def copy_tokenizer_and_config():
    print("[3/3] Copying tokenizer and config")
    keep = {"config.json", "tokenizer_config.json", "tokenizer.json",
            "special_tokens_map.json", "vocab.json", "merges.txt"}
    for f in TMP_DIR.iterdir():
        if f.name in keep:
            shutil.copy2(f, OUTPUT_DIR / f.name)


def main():
    print(f"Output: {OUTPUT_DIR}\n")

    export_onnx()
    quantize_int8()
    copy_tokenizer_and_config()

    raw_size = (TMP_DIR / "model.onnx").stat().st_size
    final_size = (OUTPUT_DIR / "model.onnx").stat().st_size
    print(f"\nRaw model:       {raw_size / 1024 / 1024:.1f} MB")
    print(f"Quantized model: {final_size / 1024 / 1024:.1f} MB ({final_size / raw_size:.0%})")

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print(f"Done! INT8 quantized model → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
