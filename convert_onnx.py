#!/usr/bin/env python3
"""Export and quantize a HuggingFace model for fastest CPU inference."""
import os
import shutil
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification
from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic
from transformers import AutoTokenizer

MODEL = os.getenv("MODEL", "ProsusAI/finbert")
BASE_DIR = Path(__file__).resolve().parent / "models"
TMP_DIR = BASE_DIR / "_conversion_tmp"
OUTPUT_DIR = BASE_DIR / "sentiment-onnx"
QUANT_REDUCE_RANGE = os.getenv("QUANT_REDUCE_RANGE", "1") == "1"


def export_onnx():
    print(f"[1/3] Exporting: {MODEL} → ONNX")
    model = ORTModelForSequenceClassification.from_pretrained(MODEL, export=True)
    model.save_pretrained(TMP_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.save_pretrained(TMP_DIR)


def quantize_int8():
    print("[2/3] Quantizing to INT8 (dynamic)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_model = TMP_DIR / "model.onnx"
    preprocessed_model = TMP_DIR / "model.preprocessed.onnx"

    # Best-effort preprocess before quantization (recommended by ORT).
    # Some transformer exports can fail symbolic shape inference, so we fallback safely.
    try:
        quant_pre_process(
            input_model=str(input_model),
            output_model_path=str(preprocessed_model),
            skip_optimization=True,
        )
        model_for_quant = preprocessed_model
        print("Preprocess: OK")
    except Exception as first_error:  # pragma: no cover - environment/model dependent
        try:
            quant_pre_process(
                input_model=str(input_model),
                output_model_path=str(preprocessed_model),
                skip_optimization=True,
                skip_symbolic_shape=True,
            )
            model_for_quant = preprocessed_model
            print("Preprocess: OK (without symbolic shape inference)")
        except Exception as second_error:  # pragma: no cover - environment/model dependent
            model_for_quant = input_model
            print(
                "Preprocess skipped (fallback to direct quantization). "
                f"Errors: {first_error} | {second_error}"
            )

    quantize_dynamic(
        model_input=str(model_for_quant),
        model_output=str(OUTPUT_DIR / "model.onnx"),
        op_types_to_quantize=["MatMul", "Gemm"],
        reduce_range=QUANT_REDUCE_RANGE,
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
    print(f"Model: {MODEL}")
    print(f"Quant reduce range: {QUANT_REDUCE_RANGE}\n")

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
