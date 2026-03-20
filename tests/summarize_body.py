#!/usr/bin/env python3
"""Envia o texto de body.txt para a API de summarizer e imprime o resumo."""
import sys
from pathlib import Path

import requests

API_URL = "http://localhost:8003/summarize"
BODY_FILE = Path(__file__).resolve().parent / "body.txt"


def main() -> None:
    if not BODY_FILE.exists():
        print(f"Arquivo não encontrado: {BODY_FILE}", file=sys.stderr)
        sys.exit(1)

    text = BODY_FILE.read_text(encoding="utf-8").strip()
    if not text:
        print("body.txt está vazio.", file=sys.stderr)
        sys.exit(1)

    try:
        r = requests.post(API_URL, json={"text": text}, timeout=60)
        r.raise_for_status()
        data = r.json()
        print(data["summary"])
    except requests.exceptions.ConnectionError:
        print("Erro: não foi possível conectar à API. Suba o servidor com:", file=sys.stderr)
        print("  python -m uvicorn tests.summarizer_server:app --host 0.0.0.0 --port 8003", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"Erro HTTP: {e}", file=sys.stderr)
        if r.text:
            print(r.text, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
