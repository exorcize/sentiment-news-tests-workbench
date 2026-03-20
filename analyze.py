#!/usr/bin/env python3
import argparse
import sys
import time

import requests


def analyze_file(filepath: str, api_url: str, batch_size: int = 1):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("No lines found in file.")
        return

    print(f"Analyzing {len(lines)} headlines from {filepath}\n")

    request_times_ms = []

    if batch_size > 1:
        for i in range(0, len(lines), batch_size):
            batch = lines[i : i + batch_size]
            start = time.perf_counter()
            resp = requests.post(
                f"{api_url}/sentiment", json={"text": batch}, timeout=30
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            request_times_ms.append(elapsed_ms)
            resp.raise_for_status()
            results = resp.json()
            for j, (text, res) in enumerate(zip(batch, results)):
                idx = i + j + 1
                print(
                    f"[{idx}] {res['label'].upper():8} ({res['confidence']:.2%}) | {text}"
                )
    else:
        for i, text in enumerate(lines, 1):
            start = time.perf_counter()
            resp = requests.post(
                f"{api_url}/sentiment", json={"text": text}, timeout=30
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            request_times_ms.append(elapsed_ms)
            resp.raise_for_status()
            res = resp.json()
            print(f"[{i}] {res['label'].upper():8} ({res['confidence']:.2%}) | {text}")

    total_requests = len(request_times_ms)
    total_time_ms = sum(request_times_ms)
    avg_ms = total_time_ms / total_requests if total_requests > 0 else 0

    print(f"\nDone. Analyzed {len(lines)} headlines.")
    print(
        f"Requests: {total_requests} | Total: {total_time_ms:.1f}ms | Avg: {avg_ms:.1f}ms/request"
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze news headlines sentiment")
    parser.add_argument("file", help="Path to .txt file with one headline per line")
    parser.add_argument("--url", default="http://localhost:8002", help="API base URL")
    parser.add_argument(
        "--batch", type=int, default=1, help="Batch size (send N at a time)"
    )
    args = parser.parse_args()

    try:
        analyze_file(args.file, args.url, args.batch)
    except FileNotFoundError:
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
    except requests.RequestException as e:
        print(f"Error: API request failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
