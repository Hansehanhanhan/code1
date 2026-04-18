from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class Sample:
    status_code: int
    latency_ms: float
    ttfb_ms: float
    bytes_read: int


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def parse_headers(raw_headers: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw in raw_headers:
        key, sep, value = raw.partition(":")
        if not sep:
            raise ValueError(f"Invalid header format: {raw}. Use 'Key: Value'.")
        headers[key.strip()] = value.strip()
    return headers


def do_request(url: str, method: str, body: dict | None, timeout: float, headers: dict[str, str]) -> Sample:
    data = None
    request_headers = dict(headers)
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    request = Request(url=url, data=data, method=method, headers=request_headers)
    start = time.perf_counter()
    first_byte_at: float | None = None
    bytes_read = 0
    try:
        with urlopen(request, timeout=timeout) as response:
            code = int(response.getcode() or 0)
            while True:
                chunk = response.read(4096)
                if not chunk:
                    break
                if first_byte_at is None:
                    first_byte_at = time.perf_counter()
                bytes_read += len(chunk)
    except HTTPError as exc:
        code = exc.code
        body_bytes = exc.read()
        bytes_read = len(body_bytes or b"")
        first_byte_at = time.perf_counter()
    except URLError:
        code = 0
    end = time.perf_counter()
    latency_ms = (end - start) * 1000
    ttfb_ms = ((first_byte_at - start) * 1000) if first_byte_at is not None else latency_ms
    return Sample(status_code=code, latency_ms=latency_ms, ttfb_ms=ttfb_ms, bytes_read=bytes_read)


def run_load_test(
    url: str,
    method: str,
    body: dict | None,
    total_requests: int,
    concurrency: int,
    timeout: float,
    headers: dict[str, str],
) -> tuple[list[Sample], float]:
    samples: list[Sample] = []
    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(do_request, url, method, body, timeout, headers) for _ in range(total_requests)]
        for future in as_completed(futures):
            samples.append(future.result())
    elapsed_s = max(1e-9, time.perf_counter() - started_at)
    return samples, elapsed_s


def summarize(samples: Iterable[Sample], duration_s: float) -> dict[str, float | int | dict[str, int]]:
    rows = list(samples)
    latencies = [row.latency_ms for row in rows]
    ttfb_values = [row.ttfb_ms for row in rows]
    success = [row for row in rows if 200 <= row.status_code < 300]
    status_counts = Counter(row.status_code for row in rows)
    bytes_total = sum(row.bytes_read for row in rows)

    return {
        "total_requests": len(rows),
        "success_requests": len(success),
        "success_rate": round((len(success) / len(rows) * 100.0), 2) if rows else 0.0,
        "duration_s": round(duration_s, 3),
        "achieved_rps": round((len(rows) / duration_s), 2) if rows else 0.0,
        "p50_ms": round(percentile(latencies, 0.5), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "p99_ms": round(percentile(latencies, 0.99), 2),
        "avg_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "max_ms": round(max(latencies), 2) if latencies else 0.0,
        "ttfb_p50_ms": round(percentile(ttfb_values, 0.5), 2),
        "ttfb_p95_ms": round(percentile(ttfb_values, 0.95), 2),
        "bytes_total": bytes_total,
        "avg_bytes": round(bytes_total / len(rows), 2) if rows else 0.0,
        "status_counts": {str(code): count for code, count in sorted(status_counts.items())},
    }


def wait_for_ready(url: str, timeout_seconds: float, headers: dict[str, str]) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        sample = do_request(url=url, method="GET", body=None, timeout=1.0, headers=headers)
        if sample.status_code == 200:
            return
        time.sleep(0.5)
    raise RuntimeError(f"Target is not ready before timeout: {url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple HTTP load test for interview demos.")
    parser.add_argument("--url", required=True, help="Target URL, e.g. http://127.0.0.1:8000/health")
    parser.add_argument("--method", default="GET", choices=["GET", "POST"], help="HTTP method")
    parser.add_argument("--body", default="", help="JSON body string for POST")
    parser.add_argument("--body-file", default="", help="JSON body file path for POST")
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Request header, repeatable. Example: --header 'Authorization: Bearer xxx'",
    )
    parser.add_argument("--requests", type=int, default=200, help="Total request count")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrent workers")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per request timeout seconds")
    parser.add_argument(
        "--ready-url",
        default="",
        help="Optional health URL. If set, the script waits until this endpoint returns 200.",
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=30.0,
        help="Max seconds to wait for ready-url.",
    )
    parser.add_argument("--label", default="", help="Optional scenario label")
    parser.add_argument("--output", default="", help="Optional output json path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    headers = parse_headers(args.header)
    if args.body_file.strip():
        with open(args.body_file.strip(), "r", encoding="utf-8") as f:
            body = json.load(f)
    else:
        body = json.loads(args.body) if args.body.strip() else None

    if args.ready_url.strip():
        wait_for_ready(args.ready_url.strip(), max(1.0, args.ready_timeout), headers)

    samples, duration_s = run_load_test(
        url=args.url,
        method=args.method,
        body=body,
        total_requests=max(1, args.requests),
        concurrency=max(1, args.concurrency),
        timeout=max(0.1, args.timeout),
        headers=headers,
    )
    summary = summarize(samples, duration_s=duration_s)
    payload = {
        "label": args.label.strip() or args.url,
        "target": {"url": args.url, "method": args.method},
        "config": {
            "requests": max(1, args.requests),
            "concurrency": max(1, args.concurrency),
            "timeout": max(0.1, args.timeout),
        },
        "summary": summary,
    }

    if args.output.strip():
        output_path = args.output.strip()
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
