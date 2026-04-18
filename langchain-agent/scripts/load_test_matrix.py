from __future__ import annotations

import argparse
import json
import os
from typing import Any

from load_test import run_load_test, summarize, wait_for_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a load-test matrix and export json/markdown reports.")
    parser.add_argument(
        "--cases-file",
        default="docs/load_test_cases.json",
        help="Path to test cases json file.",
    )
    parser.add_argument(
        "--output-json",
        default="docs/load_test_results.json",
        help="Output json report path.",
    )
    parser.add_argument(
        "--output-md",
        default="docs/load_test_results.md",
        help="Output markdown report path.",
    )
    return parser.parse_args()


def _ensure_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def _markdown_table_rows(results: list[dict[str, Any]]) -> str:
    rows = [
        "| 场景 | 请求数 | 并发 | 成功率 | RPS | P50(ms) | P95(ms) | P99(ms) | TTFB P95(ms) | 主要状态码 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        summary = item["summary"]
        status_counts = ", ".join(f"{k}:{v}" for k, v in summary["status_counts"].items()) or "-"
        rows.append(
            "| {label} | {total} | {conc} | {succ:.2f}% | {rps:.2f} | {p50:.2f} | {p95:.2f} | {p99:.2f} | {ttfb95:.2f} | {status} |".format(
                label=item["label"],
                total=item["config"]["requests"],
                conc=item["config"]["concurrency"],
                succ=summary["success_rate"],
                rps=summary["achieved_rps"],
                p50=summary["p50_ms"],
                p95=summary["p95_ms"],
                p99=summary["p99_ms"],
                ttfb95=summary["ttfb_p95_ms"],
                status=status_counts,
            )
        )
    return "\n".join(rows)


def main() -> int:
    args = parse_args()
    with open(args.cases_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cases: list[dict[str, Any]] = raw.get("cases", [])
    if not cases:
        raise ValueError("No cases found in cases-file.")

    results: list[dict[str, Any]] = []
    for case in cases:
        label = str(case.get("label", "")).strip() or str(case["url"])
        url = str(case["url"])
        method = str(case.get("method", "GET")).upper()
        body = case.get("body")
        concurrency = max(1, int(case.get("concurrency", 20)))
        requests = max(1, int(case.get("requests", 200)))
        timeout = max(0.1, float(case.get("timeout", 10.0)))
        headers = dict(case.get("headers", {}))
        ready_url = str(case.get("ready_url", "")).strip()
        ready_timeout = max(1.0, float(case.get("ready_timeout", 30.0)))

        if ready_url:
            wait_for_ready(ready_url, ready_timeout, headers=headers)

        samples, duration_s = run_load_test(
            url=url,
            method=method,
            body=body if isinstance(body, dict) else None,
            total_requests=requests,
            concurrency=concurrency,
            timeout=timeout,
            headers=headers,
        )
        results.append(
            {
                "label": label,
                "target": {"url": url, "method": method},
                "config": {
                    "requests": requests,
                    "concurrency": concurrency,
                    "timeout": timeout,
                },
                "summary": summarize(samples, duration_s=duration_s),
            }
        )

    output_json = {"cases_file": args.cases_file, "results": results}
    _ensure_dir(args.output_json)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# 压测矩阵结果",
        "",
        f"- cases file: `{args.cases_file}`",
        f"- result json: `{args.output_json}`",
        "",
        _markdown_table_rows(results),
        "",
    ]
    _ensure_dir(args.output_md)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(json.dumps(output_json, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

