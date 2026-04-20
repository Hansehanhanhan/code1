from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid bool value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate bad case regression result json.")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent.parent / "docs" / "bad_case_results.json"),
        help="Input bad_case_results.json path.",
    )
    parser.add_argument("--min-pass-rate", type=float, default=0.8, help="Minimum pass rate [0,1].")
    parser.add_argument("--max-avg-latency-ms", type=int, default=60000, help="Maximum average latency in ms.")
    parser.add_argument(
        "--require-expected",
        default="false",
        help="Whether to require expected text matched in final answer. true/false.",
    )
    parser.add_argument("--require-status", type=int, default=200, help="Expected status code for pass.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_expected = parse_bool(args.require_expected)
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"[ERROR] Input file does not exist: {input_path}")
        return 2

    try:
        rows = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to read json: {exc}")
        return 2

    if not isinstance(rows, list) or not rows:
        print("[ERROR] Input rows are empty or invalid.")
        return 2

    passed_rows: list[dict] = []
    failed_rows: list[dict] = []
    latencies: list[int] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        status_code = int(row.get("status_code", 0) or 0)
        latency_ms = int(row.get("latency_ms", 0) or 0)
        expected = str(row.get("expected", ""))
        final_answer = str(row.get("final_answer", row.get("final_answer_preview", "")))
        expected_matched = bool(expected) and (expected in final_answer)
        pass_status = status_code == args.require_status
        pass_expected = (not require_expected) or expected_matched
        row_passed = pass_status and pass_expected
        latencies.append(latency_ms)
        if row_passed:
            passed_rows.append(row)
        else:
            failed_rows.append(
                {
                    "case_id": row.get("case_id"),
                    "status_code": status_code,
                    "expected_matched": expected_matched,
                    "latency_ms": latency_ms,
                    "error": row.get("error"),
                }
            )

    total = len(rows)
    passed = len(passed_rows)
    pass_rate = (passed / total) if total else 0.0
    avg_latency = int(mean(latencies)) if latencies else 0
    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(pass_rate, 4),
        "avg_latency_ms": avg_latency,
        "min_pass_rate": args.min_pass_rate,
        "max_avg_latency_ms": args.max_avg_latency_ms,
        "require_expected": require_expected,
        "require_status": args.require_status,
    }

    print(json.dumps({"summary": summary, "failed_cases": failed_rows}, ensure_ascii=False, indent=2))

    if pass_rate < args.min_pass_rate:
        print("[FAIL] pass_rate below threshold.")
        return 1
    if avg_latency > args.max_avg_latency_ms:
        print("[FAIL] avg_latency_ms above threshold.")
        return 1

    print("[PASS] bad case evaluation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

