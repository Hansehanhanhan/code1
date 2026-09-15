from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVIDENCE_REF_KEYS = {"evidence_id", "request_id", "observation_hash"}
TERMINAL_JOB_STATUSES = {"succeeded", "degraded", "failed", "cancelled"}
ANALYSIS_TOOL_HINTS = ("traffic_analyze", "ads_analyze", "inventory_check", "product_diagnose")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end verification: real LLM x real HTTP x SSE x session state."
    )
    parser.add_argument("--base-url", default="", help="External backend base url (skip spawn when set).")
    parser.add_argument("--spawn", action="store_true", default=True, help="Spawn a backend subprocess (default).")
    parser.add_argument("--no-spawn", action="store_false", dest="spawn", help="Do not spawn; require --base-url.")
    parser.add_argument("--api-key", default="", help="X-API-Key to send when APP_AUTH_ENABLED=true.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "_e2e_result.json"),
        help="Output result json path.",
    )
    parser.add_argument("--ready-timeout", type=float, default=90.0, help="Seconds to wait for /health.")
    parser.add_argument("--request-timeout", type=float, default=180.0, help="Per request/stream timeout seconds.")
    parser.add_argument("--job-timeout", type=float, default=240.0, help="Seconds to wait for job terminal status.")
    return parser.parse_args()


def dotenv_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def wait_ready(url: str, timeout_sec: float = 60.0) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urlopen(Request(url=url, method="GET"), timeout=1.0) as response:
                if int(response.getcode() or 0) == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Service not ready: {url}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def spawn_backend(port: int, *, job_db_path: str) -> subprocess.Popen:
    env = dict(os.environ)
    env.pop("JOB_DB_PATH", None)
    env["JOB_DB_PATH"] = job_db_path
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class ApiError(RuntimeError):
    pass


def request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict | None = None,
    *,
    api_key: str | None = None,
    timeout_sec: float = 30.0,
) -> tuple[int, dict, dict]:
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip():
        headers["X-API-Key"] = api_key.strip()
    data = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = Request(url=f"{base_url}{path}", method=method, data=data, headers=headers)
    try:
        with urlopen(request, timeout=max(1.0, timeout_sec)) as response:
            code = int(response.getcode() or 0)
            body = json.loads(response.read().decode("utf-8", errors="ignore") or "{}")
            return code, body, dict(response.headers)
    except HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8", errors="ignore") or "{}")
        except Exception:
            body = {"detail": str(exc)}
        return exc.code, body, {}
    except URLError as exc:
        raise ApiError(str(exc)) from exc


def _parse_sse_block(block: str) -> dict | None:
    data_lines = [line[5:].strip() for line in block.splitlines() if line.startswith("data:")]
    if not data_lines:
        return None
    payload = "\n".join(data_lines)
    try:
        parsed = json.loads(payload)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def sse_events(
    base_url: str,
    method: str,
    path: str,
    payload: dict,
    *,
    api_key: str | None = None,
    timeout_sec: float = 60.0,
) -> list[dict]:
    parsed = _split_url(base_url)
    deadline = time.time() + max(1.0, timeout_sec)
    conn = http.client.HTTPConnection(parsed.host, parsed.port, timeout=max(1.0, timeout_sec))
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key and api_key.strip():
        headers["X-API-Key"] = api_key.strip()
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        conn.request(method, parsed.path + path, body=body, headers=headers)
        response = conn.getresponse()
        buf = ""
        events: list[dict] = []
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise ApiError(f"SSE stream timed out after {timeout_sec}s: {path}")
            conn.sock.settimeout(min(conn.timeout, remaining))
            chunk = response.read(16384)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="ignore")
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                event = _parse_sse_block(block)
                if event is not None:
                    events.append(event)
        if buf.strip():
            event = _parse_sse_block(buf)
            if event is not None:
                events.append(event)
        return events
    finally:
        conn.close()


def _split_url(base_url: str):
    rest = base_url[len("http://"):]
    host, _, port = rest.partition(":")
    return SimpleNamespace(host=host, port=int(port or 80), path="")


def add_check(checks: list[dict], name: str, ok: bool, detail: str = "") -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def event_sequence(events: list[dict]) -> tuple[list[str], dict]:
    by_type: dict[str, list[dict]] = {}
    order: list[str] = []
    for event in events:
        event_type = event.get("type")
        if isinstance(event_type, str):
            order.append(event_type)
            by_type.setdefault(event_type, []).append(event)
    return order, by_type


def verify_round1(base_url: str, session_id: str, checks: list[dict], *, api_key: str | None, timeout: float) -> dict:
    context = {"merchant_id": "demo-001", "time_range": "last_7_days"}
    payload = {
        "query": "请分析最近流量下滑的全面原因",
        "context": context,
        "session_id": session_id,
    }
    started = time.perf_counter()
    events = sse_events(base_url, "POST", "/run_stream", payload, api_key=api_key, timeout_sec=timeout)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    order, by_type = event_sequence(events)

    add_check(checks, "T1.run_stream非空", bool(events), f"{len(events)} events")
    for name in ("agent_action", "tool_observation", "final_response", "stream_metrics"):
        add_check(checks, f"T1.events含{name}", name in by_type, f"order={order[:20]}")
    key_steps = by_type.get("key_step", [])
    first_evidence = [e for e in key_steps if isinstance(e.get("content"), dict) and e["content"].get("kind") == "first_evidence"]
    add_check(checks, "T1.key_step(first_evidence)", bool(first_evidence), str(first_evidence[:1]))

    terminal_idx = None
    for idx, event in enumerate(events):
        if event.get("type") == "final_response":
            terminal_idx = idx
            break
    before_terminal = events[:terminal_idx] if terminal_idx is not None else events
    has_tool_before_final = any(
        e.get("type") in ("agent_action", "tool_observation", "key_step") for e in before_terminal
    )
    add_check(checks, "T1.final_response前有工具事件", has_tool_before_final)

    obs_evidence_ok = []
    for event in by_type.get("tool_observation", []):
        content = event.get("content") or {}
        observation = content.get("observation") or {}
        embedded = observation.get("evidence") if isinstance(observation, dict) else None
        if isinstance(embedded, dict):
            obs_evidence_ok.append(set(embedded.keys()) == EVIDENCE_REF_KEYS and str(embedded.get("observation_hash", "")).startswith("sha256:"))
    add_check(checks, "T1.tool_observation嵌3字段证据", bool(obs_evidence_ok) and all(obs_evidence_ok), f"{len(obs_evidence_ok)} observations")

    stream_metrics = next((e.get("content") for e in by_type.get("stream_metrics", []) if isinstance(e.get("content"), dict)), None)
    add_check(
        checks,
        "T1.stream_metrics完整性",
        bool(stream_metrics)
        and stream_metrics.get("ttfb_ms", -1) >= 0
        and stream_metrics.get("event_count", 0) >= 1
        and stream_metrics.get("event_completeness") is True,
        json.dumps(stream_metrics or {}, ensure_ascii=False),
    )

    final_response = next((e.get("content") for e in by_type.get("final_response", []) if isinstance(e.get("content"), dict)), None)
    final_content = (final_response or {}).get("content") or (final_response or {})
    final_answer = str((final_content or {}).get("final_answer", ""))
    force_stopped = bool(final_answer.startswith("Agent stopped due to max iterations."))
    add_check(checks, "T1.final_answer非空未截断", bool(final_answer) and not force_stopped, final_answer[:120])

    return {
        "steps": len(order),
        "event_types": order,
        "latency_ms": elapsed_ms,
        "force_stopped": force_stopped,
        "final_answer_preview": final_answer[:160],
    }


def verify_session_state(base_url: str, session_id: str, checks: list[dict], *, api_key: str | None) -> dict:
    code, body, _ = request_json(base_url, "GET", f"/sessions/{session_id}", api_key=api_key, timeout_sec=30)
    add_check(checks, "S.state可读", code == 200 and isinstance(body, dict), f"http={code}")
    if code != 200:
        return {"http": code}
    findings = body.get("verified_findings") or []
    constraint_status = body.get("constraint_status") or {}
    citations = body.get("verified_citations") or []
    answer_confidence = body.get("answer_confidence")
    legal_status = bool(constraint_status) and all(v in {"satisfied", "unsatisfied"} for v in constraint_status.values())
    add_check(checks, "S.finding_count>=1", len(findings) >= 1, f"findings={len(findings)}")
    add_check(checks, "S.constraint_status非空合法", legal_status, json.dumps(constraint_status, ensure_ascii=False))
    add_check(checks, "S.verified_citations非空", bool(citations), f"citations={len(citations)}")
    add_check(
        checks,
        "S.answer_confidence合法",
        answer_confidence is None or (isinstance(answer_confidence, (int, float)) and 0 <= answer_confidence <= 1),
        f"confidence={answer_confidence}",
    )
    add_check(checks, "S.context_slots.merchant_id", (body.get("context_slots") or {}).get("merchant_id") == "demo-001")
    return {
        "http": code,
        "state_version": body.get("version"),
        "finding_count": len(findings),
        "candidate_count": len(body.get("current_candidates") or []),
        "constraint_status": constraint_status,
        "answer_confidence": answer_confidence,
        "verified_citations": citations,
        "context_slots": body.get("context_slots") or {},
    }


def verify_round2(base_url: str, session_id: str, checks: list[dict], *, api_key: str | None, timeout: float) -> dict:
    payload = {
        "query": "根据上面的分析，给出接下来改进措施的优先级排序",
        "context": {},
        "session_id": session_id,
    }
    started = time.perf_counter()
    code, body, _ = request_json(base_url, "POST", "/run", payload, api_key=api_key, timeout_sec=timeout)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    steps = body.get("steps") or []
    final_answer = str(body.get("final_answer", ""))
    is_clarification = bool(steps and steps[0].get("name") == "Clarification")
    add_check(checks, "T2.run状态200非澄清", code == 200 and not is_clarification, f"http={code} steps={len(steps)}")
    add_check(checks, "T2.final_answer非空", bool(final_answer), final_answer[:120])
    return {
        "http": code,
        "steps": len(steps),
        "latency_ms": elapsed_ms,
        "clarification": is_clarification,
        "final_answer_preview": final_answer[:160],
    }


def verify_round3_jobs(base_url: str, session_id: str, checks: list[dict], *, api_key: str | None, timeout: float, job_timeout: float) -> dict:
    payload = {
        "query": "请综合排查广告投放效率与预算分配的整体情况",
        "context": {"merchant_id": "demo-001", "time_range": "last_7_days"},
        "session_id": session_id,
    }
    code, body, _ = request_json(base_url, "POST", "/jobs", payload, api_key=api_key, timeout_sec=timeout)
    job_id = body.get("job_id") if isinstance(body, dict) else None
    add_check(checks, "T3.任务创建", code == 200 and isinstance(job_id, str) and job_id, f"http={code}")

    status = body.get("status", "")
    deadline = time.time() + job_timeout
    while time.time() < deadline:
        code, body, _ = request_json(base_url, "GET", f"/jobs/{job_id}", api_key=api_key, timeout_sec=30)
        status = body.get("status", "")
        if status in TERMINAL_JOB_STATUSES:
            break
        time.sleep(0.5)

    add_check(checks, "T3.任务终态succeeded", status == "succeeded", f"status={status}")
    code, events_body, _ = request_json(base_url, "GET", f"/jobs/{job_id}/events", api_key=api_key, timeout_sec=30)
    events = events_body.get("events") or []
    event_types = [e.get("type") for e in events if isinstance(e, dict)]
    for name in ("tool_observation", "final_response", "stream_metrics"):
        add_check(checks, f"T3.events含{name}", name in event_types, f"types={event_types[:12]}")
    job_response = body.get("response") if status == "succeeded" else None
    job_final = str((job_response or {}).get("final_answer", "")) if isinstance(job_response, dict) else ""
    add_check(checks, "T3.job_response.final_answer", bool(job_final), job_final[:120])

    stream_replays = sse_events(base_url, "GET", f"/jobs/{job_id}/stream", {}, api_key=api_key, timeout_sec=30)
    replay_types = [e.get("type") for e in stream_replays if isinstance(e, dict)]
    replay_ok = "final_response" in replay_types and "stream_metrics" in replay_types
    add_check(checks, "T3./stream可回放", replay_ok, f"replay={replay_types[:12]}")

    return {
        "job_id": job_id,
        "status": status,
        "event_types": event_types,
        "replay_types": replay_types,
    }


def build_report(round1: dict, state1: dict, state2: dict, round2: dict, round3: dict, checks: list[dict], spawned: bool) -> dict:
    ok = all(check["ok"] for check in checks)
    return {
        "ok": ok,
        "spawned": spawned,
        "round1": round1,
        "session_after_round1": state1,
        "round2": round2,
        "session_after_round2": state2,
        "round3_jobs": round3,
        "checks": checks,
        "total_checks": len(checks),
        "passed_checks": sum(1 for check in checks if check["ok"]),
    }


def main() -> int:
    args = parse_args()
    env = dotenv_values(PROJECT_ROOT / ".env")
    api_key = args.api_key.strip() or (env.get("APP_API_KEY", "") if env.get("APP_AUTH_ENABLED", "") == "true" else "")

    spawned = False
    process: subprocess.Popen | None = None
    if args.base_url:
        base_url = args.base_url.rstrip("/")
        print(f"[e2e] external server: {base_url}")
    elif args.spawn:
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        job_db_path = str(Path(tempfile.mkdtemp(prefix="e2e_")) / "jobs.db")
        process = spawn_backend(port, job_db_path=job_db_path)
        spawned = True
        print(f"[e2e] spawned backend: {base_url}")
        try:
            wait_ready(f"{base_url}/health", timeout_sec=max(1.0, args.ready_timeout))
        except Exception:
            process.terminate()
            process.wait(timeout=5)
            raise
    else:
        raise SystemExit("--base-url is required when --no-spawn is set.")

    checks: list[dict] = []
    session_id = f"e2e_{uuid.uuid4().hex[:12]}"
    try:
        round1 = verify_round1(base_url, session_id, checks, api_key=api_key, timeout=args.request_timeout)
        print("[e2e] T1 /run_stream:", json.dumps(round1, ensure_ascii=False))
        state1 = verify_session_state(base_url, session_id, checks, api_key=api_key)
        print("[e2e] session after T1:", json.dumps(state1, ensure_ascii=False))
        round2 = verify_round2(base_url, session_id, checks, api_key=api_key, timeout=args.request_timeout)
        print("[e2e] T2 /run:", json.dumps(round2, ensure_ascii=False))
        state2 = verify_session_state(base_url, session_id, checks, api_key=api_key)
        print("[e2e] session after T2:", json.dumps(state2, ensure_ascii=False))
        state_evolved = (
            isinstance(state2.get("state_version"), int)
            and isinstance(state1.get("state_version"), int)
            and state2["state_version"] > state1["state_version"]
        )
        state_merged = (_keys(state2.get("context_slots") or {}) & _keys(state1.get("context_slots") or {})) == _keys(
            state1.get("context_slots") or {}
        )
        channels_grew = int(state2.get("candidate_count") or 0) >= int(state1.get("candidate_count") or 0) and int(
            state2.get("finding_count") or 0
        ) >= int(state1.get("finding_count") or 0)
        add_check(
            checks,
            "T2.记忆闭环(版本增长/槽保留/计数非降)",
            state_evolved and bool(state_merged) and channels_grew,
            f"v{state1.get('state_version')}->v{state2.get('state_version')}",
        )
        job_session = f"e2e_job_{uuid.uuid4().hex[:12]}"
        round3 = verify_round3_jobs(base_url, job_session, checks, api_key=api_key, timeout=args.request_timeout, job_timeout=args.job_timeout)
        print("[e2e] T3 /jobs:", json.dumps(round3, ensure_ascii=False))
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    report = build_report(round1, state1, state2, round2, round3, checks, spawned=spawned)
    output_path = Path(args.output).expanduser().resolve()
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[e2e] checks {report['passed_checks']}/{report['total_checks']}")
    for check in checks:
        print(f"     [{'PASS' if check['ok'] else 'FAIL'}] {check['name']} {check['detail']}")
    print(f"[e2e] result written to {output_path}")
    if report["ok"]:
        print("[PASS] end-to-end verification passed.")
        return 0
    print("[FAIL] end-to-end verification failed.")
    return 1


def _keys(mapping: dict) -> set:
    return set(mapping.keys())


if __name__ == "__main__":
    raise SystemExit(main())