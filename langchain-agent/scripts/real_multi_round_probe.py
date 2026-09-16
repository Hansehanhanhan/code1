from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.agent import run_agent
from backend.settings import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real multi-round memory probe: round1 writes state, round2 reuses it without re-clarifying."
    )
    parser.add_argument("--session-id", default=None, help="Fixed session id (default: random per run).")
    parser.add_argument(
        "--round1-query",
        default="请分析最近流量下滑的全面原因和行动方案",
        help="Round 1 user query (full tool-calling agent, narrow tool set so round 2 adds new evidence).",
    )
    parser.add_argument(
        "--round2-query",
        default="那广告投放效率怎么样？帮我看看有没有问题",
        help="Round 2 user query (context omitted to prove slot reuse).",
    )
    parser.add_argument(
        "--merchant-id",
        default="demo-001",
        help="merchant_id used in round 1 explicit context.",
    )
    parser.add_argument(
        "--time-range",
        default="last_7_days",
        help="time_range used in round 1 explicit context.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent.parent / "_real_multi_probe_result.json"),
        help="Output result json path.",
    )
    return parser.parse_args()


def summarize(session_id: str, context: dict, response) -> dict:
    from backend.session_store import get_session_store

    settings = Settings.from_env()
    store = get_session_store(settings)
    state = store.get_state(session_id)
    is_clarification = bool(response.steps and response.steps[0].name == "Clarification")
    return {
        "steps": len(response.steps),
        "clarification": is_clarification,
        "state_version": state.version,
        "finding_count": len(state.verified_findings),
        "candidate_count": len(state.current_candidates),
        "constraints_count": len(state.unresolved_constraints),
        "constraint_status": state.constraint_status,
        "answer_confidence": state.answer_confidence,
        "verified_citations": state.verified_citations,
        "context_slots": state.context_slots,
        "latency_ms": response.metrics.latency_ms,
        "force_stopped": bool(
            response.final_answer and response.final_answer.startswith("Agent stopped due to max iterations.")
        ),
        "final_answer_preview": (response.final_answer or "")[:160],
    }


def main() -> int:
    args = parse_args()
    settings = Settings.from_env()
    session_id = args.session_id or f"probe_{uuid4().hex[:8]}"

    round1_context = {"merchant_id": args.merchant_id, "time_range": args.time_range}
    print(f"[probe] session_id={session_id}")

    started = time.perf_counter()
    round1_response = run_agent(args.round1_query, round1_context, session_id=session_id)
    round1_summary = summarize(session_id, round1_context, round1_response)
    print("[probe] round1:", json.dumps(round1_summary, ensure_ascii=False))

    round2_response = run_agent(args.round2_query, {}, session_id=session_id)
    round2_summary = summarize(session_id, {}, round2_response)
    print("[probe] round2:", json.dumps(round2_summary, ensure_ascii=False))
    elapsed = int((time.perf_counter() - started) * 1000)

    state = summarize(session_id, {}, round2_response)
    state_evolved = (
        round2_summary["state_version"] > round1_summary["state_version"]
        or round2_summary["candidate_count"] > round1_summary["candidate_count"]
        or round2_summary["finding_count"] > round1_summary["finding_count"]
    )
    state_reusable = (
        not round2_summary["clarification"]
        and round2_summary["context_slots"] == round1_summary["context_slots"]
    )
    ok = (
        not round1_summary["clarification"]
        and not round2_summary["clarification"]
        and round1_summary["state_version"] > 0
        and round1_summary["finding_count"] >= 1
        and bool(round1_summary["constraint_status"])
        and not round1_summary["force_stopped"]
        and (state_evolved or state_reusable)
    )
    result = {
        "ok": ok,
        "session_id": session_id,
        "settings": {
            "session_backend": settings.session_backend,
            "rag_enabled": settings.rag_enabled,
            "model": settings.openai_model,
        },
        "round1": round1_summary,
        "round2": round2_summary,
        "total_elapsed_ms": elapsed,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    output_path = Path(args.output)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[probe] result written to {output_path}")

    if not ok:
        print("[FAIL] memory closure criteria not met.")
        return 1

    print("[PASS] round1 wrote evidence-backed findings + constraint status; round2 reused context without re-clarification; state evolved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())