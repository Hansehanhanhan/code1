from __future__ import annotations

import json

import agent.agent as agent_mod
from agent.agent import (
    _apply_restart,
    _audit_confidence,
    _emit_outer_decision,
    _outer_decision,
    _restart_recoverable,
    _run_deterministic_verify,
    _run_llm_verify,
    _run_outer_loop,
)
from backend.diagnostic_state import (
    DiagnosticPatch,
    EvidenceRef,
    FindingProposal,
)
from backend.models import Metrics, StepRecord
from backend.session_store import InMemorySessionStore


class OuterLoopSettings:
    session_ttl_seconds = 3600
    outer_loop_enabled = True
    openai_api_key = ""
    openai_base_url = "https://example.invalid"
    openai_model = "gpt-test"
    request_timeout_seconds = 10
    rag_enabled = False


def _evidence() -> EvidenceRef:
    return EvidenceRef(
        evidence_id="e1",
        request_id="req-x",
        observation_hash="h1",
    )


def _seed_low_finding(store: InMemorySessionStore, claim: str = "综合结论待复核") -> None:
    state = store.get_state("s1")
    store.update_state(
        "s1",
        DiagnosticPatch(
            reason="seed_finding",
            expected_version=state.version,
            add_findings=[
                FindingProposal(id="f1", claim=claim, confidence="low", evidence=_evidence())
            ],
            update_constraint_status={"约束A": "satisfied"},
        ),
        ttl_seconds=3600,
    )


def test_outer_decision_threshold_split() -> None:
    assert _outer_decision(None) == "verify"
    assert _outer_decision(1.0) == "accept"
    assert _outer_decision(0.7) == "accept"
    assert _outer_decision(0.69) == "verify"
    assert _outer_decision(0.4) == "verify"
    assert _outer_decision(0.39) == "restart"
    assert _outer_decision(0.0) == "restart"


def test_audit_confidence_uses_constraint_ratio() -> None:
    store = InMemorySessionStore()
    state = store.get_state("s-a")
    assert _audit_confidence(state) is None
    store.update_state(
        "s-a",
        DiagnosticPatch(
            reason="seed",
            expected_version=0,
            update_constraint_status={"a": "satisfied", "b": "unsatisfied", "c": "unchecked"},
        ),
        ttl_seconds=3600,
    )
    refreshed = store.get_state("s-a")
    assert _audit_confidence(refreshed) == 0.33
    store.update_state(
        "s-a",
        DiagnosticPatch(
            reason="set_conf",
            expected_version=1,
            set_answer_confidence=0.9,
        ),
        ttl_seconds=3600,
    )
    with_conf = store.get_state("s-a")
    assert _audit_confidence(with_conf) == 0.33


def test_audit_confidence_falls_back_to_model_report() -> None:
    store = InMemorySessionStore()
    store.update_state(
        "s-b",
        DiagnosticPatch(reason="set_conf", expected_version=0, set_answer_confidence=0.85),
        ttl_seconds=3600,
    )
    assert _audit_confidence(store.get_state("s-b")) == 0.85


def test_restart_recoverable_thresholds() -> None:
    store = InMemorySessionStore()
    state = store.get_state("s-c")
    assert _restart_recoverable(state) is False
    store.update_state(
        "s-c",
        DiagnosticPatch(
            reason="reject",
            expected_version=0,
            reject_candidates=["x1", "x2", "x3"],
        ),
        ttl_seconds=3600,
    )
    assert _restart_recoverable(store.get_state("s-c")) is True
    store2 = InMemorySessionStore()
    store2.update_state(
        "s-d",
        DiagnosticPatch(
            reason="concerns",
            expected_version=0,
            add_validity_concerns=["c1", "c2"],
        ),
        ttl_seconds=3600,
    )
    assert _restart_recoverable(store2.get_state("s-d")) is True


def test_clear_current_candidates_patch_applied_by_store() -> None:
    store = InMemorySessionStore()
    store.update_state(
        "s-cc",
        DiagnosticPatch(
            reason="seed_cands",
            expected_version=0,
            add_candidates=["c1", "c2"],
            reject_candidates=["r1"],
            replace_next_step_plan=["A"],
        ),
        ttl_seconds=3600,
    )
    store.update_state(
        "s-cc",
        DiagnosticPatch(reason="clear", expected_version=1, clear_current_candidates=True),
        ttl_seconds=3600,
    )
    state = store.get_state("s-cc")
    assert state.current_candidates == []
    assert state.rejected_candidates == ["r1"]


def test_apply_restart_preserves_verified_progress() -> None:
    store = InMemorySessionStore()
    _seed_low_finding(store)
    store.update_state(
        "s1",
        DiagnosticPatch(
            reason="seed_plan",
            expected_version=1,
            add_candidates=["C1", "C2", "C3"],
            replace_next_step_plan=["先看广告", "再看流量"],
        ),
        ttl_seconds=3600,
    )
    assert _apply_restart(store, OuterLoopSettings(), "s1", "req-r", {"merchant_id": "m"}) is True

    state = store.get_state("s1")
    assert state.current_candidates == []
    assert state.next_step_plan == []
    assert state.rejected_candidates == ["C1", "C2", "C3"]
    assert any(f.id == "f1" and f.status == "active" for f in state.verified_findings)
    assert state.constraint_status == {"约束A": "satisfied"}


def test_deterministic_verify_upgrades_low_confidence_finding(monkeypatch) -> None:
    store = InMemorySessionStore()
    _seed_low_finding(store, claim="广告ROI下降")
    monkeypatch.setattr(
        agent_mod,
        "_run_single_tool",
        lambda tool_name, settings, query, context: {"status": "ok", "summary": "复核支持"},
    )
    steps: list[StepRecord] = []

    acted = _run_deterministic_verify(
        OuterLoopSettings(),
        store,
        "s1",
        "req-v",
        "请查广告ROI",
        {"merchant_id": "m", "time_range": "t"},
        {},
        {"index": 0},
        steps,
        2,
    )

    assert acted is True
    assert any(step.name == "Verify Loop 2" for step in steps)
    state = store.get_state("s1")
    assert any(f.id == "v-f1" and f.confidence == "high" for f in state.verified_findings)
    assert state.verified_findings[0].status == "superseded"


def test_deterministic_verify_raises_concern_when_unconfirmed(monkeypatch) -> None:
    store = InMemorySessionStore()
    _seed_low_finding(store, claim="广告ROI下降")
    monkeypatch.setattr(
        agent_mod,
        "_run_single_tool",
        lambda tool_name, settings, query, context: {"status": "error", "summary": ""},
    )
    steps: list[StepRecord] = []

    acted = _run_deterministic_verify(
        OuterLoopSettings(),
        store,
        "s1",
        "req-v2",
        "请查广告ROI",
        {},
        {},
        {"index": 0},
        steps,
        1,
    )

    assert acted is True
    state = store.get_state("s1")
    assert any("复核未获支持" in concern for concern in state.validity_concerns)
    assert any(f.id == "f1" and f.status == "active" for f in state.verified_findings)


def test_llm_verify_parses_and_applies_constraint_verdict(monkeypatch) -> None:
    store = InMemorySessionStore()
    _seed_low_finding(store)
    state = store.get_state("s1")
    assert state.constraint_status == {"约束A": "satisfied"}

    verdict = {
        "constraints": {"约束A": "supported"},
        "note": "证据充分，确认结论",
    }

    class FakeChat:
        def __init__(self, **kwargs):
            pass

        def invoke(self, prompt: str):
            return type("Msg", (), {"content": json.dumps(verdict, ensure_ascii=False)})()

    monkeypatch.setattr(agent_mod, "ChatOpenAI", FakeChat)
    settings = OuterLoopSettings()
    settings.openai_api_key = "test-key"

    result = _run_llm_verify(
        settings, store, "s1", "req-llm", store.get_state("s1"), "答案",
    )
    assert result is not None
    assert result["constraints"]["约束A"] == "supported"
    updated = store.get_state("s1")
    assert updated.constraint_status == {"约束A": "satisfied"}
    assert any("证据充分" in concern for concern in updated.validity_concerns)


def test_llm_verify_ignores_unknown_constraints_and_collapses(monkeypatch) -> None:
    import copy

    store = InMemorySessionStore()
    _seed_low_finding(store)

    class FakeChat:
        def __init__(self, **kwargs):
            pass

        def invoke(self, prompt: str):
            payload = {"constraints": {"约束A": "unsupported", "不存在键": "supported"}, "note": ""}
            return type("Msg", (), {"content": json.dumps(payload, ensure_ascii=False)})()

    monkeypatch.setattr(agent_mod, "ChatOpenAI", FakeChat)
    settings = OuterLoopSettings()
    settings.openai_api_key = "test-key"
    before = copy.deepcopy(store.get_state("s1").constraint_status)

    result = _run_llm_verify(
        settings, store, "s1", "req-llm2", store.get_state("s1"), "答案",
    )
    assert result is not None
    updated = store.get_state("s1")
    assert "不存在键" not in updated.constraint_status
    assert updated.constraint_status["约束A"] == "unsatisfied"


def test_llm_verify_falls_back_on_malformed_response(monkeypatch) -> None:
    store = InMemorySessionStore()
    _seed_low_finding(store)

    class FakeChat:
        def __init__(self, **kwargs):
            pass

        def invoke(self, prompt: str):
            return type("Msg", (), {"content": "抱歉我无法回答"})()  # noqa: E501

    monkeypatch.setattr(agent_mod, "ChatOpenAI", FakeChat)
    settings = OuterLoopSettings()
    settings.openai_api_key = "test-key"

    assert _run_llm_verify(settings, store, "s1", "req-bad", store.get_state("s1"), "答案") is None
    assert store.get_state("s1").constraint_status == {"约束A": "satisfied"}


def test_llm_verify_skips_without_api_key() -> None:
    store = InMemorySessionStore()
    _seed_low_finding(store)
    assert (
        _run_llm_verify(
            OuterLoopSettings(), store, "s1", "req-nokey", store.get_state("s1"), "答案"
        )
        is None
    )


def test_emit_outer_decision_streams_sse_event() -> None:
    seen: list[dict] = []
    _emit_outer_decision(
        lambda event: seen.append(event),
        decision="verify",
        confidence=0.5,
        reason="测试理由",
        round_no=1,
    )
    assert len(seen) == 1
    event = seen[0]
    assert event["type"] == "outer_decision"
    assert event["content"]["decision"] == "verify"
    assert event["content"]["confidence"] == 0.5
    assert event["content"]["round"] == 1


def test_outer_loop_accepts_immediately_on_high_confidence() -> None:
    store = InMemorySessionStore()
    _seed_low_finding(store)
    store.update_state(
        "s1",
        DiagnosticPatch(
            reason="make_confident",
            expected_version=1,
            update_constraint_status={"约束A": "satisfied", "约束B": "satisfied"},
        ),
        ttl_seconds=3600,
    )
    calls: list[str] = []

    def run_once() -> str:
        calls.append("invoked")
        return "第二次答案"

    steps: list[StepRecord] = []
    result = _run_outer_loop(
        OuterLoopSettings(), store, "s1", "req-o", "问题", {}, {}, {"index": 0}, None,
        steps, "第一次答案", run_once,
    )
    assert result["final_answer"] == "第一次答案"
    assert result["outer_rounds"] == 1
    assert result["outer_decisions"] == ["accept"]
    assert calls == []


def test_outer_loop_verify_rounds_and_falls_back_to_best() -> None:
    store = InMemorySessionStore()
    _seed_low_finding(store)
    store.update_state(
        "s1",
        DiagnosticPatch(
            reason="half",
            expected_version=1,
            update_constraint_status={"约束A": "satisfied", "约束B": "unsatisfied"},
        ),
        ttl_seconds=3600,
    )
    steps: list[StepRecord] = []
    result = _run_outer_loop(
        OuterLoopSettings(), store, "s1", "req-o2", "问题", {"merchant_id": "m"}, {},
        {"index": 0}, None, steps, "答案一", lambda: "不应再跑",
    )
    assert result["outer_rounds"] == 3
    assert result["outer_decisions"] == ["verify", "verify", "verify"]
    assert result["final_answer"] == "答案一"


def test_outer_loop_restart_runs_once_again_and_keeps_accepted_latest() -> None:
    store = InMemorySessionStore()
    _seed_low_finding(store)
    store.update_state(
        "s1",
        DiagnosticPatch(
            reason="low_conf",
            expected_version=1,
            update_constraint_status={"约束A": "satisfied", "约束B": "unsatisfied", "约束C": "unsatisfied"},
            add_validity_concerns=["疑点1", "疑点2"],
            add_candidates=["C1", "C2", "C3"],
            replace_next_step_plan=["计划"],
        ),
        ttl_seconds=3600,
    )
    calls: list[str] = []

    def run_once() -> str:
        calls.append("invoked")
        return "重启后答案"

    steps: list[StepRecord] = []
    result = _run_outer_loop(
        OuterLoopSettings(), store, "s1", "req-o3", "问题", {}, {}, {"index": 0}, None,
        steps, "低置信答案", run_once,
    )
    assert len(calls) == 1
    assert result["outer_decisions"][0] == "restart"
    assert result["final_answer"] == "重启后答案"
    state = store.get_state("s1")
    assert state.rejected_candidates == ["C1", "C2", "C3"]


def test_metrics_outer_fields_defaults() -> None:
    metrics = Metrics(latency_ms=0, fallback_used=False)
    assert metrics.outer_rounds == 0
    assert metrics.outer_decisions == []
    assert metrics.tool_loop_count == 0