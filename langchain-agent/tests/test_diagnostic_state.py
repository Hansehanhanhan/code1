from __future__ import annotations

import pytest

from backend.diagnostic_state import (
    DiagnosticPatch,
    DiagnosticState,
    EvidenceRecord,
    EvidenceValidationError,
    StateConflictError,
    apply_diagnostic_patch,
    normalize_and_limit_state,
    validate_patch_evidence,
)


def finding_patch(*, expected_version: int, finding_id: str = "f-001") -> DiagnosticPatch:
    return DiagnosticPatch.model_validate(
        {
            "reason": "traffic_analysis_completed",
            "expected_version": expected_version,
            "add_findings": [
                {
                    "id": finding_id,
                    "claim": "流量下降 22%",
                    "confidence": "high",
                    "evidence": {
                        "evidence_id": "req-1:tool-1",
                        "request_id": "req-1",
                        "observation_hash": "sha256:abc",
                    },
                }
            ],
        }
    )


def test_patch_appends_without_overwriting_existing_finding() -> None:
    state = DiagnosticState(
        version=3,
        verified_findings=[
            {
                "id": "f-old",
                "claim": "历史结论",
                "confidence": "medium",
                "evidence": {
                    "evidence_id": "req-0:tool-0",
                    "request_id": "req-0",
                    "observation_hash": "sha256:old",
                },
            }
        ],
    )

    updated = apply_diagnostic_patch(state, finding_patch(expected_version=3))

    assert updated.version == 4
    assert [item.id for item in updated.verified_findings] == ["f-old", "f-001"]
    assert state.version == 3


def test_patch_rejects_stale_version() -> None:
    with pytest.raises(StateConflictError):
        apply_diagnostic_patch(DiagnosticState(version=2), finding_patch(expected_version=1))


def test_supersede_keeps_old_finding_for_audit() -> None:
    state = apply_diagnostic_patch(DiagnosticState(), finding_patch(expected_version=0))
    patch = DiagnosticPatch.model_validate(
        {
            "reason": "conflicting_evidence_resolved",
            "expected_version": 1,
            "supersede_findings": [
                {
                    "finding_id": "f-001",
                    "replacement": {
                        "id": "f-002",
                        "claim": "流量下降 18%",
                        "confidence": "high",
                        "evidence": {
                            "evidence_id": "req-2:tool-1",
                            "request_id": "req-2",
                            "observation_hash": "sha256:new",
                        },
                    },
                }
            ],
        }
    )

    updated = apply_diagnostic_patch(state, patch)

    assert [(item.id, item.status) for item in updated.verified_findings] == [
        ("f-001", "superseded"),
        ("f-002", "active"),
    ]


def test_patch_evidence_must_match_current_request_and_hash() -> None:
    patch = finding_patch(expected_version=0)
    records = [
        EvidenceRecord(
            evidence_id="req-1:tool-1",
            request_id="req-1",
            tool_call_id="tool-1",
            tool="traffic_analyze",
            observation_hash="sha256:abc",
        )
    ]

    validate_patch_evidence(patch, records, request_id="req-1")

    with pytest.raises(EvidenceValidationError, match="request mismatch"):
        validate_patch_evidence(patch, records, request_id="req-2")

    records[0].observation_hash = "sha256:wrong"
    with pytest.raises(EvidenceValidationError, match="hash mismatch"):
        validate_patch_evidence(patch, records, request_id="req-1")


def test_state_limits_items_and_total_json_size() -> None:
    state = DiagnosticState(
        current_candidates=[f"candidate-{index}" for index in range(20)],
        validity_concerns=["x" * 500 for _ in range(20)],
        next_step_plan=["plan-1", "plan-1", "plan-2"],
    )

    normalized = normalize_and_limit_state(state)

    assert len(normalized.current_candidates) == 10
    assert len(normalized.validity_concerns) <= 10
    assert normalized.next_step_plan == ["plan-1", "plan-2"]
    assert len(normalized.model_dump_json().encode("utf-8")) <= 8 * 1024


def test_patch_applies_constraint_status_confidence_and_citations() -> None:
    state = DiagnosticState(version=3)
    patch = DiagnosticPatch.model_validate(
        {
            "reason": "audit_completed",
            "expected_version": 3,
            "update_constraint_status": {
                "merchant_id": "satisfied",
                "time_range": "unchecked",
                "库存数据缺失": "unsatisfied",
            },
            "set_answer_confidence": 0.67,
            "add_citations": ["req-1:tool-1", "req-1:tool-1"],
            "supersede_findings": [],
            "replace_next_step_plan": None,
        }
    )

    updated = apply_diagnostic_patch(state, patch)

    assert updated.version == 4
    assert updated.constraint_status == {
        "merchant_id": "satisfied",
        "time_range": "unchecked",
        "库存数据缺失": "unsatisfied",
    }
    assert updated.answer_confidence == 0.67
    assert updated.verified_citations == ["req-1:tool-1"]


def test_state_limits_constraint_status_and_citations() -> None:
    state = DiagnosticState(
        constraint_status={f"c-{index}": "satisfied" for index in range(15)},
        verified_citations=[f"cite-{index}" for index in range(40)],
    )

    normalized = normalize_and_limit_state(state)

    assert len(normalized.constraint_status) == 10
    assert len(normalized.verified_citations) == 30