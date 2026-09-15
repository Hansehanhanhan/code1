from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


MAX_STATE_BYTES = 8 * 1024
MAX_FINDINGS = 20
MAX_CANDIDATES = 10
MAX_CONSTRAINTS = 10
MAX_CONCERNS = 10
MAX_REJECTED = 20
MAX_PLAN_ITEMS = 10
MAX_STATE_TEXT_LENGTH = 200
MAX_CITATIONS = 30


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    observation_hash: str = Field(min_length=1)


class EvidenceRecord(EvidenceRef):
    tool_call_id: str = Field(min_length=1)
    tool: str = Field(min_length=1)


class VerifiedFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    claim: str = Field(min_length=1, max_length=200)
    confidence: Literal["low", "medium", "high"]
    evidence: EvidenceRef
    status: Literal["active", "superseded"] = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FindingProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    claim: str = Field(min_length=1, max_length=200)
    confidence: Literal["low", "medium", "high"]
    evidence: EvidenceRef


class SupersedeProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(min_length=1)
    replacement: FindingProposal


class DiagnosticState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=0, ge=0)
    context_slots: dict[str, object] = Field(default_factory=dict)
    verified_findings: list[VerifiedFinding] = Field(default_factory=list)
    current_candidates: list[str] = Field(default_factory=list)
    unresolved_constraints: list[str] = Field(default_factory=list)
    validity_concerns: list[str] = Field(default_factory=list)
    rejected_candidates: list[str] = Field(default_factory=list)
    next_step_plan: list[str] = Field(default_factory=list)
    constraint_status: dict[str, Literal["satisfied", "unsatisfied", "unchecked"]] = Field(default_factory=dict)
    answer_confidence: float | None = Field(default=None, ge=0, le=1)
    verified_citations: list[str] = Field(default_factory=list)


class DiagnosticPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1)
    expected_version: int = Field(ge=0)
    context_slots: dict[str, object] = Field(default_factory=dict)
    add_findings: list[FindingProposal] = Field(default_factory=list)
    add_candidates: list[str] = Field(default_factory=list)
    reject_candidates: list[str] = Field(default_factory=list)
    add_validity_concerns: list[str] = Field(default_factory=list)
    add_constraints: list[str] = Field(default_factory=list)
    replace_next_step_plan: list[str] | None = None
    supersede_findings: list[SupersedeProposal] = Field(default_factory=list)
    update_constraint_status: dict[str, Literal["satisfied", "unsatisfied", "unchecked"]] = Field(default_factory=dict)
    set_answer_confidence: float | None = Field(default=None, ge=0, le=1)
    add_citations: list[str] = Field(default_factory=list)


class StateConflictError(ValueError):
    pass


class EvidenceValidationError(ValueError):
    pass


def _append_unique(values: list[str], additions: list[str]) -> None:
    for value in additions:
        if value not in values:
            values.append(value)


def _normalize_text_list(values: list[str], limit: int) -> list[str]:
    normalized: list[str] = []
    for value in values:
        text = value.strip()[:MAX_STATE_TEXT_LENGTH]
        if text and text not in normalized:
            normalized.append(text)
    return normalized[-limit:]


def _finding_priority(finding: VerifiedFinding) -> tuple[int, int, float]:
    confidence_rank = {"high": 2, "medium": 1, "low": 0}[finding.confidence]
    created_at = finding.created_at.timestamp()
    return (1 if finding.status == "active" else 0, confidence_rank, created_at)


def normalize_and_limit_state(state: DiagnosticState) -> DiagnosticState:
    normalized = state.model_copy(deep=True)
    normalized.context_slots = dict(list(normalized.context_slots.items())[:MAX_CANDIDATES])
    normalized.current_candidates = _normalize_text_list(normalized.current_candidates, MAX_CANDIDATES)
    normalized.unresolved_constraints = _normalize_text_list(normalized.unresolved_constraints, MAX_CONSTRAINTS)
    normalized.validity_concerns = _normalize_text_list(normalized.validity_concerns, MAX_CONCERNS)
    normalized.rejected_candidates = _normalize_text_list(normalized.rejected_candidates, MAX_REJECTED)
    normalized.next_step_plan = _normalize_text_list(normalized.next_step_plan, MAX_PLAN_ITEMS)
    normalized.verified_citations = _normalize_text_list(normalized.verified_citations, MAX_CITATIONS)
    normalized.constraint_status = {
        key: value for key, value in list(normalized.constraint_status.items())[:MAX_CONSTRAINTS]
    }
    if len(normalized.verified_findings) > MAX_FINDINGS:
        prioritized = sorted(
            normalized.verified_findings,
            key=_finding_priority,
            reverse=True,
        )[:MAX_FINDINGS]
        keep_ids = {finding.id for finding in prioritized}
        normalized.verified_findings = [
            finding for finding in normalized.verified_findings if finding.id in keep_ids
        ]

    while len(normalized.model_dump_json().encode("utf-8")) > MAX_STATE_BYTES:
        if normalized.verified_findings:
            normalized.verified_findings.pop()
            continue
        removable = (
            normalized.validity_concerns,
            normalized.unresolved_constraints,
            normalized.rejected_candidates,
            normalized.current_candidates,
            normalized.next_step_plan,
        )
        target = next((items for items in removable if items), None)
        if target is None:
            break
        target.pop(0)
    return normalized


def validate_patch_evidence(
    patch: DiagnosticPatch,
    evidence_records: list[EvidenceRecord],
    *,
    request_id: str,
) -> None:
    records = {record.evidence_id: record for record in evidence_records}
    proposals = [*patch.add_findings, *(item.replacement for item in patch.supersede_findings)]
    for proposal in proposals:
        reference = proposal.evidence
        record = records.get(reference.evidence_id)
        if record is None:
            raise EvidenceValidationError(f"evidence not found: {reference.evidence_id}")
        if reference.request_id != request_id or record.request_id != request_id:
            raise EvidenceValidationError(f"evidence request mismatch: {reference.evidence_id}")
        if reference.observation_hash != record.observation_hash:
            raise EvidenceValidationError(f"evidence hash mismatch: {reference.evidence_id}")


def apply_diagnostic_patch(state: DiagnosticState, patch: DiagnosticPatch) -> DiagnosticState:
    if patch.expected_version != state.version:
        raise StateConflictError(
            f"state version conflict: expected {patch.expected_version}, current {state.version}"
        )

    updated = state.model_copy(deep=True)
    updated.context_slots.update(patch.context_slots)

    existing_ids = {finding.id for finding in updated.verified_findings}
    existing_claims = {finding.claim for finding in updated.verified_findings}
    for proposal in patch.add_findings:
        if proposal.id in existing_ids or proposal.claim in existing_claims:
            continue
        updated.verified_findings.append(VerifiedFinding(**proposal.model_dump()))
        existing_ids.add(proposal.id)
        existing_claims.add(proposal.claim)

    _append_unique(updated.current_candidates, patch.add_candidates)
    _append_unique(updated.rejected_candidates, patch.reject_candidates)
    _append_unique(updated.validity_concerns, patch.add_validity_concerns)
    _append_unique(updated.unresolved_constraints, patch.add_constraints)
    _append_unique(updated.verified_citations, patch.add_citations)
    rejected = set(updated.rejected_candidates)
    updated.current_candidates = [item for item in updated.current_candidates if item not in rejected]

    for constraint, status in patch.update_constraint_status.items():
        updated.constraint_status[constraint] = status
    if patch.set_answer_confidence is not None:
        updated.answer_confidence = patch.set_answer_confidence

    if patch.replace_next_step_plan is not None:
        updated.next_step_plan = list(dict.fromkeys(patch.replace_next_step_plan))

    finding_by_id = {finding.id: finding for finding in updated.verified_findings}
    for proposal in patch.supersede_findings:
        current = finding_by_id.get(proposal.finding_id)
        if current is None or current.status != "active":
            continue
        current.status = "superseded"
        replacement = proposal.replacement
        if replacement.id not in finding_by_id:
            new_finding = VerifiedFinding(**replacement.model_dump())
            updated.verified_findings.append(new_finding)
            finding_by_id[new_finding.id] = new_finding

    updated.version += 1
    return normalize_and_limit_state(updated)
