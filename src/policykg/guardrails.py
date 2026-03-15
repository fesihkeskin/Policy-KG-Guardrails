from __future__ import annotations

from typing import Iterable

from .evaluator import evaluate
from .kg import PolicyKG
from .llm import LLMClient
from .response_contract import parse_response_contract
from .types import (
    Decision,
    DecisionTrace,
    EvidenceSubgraph,
    ExplanationClaim,
    GuardrailedResponse,
    ModelDraft,
    RequestContext,
)


def _flatten_allowed_predicates(decisive_predicates: dict[str, list[str]]) -> set[str]:
    out: set[str] = set()
    for items in decisive_predicates.values():
        out.update(items)
    return out


def _align_claims(
    claims: list[ExplanationClaim],
    *,
    allowed_sg: set[str],
    allowed_rules: set[str],
    allowed_predicates: set[str],
) -> tuple[list[ExplanationClaim], int]:
    aligned_claims: list[ExplanationClaim] = []
    unsupported = 0

    for claim in claims:
        reason = ""
        supports = set(claim.supports)
        if not supports:
            claim.aligned = False
            claim.alignment_reason = "empty-support"
            unsupported += 1
            aligned_claims.append(claim)
            continue

        bad_tokens: list[str] = []
        for token in supports:
            if token.startswith("SG") and token not in allowed_sg:
                bad_tokens.append(token)
            if token.startswith("R") and token not in allowed_rules:
                bad_tokens.append(token)
            if token.startswith("P") and token not in allowed_predicates:
                bad_tokens.append(token)
            if token.startswith("C") and token not in allowed_predicates:
                bad_tokens.append(token)

        if bad_tokens:
            claim.aligned = False
            reason = f"unsupported-tokens:{','.join(sorted(set(bad_tokens)))}"
            claim.alignment_reason = reason
            unsupported += 1
        else:
            claim.aligned = True
            claim.alignment_reason = "aligned"
        aligned_claims.append(claim)

    return aligned_claims, unsupported


def _decision_trace_predicates(trace: DecisionTrace) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for rule_id, preds in trace.decisive_predicates.items():
        out[rule_id] = sorted(preds.keys())
    return out


def _build_revision_prompt(
    draft: ModelDraft,
    trace: DecisionTrace,
    evidence: list[EvidenceSubgraph],
    unsupported_claims: list[ExplanationClaim],
) -> str:
    allowed_sg = [item.evidence_id for item in evidence]
    allowed_rules = trace.decisive_rules
    allowed_preds = sorted({pid for pmap in trace.decisive_predicates.values() for pid in pmap.keys()})

    issues = []
    if draft.decision != trace.decision:
        issues.append(f"Decision mismatch: draft={draft.decision.value} expected={trace.decision.value}")
    if unsupported_claims:
        issues.append("Unsupported claims: " + "; ".join(c.text for c in unsupported_claims))

    return (
        "Revise the response to satisfy policy guardrails.\n"
        + "\n".join(issues)
        + "\nHard constraints:\n"
        + f"- decision must be {trace.decision.value}\n"
        + f"- decisive_rules must only use {allowed_rules}\n"
        + f"- supports tokens must only use subgraphs {allowed_sg}\n"
        + f"- supports predicate tokens must only use {allowed_preds}\n"
        + "- do not invent new rules, attributes, or citations\n"
    )


def verify_and_revise(
    draft: ModelDraft,
    decision_trace: DecisionTrace,
    evidence: list[EvidenceSubgraph],
    llm_client: LLMClient | None = None,
    allow_revise: bool = True,
) -> GuardrailedResponse:
    evidence_ids = {item.evidence_id for item in evidence}
    allowed_rules = set(decision_trace.decisive_rules)
    allowed_predicates_map = _decision_trace_predicates(decision_trace)
    allowed_predicates = _flatten_allowed_predicates(allowed_predicates_map)

    current = draft
    was_overridden = False
    revised = False

    if current.decision != decision_trace.decision:
        was_overridden = True

    aligned_claims, unsupported_count = _align_claims(
        current.claims,
        allowed_sg=evidence_ids,
        allowed_rules=allowed_rules,
        allowed_predicates=allowed_predicates,
    )

    unsupported_claims = [claim for claim in aligned_claims if not claim.aligned]

    if allow_revise and llm_client is not None and (was_overridden or unsupported_claims):
        revised = True
        revision_prompt = _build_revision_prompt(current, decision_trace, evidence, unsupported_claims)
        revised_text = llm_client.generate(
            system_prompt="You are a policy explanation reviser.",
            user_prompt=revision_prompt,
            temperature=0.0,
            top_p=1.0,
            max_tokens=600,
        )
        revised_draft = parse_response_contract(
            revised_text,
            default_policy_id=current.policy_id,
            default_subject=current.request_subject,
            default_resource=current.request_resource,
            default_action=current.request_action,
            default_environment=current.request_environment,
        )
        current = revised_draft
        aligned_claims, unsupported_count = _align_claims(
            current.claims,
            allowed_sg=evidence_ids,
            allowed_rules=allowed_rules,
            allowed_predicates=allowed_predicates,
        )

    final_claims = [claim for claim in aligned_claims if claim.aligned]
    final_citations = [cite for cite in current.citations if cite in evidence_ids]

    return GuardrailedResponse(
        decision=decision_trace.decision if was_overridden else current.decision,
        confidence=current.confidence,
        policy_id=current.policy_id,
        request_subject=current.request_subject,
        request_resource=current.request_resource,
        request_action=current.request_action,
        request_environment=current.request_environment,
        evidence_subgraphs=[item.evidence_id for item in evidence],
        decisive_rules=decision_trace.decisive_rules,
        decisive_predicates=allowed_predicates_map,
        claims=final_claims,
        citations=final_citations,
        missing_attributes=decision_trace.missing_attributes,
        was_overridden=was_overridden,
        revised=revised,
        unsupported_claim_count=unsupported_count,
        notes=list(decision_trace.notes),
    )


def run_guardrailed_query(
    policy_kg: PolicyKG,
    query: str,
    request_context: RequestContext,
    draft: ModelDraft,
    evidence: list[EvidenceSubgraph],
    llm_client: LLMClient | None = None,
) -> GuardrailedResponse:
    trace = evaluate(policy_kg, request_context)
    return verify_and_revise(draft, trace, evidence, llm_client=llm_client, allow_revise=True)
