from __future__ import annotations

import json
import re
from typing import Any

from .types import Decision, ExplanationClaim, ModelDraft


def _to_decision(value: str | None) -> Decision:
    if value is None:
        return Decision.DENY
    normalized = value.strip().lower()
    if normalized == "permit":
        return Decision.PERMIT
    if normalized == "deny":
        return Decision.DENY
    if normalized in {"insufficientpolicyevidence", "insufficient_policy_evidence", "insufficient"}:
        return Decision.INSUFFICIENT
    return Decision.DENY


def _parse_list_expr(value: str) -> list[str]:
    token = value.strip()
    if token.startswith("[") and token.endswith("]"):
        token = token[1:-1]
    if not token.strip():
        return []
    out: list[str] = []
    for raw in token.split(","):
        piece = raw.strip().strip("'\"")
        if piece:
            out.append(piece)
    return out


def _parse_claim_lines(text: str) -> list[ExplanationClaim]:
    claims: list[ExplanationClaim] = []
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\s+(.+?)\s+supports\s*=\s*(.+)$", line.strip())
        if not m:
            continue
        idx = int(m.group(1))
        claim_text = m.group(2).strip()
        supports = [token for token in re.split(r"[\s,]+", m.group(3).strip()) if token]
        claims.append(ExplanationClaim(index=idx, text=claim_text, supports=supports))
    return claims


def parse_response_contract(
    text: str,
    *,
    default_policy_id: str,
    default_subject: dict[str, Any],
    default_resource: dict[str, Any],
    default_action: str | None,
    default_environment: dict[str, Any],
) -> ModelDraft:
    stripped = text.strip()

    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            claims_payload = payload.get("claims", [])
            claims = [
                ExplanationClaim(
                    index=int(item.get("index", idx + 1)),
                    text=str(item.get("text", "")),
                    supports=[str(tok) for tok in item.get("supports", [])],
                )
                for idx, item in enumerate(claims_payload)
            ]
            decisive_predicates = {
                str(k): [str(x) for x in v] for k, v in payload.get("decisive_predicates", {}).items()
            }
            return ModelDraft(
                decision=_to_decision(payload.get("decision")),
                confidence=str(payload.get("confidence", "Medium")),
                policy_id=str(payload.get("policy_id", default_policy_id)),
                request_subject=dict(payload.get("request_subject", default_subject)),
                request_resource=dict(payload.get("request_resource", default_resource)),
                request_action=payload.get("request_action", default_action),
                request_environment=dict(payload.get("request_environment", default_environment)),
                evidence_subgraphs=[str(x) for x in payload.get("evidence_subgraphs", [])],
                decisive_rules=[str(x) for x in payload.get("decisive_rules", [])],
                decisive_predicates=decisive_predicates,
                claims=claims,
                citations=[str(x) for x in payload.get("citations", [])],
                raw_text=text,
            )
        except json.JSONDecodeError:
            pass

    decision_match = re.search(r"decision\s*=\s*([^\n]+)", text, flags=re.IGNORECASE)
    confidence_match = re.search(r"confidence\s*=\s*([^\n]+)", text, flags=re.IGNORECASE)
    policy_match = re.search(r"policy_id\s*=\s*([^\n]+)", text, flags=re.IGNORECASE)
    action_match = re.search(r"request_action\s*=\s*([^\n]+)", text, flags=re.IGNORECASE)
    decisive_rules_match = re.search(r"decisive_rules\s*=\s*([^\n]+)", text, flags=re.IGNORECASE)
    evidence_match = re.search(r"evidence_subgraphs\s*=\s*([^\n]+)", text, flags=re.IGNORECASE)
    citations_match = re.search(r"citations\s*=\s*([^\n]+)", text, flags=re.IGNORECASE)

    decisive_rules = _parse_list_expr(decisive_rules_match.group(1)) if decisive_rules_match else []
    evidence_subgraphs = _parse_list_expr(evidence_match.group(1)) if evidence_match else []
    citations = _parse_list_expr(citations_match.group(1)) if citations_match else []
    claims = _parse_claim_lines(text)

    decisive_predicates: dict[str, list[str]] = {}
    for line in text.splitlines():
        m = re.match(
            r"^\s*\{?\s*rule\s*=\s*([^,\s]+)[,\s]+predicate_ids\s*=\s*([^\}]+)\}?\s*$",
            line.strip(),
            flags=re.IGNORECASE,
        )
        if not m:
            continue
        decisive_predicates[m.group(1).strip()] = _parse_list_expr(m.group(2).strip())

    return ModelDraft(
        decision=_to_decision(decision_match.group(1).strip() if decision_match else None),
        confidence=confidence_match.group(1).strip() if confidence_match else "Medium",
        policy_id=policy_match.group(1).strip() if policy_match else default_policy_id,
        request_subject=default_subject,
        request_resource=default_resource,
        request_action=action_match.group(1).strip() if action_match else default_action,
        request_environment=default_environment,
        evidence_subgraphs=evidence_subgraphs,
        decisive_rules=decisive_rules,
        decisive_predicates=decisive_predicates,
        claims=claims,
        citations=citations,
        raw_text=text,
    )


def format_contract_request(
    *,
    policy_id: str,
    subject: dict[str, Any],
    resource: dict[str, Any],
    action: str | None,
    environment: dict[str, Any],
    evidence_ids: list[str],
    rule_ids: list[str],
) -> str:
    return (
        "Return output using this exact structure:\n"
        "Decision\n"
        "decision = Permit | Deny | InsufficientPolicyEvidence\n"
        "confidence = High | Medium | Low\n\n"
        "PolicyContext\n"
        f"policy_id = {policy_id}\n"
        f"request_subject = {subject}\n"
        f"request_resource = {resource}\n"
        f"request_action = {action}\n"
        f"request_environment = {environment}\n\n"
        "Evidence\n"
        f"evidence_subgraphs = {evidence_ids}\n"
        f"decisive_rules = {rule_ids}\n"
        "decisive_predicates = [{ rule = <R>, predicate_ids = [<C1>, <C2>] }]\n\n"
        "Explanation\n"
        "1 <sentence> supports = <SGx> <Ry> <Cz>\n"
        "2 <sentence> supports = <SGx> <Ry> <Cw>\n\n"
        "Citations\n"
        "citations = [<SGx>, ...]\n"
    )
