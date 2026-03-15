from __future__ import annotations

from pathlib import Path

from policykg.evaluator import evaluate
from policykg.kg import compile_policy_kg
from policykg.parser import parse_abac
from policykg.types import Decision, RequestContext


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "examples" / "healthcare.abac"
README_PATH = ROOT / "examples" / "healthcare_README.md"


def _ctx(policy, uid: str, rid: str, action: str) -> RequestContext:
    return RequestContext(
        subject=dict(policy.users[uid]),
        resource=dict(policy.resources[rid]),
        action=action,
        environment={},
    )


def test_healthcare_permit_and_deny_paths() -> None:
    policy = parse_abac(POLICY_PATH, readme_path=README_PATH)
    kg = compile_policy_kg(policy)

    permit_ctx = _ctx(policy, "oncNurse1", "oncPat1HR", "addItem")
    permit_trace = evaluate(kg, permit_ctx)
    assert permit_trace.decision == Decision.PERMIT

    deny_ctx = _ctx(policy, "oncNurse1", "carPat1HR", "addItem")
    deny_trace = evaluate(kg, deny_ctx)
    assert deny_trace.decision == Decision.DENY


def test_healthcare_author_and_team_rules() -> None:
    policy = parse_abac(POLICY_PATH, readme_path=README_PATH)
    kg = compile_policy_kg(policy)

    author_ctx = _ctx(policy, "doc1", "oncPat2oncItem", "read")
    author_trace = evaluate(kg, author_ctx)
    assert author_trace.decision == Decision.PERMIT

    team_ctx = _ctx(policy, "oncDoc1", "oncPat1oncItem", "read")
    team_trace = evaluate(kg, team_ctx)
    assert team_trace.decision == Decision.PERMIT


def test_insufficient_when_required_attribute_missing() -> None:
    policy = parse_abac(POLICY_PATH, readme_path=README_PATH)
    kg = compile_policy_kg(policy)

    ctx = _ctx(policy, "oncNurse1", "oncPat1HR", "addItem")
    ctx.subject["ward"] = ctx.unknown_token
    trace = evaluate(kg, ctx)

    assert trace.decision == Decision.INSUFFICIENT
    assert "subject.ward" in trace.missing_attributes


def test_insufficient_when_action_missing() -> None:
    policy = parse_abac(POLICY_PATH, readme_path=README_PATH)
    kg = compile_policy_kg(policy)

    ctx = RequestContext(subject={}, resource={}, action=None, environment={})
    trace = evaluate(kg, ctx)
    assert trace.decision == Decision.INSUFFICIENT
    assert "action" in trace.missing_attributes
