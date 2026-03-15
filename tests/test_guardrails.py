from __future__ import annotations

from policykg.guardrails import verify_and_revise
from policykg.llm import ScriptedLLMClient
from policykg.types import (
    Decision,
    DecisionTrace,
    EvidenceSubgraph,
    ExplanationClaim,
    ModelDraft,
    TriValue,
)


def test_guardrails_override_and_revision() -> None:
    draft = ModelDraft(
        decision=Decision.PERMIT,
        confidence="High",
        policy_id="healthcare",
        request_subject={"uid": "u1"},
        request_resource={"rid": "r1"},
        request_action="read",
        request_environment={},
        evidence_subgraphs=["SG9"],
        decisive_rules=["R9"],
        decisive_predicates={"R9": ["P9"]},
        claims=[ExplanationClaim(index=1, text="Unsupported claim", supports=["SG9", "R9", "P9"])],
        citations=["SG9"],
        raw_text="",
    )

    trace = DecisionTrace(
        decision=Decision.DENY,
        decisive_rules=["R1"],
        decisive_predicates={"R1": {"P1": TriValue.TRUE}},
        rule_results={"R1": TriValue.TRUE},
        missing_attributes=[],
        notes=[],
    )

    evidence = [
        EvidenceSubgraph(
            evidence_id="SG1",
            rule_ids=("R1",),
            node_ids=("Rule:R1",),
            edges=(("Rule:R1", "HAS_COND", "Condition:R1:P1"),),
            score=1.0,
        )
    ]

    revised_text = """Decision

decision = Deny
confidence = High

PolicyContext
policy_id = healthcare
request_subject = {'uid': 'u1'}
request_resource = {'rid': 'r1'}
request_action = read
request_environment = {}

Evidence
evidence_subgraphs = [SG1]
decisive_rules = [R1]
decisive_predicates = [{ rule = R1, predicate_ids = [P1] }]

Explanation
1 Rule R1 blocks this request supports = SG1 R1 P1

Citations
citations = [SG1]
"""

    llm = ScriptedLLMClient(responses=[revised_text])
    out = verify_and_revise(draft, trace, evidence, llm_client=llm, allow_revise=True)

    assert out.decision == Decision.DENY
    assert out.was_overridden is True
    assert out.revised is True
    assert out.unsupported_claim_count == 0
    assert len(out.claims) == 1
    assert out.claims[0].aligned is True
    assert out.citations == ["SG1"]


def test_guardrails_without_revision_drops_unsupported_claims() -> None:
    draft = ModelDraft(
        decision=Decision.DENY,
        confidence="Medium",
        policy_id="healthcare",
        request_subject={},
        request_resource={},
        request_action="read",
        request_environment={},
        evidence_subgraphs=["SG1"],
        decisive_rules=["R1"],
        decisive_predicates={"R1": ["P1"]},
        claims=[ExplanationClaim(index=1, text="Bad", supports=["SG2", "R2", "P2"])],
        citations=["SG2"],
        raw_text="",
    )

    trace = DecisionTrace(
        decision=Decision.DENY,
        decisive_rules=["R1"],
        decisive_predicates={"R1": {"P1": TriValue.TRUE}},
        rule_results={"R1": TriValue.TRUE},
    )

    evidence = [
        EvidenceSubgraph(
            evidence_id="SG1",
            rule_ids=("R1",),
            node_ids=("Rule:R1",),
            edges=(),
            score=1.0,
        )
    ]

    out = verify_and_revise(draft, trace, evidence, llm_client=None, allow_revise=False)
    assert out.decision == Decision.DENY
    assert len(out.claims) == 0
    assert out.unsupported_claim_count == 1
    assert out.citations == []
