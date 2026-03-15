from __future__ import annotations

from pathlib import Path

from policykg.kg import compile_policy_kg
from policykg.parser import parse_abac


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "examples" / "healthcare.abac"
README_PATH = ROOT / "examples" / "healthcare_README.md"


def test_parse_healthcare_abac_counts() -> None:
    policy = parse_abac(POLICY_PATH, readme_path=README_PATH)

    assert policy.meta.policy_id == "healthcare"
    assert "healthcare" in policy.meta.domain
    assert len(policy.users) == 21
    assert len(policy.resources) == 16
    assert len(policy.rules) == 6
    assert set(policy.actions) == {"addItem", "addNote", "read"}

    rule6 = next(rule for rule in policy.rules if rule.rule_id == "R6")
    ops = {pred.op for pred in rule6.predicates.values()}
    assert "SUPERSET" in ops
    assert "CONTAINS" in ops


def test_compile_policy_kg_has_schema_edges() -> None:
    policy = parse_abac(POLICY_PATH, readme_path=README_PATH)
    policy_kg = compile_policy_kg(policy)
    graph = policy_kg.graph

    rule_nodes = [n for n, data in graph.nodes(data=True) if data.get("kind") == "Rule"]
    assert len(rule_nodes) == 6

    has_rule_edges = [
        (u, v, d)
        for u, v, d in graph.edges(data=True)
        if d.get("label") == "HAS_RULE"
    ]
    assert len(has_rule_edges) == 6

    condition_nodes = [n for n, data in graph.nodes(data=True) if data.get("kind") == "Condition"]
    assert len(condition_nodes) >= 6

    ref_attr_edges = [
        (u, v, d)
        for u, v, d in graph.edges(data=True)
        if d.get("label") == "REF_ATTR"
    ]
    assert len(ref_attr_edges) >= 10
