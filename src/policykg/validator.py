from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from .types import CanonicalPolicyIR, PolicyValidationError, Rule


def validate_policy_ir(policy: CanonicalPolicyIR) -> None:
    _validate_rule_ids(policy.rules)
    _validate_actions(policy)
    _validate_rule_graphs(policy.rules)
    _validate_predicate_refs(policy)
    _validate_types(policy)


def _validate_rule_ids(rules: Iterable[Rule]) -> None:
    ids = [r.rule_id for r in rules]
    dupes = [rule_id for rule_id, count in Counter(ids).items() if count > 1]
    if dupes:
        raise PolicyValidationError(f"Duplicate rule IDs detected: {dupes}")


def _validate_actions(policy: CanonicalPolicyIR) -> None:
    known_actions = set(policy.actions.keys())
    for rule in policy.rules:
        missing = [action for action in rule.actions if action not in known_actions]
        if missing:
            raise PolicyValidationError(
                f"Rule {rule.rule_id} references unknown actions: {missing}"
            )


def _validate_rule_graphs(rules: Iterable[Rule]) -> None:
    for rule in rules:
        if rule.root_expr_id not in rule.expr_nodes:
            raise PolicyValidationError(
                f"Rule {rule.rule_id} root expression {rule.root_expr_id} is missing"
            )
        _check_expr_tree_acyclic(rule)


def _check_expr_tree_acyclic(rule: Rule) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node_id: str) -> None:
        if node_id in visiting:
            raise PolicyValidationError(
                f"Cycle detected in expression tree of rule {rule.rule_id} at node {node_id}"
            )
        if node_id in visited:
            return
        visiting.add(node_id)
        node = rule.expr_nodes.get(node_id)
        if node is None:
            raise PolicyValidationError(
                f"Rule {rule.rule_id} references missing expr node {node_id}"
            )
        for child in node.children:
            dfs(child)
        visiting.remove(node_id)
        visited.add(node_id)

    dfs(rule.root_expr_id)


def _validate_predicate_refs(policy: CanonicalPolicyIR) -> None:
    known = {
        "subject": set(policy.subject_attrs.keys()),
        "resource": set(policy.resource_attrs.keys()),
        "environment": set(policy.environment_attrs.keys()),
        "action": {"action"},
    }
    for rule in policy.rules:
        for predicate in rule.predicates.values():
            if predicate.left_attr not in known.get(predicate.left_scope, set()):
                raise PolicyValidationError(
                    f"Rule {rule.rule_id} predicate {predicate.predicate_id} references unknown "
                    f"{predicate.left_scope} attribute '{predicate.left_attr}'"
                )
            if predicate.right_attr is not None and predicate.right_scope is not None:
                if predicate.right_attr not in known.get(predicate.right_scope, set()):
                    raise PolicyValidationError(
                        f"Rule {rule.rule_id} predicate {predicate.predicate_id} references unknown "
                        f"{predicate.right_scope} attribute '{predicate.right_attr}'"
                    )


def _base_type(datatype: str) -> str:
    if datatype.startswith("set[") and datatype.endswith("]"):
        return datatype[4:-1]
    return datatype


def _is_multivalue(datatype: str) -> bool:
    return datatype.startswith("set[") and datatype.endswith("]")


def _value_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    return "string"


def _compatible_literal(attr_datatype: str, literal: Any) -> bool:
    if isinstance(literal, set):
        lit_values = list(literal)
    elif isinstance(literal, (list, tuple)):
        lit_values = list(literal)
    else:
        lit_values = [literal]

    target_base = _base_type(attr_datatype)
    if target_base == "float":
        valid_types = {"int", "float"}
    else:
        valid_types = {target_base}
    return all(_value_type_name(v) in valid_types for v in lit_values)


def _validate_types(policy: CanonicalPolicyIR) -> None:
    scope_defs = {
        "subject": policy.subject_attrs,
        "resource": policy.resource_attrs,
        "environment": policy.environment_attrs,
    }
    for rule in policy.rules:
        for predicate in rule.predicates.values():
            left_def = scope_defs.get(predicate.left_scope, {}).get(predicate.left_attr)
            if left_def is None:
                continue
            if predicate.right_value is not None and not _compatible_literal(
                left_def.datatype, predicate.right_value
            ):
                raise PolicyValidationError(
                    f"Rule {rule.rule_id} predicate {predicate.predicate_id} has incompatible literal "
                    f"for {predicate.left_scope}.{predicate.left_attr}: {predicate.right_value}"
                )
            if predicate.right_attr is not None and predicate.right_scope is not None:
                right_def = scope_defs.get(predicate.right_scope, {}).get(predicate.right_attr)
                if right_def is None:
                    continue
                if _base_type(left_def.datatype) != _base_type(right_def.datatype):
                    raise PolicyValidationError(
                        f"Rule {rule.rule_id} predicate {predicate.predicate_id} compares "
                        f"incompatible types {left_def.datatype} and {right_def.datatype}"
                    )
            if predicate.op in {"CONTAINS", "SUPERSET"} and not _is_multivalue(
                left_def.datatype
            ):
                raise PolicyValidationError(
                    f"Rule {rule.rule_id} predicate {predicate.predicate_id} uses {predicate.op} on "
                    f"non-multivalue attribute {predicate.left_scope}.{predicate.left_attr}"
                )
