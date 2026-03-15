from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .kg import PolicyKG
from .types import AttrDef, Decision, DecisionTrace, ExprNode, Predicate, RequestContext, Rule, TriValue


@dataclass
class _RuleEval:
    value: TriValue
    predicate_values: dict[str, TriValue]
    missing_attrs: set[str]


def _to_set(value: Any) -> set[Any]:
    if isinstance(value, set):
        return set(value)
    if isinstance(value, (list, tuple)):
        return set(value)
    return {value}


def _truth_and(values: list[TriValue]) -> TriValue:
    if any(v == TriValue.FALSE for v in values):
        return TriValue.FALSE
    if all(v == TriValue.TRUE for v in values):
        return TriValue.TRUE
    return TriValue.UNKNOWN


def _truth_or(values: list[TriValue]) -> TriValue:
    if any(v == TriValue.TRUE for v in values):
        return TriValue.TRUE
    if all(v == TriValue.FALSE for v in values):
        return TriValue.FALSE
    return TriValue.UNKNOWN


def _truth_not(value: TriValue) -> TriValue:
    if value == TriValue.TRUE:
        return TriValue.FALSE
    if value == TriValue.FALSE:
        return TriValue.TRUE
    return TriValue.UNKNOWN


def _resolve_attr(
    ctx: RequestContext,
    scope: str,
    attr: str,
    attr_defs: dict[str, dict[str, AttrDef]],
) -> tuple[bool, Any]:
    if scope == "subject":
        source = ctx.subject
    elif scope == "resource":
        source = ctx.resource
    elif scope == "environment":
        source = ctx.environment
    elif scope == "action":
        source = {"action": ctx.action}
    else:
        source = {}

    if attr not in source:
        attr_def = attr_defs.get(scope, {}).get(attr)
        if attr_def is None:
            return False, None
        # In ABAC datasets, an omitted attribute for an entity usually means
        # that attribute is not applicable (e.g., no teams for nurses), not
        # that user input is incomplete.
        if attr_def.multivalue:
            return True, set()
        return True, None
    value = source[attr]
    if value == ctx.unknown_token:
        return False, None
    return True, value


def evaluate_predicate(
    predicate: Predicate,
    ctx: RequestContext,
    attr_defs: dict[str, dict[str, AttrDef]],
) -> tuple[TriValue, set[str]]:
    missing: set[str] = set()

    left_ok, left_value = _resolve_attr(ctx, predicate.left_scope, predicate.left_attr, attr_defs)
    if not left_ok:
        missing.add(f"{predicate.left_scope}.{predicate.left_attr}")
        return TriValue.UNKNOWN, missing

    if predicate.right_attr is not None and predicate.right_scope is not None:
        right_ok, right_value = _resolve_attr(
            ctx, predicate.right_scope, predicate.right_attr, attr_defs
        )
        if not right_ok:
            missing.add(f"{predicate.right_scope}.{predicate.right_attr}")
            return TriValue.UNKNOWN, missing
    else:
        right_value = predicate.right_value

    op = predicate.op.upper()

    if op == "EQ":
        return (TriValue.TRUE if left_value == right_value else TriValue.FALSE), missing
    if op == "NEQ":
        return (TriValue.TRUE if left_value != right_value else TriValue.FALSE), missing
    if op == "IN":
        rhs = _to_set(right_value)
        if isinstance(left_value, (set, list, tuple)):
            result = len(_to_set(left_value).intersection(rhs)) > 0
        else:
            result = left_value in rhs
        return (TriValue.TRUE if result else TriValue.FALSE), missing
    if op == "NIN":
        rhs = _to_set(right_value)
        if isinstance(left_value, (set, list, tuple)):
            result = len(_to_set(left_value).intersection(rhs)) == 0
        else:
            result = left_value not in rhs
        return (TriValue.TRUE if result else TriValue.FALSE), missing
    if op == "CONTAINS":
        left_set = _to_set(left_value)
        right_set = _to_set(right_value)
        result = right_set.issubset(left_set)
        return (TriValue.TRUE if result else TriValue.FALSE), missing
    if op == "SUPERSET":
        left_set = _to_set(left_value)
        right_set = _to_set(right_value)
        result = left_set.issuperset(right_set)
        return (TriValue.TRUE if result else TriValue.FALSE), missing
    if op == "LT":
        return (TriValue.TRUE if left_value < right_value else TriValue.FALSE), missing
    if op == "LE":
        return (TriValue.TRUE if left_value <= right_value else TriValue.FALSE), missing
    if op == "GT":
        return (TriValue.TRUE if left_value > right_value else TriValue.FALSE), missing
    if op == "GE":
        return (TriValue.TRUE if left_value >= right_value else TriValue.FALSE), missing
    if op == "MATCH":
        result = str(right_value) in str(left_value)
        return (TriValue.TRUE if result else TriValue.FALSE), missing

    return TriValue.UNKNOWN, missing


def _eval_expr(
    rule: Rule,
    node_id: str,
    ctx: RequestContext,
    attr_defs: dict[str, dict[str, AttrDef]],
    pred_values: dict[str, TriValue],
    missing: set[str],
) -> TriValue:
    node = rule.expr_nodes[node_id]
    if node.op == "PRED":
        if node.predicate_id is None:
            return TriValue.UNKNOWN
        pred = rule.predicates[node.predicate_id]
        value, pred_missing = evaluate_predicate(pred, ctx, attr_defs)
        pred_values[pred.predicate_id] = value
        missing.update(pred_missing)
        return value

    child_values = [
        _eval_expr(rule, child, ctx, attr_defs, pred_values, missing)
        for child in node.children
    ]

    op = node.op.upper()
    if op == "AND":
        if not child_values:
            return TriValue.TRUE
        return _truth_and(child_values)
    if op == "OR":
        if not child_values:
            return TriValue.FALSE
        return _truth_or(child_values)
    if op == "NOT":
        if not child_values:
            return TriValue.UNKNOWN
        return _truth_not(child_values[0])
    return TriValue.UNKNOWN


def _evaluate_rule(
    rule: Rule,
    ctx: RequestContext,
    attr_defs: dict[str, dict[str, AttrDef]],
) -> _RuleEval:
    pred_values: dict[str, TriValue] = {}
    missing: set[str] = set()
    value = _eval_expr(rule, rule.root_expr_id, ctx, attr_defs, pred_values, missing)
    return _RuleEval(value=value, predicate_values=pred_values, missing_attrs=missing)


def evaluate(policy_kg: PolicyKG, request_context: RequestContext) -> DecisionTrace:
    policy = policy_kg.policy
    notes: list[str] = []
    attr_defs = {
        "subject": policy.subject_attrs,
        "resource": policy.resource_attrs,
        "environment": policy.environment_attrs,
    }

    if request_context.action is None or request_context.action == request_context.unknown_token:
        return DecisionTrace(
            decision=Decision.INSUFFICIENT,
            decisive_rules=[],
            decisive_predicates={},
            rule_results={},
            missing_attributes=["action"],
            combining=policy.meta.combining,
            notes=["Action is missing"],
        )

    candidate_rules = [
        rule for rule in policy.rules if request_context.action in set(rule.actions)
    ]
    if not candidate_rules:
        notes.append(f"No rules match action '{request_context.action}'")
        return DecisionTrace(
            decision=Decision.DENY,
            decisive_rules=[],
            decisive_predicates={},
            rule_results={},
            missing_attributes=[],
            combining=policy.meta.combining,
            notes=notes,
        )

    rule_results: dict[str, TriValue] = {}
    rule_predicates: dict[str, dict[str, TriValue]] = {}
    rule_missing: dict[str, set[str]] = {}

    true_permit: list[str] = []
    true_deny: list[str] = []
    unknown_rules: list[str] = []

    for rule in candidate_rules:
        rr = _evaluate_rule(rule, request_context, attr_defs)
        rule_results[rule.rule_id] = rr.value
        rule_predicates[rule.rule_id] = rr.predicate_values
        rule_missing[rule.rule_id] = rr.missing_attrs

        if rr.value == TriValue.TRUE:
            if rule.effect.lower() == "deny":
                true_deny.append(rule.rule_id)
            else:
                true_permit.append(rule.rule_id)
        elif rr.value == TriValue.UNKNOWN:
            unknown_rules.append(rule.rule_id)

    if true_deny:
        decisive = sorted(true_deny)
        return DecisionTrace(
            decision=Decision.DENY,
            decisive_rules=decisive,
            decisive_predicates={rid: rule_predicates[rid] for rid in decisive},
            rule_results=rule_results,
            missing_attributes=[],
            combining=policy.meta.combining,
            notes=notes,
        )

    if true_permit:
        decisive = sorted(true_permit)
        return DecisionTrace(
            decision=Decision.PERMIT,
            decisive_rules=decisive,
            decisive_predicates={rid: rule_predicates[rid] for rid in decisive},
            rule_results=rule_results,
            missing_attributes=[],
            combining=policy.meta.combining,
            notes=notes,
        )

    if unknown_rules:
        unknown_sorted = sorted(unknown_rules, key=lambda rid: (len(rule_missing[rid]), rid))
        chosen = unknown_sorted[0]
        missing_attrs = sorted(rule_missing[chosen])
        notes.append("Evaluation requires additional attributes")
        return DecisionTrace(
            decision=Decision.INSUFFICIENT,
            decisive_rules=[chosen],
            decisive_predicates={chosen: rule_predicates[chosen]},
            rule_results=rule_results,
            missing_attributes=missing_attrs,
            combining=policy.meta.combining,
            notes=notes,
        )

    notes.append("Default deny: no applicable permit/deny rule evaluated to TRUE")
    return DecisionTrace(
        decision=Decision.DENY,
        decisive_rules=[],
        decisive_predicates={},
        rule_results=rule_results,
        missing_attributes=[],
        combining=policy.meta.combining,
        notes=notes,
    )
