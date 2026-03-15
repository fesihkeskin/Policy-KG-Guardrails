from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from .types import CanonicalPolicyIR, Predicate, Rule


@dataclass
class PolicyKG:
    graph: nx.MultiDiGraph
    policy: CanonicalPolicyIR
    rule_node_ids: dict[str, str]
    condition_node_ids: dict[tuple[str, str], str]
    rule_summaries: dict[str, str]


def render_rule_summary(rule: Rule) -> str:
    pred_parts: list[str] = []
    for pred in rule.predicates.values():
        right = (
            f"{pred.right_scope}.{pred.right_attr}"
            if pred.right_attr is not None and pred.right_scope is not None
            else str(pred.right_value)
        )
        pred_parts.append(f"{pred.left_scope}.{pred.left_attr} {pred.op} {right}")
    actions = ", ".join(rule.actions)
    predicates = " AND ".join(pred_parts) if pred_parts else "TRUE"
    return f"{rule.rule_id}: effect={rule.effect}; actions=[{actions}]; when {predicates}"


def _add_attr_node(graph: nx.MultiDiGraph, scope: str, attr: str, datatype: str) -> str:
    if scope == "subject":
        kind = "SubjectAttr"
    elif scope == "resource":
        kind = "ResourceAttr"
    else:
        kind = "EnvAttr"
    node_id = f"{kind}:{attr}"
    if node_id not in graph:
        graph.add_node(node_id, kind=kind, name=attr, datatype=datatype)
    return node_id


def _add_value_node(graph: nx.MultiDiGraph, attr_name: str, value: Any) -> str:
    node_id = f"AttrValue:{attr_name}:{repr(value)}"
    if node_id not in graph:
        graph.add_node(node_id, kind="AttrValue", attr_name=attr_name, value=value)
    return node_id


def _add_condition_node(
    graph: nx.MultiDiGraph,
    rule_id: str,
    predicate: Predicate,
) -> str:
    node_id = f"Condition:{rule_id}:{predicate.predicate_id}"
    if node_id not in graph:
        graph.add_node(
            node_id,
            kind="Condition",
            id=predicate.predicate_id,
            op=predicate.op,
            left_scope=predicate.left_scope,
            left_attr=predicate.left_attr,
            right_scope=predicate.right_scope,
            right_attr=predicate.right_attr,
            right_value=predicate.right_value,
        )
    return node_id


def compile_policy_kg(policy_ir: CanonicalPolicyIR) -> PolicyKG:
    graph = nx.MultiDiGraph()
    policy_node = f"Policy:{policy_ir.meta.policy_id}"
    graph.add_node(
        policy_node,
        kind="Policy",
        id=policy_ir.meta.policy_id,
        domain=policy_ir.meta.domain,
        version=policy_ir.meta.version,
        dataset=policy_ir.meta.dataset,
    )

    for attr in policy_ir.subject_attrs.values():
        _add_attr_node(graph, "subject", attr.name, attr.datatype)
    for attr in policy_ir.resource_attrs.values():
        _add_attr_node(graph, "resource", attr.name, attr.datatype)
    for attr in policy_ir.environment_attrs.values():
        _add_attr_node(graph, "environment", attr.name, attr.datatype)

    for action in policy_ir.actions.values():
        graph.add_node(f"Action:{action.name}", kind="Action", name=action.name)

    rule_node_ids: dict[str, str] = {}
    condition_node_ids: dict[tuple[str, str], str] = {}
    rule_summaries: dict[str, str] = {}

    for rule in policy_ir.rules:
        rule_node = f"Rule:{rule.rule_id}"
        rule_node_ids[rule.rule_id] = rule_node
        graph.add_node(
            rule_node,
            kind="Rule",
            id=rule.rule_id,
            effect=rule.effect,
            priority=rule.priority,
        )
        graph.add_edge(policy_node, rule_node, label="HAS_RULE")

        for action in rule.actions:
            graph.add_edge(rule_node, f"Action:{action}", label="USES_ACTION")

        expr_node_ids: dict[str, str] = {}
        for expr in rule.expr_nodes.values():
            if expr.op == "PRED":
                pred = rule.predicates[expr.predicate_id or ""]
                cond_node = _add_condition_node(graph, rule.rule_id, pred)
                expr_node_ids[expr.node_id] = cond_node
                condition_node_ids[(rule.rule_id, pred.predicate_id)] = cond_node
            else:
                expr_node = f"Expr:{rule.rule_id}:{expr.node_id}"
                expr_node_ids[expr.node_id] = expr_node
                graph.add_node(expr_node, kind="Expr", id=expr.node_id, op=expr.op)

        root_graph_node = expr_node_ids[rule.root_expr_id]
        graph.add_edge(rule_node, root_graph_node, label="HAS_COND")

        for expr in rule.expr_nodes.values():
            if expr.op == "PRED":
                continue
            parent = expr_node_ids[expr.node_id]
            for child in expr.children:
                child_node = expr_node_ids[child]
                graph.add_edge(parent, child_node, label="CHILD")

        for pred in rule.predicates.values():
            cond_node = condition_node_ids[(rule.rule_id, pred.predicate_id)]
            if pred.left_scope == "subject":
                left_dtype = policy_ir.subject_attrs.get(pred.left_attr, None)
            elif pred.left_scope == "resource":
                left_dtype = policy_ir.resource_attrs.get(pred.left_attr, None)
            else:
                left_dtype = policy_ir.environment_attrs.get(pred.left_attr, None)
            left_node = _add_attr_node(
                graph,
                pred.left_scope,
                pred.left_attr,
                left_dtype.datatype if left_dtype else "string",
            )
            graph.add_edge(cond_node, left_node, label="REF_ATTR", side="left")

            if pred.right_attr is not None and pred.right_scope is not None:
                if pred.right_scope == "subject":
                    right_dtype = policy_ir.subject_attrs.get(pred.right_attr, None)
                elif pred.right_scope == "resource":
                    right_dtype = policy_ir.resource_attrs.get(pred.right_attr, None)
                else:
                    right_dtype = policy_ir.environment_attrs.get(pred.right_attr, None)
                right_node = _add_attr_node(
                    graph,
                    pred.right_scope,
                    pred.right_attr,
                    right_dtype.datatype if right_dtype else "string",
                )
                graph.add_edge(cond_node, right_node, label="REF_ATTR", side="right")
            else:
                right_value = pred.right_value
                if isinstance(right_value, set):
                    for value in sorted(right_value, key=lambda x: str(x)):
                        value_node = _add_value_node(graph, pred.left_attr, value)
                        graph.add_edge(cond_node, value_node, label="REF_VALUE")
                elif right_value is not None:
                    value_node = _add_value_node(graph, pred.left_attr, right_value)
                    graph.add_edge(cond_node, value_node, label="REF_VALUE")

        rule_summaries[rule.rule_id] = render_rule_summary(rule)

    return PolicyKG(
        graph=graph,
        policy=policy_ir,
        rule_node_ids=rule_node_ids,
        condition_node_ids=condition_node_ids,
        rule_summaries=rule_summaries,
    )
