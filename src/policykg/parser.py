from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from .types import (
    ActionDef,
    AttrDef,
    CanonicalPolicyIR,
    ExprNode,
    PolicyMeta,
    PolicyParseError,
    Predicate,
    Rule,
)
from .utils import infer_datatype, parse_value, split_top_level
from .validator import validate_policy_ir


_USER_RE = re.compile(r"^userAttrib\(\s*([^,\s]+)\s*,\s*(.*)\)\s*$")
_RESOURCE_RE = re.compile(r"^resourceAttrib\(\s*([^,\s]+)\s*,\s*(.*)\)\s*$")
_RULE_RE = re.compile(r"^rule\(\s*(.*)\)\s*$")


def parse_abac(path: str | Path, readme_path: str | Path | None = None) -> CanonicalPolicyIR:
    policy_path = Path(path)
    if not policy_path.exists():
        raise PolicyParseError(f"Policy file not found: {policy_path}")
    raw = policy_path.read_text(encoding="utf-8")

    if _looks_like_json(raw):
        policy = _parse_json_abac(raw, policy_path.stem)
    else:
        policy = _parse_custom_abac(raw, policy_path.stem)

    if readme_path is not None:
        readme = Path(readme_path)
        if readme.exists():
            _apply_readme_meta(policy, readme.read_text(encoding="utf-8"))

    validate_policy_ir(policy)
    return policy


def parse_abac_text(text: str, policy_id: str = "inline") -> CanonicalPolicyIR:
    if _looks_like_json(text):
        policy = _parse_json_abac(text, policy_id)
    else:
        policy = _parse_custom_abac(text, policy_id)
    validate_policy_ir(policy)
    return policy


def _looks_like_json(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("{")


def _strip_comment(line: str) -> str:
    if "#" in line:
        return line.split("#", 1)[0].strip()
    return line.strip()


def _parse_attr_blob(blob: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for item in split_top_level(blob, ","):
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            raise PolicyParseError(f"Invalid attribute token: '{token}'")
        key, value = token.split("=", 1)
        attrs[key.strip()] = parse_value(value.strip())
    return attrs


def _split_conditions(part: str) -> list[str]:
    if not part.strip():
        return []
    return [token.strip() for token in split_top_level(part, ",") if token.strip()]


def _parse_set_literal(text: str) -> set[Any]:
    token = text.strip()
    if token.startswith("{") and token.endswith("}"):
        return set(parse_value(token))
    return set(parse_value("{" + token + "}"))


def _parse_atomic_condition(
    atom: str,
    *,
    default_left_scope: str,
    default_right_scope: str,
    pred_idx: int,
) -> Predicate:
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\{[^}]*\})\s*$", atom)
    if m:
        attr = m.group(1)
        values = _parse_set_literal(m.group(2))
        return Predicate(
            predicate_id=f"P{pred_idx}",
            op="IN",
            left_scope=default_left_scope,
            left_attr=attr,
            right_value=values,
        )

    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*$", atom)
    if m:
        return Predicate(
            predicate_id=f"P{pred_idx}",
            op="EQ",
            left_scope=default_left_scope,
            left_attr=m.group(1),
            right_scope=default_right_scope,
            right_attr=m.group(2),
        )

    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\]\s*([A-Za-z_][A-Za-z0-9_]*)\s*$", atom)
    if m:
        return Predicate(
            predicate_id=f"P{pred_idx}",
            op="CONTAINS",
            left_scope=default_left_scope,
            left_attr=m.group(1),
            right_scope=default_right_scope,
            right_attr=m.group(2),
        )

    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*>\s*([A-Za-z_][A-Za-z0-9_]*)\s*$", atom)
    if m:
        return Predicate(
            predicate_id=f"P{pred_idx}",
            op="SUPERSET",
            left_scope=default_left_scope,
            left_attr=m.group(1),
            right_scope=default_right_scope,
            right_attr=m.group(2),
        )

    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", atom)
    if m:
        return Predicate(
            predicate_id=f"P{pred_idx}",
            op="EQ",
            left_scope=default_left_scope,
            left_attr=m.group(1),
            right_value=parse_value(m.group(2).strip()),
        )

    raise PolicyParseError(f"Unsupported rule condition syntax: '{atom}'")


def _parse_actions(actions_part: str) -> tuple[str, ...]:
    m = re.search(r"\{([^}]*)\}", actions_part)
    if not m:
        token = actions_part.strip()
        if not token:
            raise PolicyParseError("Rule actions segment is empty")
        return (token,)
    payload = m.group(1).strip()
    actions = [tok for tok in re.split(r"[\s,]+", payload) if tok]
    if not actions:
        raise PolicyParseError("Rule actions segment has no actions")
    return tuple(actions)


def _build_rule(rule_idx: int, body: str) -> Rule:
    parts = [p.strip() for p in split_top_level(body, ";")]
    if len(parts) < 4:
        raise PolicyParseError(
            f"Invalid rule format (expected 4 ';' sections): '{body}'"
        )
    # Some ABAC files may contain extra ';' in comments/copies.
    if len(parts) > 4:
        cons_part = ";".join(parts[3:]).strip()
        parts = parts[:3] + [cons_part]

    sub_part, res_part, action_part, cons_part = parts

    predicates: dict[str, Predicate] = {}
    pred_idx = 1

    for atom in _split_conditions(sub_part):
        pred = _parse_atomic_condition(
            atom,
            default_left_scope="subject",
            default_right_scope="resource",
            pred_idx=pred_idx,
        )
        predicates[pred.predicate_id] = pred
        pred_idx += 1

    for atom in _split_conditions(res_part):
        pred = _parse_atomic_condition(
            atom,
            default_left_scope="resource",
            default_right_scope="subject",
            pred_idx=pred_idx,
        )
        predicates[pred.predicate_id] = pred
        pred_idx += 1

    for atom in _split_conditions(cons_part):
        pred = _parse_atomic_condition(
            atom,
            default_left_scope="subject",
            default_right_scope="resource",
            pred_idx=pred_idx,
        )
        predicates[pred.predicate_id] = pred
        pred_idx += 1

    expr_nodes: dict[str, ExprNode] = {}
    leaf_ids: list[str] = []
    for idx, predicate_id in enumerate(predicates.keys(), start=1):
        node_id = f"E{idx}"
        expr_nodes[node_id] = ExprNode(node_id=node_id, op="PRED", predicate_id=predicate_id)
        leaf_ids.append(node_id)

    root_id = "EROOT"
    expr_nodes[root_id] = ExprNode(node_id=root_id, op="AND", children=tuple(leaf_ids))

    return Rule(
        rule_id=f"R{rule_idx}",
        effect="Permit",
        actions=_parse_actions(action_part),
        root_expr_id=root_id,
        expr_nodes=expr_nodes,
        predicates=predicates,
        source=body,
    )


def _infer_attr_defs(
    entities: dict[str, dict[str, Any]],
    scope: str,
    rule_preds: list[Predicate],
    *,
    include_rule_refs: bool = False,
) -> dict[str, AttrDef]:
    observed: dict[str, list[Any]] = {}
    for row in entities.values():
        for key, value in row.items():
            observed.setdefault(key, []).append(value)

    if include_rule_refs:
        for pred in rule_preds:
            if pred.left_scope == scope:
                observed.setdefault(pred.left_attr, [])
            if pred.right_scope == scope and pred.right_attr is not None:
                observed.setdefault(pred.right_attr, [])

    out: dict[str, AttrDef] = {}
    for key, values in sorted(observed.items()):
        dtype, multivalue = infer_datatype(values)
        out[key] = AttrDef(scope=scope, name=key, datatype=dtype, multivalue=multivalue)
    return out


def _parse_custom_abac(raw: str, policy_id: str) -> CanonicalPolicyIR:
    users: dict[str, dict[str, Any]] = {}
    resources: dict[str, dict[str, Any]] = {}
    rules: list[Rule] = []

    for raw_line in raw.splitlines():
        line = _strip_comment(raw_line)
        if not line:
            continue

        user_m = _USER_RE.match(line)
        if user_m:
            uid, blob = user_m.groups()
            attrs = _parse_attr_blob(blob)
            attrs["uid"] = uid
            users[uid] = attrs
            continue

        res_m = _RESOURCE_RE.match(line)
        if res_m:
            rid, blob = res_m.groups()
            attrs = _parse_attr_blob(blob)
            attrs["rid"] = rid
            resources[rid] = attrs
            continue

        rule_m = _RULE_RE.match(line)
        if rule_m:
            rules.append(_build_rule(len(rules) + 1, rule_m.group(1).strip()))
            continue

    if not rules:
        raise PolicyParseError("No rule(...) entries found in ABAC policy")

    rule_preds = [pred for rule in rules for pred in rule.predicates.values()]
    subject_attrs = _infer_attr_defs(users, "subject", rule_preds)
    resource_attrs = _infer_attr_defs(resources, "resource", rule_preds)

    # Build action vocabulary from rules.
    action_names = sorted({action for rule in rules for action in rule.actions})
    actions = {name: ActionDef(name=name) for name in action_names}

    meta = PolicyMeta(
        policy_id=policy_id,
        domain="healthcare" if "health" in policy_id.lower() else "unknown",
        version="unknown",
        dataset="abac-lab",
        combining="deny-overrides",
    )

    return CanonicalPolicyIR(
        meta=meta,
        subject_attrs=subject_attrs,
        resource_attrs=resource_attrs,
        environment_attrs={},
        actions=actions,
        rules=rules,
        users=users,
        resources=resources,
    )


def _json_rule_to_rule(rule_idx: int, payload: dict[str, Any]) -> Rule:
    rule_id = str(payload.get("id", f"R{rule_idx}"))
    effect = str(payload.get("effect", "Permit"))
    actions_raw = payload.get("actions", [])
    if isinstance(actions_raw, str):
        actions = (actions_raw,)
    else:
        actions = tuple(str(x) for x in actions_raw)

    predicates_payload = payload.get("predicates", [])
    predicates: dict[str, Predicate] = {}
    expr_nodes: dict[str, ExprNode] = {}
    leaf_ids: list[str] = []

    for idx, pp in enumerate(predicates_payload, start=1):
        pid = str(pp.get("id", f"P{idx}"))
        pred = Predicate(
            predicate_id=pid,
            op=str(pp["op"]),
            left_scope=str(pp["left_scope"]),
            left_attr=str(pp["left_attr"]),
            right_value=pp.get("right_value"),
            right_scope=pp.get("right_scope"),
            right_attr=pp.get("right_attr"),
        )
        predicates[pid] = pred
        node_id = f"E{idx}"
        expr_nodes[node_id] = ExprNode(node_id=node_id, op="PRED", predicate_id=pid)
        leaf_ids.append(node_id)

    root_id = "EROOT"
    expr_nodes[root_id] = ExprNode(node_id=root_id, op="AND", children=tuple(leaf_ids))

    return Rule(
        rule_id=rule_id,
        effect=effect,
        actions=actions,
        root_expr_id=root_id,
        expr_nodes=expr_nodes,
        predicates=predicates,
        source="json",
    )


def _parse_json_abac(raw: str, policy_id: str) -> CanonicalPolicyIR:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PolicyParseError(f"Invalid JSON policy: {exc}") from exc

    users = payload.get("users", {})
    resources = payload.get("resources", {})
    for uid, attrs in users.items():
        attrs.setdefault("uid", uid)
    for rid, attrs in resources.items():
        attrs.setdefault("rid", rid)

    rules_payload = payload.get("rules", [])
    rules = [_json_rule_to_rule(i + 1, rule_payload) for i, rule_payload in enumerate(rules_payload)]
    if not rules:
        raise PolicyParseError("JSON policy has no rules")

    all_preds = [pred for rule in rules for pred in rule.predicates.values()]
    subject_attrs = _infer_attr_defs(users, "subject", all_preds)
    resource_attrs = _infer_attr_defs(resources, "resource", all_preds)

    actions_from_rules = sorted({action for rule in rules for action in rule.actions})
    actions = {name: ActionDef(name=name) for name in actions_from_rules}

    meta_payload = payload.get("meta", {})
    meta = PolicyMeta(
        policy_id=str(meta_payload.get("policy_id", policy_id)),
        domain=str(meta_payload.get("domain", "unknown")),
        version=str(meta_payload.get("version", "unknown")),
        dataset=str(meta_payload.get("dataset", "unknown")),
        combining=str(meta_payload.get("combining", "deny-overrides")),
    )

    return CanonicalPolicyIR(
        meta=meta,
        subject_attrs=subject_attrs,
        resource_attrs=resource_attrs,
        environment_attrs={},
        actions=actions,
        rules=rules,
        users=users,
        resources=resources,
    )


def _apply_readme_meta(policy: CanonicalPolicyIR, readme: str) -> None:
    title_match = re.search(r"#\s*Policy Description:\s*(.+)", readme, flags=re.IGNORECASE)
    version_match = re.search(r"Ves(?:i|e)on:\s*([^*\n]+)", readme, flags=re.IGNORECASE)
    if title_match:
        policy.meta = PolicyMeta(
            policy_id=policy.meta.policy_id,
            domain=title_match.group(1).strip().lower(),
            version=policy.meta.version,
            dataset=policy.meta.dataset,
            combining=policy.meta.combining,
        )
    if version_match:
        policy.meta = PolicyMeta(
            policy_id=policy.meta.policy_id,
            domain=policy.meta.domain,
            version=version_match.group(1).strip(),
            dataset=policy.meta.dataset,
            combining=policy.meta.combining,
        )
