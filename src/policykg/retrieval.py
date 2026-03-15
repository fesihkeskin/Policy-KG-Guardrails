from __future__ import annotations

from collections import defaultdict
from typing import Any

from .kg import PolicyKG, render_rule_summary
from .types import EvidenceSubgraph, RequestContext, TextEvidence
from .utils import safe_div, tokenize


def build_text_corpus(policy_kg: PolicyKG) -> list[TextEvidence]:
    corpus: list[TextEvidence] = []
    policy = policy_kg.policy

    for rule in policy.rules:
        summary = render_rule_summary(rule)
        corpus.append(
            TextEvidence(
                evidence_id=f"TXT_RULE_{rule.rule_id}",
                text=summary,
                rule_ids=(rule.rule_id,),
            )
        )

    for attr in policy.subject_attrs.values():
        corpus.append(
            TextEvidence(
                evidence_id=f"TXT_SUBATTR_{attr.name}",
                text=f"subject attribute {attr.name} datatype={attr.datatype}",
                rule_ids=(),
            )
        )

    for attr in policy.resource_attrs.values():
        corpus.append(
            TextEvidence(
                evidence_id=f"TXT_RESATTR_{attr.name}",
                text=f"resource attribute {attr.name} datatype={attr.datatype}",
                rule_ids=(),
            )
        )

    return corpus


def _score_text(query: str, text: str) -> float:
    q_tokens = set(tokenize(query))
    t_tokens = set(tokenize(text))
    overlap = len(q_tokens.intersection(t_tokens))
    base = safe_div(overlap, max(1, len(q_tokens)))
    return float(base)


def retrieve_text(query: str, k: int, policy_kg: PolicyKG) -> list[TextEvidence]:
    corpus = build_text_corpus(policy_kg)
    ranked = sorted(
        (
            TextEvidence(
                evidence_id=item.evidence_id,
                text=item.text,
                rule_ids=item.rule_ids,
                score=_score_text(query, item.text),
            )
            for item in corpus
        ),
        key=lambda item: (item.score, item.evidence_id),
        reverse=True,
    )
    return ranked[:k]


def _rule_score(query_tokens: set[str], rule_summary: str, action_names: set[str]) -> float:
    summary_tokens = set(tokenize(rule_summary))
    overlap = len(query_tokens.intersection(summary_tokens))
    score = safe_div(overlap, max(1, len(query_tokens)))
    action_boost = 0.0
    for action in action_names:
        if action.lower() in query_tokens and action.lower() in summary_tokens:
            action_boost += 0.35
    return float(score + action_boost)


def _collect_neighborhood_edges(policy_kg: PolicyKG, rule_node: str, hops: int = 2) -> tuple[set[str], list[tuple[str, str, str]]]:
    graph = policy_kg.graph
    visited = {rule_node}
    frontier = {rule_node}
    edges: list[tuple[str, str, str]] = []

    for _ in range(hops):
        next_frontier: set[str] = set()
        for node in frontier:
            out_edges = graph.out_edges(node, keys=True, data=True)
            in_edges = graph.in_edges(node, keys=True, data=True)
            for src, dst, _, data in out_edges:
                label = str(data.get("label", ""))
                edges.append((str(src), label, str(dst)))
                if dst not in visited:
                    next_frontier.add(dst)
            for src, dst, _, data in in_edges:
                label = str(data.get("label", ""))
                edges.append((str(src), label, str(dst)))
                if src not in visited:
                    next_frontier.add(src)
        visited.update(next_frontier)
        frontier = next_frontier

    dedup_edges = list(dict.fromkeys(edges))
    return visited, dedup_edges


def retrieve_graph(policy_kg: PolicyKG, query: str, k_rules: int = 5) -> list[EvidenceSubgraph]:
    query_tokens = set(tokenize(query))
    action_names = set(policy_kg.policy.actions.keys())

    scored: list[tuple[float, str]] = []
    for rule in policy_kg.policy.rules:
        summary = policy_kg.rule_summaries[rule.rule_id]
        score = _rule_score(query_tokens, summary, action_names)
        scored.append((score, rule.rule_id))

    top_rules = sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)[:k_rules]

    out: list[EvidenceSubgraph] = []
    for idx, (score, rule_id) in enumerate(top_rules, start=1):
        rule_node = policy_kg.rule_node_ids[rule_id]
        node_ids, edges = _collect_neighborhood_edges(policy_kg, rule_node, hops=2)
        out.append(
            EvidenceSubgraph(
                evidence_id=f"SG{idx}",
                rule_ids=(rule_id,),
                node_ids=tuple(sorted(node_ids)),
                edges=tuple(edges[:120]),
                score=float(score),
            )
        )
    return out


def extract_request_sketch(policy_kg: PolicyKG, query: str) -> RequestContext:
    tokens = tokenize(query)
    token_set = set(tokens)

    action = None
    for candidate in policy_kg.policy.actions:
        if candidate.lower() in token_set:
            action = candidate
            break

    subject: dict[str, Any] = {}
    resource: dict[str, Any] = {}

    # Try explicit IDs first.
    for uid, attrs in policy_kg.policy.users.items():
        if uid.lower() in token_set:
            subject = dict(attrs)
            break

    for rid, attrs in policy_kg.policy.resources.items():
        if rid.lower() in token_set:
            resource = dict(attrs)
            break

    # Heuristic extraction for key=value style mentions in free text.
    pairs: dict[str, str] = {}
    for token in query.replace(",", " ").split():
        if "=" in token:
            left, right = token.split("=", 1)
            pairs[left.strip().lower()] = right.strip()

    for key, value in pairs.items():
        if key.startswith("subject."):
            subject[key.split(".", 1)[1]] = value
        elif key.startswith("resource."):
            resource[key.split(".", 1)[1]] = value
        elif key == "action":
            action = value

    return RequestContext(subject=subject, resource=resource, action=action, environment={})


def evidence_rule_map(evidence: list[EvidenceSubgraph]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for item in evidence:
        for rule_id in item.rule_ids:
            mapping[rule_id].append(item.evidence_id)
    return dict(mapping)
