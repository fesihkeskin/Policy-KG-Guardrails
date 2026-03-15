from __future__ import annotations

from typing import Any

from .types import Decision, GuardrailedResponse, MetricsBundle, VariantPrediction
from .utils import mean, safe_div


def _extract_rows(predictions: list[Any]) -> list[tuple[GuardrailedResponse, Decision, dict[str, Any]]]:
    rows: list[tuple[GuardrailedResponse, Decision, dict[str, Any]]] = []
    for item in predictions:
        if isinstance(item, VariantPrediction):
            rows.append((item.response, item.oracle_decision, item.metadata))
            continue
        if isinstance(item, dict):
            rows.append((item["response"], item["oracle_decision"], item.get("metadata", {})))
            continue
        raise TypeError(f"Unsupported prediction row type: {type(item)}")
    return rows


def score_run(predictions: list[Any], oracle_outputs: list[Any] | None = None) -> MetricsBundle:
    _ = oracle_outputs
    rows = _extract_rows(predictions)
    total = len(rows)
    if total == 0:
        return MetricsBundle(
            accuracy=0.0,
            false_permit_rate=0.0,
            false_deny_rate=0.0,
            abstain_rate=0.0,
            graph_faithfulness=0.0,
            rule_precision=0.0,
            attr_precision=0.0,
            kg_hallucination_rate=0.0,
            citation_correctness=0.0,
            citation_faithfulness=0.0,
            unsupported_claim_rate=0.0,
        )

    correct = 0
    false_permit = 0
    false_deny = 0
    abstain = 0

    claim_alignment_scores: list[float] = []
    rule_precision_scores: list[float] = []
    attr_precision_scores: list[float] = []
    citation_correct_scores: list[float] = []
    citation_faith_scores: list[float] = []
    unsupported_claim_scores: list[float] = []

    cf_validity: list[float] = []
    cf_minimality: list[float] = []
    cf_changes: list[float] = []

    for response, oracle_decision, metadata in rows:
        pred = response.decision
        oracle = oracle_decision

        if pred == oracle:
            correct += 1
        if pred == Decision.PERMIT and oracle != Decision.PERMIT:
            false_permit += 1
        if pred == Decision.DENY and oracle == Decision.PERMIT:
            false_deny += 1
        if pred == Decision.INSUFFICIENT:
            abstain += 1

        total_claims = max(1, len(response.claims))
        aligned_claims = sum(1 for claim in response.claims if claim.aligned)
        claim_alignment_scores.append(safe_div(aligned_claims, total_claims))

        rule_mentions = 0
        rule_supported = 0
        for claim in response.claims:
            for token in claim.supports:
                if token.startswith("R"):
                    rule_mentions += 1
                    if token in set(response.decisive_rules):
                        rule_supported += 1
        if rule_mentions == 0:
            rule_precision_scores.append(1.0)
        else:
            rule_precision_scores.append(safe_div(rule_supported, rule_mentions))

        # Attr precision is approximated from metadata if available.
        mentioned_attrs = metadata.get("mentioned_attrs", [])
        supported_attrs = metadata.get("supported_attrs", [])
        if mentioned_attrs:
            attr_precision_scores.append(
                safe_div(len(set(supported_attrs).intersection(set(mentioned_attrs))), len(set(mentioned_attrs)))
            )
        else:
            attr_precision_scores.append(1.0)

        citation_total = max(1, len(response.citations))
        citation_correct = sum(1 for cite in response.citations if cite in set(response.evidence_subgraphs))
        citation_correct_scores.append(safe_div(citation_correct, citation_total))

        cited_in_aligned = set()
        for claim in response.claims:
            if claim.aligned:
                cited_in_aligned.update(tok for tok in claim.supports if tok.startswith("SG"))
        if response.citations:
            citation_faith_scores.append(
                safe_div(len(set(response.citations).intersection(cited_in_aligned)), len(set(response.citations)))
            )
        else:
            citation_faith_scores.append(1.0)

        unsupported_claim_scores.append(safe_div(response.unsupported_claim_count, total_claims))

        if "counterfactual_valid" in metadata:
            cf_validity.append(1.0 if metadata.get("counterfactual_valid") else 0.0)
            cf_minimality.append(1.0 if metadata.get("counterfactual_minimal") else 0.0)
            cf_changes.append(float(metadata.get("counterfactual_attr_changes", 0)))

    acc = safe_div(correct, total)
    fpr = safe_div(false_permit, total)
    fnr = safe_div(false_deny, total)
    abst = safe_div(abstain, total)
    gf = mean(claim_alignment_scores)
    rule_p = mean(rule_precision_scores)
    attr_p = mean(attr_precision_scores)
    cite_corr = mean(citation_correct_scores)
    cite_faith = mean(citation_faith_scores)
    unsupported = mean(unsupported_claim_scores)

    return MetricsBundle(
        accuracy=acc * 100.0,
        false_permit_rate=fpr * 100.0,
        false_deny_rate=fnr * 100.0,
        abstain_rate=abst * 100.0,
        graph_faithfulness=gf * 100.0,
        rule_precision=rule_p * 100.0,
        attr_precision=attr_p * 100.0,
        kg_hallucination_rate=(1.0 - gf) * 100.0,
        citation_correctness=cite_corr * 100.0,
        citation_faithfulness=cite_faith * 100.0,
        unsupported_claim_rate=unsupported * 100.0,
        counterfactual_validity=mean(cf_validity) * 100.0 if cf_validity else 0.0,
        counterfactual_minimality=mean(cf_minimality) * 100.0 if cf_minimality else 0.0,
        avg_attr_changes=mean(cf_changes) if cf_changes else 0.0,
    )
