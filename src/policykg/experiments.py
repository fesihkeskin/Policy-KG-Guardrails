from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluator import evaluate
from .kg import compile_policy_kg
from .llm import HeuristicLLMClient, LLMClient
from .metrics import score_run
from .parser import parse_abac
from .tasks import (
    generate_adversarial_tasks,
    generate_counterfactual_tasks,
    generate_policy_qa_tasks,
    stratified_split,
)
from .types import (
    Decision,
    GuardrailedResponse,
    MetricsBundle,
    ModelDraft,
    VariantPrediction,
)
from .variants import DecodingConfig, VariantRunner


@dataclass
class ExperimentConfig:
    seed: int = 7
    dev_ratio: float = 0.2
    max_phase1_samples: int = 400
    max_phase2_samples: int = 120
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 900


def _draft_to_response(draft: ModelDraft) -> GuardrailedResponse:
    return GuardrailedResponse(
        decision=draft.decision,
        confidence=draft.confidence,
        policy_id=draft.policy_id,
        request_subject=draft.request_subject,
        request_resource=draft.request_resource,
        request_action=draft.request_action,
        request_environment=draft.request_environment,
        evidence_subgraphs=draft.evidence_subgraphs,
        decisive_rules=draft.decisive_rules,
        decisive_predicates=draft.decisive_predicates,
        claims=draft.claims,
        citations=draft.citations,
        missing_attributes=[],
        was_overridden=False,
        revised=False,
        unsupported_claim_count=sum(1 for claim in draft.claims if not claim.aligned),
        notes=[],
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _table_markdown(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _write_tables(path: Path, phase1: dict[str, MetricsBundle], adversarial: dict[str, MetricsBundle], counterfactual: dict[str, MetricsBundle]) -> None:
    lines: list[str] = ["# Experiment Tables", ""]

    headers = ["Method", "Acc (%)", "FPR (%)", "FNR (%)", "Abstain (%)"]
    rows = []
    for method, m in phase1.items():
        rows.append([method, _fmt(m.accuracy), _fmt(m.false_permit_rate), _fmt(m.false_deny_rate), _fmt(m.abstain_rate)])
    lines.append("## Decision reliability")
    lines.append(_table_markdown(headers, rows))
    lines.append("")

    headers = ["Method", "GF (%)", "Rule Prec (%)", "Attr Prec (%)", "KGHallu (%)"]
    rows = []
    for method, m in phase1.items():
        rows.append(
            [
                method,
                _fmt(m.graph_faithfulness),
                _fmt(m.rule_precision),
                _fmt(m.attr_precision),
                _fmt(m.kg_hallucination_rate),
            ]
        )
    lines.append("## Grounding and faithfulness")
    lines.append(_table_markdown(headers, rows))
    lines.append("")

    headers = ["Method", "CiteCorr (%)", "CiteFaith (%)", "Unsupported Claims (%)"]
    rows = []
    for method, m in phase1.items():
        rows.append([method, _fmt(m.citation_correctness), _fmt(m.citation_faithfulness), _fmt(m.unsupported_claim_rate)])
    lines.append("## Citation quality")
    lines.append(_table_markdown(headers, rows))
    lines.append("")

    headers = ["Method", "Acc (%)", "FPR (%)", "GF (%)", "Abstain (%)"]
    rows = []
    for method, m in adversarial.items():
        rows.append([method, _fmt(m.accuracy), _fmt(m.false_permit_rate), _fmt(m.graph_faithfulness), _fmt(m.abstain_rate)])
    lines.append("## Adversarial robustness")
    lines.append(_table_markdown(headers, rows))
    lines.append("")

    headers = ["Method", "CF Validity (%)", "CF Minimality (%)", "Avg Attr Changes"]
    rows = []
    for method, m in counterfactual.items():
        rows.append([method, _fmt(m.counterfactual_validity), _fmt(m.counterfactual_minimality), _fmt(m.avg_attr_changes)])
    lines.append("## Counterfactual quality")
    lines.append(_table_markdown(headers, rows))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiments(
    *,
    policy_path: str,
    readme_path: str | None = None,
    output_dir: str = "outputs",
    llm_client: LLMClient | None = None,
    config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ExperimentConfig()
    policy = parse_abac(policy_path, readme_path=readme_path)
    policy_kg = compile_policy_kg(policy)

    model = llm_client or HeuristicLLMClient()
    runner = VariantRunner(
        policy_kg=policy_kg,
        llm_client=model,
        decoding=DecodingConfig(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
        ),
    )

    phase1_tasks = generate_policy_qa_tasks(
        policy_kg,
        seed=cfg.seed,
        max_samples=cfg.max_phase1_samples,
    )

    labels: dict[str, Decision] = {}
    for task in phase1_tasks:
        trace = evaluate(policy_kg, task.request)
        labels[task.task_id] = trace.decision

    _, test_tasks = stratified_split(
        phase1_tasks,
        labels,
        dev_ratio=cfg.dev_ratio,
        seed=cfg.seed,
    )

    base_for_phase2 = [task for task in test_tasks if task.task_type == "qa"]
    adversarial_tasks = generate_adversarial_tasks(
        base_for_phase2,
        seed=cfg.seed,
        max_tasks=cfg.max_phase2_samples,
    )
    counterfactual_tasks = generate_counterfactual_tasks(
        policy_kg,
        base_for_phase2,
        seed=cfg.seed,
        max_tasks=cfg.max_phase2_samples,
    )

    variants = ["vanilla", "text-rag", "kg-rag", "policy-kg-guardrails"]

    all_outputs = Path(output_dir)
    all_outputs.mkdir(parents=True, exist_ok=True)

    phase1_metrics: dict[str, MetricsBundle] = {}
    adversarial_metrics: dict[str, MetricsBundle] = {}
    counterfactual_metrics: dict[str, MetricsBundle] = {}

    for variant in variants:
        phase1_predictions: list[VariantPrediction] = []
        for task in test_tasks:
            oracle = labels[task.task_id]
            if variant == "policy-kg-guardrails":
                response = runner.run_guardrails(task.query, task.request)
            else:
                draft = runner.run_variant(variant, task.query, task.request)
                response = _draft_to_response(draft)
            phase1_predictions.append(
                VariantPrediction(
                    task_id=task.task_id,
                    variant=variant,
                    response=response,
                    oracle_decision=oracle,
                    metadata=task.metadata,
                )
            )

        phase1_metrics[variant] = score_run(phase1_predictions, None)

        adversarial_predictions: list[VariantPrediction] = []
        for task in adversarial_tasks:
            oracle = evaluate(policy_kg, task.request).decision
            if variant == "policy-kg-guardrails":
                response = runner.run_guardrails(task.query, task.request)
            else:
                draft = runner.run_variant(variant, task.query, task.request)
                response = _draft_to_response(draft)
            adversarial_predictions.append(
                VariantPrediction(
                    task_id=task.task_id,
                    variant=variant,
                    response=response,
                    oracle_decision=oracle,
                    metadata=task.metadata,
                )
            )
        adversarial_metrics[variant] = score_run(adversarial_predictions, None)

        counterfactual_predictions: list[VariantPrediction] = []
        for task in counterfactual_tasks:
            oracle = evaluate(policy_kg, task.request).decision
            if variant == "policy-kg-guardrails":
                response = runner.run_guardrails(task.query, task.request)
            else:
                draft = runner.run_variant(variant, task.query, task.request)
                response = _draft_to_response(draft)

            meta = dict(task.metadata)
            # Model-side counterfactual extraction can be added later; default to invalid.
            meta.setdefault("counterfactual_valid", False)
            meta.setdefault("counterfactual_minimal", False)
            meta.setdefault("counterfactual_attr_changes", 0)

            counterfactual_predictions.append(
                VariantPrediction(
                    task_id=task.task_id,
                    variant=variant,
                    response=response,
                    oracle_decision=oracle,
                    metadata=meta,
                )
            )
        counterfactual_metrics[variant] = score_run(counterfactual_predictions, None)

        json_rows = []
        for row in phase1_predictions:
            json_rows.append(
                {
                    "task_id": row.task_id,
                    "variant": row.variant,
                    "oracle": row.oracle_decision.value,
                    "prediction": row.response.decision.value,
                    "citations": row.response.citations,
                    "decisive_rules": row.response.decisive_rules,
                    "unsupported_claim_count": row.response.unsupported_claim_count,
                }
            )
        _write_jsonl(all_outputs / f"predictions_{variant}.jsonl", json_rows)

    _write_tables(all_outputs / "tables.md", phase1_metrics, adversarial_metrics, counterfactual_metrics)

    return {
        "phase1_metrics": phase1_metrics,
        "adversarial_metrics": adversarial_metrics,
        "counterfactual_metrics": counterfactual_metrics,
        "test_size": len(test_tasks),
        "adversarial_size": len(adversarial_tasks),
        "counterfactual_size": len(counterfactual_tasks),
        "output_dir": str(all_outputs),
    }
