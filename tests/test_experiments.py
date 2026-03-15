from __future__ import annotations

from pathlib import Path

from policykg.experiments import ExperimentConfig, run_experiments
from policykg.llm import HeuristicLLMClient


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "examples" / "healthcare.abac"
README_PATH = ROOT / "examples" / "healthcare_README.md"


def test_run_experiments_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    result = run_experiments(
        policy_path=str(POLICY_PATH),
        readme_path=str(README_PATH),
        output_dir=str(out_dir),
        llm_client=HeuristicLLMClient(),
        config=ExperimentConfig(seed=13, max_phase1_samples=40, max_phase2_samples=12),
    )

    assert result["test_size"] > 0
    assert result["adversarial_size"] > 0
    assert result["counterfactual_size"] > 0

    table_path = out_dir / "tables.md"
    assert table_path.exists()
    text = table_path.read_text(encoding="utf-8")
    assert "Decision reliability" in text
    assert "Adversarial robustness" in text

    for variant in ["vanilla", "text-rag", "kg-rag", "policy-kg-guardrails"]:
        pred_file = out_dir / f"predictions_{variant}.jsonl"
        assert pred_file.exists()

    for metrics in result["phase1_metrics"].values():
        assert 0.0 <= metrics.accuracy <= 100.0
        assert 0.0 <= metrics.false_permit_rate <= 100.0
        assert 0.0 <= metrics.graph_faithfulness <= 100.0
