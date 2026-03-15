# Policy-KG Guardrails Prototype

Research prototype implementing Policy Knowledge Graph guardrails for ABAC decision support.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

For local LLM backends (Qwen via Hugging Face):

```bash
pip install -e .[dev,llm]
```

## Core APIs

- `parse_abac(path) -> CanonicalPolicyIR`
- `compile_policy_kg(policy_ir) -> PolicyKG`
- `evaluate(policy_kg, request_context) -> DecisionTrace`
- `retrieve_text(query, k) -> list[TextEvidence]`
- `retrieve_graph(policy_kg, query, k_rules) -> list[EvidenceSubgraph]`
- `run_variant(variant_name, query, request_context, evidence) -> ModelDraft`
- `verify_and_revise(draft, decision_trace, evidence) -> GuardrailedResponse`
- `score_run(predictions, oracle_outputs) -> MetricsBundle`

## Run experiments

Heuristic backend:

```bash
python3 scripts/run_experiments.py \
  --policy examples/healthcare.abac \
  --readme examples/healthcare_README.md \
  --output-dir outputs
```

Local Qwen backend (4-bit quantized):

```bash
python3 scripts/run_experiments.py \
  --policy examples/healthcare.abac \
  --readme examples/healthcare_README.md \
  --output-dir outputs_qwen \
  --llm-backend hf-local \
  --model-id Qwen/Qwen2.5-7B-Instruct
```

## TRUBA / SLURM

SLURM templates are in `slurm/`:

- `slurm/warm_cache.slurm`: download/cache model and run a minimal generation check.
- `slurm/qwen_smoke.slurm`: run prompt smoke test only (`scripts/qwen_smoke_test.py`).
- `slurm/run_all.slurm`: smoke test + full experiment run + artifact validation (`scripts/check_results.py`).

Run examples:

```bash
sbatch --export=ALL,MODEL_ID=Qwen/Qwen2.5-7B-Instruct slurm/warm_cache.slurm
sbatch --export=ALL,MODEL_ID=Qwen/Qwen2.5-7B-Instruct slurm/qwen_smoke.slurm
sbatch --export=ALL,MODEL_ID=Qwen/Qwen2.5-7B-Instruct,MAX_PHASE1=400,MAX_PHASE2=120 slurm/run_all.slurm
```
