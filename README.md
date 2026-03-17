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
  --model-id Qwen/Qwen3.5-4B \
  --temperature 0.1 \
  --max-tokens 384 \
  --trust-remote-code
```

All-in-one local pipeline (no SLURM):

```bash
python3 scripts/run_local_pipeline.py \
  --llm-backend heuristic \
  --policy examples/healthcare.abac \
  --readme examples/healthcare_README.md
```

HF local all-in-one pipeline:

```bash
python3 scripts/run_local_pipeline.py \
  --llm-backend hf-local \
  --model-id Qwen/Qwen3.5-4B \
  --trust-remote-code \
  --policy examples/healthcare.abac \
  --readme examples/healthcare_README.md
```

## TRUBA / SLURM

SLURM templates are in `slurm/`:

- `slurm/policy_warm_cache.slurm`: warm model/cache and run minimal generation check.
- `slurm/policy_qwen_smoke_debug.slurm`: quick smoke test for Qwen responses.
- `slurm/policy_run_all.slurm`: smoke test + full experiment run + artifact validation (`scripts/check_results.py`).

Run examples:

```bash
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3.5-4B,TRUST_REMOTE_CODE=1 slurm/policy_qwen_smoke_debug.slurm
sbatch --export=ALL,MODEL_ID=Qwen/Qwen3.5-4B,TRUST_REMOTE_CODE=1,MAX_PHASE1=400,MAX_PHASE2=120,MAX_TOKENS=384 slurm/policy_run_all.slurm
```
