# Policy-KG Guardrails CLI Cookbook

This file collects practical command-line recipes for submitting SLURM jobs and running scripts directly in this repo.

## 1) One-time setup (interactive shell)

```bash
cd /arf/scratch/fekeskin/Policy-KG-Guardrails
eval "$(/arf/home/${USER}/miniconda3/bin/conda shell.bash hook)"
conda activate pytorchgpu_env
python -m pip install -e .[dev,llm]
```

Optional for gated Hugging Face models:

```bash
export HF_TOKEN=hf_xxx
```

Local cache placement (optional, useful on small home directories):

```bash
export HF_HOME=$PWD/.hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
```

## 2) Quick SLURM submissions

### 2.1 Warm cache only

Default:

```bash
sbatch -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_warm_cache_%j.out \
	-e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_warm_cache_%j.err \
	slurm/policy_warm_cache.slurm
```

With model/token overrides:

```bash
sbatch -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_warm_cache_%j.out \
	-e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_warm_cache_%j.err \
	--export=ALL,HF_TOKEN=${HF_TOKEN},MODEL_ID=Qwen/Qwen3.5-4B,MAX_TOKENS=96 \
	slurm/policy_warm_cache.slurm
```

Use a different conda env:

```bash
sbatch -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_warm_cache_%j.out \
	-e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_warm_cache_%j.err \
	--export=ALL,CONDA_ENV=pytorchgpu_env \
	slurm/policy_warm_cache.slurm
```

### 2.2 Qwen smoke debug

Default:

```bash
sbatch -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_qwen_smoke_debug_%j.out \
	-e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_qwen_smoke_debug_%j.err \
	slurm/policy_qwen_smoke_debug.slurm
```

Tune generation and remote code trust:

```bash
sbatch -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_qwen_smoke_debug_%j.out \
	-e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_qwen_smoke_debug_%j.err \
	--export=ALL,HF_TOKEN=${HF_TOKEN},MODEL_ID=Qwen/Qwen3.5-4B,MAX_TOKENS=128,TEMP=0.1,TOP_P=0.9,TRUST_REMOTE_CODE=1 \
	slurm/policy_qwen_smoke_debug.slurm
```

Disable trust_remote_code:

```bash
sbatch -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_qwen_smoke_debug_%j.out \
	-e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_qwen_smoke_debug_%j.err \
	--export=ALL,MODEL_ID=Qwen/Qwen2.5-7B-Instruct,TRUST_REMOTE_CODE=0 \
	slurm/policy_qwen_smoke_debug.slurm
```

### 2.3 Full pipeline (smoke + experiments + validation)

Default:

```bash
sbatch -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_run_all_%j.out \
	-e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_run_all_%j.err \
	slurm/policy_run_all.slurm
```

Common tuned run:

```bash
sbatch -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_run_all_%j.out \
	-e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_run_all_%j.err \
	--export=ALL,HF_TOKEN=${HF_TOKEN},MODEL_ID=Qwen/Qwen3.5-4B,TRUST_REMOTE_CODE=1,MAX_SMOKE_TOKENS=96,MAX_PHASE1=400,MAX_PHASE2=120,TEMP=0.1,TOP_P=0.9,MAX_TOKENS=384 \
	slurm/policy_run_all.slurm
```

Custom policy/readme inputs:

```bash
sbatch -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_run_all_%j.out \
	-e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_run_all_%j.err \
	--export=ALL,HF_TOKEN=${HF_TOKEN},POLICY_FILE=data/healthcare/healthcare.abac,README_FILE=data/healthcare/healthcare_README.md \
	slurm/policy_run_all.slurm
```

## 3) Override SLURM resources at submit time

You can override `#SBATCH` defaults without editing files:

```bash
sbatch -p akya-cuda --cpus-per-task=16 --mem=64G --time=02:00:00 --gres=gpu:1 \
	-o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_qwen_smoke_debug_%j.out \
	-e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_qwen_smoke_debug_%j.err \
	slurm/policy_qwen_smoke_debug.slurm
```

Custom log names per submission:

```bash
TS=$(date +%Y%m%d_%H%M)
sbatch -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/${TS}_policy_run_all_%j.out \
	-e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/${TS}_policy_run_all_%j.err \
	slurm/policy_run_all.slurm
```

## 4) Local (non-SLURM) script runs

These are the complete local entrypoints in this repo:

- `scripts/qwen_smoke_test.py`
- `scripts/run_experiments.py`
- `scripts/check_results.py`
- `scripts/run_local_pipeline.py` (all-in-one local orchestrator)

### 4.1 `scripts/qwen_smoke_test.py`

Minimal:

```bash
python scripts/qwen_smoke_test.py --model-id Qwen/Qwen3.5-4B --output outputs/qwen_smoke_local.jsonl
```

All useful options:

```bash
python scripts/qwen_smoke_test.py \
	--model-id Qwen/Qwen3.5-4B \
	--output outputs/qwen_smoke_local.jsonl \
	--temperature 0.1 \
	--top-p 0.9 \
	--max-tokens 160 \
	--trust-remote-code
```

Disable 4-bit quantization:

```bash
python scripts/qwen_smoke_test.py --model-id Qwen/Qwen3.5-4B --output outputs/qwen_smoke_fp16.jsonl --no-4bit
```

### 4.2 `scripts/run_experiments.py`

Heuristic backend:

```bash
python scripts/run_experiments.py \
	--policy data/healthcare/healthcare.abac \
	--readme data/healthcare/healthcare_README.md \
	--output-dir outputs/heuristic_run \
	--llm-backend heuristic
```

HF local backend:

```bash
python scripts/run_experiments.py \
	--policy data/healthcare/healthcare.abac \
	--readme data/healthcare/healthcare_README.md \
	--output-dir outputs/hf_local_run \
	--llm-backend hf-local \
	--model-id Qwen/Qwen3.5-4B \
	--max-phase1 400 \
	--max-phase2 120 \
	--temperature 0.1 \
	--top-p 0.9 \
	--max-tokens 384 \
	--trust-remote-code
```

Disable 4-bit quantization:

```bash
python scripts/run_experiments.py \
	--policy data/healthcare/healthcare.abac \
	--readme data/healthcare/healthcare_README.md \
	--output-dir outputs/hf_local_no4bit \
	--llm-backend hf-local \
	--model-id Qwen/Qwen3.5-4B \
	--no-4bit
```

### 4.3 `scripts/check_results.py`

Validate outputs:

```bash
python scripts/check_results.py --output-dir outputs/hf_local_run --min-lines 1
```

Stricter check:

```bash
python scripts/check_results.py --output-dir outputs/hf_local_run --min-lines 5
```

### 4.4 `scripts/run_local_pipeline.py` (all-in-one local run)

Heuristic-only full local run (no model download):

```bash
python scripts/run_local_pipeline.py \
	--llm-backend heuristic \
	--policy examples/healthcare.abac \
	--readme examples/healthcare_README.md
```

HF local full run with smoke + experiments + checks:

```bash
python scripts/run_local_pipeline.py \
	--llm-backend hf-local \
	--model-id Qwen/Qwen3.5-4B \
	--trust-remote-code \
	--policy examples/healthcare.abac \
	--readme examples/healthcare_README.md
```

HF local run without smoke phase:

```bash
python scripts/run_local_pipeline.py \
	--llm-backend hf-local \
	--model-id Qwen/Qwen3.5-4B \
	--no-run-smoke
```

HF local run in FP16 (no 4-bit quantization):

```bash
python scripts/run_local_pipeline.py \
	--llm-backend hf-local \
	--model-id Qwen/Qwen3.5-4B \
	--no-4bit \
	--trust-remote-code
```

Use a custom output directory and stricter artifact validation:

```bash
python scripts/run_local_pipeline.py \
	--llm-backend heuristic \
	--output-dir outputs/local_custom \
	--check-results \
	--min-lines 5
```

### 4.5 Direct module invocation (equivalent local style)

```bash
python -m scripts.qwen_smoke_test --model-id Qwen/Qwen3.5-4B --output outputs/qwen_smoke_module.jsonl
python -m scripts.run_experiments --policy examples/healthcare.abac --readme examples/healthcare_README.md --output-dir outputs/module_run
python -m scripts.check_results --output-dir outputs/module_run --min-lines 1
```

## 5) Helpful monitoring commands

```bash
squeue -u ${USER}
scontrol show job <job_id>
tail -f logs/run_all_<job_id>.out
tail -f logs/run_all_<job_id>.err
```

Cancel a job:

```bash
scancel <job_id>
```

## 6) Troubleshooting quick checks

Verify script syntax:

```bash
bash -n slurm/policy_warm_cache.slurm
bash -n slurm/policy_qwen_smoke_debug.slurm
bash -n slurm/policy_run_all.slurm
bash -n slurm/policy_qwen_smoke_akya.slurm
```

Verify policy/readme paths used by `policy_run_all.slurm`:

```bash
ls -l data/healthcare/healthcare.abac data/healthcare/healthcare_README.md
```

Check GPU visibility in interactive shell:

```bash
nvidia-smi -L
```

## 7) Recommended submission profiles

Small smoke profile:

```bash
sbatch -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_warm_cache_%j.out \
	-e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_warm_cache_%j.err \
	--export=ALL,HF_TOKEN=${HF_TOKEN},MODEL_ID=Qwen/Qwen3.5-4B,MAX_TOKENS=64 \
	slurm/policy_warm_cache.slurm
```

Standard full run:

```bash
sbatch -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_run_all_%j.out \
	-e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_run_all_%j.err \
	--export=ALL,HF_TOKEN=${HF_TOKEN},MODEL_ID=Qwen/Qwen3.5-4B,TRUST_REMOTE_CODE=1,MAX_PHASE1=400,MAX_PHASE2=120,TEMP=0.1,TOP_P=0.9,MAX_TOKENS=384 \
	slurm/policy_run_all.slurm
```

Conservative memory/token profile:

```bash
sbatch -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_run_all_%j.out \
        -e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_run_all_%j.err \
		--export=ALL,HF_TOKEN=${HF_TOKEN},MODEL_ID=Qwen/Qwen2.5-7B-Instruct,TRUST_REMOTE_CODE=0,MAX_SMOKE_TOKENS=64,MAX_PHASE1=250,MAX_PHASE2=80,MAX_TOKENS=256 slurm/policy_run_all.slurm
```

```bash
sbatch -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_qwen_smoke_akya_%j.out \
	-e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_qwen_smoke_akya_%j.err \
	slurm/policy_qwen_smoke_akya.slurm
		
```