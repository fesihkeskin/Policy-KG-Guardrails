#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -A fekeskin
#SBATCH -J policykg-run-all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/run_all_%j.out
#SBATCH -e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/run_all_%j.err

set -euo pipefail

echo "Job Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

# -------------------------
# Paths / env
# -------------------------
export SCRATCH_DIR="${SCRATCH_DIR:-/arf/scratch/${USER}}"
export PROJECT_DIR="${PROJECT_DIR:-${SCRATCH_DIR}/Policy-KG-Guardrails}"
export HF_HOME="${HF_HOME:-${SCRATCH_DIR}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_CACHE="${HF_HOME}"

# Set to your Qwen model id.
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"

DATE_TAG=$(date +%Y%m%d_%H%M)
OUT_DIR="${PROJECT_DIR}/outputs/${DATE_TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${OUT_DIR}"

# -------------------------
# Conda
# -------------------------
eval "$(${CONDA_ROOT:-/arf/home/${USER}/miniconda3}/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-pytorchgpu_env}"

cd "${PROJECT_DIR}"
python -m pip install -e .[dev,llm]

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

nvidia-smi -L || true

# -------------------------
# 0) Qwen smoke test
# -------------------------
python scripts/qwen_smoke_test.py \
  --model-id "${MODEL_ID}" \
  --output "${OUT_DIR}/qwen_smoke.jsonl" \
  --max-tokens 128

# -------------------------
# 1) Main experiments
# -------------------------
python scripts/run_experiments.py \
  --policy examples/healthcare.abac \
  --readme examples/healthcare_README.md \
  --output-dir "${OUT_DIR}" \
  --llm-backend hf-local \
  --model-id "${MODEL_ID}" \
  --max-phase1 ${MAX_PHASE1:-400} \
  --max-phase2 ${MAX_PHASE2:-120} \
  --temperature ${TEMP:-0.2} \
  --top-p ${TOP_P:-0.9} \
  --max-tokens ${MAX_TOKENS:-900}

# -------------------------
# 2) Verify outputs
# -------------------------
python scripts/check_results.py --output-dir "${OUT_DIR}" --min-lines 1

echo "All done. Output dir: ${OUT_DIR}"
echo "Job Finished: $(date)"
