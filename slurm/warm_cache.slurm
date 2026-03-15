#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -A fekeskin
#SBATCH -J policykg-warm-cache
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/warm_cache_%j.out
#SBATCH -e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/warm_cache_%j.err

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

# Set to your preferred Qwen model id before sbatch or edit here.
# Example candidates:
# - Qwen/Qwen2.5-7B-Instruct
# - Qwen/Qwen2.5-3B-Instruct
# - <your Qwen 3.5 model id>
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"

mkdir -p "${PROJECT_DIR}/logs" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${PROJECT_DIR}/outputs"

# -------------------------
# Conda
# -------------------------
eval "$(${CONDA_ROOT:-/arf/home/${USER}/miniconda3}/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-pytorchgpu_env}"

cd "${PROJECT_DIR}"

# Install project + llm deps in current env (idempotent)
python -m pip install -e .[dev,llm]

# Optional gated access
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

nvidia-smi -L || true

# Download model weights and verify generation works
python scripts/qwen_smoke_test.py \
  --model-id "${MODEL_ID}" \
  --output "outputs/qwen_smoke_${SLURM_JOB_ID}.jsonl" \
  --max-tokens 96

echo "Warm-cache complete."
echo "Job Finished: $(date)"
