#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -A fekeskin
#SBATCH -J policykg-qwen-smoke
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/qwen_smoke_%j.out
#SBATCH -e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/qwen_smoke_%j.err

set -euo pipefail

echo "Job Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

export SCRATCH_DIR="${SCRATCH_DIR:-/arf/scratch/${USER}}"
export PROJECT_DIR="${PROJECT_DIR:-${SCRATCH_DIR}/Policy-KG-Guardrails}"
export HF_HOME="${HF_HOME:-${SCRATCH_DIR}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/outputs" "${HF_HOME}" "${HF_DATASETS_CACHE}"

eval "$(${CONDA_ROOT:-/arf/home/${USER}/miniconda3}/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-pytorchgpu_env}"

cd "${PROJECT_DIR}"
python -m pip install -e .[dev,llm]

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

nvidia-smi -L || true

python scripts/qwen_smoke_test.py \
  --model-id "${MODEL_ID}" \
  --output "outputs/qwen_smoke_${SLURM_JOB_ID}.jsonl" \
  --max-tokens 128

echo "Qwen smoke test completed successfully."
echo "Job Finished: $(date)"
