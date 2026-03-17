#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A fekeskin
#SBATCH -J policykg-qwen-smoke-akya
#SBATCH -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/policy_qwen_smoke_akya_%j.out
#SBATCH -e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/policy_qwen_smoke_akya_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

set -euo pipefail

echo "Job Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "=== SLURM SETTINGS ==="
echo "Job Partition: ${SLURM_JOB_PARTITION:-${SLURM_PARTITION:-N/A}}"
echo "Job Account: ${SLURM_JOB_ACCOUNT:-N/A}"
echo "Job Name: ${SLURM_JOB_NAME:-N/A}"
echo "Job User: ${SLURM_JOB_USER:-${USER:-N/A}}"
echo "Submit Dir: ${SLURM_SUBMIT_DIR:-N/A}"
echo "Nodes Requested (NNODES): ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-N/A}}"
echo "Node List: ${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-N/A}}"
echo "NTASKS: ${SLURM_NTASKS:-N/A}"
echo "NTASKS per node: ${SLURM_NTASKS_PER_NODE:-N/A}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "CPUs on node: ${SLURM_CPUS_ON_NODE:-N/A}"
echo "Memory per node: ${SLURM_MEM_PER_NODE:-N/A}"
echo "Memory per CPU: ${SLURM_MEM_PER_CPU:-N/A}"
echo "GRES (GPUs): ${SLURM_GPUS:-${SLURM_JOB_GPUS:-N/A}}"
echo "GRES on node: ${SLURM_GPUS_ON_NODE:-N/A}"
echo "Time limit: ${SLURM_TIMELIMIT:-${SLURM_JOB_TIME_LIMIT:-N/A}}"
echo "QOS: ${SLURM_JOB_QOS:-N/A}"
echo "Constraint: ${SLURM_JOB_CONSTRAINT:-N/A}"
echo "=== ENV SUMMARY ==="
echo "PWD: $(pwd)"
echo "PATH: ${PATH}"
echo "Python (pre-conda): $(which python 2>/dev/null || echo N/A)"
echo "======================"

export SCRATCH_DIR="${SCRATCH_DIR:-/arf/scratch/${USER}}"
export PROJECT_DIR="${PROJECT_DIR:-${SCRATCH_DIR}/Policy-KG-Guardrails}"
export HF_HOME="${HF_HOME:-${SCRATCH_DIR}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-4B}"
export MAX_TOKENS="${MAX_TOKENS:-128}"
export DATE_TAG="$(date +%Y%m%d_%H%M)"
export OUTPUT_FILE="${PROJECT_DIR}/outputs/policy_qwen_smoke_akya_${DATE_TAG}_${SLURM_JOB_ID:-manual}.jsonl"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/outputs" "${HF_HOME}" "${HF_DATASETS_CACHE}"

echo "Project Dir: ${PROJECT_DIR}"
echo "HF Cache: ${HF_HOME}"
echo "Output File: ${OUTPUT_FILE}"
echo "Model ID: ${MODEL_ID}"
echo "Max Tokens: ${MAX_TOKENS}"

if [[ "${MODEL_ID}" == "Qwen/Qwen3.5-7B-Instruct" ]]; then
  echo "Error: MODEL_ID=${MODEL_ID} is not available on Hugging Face (404)."
  echo "Use a valid model id, e.g. Qwen/Qwen3.5-4B (default), Qwen/Qwen3.5-9B, or Qwen/Qwen2.5-7B-Instruct."
  exit 2
fi

MODEL_CACHE_ROOT="${HF_HOME}/hub/models--${MODEL_ID//\//--}"
if [[ -d "${MODEL_CACHE_ROOT}" ]]; then
  MODEL_SNAPSHOT_COUNT=0
  if [[ -d "${MODEL_CACHE_ROOT}/snapshots" ]]; then
    MODEL_SNAPSHOT_COUNT=$(find "${MODEL_CACHE_ROOT}/snapshots" -mindepth 1 -maxdepth 1 -type d | wc -l)
  fi
  echo "[CACHE] Model cache found for ${MODEL_ID}: ${MODEL_CACHE_ROOT} (snapshots: ${MODEL_SNAPSHOT_COUNT})"
else
  echo "[CACHE] Model cache NOT found for ${MODEL_ID} under ${HF_HOME}/hub (first run may download)."
fi

if [[ -d "${HF_DATASETS_CACHE}" ]]; then
  DATASET_CACHE_ITEMS=$(find "${HF_DATASETS_CACHE}" -mindepth 1 -maxdepth 1 | wc -l)
  if [[ "${DATASET_CACHE_ITEMS}" -gt 0 ]]; then
    echo "[CACHE] Dataset cache contains ${DATASET_CACHE_ITEMS} top-level entries at ${HF_DATASETS_CACHE}."
  else
    echo "[CACHE] Dataset cache directory exists but is currently empty: ${HF_DATASETS_CACHE}."
  fi
else
  echo "[CACHE] Dataset cache directory does not exist yet: ${HF_DATASETS_CACHE}."
fi

echo "[CACHE] Hugging Face will reuse local snapshots automatically when available and skip re-download behavior."

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "Error: PROJECT_DIR does not exist: ${PROJECT_DIR}"
  exit 1
fi

if [[ ! -f "${PROJECT_DIR}/scripts/qwen_smoke_test.py" ]]; then
  echo "Error: Missing smoke script at ${PROJECT_DIR}/scripts/qwen_smoke_test.py"
  exit 1
fi

eval "$(${CONDA_ROOT:-/arf/home/${USER}/miniconda3}/bin/conda shell.bash hook)"
conda activate "${CONDA_ENV:-pytorchgpu_env}"

cd "${PROJECT_DIR}"
python -m pip install -e .[dev,llm]

echo "Python (post-conda): $(which python)"
python --version || true

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
  echo "HF token detected and exported."
else
  echo "Warning: HF_TOKEN is not set. Model download may fail for gated repos."
fi

echo "Visible CUDA devices: ${CUDA_VISIBLE_DEVICES:-N/A}"
nvidia-smi -L || true

python scripts/qwen_smoke_test.py \
  --model-id "${MODEL_ID}" \
  --output "${OUTPUT_FILE}" \
  --max-tokens "${MAX_TOKENS}"

echo "Qwen smoke test completed successfully."
echo "Results saved at: ${OUTPUT_FILE}"
echo "Job Finished: $(date)"
# Usage: 
# sbatch -o /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_qwen_smoke_akya_%j.out \
#         -e /arf/scratch/fekeskin/Policy-KG-Guardrails/logs/$(date +%Y%m%d_%H%M)_policy_qwen_smoke_akya_%j.err \
#         slurm/policy_qwen_smoke_akya.slurm