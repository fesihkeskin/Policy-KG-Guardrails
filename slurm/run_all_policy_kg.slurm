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
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Set to your Qwen model id.
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
export MAX_SMOKE_TOKENS="${MAX_SMOKE_TOKENS:-128}"
export MAX_PHASE1="${MAX_PHASE1:-400}"
export MAX_PHASE2="${MAX_PHASE2:-120}"
export TEMP="${TEMP:-0.2}"
export TOP_P="${TOP_P:-0.9}"
export MAX_TOKENS="${MAX_TOKENS:-900}"
export POLICY_FILE="${POLICY_FILE:-examples/healthcare.abac}"
export README_FILE="${README_FILE:-examples/healthcare_README.md}"

# Fallback for repos/branches where healthcare assets live under data/healthcare.
if [[ "${POLICY_FILE}" == "examples/healthcare.abac" ]] && [[ ! -f "${PROJECT_DIR}/${POLICY_FILE}" ]] && [[ -f "${PROJECT_DIR}/data/healthcare/healthcare.abac" ]]; then
  POLICY_FILE="data/healthcare/healthcare.abac"
  echo "Info: POLICY_FILE fallback applied -> ${POLICY_FILE}"
fi

if [[ "${README_FILE}" == "examples/healthcare_README.md" ]] && [[ ! -f "${PROJECT_DIR}/${README_FILE}" ]] && [[ -f "${PROJECT_DIR}/data/healthcare/healthcare_README.md" ]]; then
  README_FILE="data/healthcare/healthcare_README.md"
  echo "Info: README_FILE fallback applied -> ${README_FILE}"
fi

DATE_TAG=$(date +%Y%m%d_%H%M)
OUT_DIR="${PROJECT_DIR}/outputs/${DATE_TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${OUT_DIR}"

echo "Project Dir: ${PROJECT_DIR}"
echo "HF Cache: ${HF_HOME}"
echo "Output Dir: ${OUT_DIR}"
echo "Model ID: ${MODEL_ID}"
echo "Policy File: ${POLICY_FILE}"
echo "README File: ${README_FILE}"
echo "Phase1 Limit: ${MAX_PHASE1}"
echo "Phase2 Limit: ${MAX_PHASE2}"
echo "Temperature: ${TEMP}"
echo "Top-p: ${TOP_P}"
echo "Max Tokens: ${MAX_TOKENS}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "Error: PROJECT_DIR does not exist: ${PROJECT_DIR}"
  exit 1
fi

if [[ ! -f "${PROJECT_DIR}/scripts/qwen_smoke_test.py" ]]; then
  echo "Error: Missing script: ${PROJECT_DIR}/scripts/qwen_smoke_test.py"
  exit 1
fi

if [[ ! -f "${PROJECT_DIR}/scripts/run_experiments.py" ]]; then
  echo "Error: Missing script: ${PROJECT_DIR}/scripts/run_experiments.py"
  exit 1
fi

if [[ ! -f "${PROJECT_DIR}/scripts/check_results.py" ]]; then
  echo "Error: Missing script: ${PROJECT_DIR}/scripts/check_results.py"
  exit 1
fi

if [[ ! -f "${PROJECT_DIR}/${POLICY_FILE}" ]]; then
  echo "Error: Policy file not found: ${PROJECT_DIR}/${POLICY_FILE}"
  exit 1
fi

if [[ ! -f "${PROJECT_DIR}/${README_FILE}" ]]; then
  echo "Error: README file not found: ${PROJECT_DIR}/${README_FILE}"
  exit 1
fi

# -------------------------
# Conda
# -------------------------
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

# -------------------------
# 0) Qwen smoke test
# -------------------------
python scripts/qwen_smoke_test.py \
  --model-id "${MODEL_ID}" \
  --output "${OUT_DIR}/qwen_smoke.jsonl" \
  --max-tokens "${MAX_SMOKE_TOKENS}"

# -------------------------
# 1) Main experiments
# -------------------------
python scripts/run_experiments.py \
  --policy "${POLICY_FILE}" \
  --readme "${README_FILE}" \
  --output-dir "${OUT_DIR}" \
  --llm-backend hf-local \
  --model-id "${MODEL_ID}" \
  --max-phase1 "${MAX_PHASE1}" \
  --max-phase2 "${MAX_PHASE2}" \
  --temperature "${TEMP}" \
  --top-p "${TOP_P}" \
  --max-tokens "${MAX_TOKENS}"

# -------------------------
# 2) Verify outputs
# -------------------------
python scripts/check_results.py --output-dir "${OUT_DIR}" --min-lines 1

echo "All done. Output dir: ${OUT_DIR}"
echo "Job Finished: $(date)"
