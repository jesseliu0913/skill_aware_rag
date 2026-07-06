#!/bin/bash
#SBATCH --job-name=qa_lora_train
#SBATCH --partition=ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /playpen-jfs/jesse/drug_microbiome
mkdir -p outputs/slurm outputs/qa_skill_lora outputs/triton_cache
export TRITON_CACHE_DIR=/playpen-jfs/jesse/drug_microbiome/outputs/triton_cache

: "${MODEL:?Set MODEL to a local model path or HF repo id}"
: "${TRAIN:?Set TRAIN to a QA JSONL train file}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the adapter output directory}"

DEV="${DEV:-}"
MAX_STEPS="${MAX_STEPS:-1125}"
SAVE_STEPS="${SAVE_STEPS:-375}"
LOGGING_STEPS="${LOGGING_STEPS:-25}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

cmd=(
  /playpen-shared/jesse/miniconda3/envs/agentskill/bin/python
  skill_aware_rag/train/train_qa_skill_lora.py
  --model "$MODEL"
  --train "$TRAIN"
  --dev "$DEV"
  --output-dir "$OUTPUT_DIR"
  --max-steps "$MAX_STEPS"
  --save-steps "$SAVE_STEPS"
  --logging-steps "$LOGGING_STEPS"
  --max-length "$MAX_LENGTH"
)

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  cmd+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

"${cmd[@]}"
