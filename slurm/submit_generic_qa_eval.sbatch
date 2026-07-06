#!/bin/bash
#SBATCH --job-name=qa_eval
#SBATCH --partition=ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /playpen-jfs/jesse/drug_microbiome
mkdir -p outputs/slurm outputs/qa_skill_predictions outputs/triton_cache
export TRITON_CACHE_DIR=/playpen-jfs/jesse/drug_microbiome/outputs/triton_cache

: "${MODEL:?Set MODEL to a local model path or HF repo id}"
: "${INPUT:?Set INPUT to a QA JSONL test file}"
: "${OUTPUT:?Set OUTPUT to the prediction JSONL path}"

LIMIT="${LIMIT:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
ADAPTER="${ADAPTER:-}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${OUTPUT%.jsonl}_eval.json}"

cmd=(
  /playpen-shared/jesse/miniconda3/envs/agentskill/bin/python
  skill_aware_rag/eval/run_qa_jsonl_inference.py
  --model "$MODEL"
  --input "$INPUT"
  --output "$OUTPUT"
  --limit "$LIMIT"
  --max-new-tokens "$MAX_NEW_TOKENS"
)

if [[ -n "$ADAPTER" ]]; then
  cmd+=(--adapter "$ADAPTER")
fi

"${cmd[@]}"

/playpen-shared/jesse/miniconda3/envs/agentskill/bin/python skill_aware_rag/eval/evaluate_instruction_outputs.py \
  "$OUTPUT" \
  --output "$EVAL_OUTPUT"
