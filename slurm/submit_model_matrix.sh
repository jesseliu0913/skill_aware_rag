#!/bin/bash
set -euo pipefail

cd /playpen-jfs/jesse/drug_microbiome

DATASET="${DATASET:-mol_instructions}"
LIMIT="${LIMIT:-1000}"
MAX_STEPS="${MAX_STEPS:-1125}"
SAVE_STEPS="${SAVE_STEPS:-375}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
DRY_RUN="${DRY_RUN:-0}"

case "$DATASET" in
  mol_instructions)
    TRAIN="outputs/qa_skill_data_corrected/mol_instructions/train.jsonl"
    TEST="outputs/qa_skill_data_corrected/mol_instructions/test.jsonl"
    ;;
  fdarxbench)
    TRAIN="outputs/qa_skill_data_corrected/fdarxbench/train.jsonl"
    TEST="outputs/qa_skill_data_corrected/fdarxbench/test.jsonl"
    ;;
  *)
    echo "Unknown DATASET=$DATASET. Use mol_instructions or fdarxbench." >&2
    exit 2
    ;;
esac

declare -A MODELS=(
  [qwen2_5_7b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
  [qwen2_5_3b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
  [llama3_2_3b]="/playpen-shared/jesse/cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/main"
)

for model_name in qwen2_5_7b qwen2_5_3b llama3_2_3b; do
  model_path="${MODELS[$model_name]}"
  if [[ ! -f "$model_path/tokenizer_config.json" ]]; then
    echo "Missing model snapshot for $model_name: $model_path" >&2
    exit 1
  fi

  adapter_dir="outputs/qa_skill_lora/${model_name}_${DATASET}_lora"
  raw_output="outputs/qa_skill_predictions/${model_name}_${DATASET}_raw_test.jsonl"
  lora_output="outputs/qa_skill_predictions/${model_name}_${DATASET}_lora_test.jsonl"

  raw_cmd=(
    sbatch --parsable
    --job-name="${model_name}_${DATASET}_raw"
    --export=ALL,MODEL="$model_path",INPUT="$TEST",OUTPUT="$raw_output",LIMIT="$LIMIT",MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
    skill_aware_rag/slurm/submit_generic_qa_eval.sbatch
  )
  train_cmd=(
    sbatch --parsable
    --job-name="${model_name}_${DATASET}_train"
    --export=ALL,MODEL="$model_path",TRAIN="$TRAIN",DEV="",OUTPUT_DIR="$adapter_dir",MAX_STEPS="$MAX_STEPS",SAVE_STEPS="$SAVE_STEPS",MAX_LENGTH="$MAX_LENGTH"
    skill_aware_rag/slurm/submit_generic_qa_lora_train.sbatch
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf "DRY_RUN raw: %q " "${raw_cmd[@]}"
    printf "\n"
    printf "DRY_RUN train: %q " "${train_cmd[@]}"
    printf "\n"
    raw_job="dry_raw_${model_name}"
    train_job="dry_train_${model_name}"
  else
    raw_job=$("${raw_cmd[@]}")
    train_job=$("${train_cmd[@]}")
  fi

  trained_eval_cmd=(
    sbatch --parsable
    --dependency=afterok:"$train_job"
    --job-name="${model_name}_${DATASET}_lora_eval"
    --export=ALL,MODEL="$model_path",INPUT="$TEST",OUTPUT="$lora_output",ADAPTER="$adapter_dir",LIMIT="$LIMIT",MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
    skill_aware_rag/slurm/submit_generic_qa_eval.sbatch
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf "DRY_RUN trained_eval: %q " "${trained_eval_cmd[@]}"
    printf "\n"
    trained_eval_job="dry_lora_eval_${model_name}"
  else
    trained_eval_job=$("${trained_eval_cmd[@]}")
  fi

  printf "%s\t%s\traw_eval=%s\ttrain=%s\ttrained_eval=%s\n" \
    "$DATASET" "$model_name" "$raw_job" "$train_job" "$trained_eval_job"
done
