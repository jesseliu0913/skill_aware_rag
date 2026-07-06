#!/bin/bash
set -euo pipefail

cd /playpen-jfs/jesse/drug_microbiome

DATASET="${DATASET:-mol_instructions}"
METHOD="${METHOD:-skill_hybrid}"
LIMIT="${LIMIT:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
DRY_RUN="${DRY_RUN:-0}"

case "$DATASET" in
  mol_instructions|fdarxbench)
    INPUT="outputs/qa_skill_data_skillkb/${DATASET}/test_${METHOD}.jsonl"
    ;;
  *)
    echo "Unknown DATASET=$DATASET. Use mol_instructions or fdarxbench." >&2
    exit 2
    ;;
esac

if [[ ! -f "$INPUT" ]]; then
  echo "Missing augmented input: $INPUT" >&2
  echo "Build it with skill_aware_rag/retrieval/augment_with_skill_kb.py first." >&2
  exit 1
fi

declare -A MODELS=(
  [qwen2_5_7b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
  [qwen2_5_3b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
  [llama3_2_3b]="/playpen-shared/jesse/cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/main"
)

for model_name in qwen2_5_7b qwen2_5_3b llama3_2_3b; do
  model_path="${MODELS[$model_name]}"
  adapter_dir="outputs/qa_skill_lora/${model_name}_${DATASET}_lora"

  raw_output="outputs/qa_skill_predictions/${model_name}_${DATASET}_${METHOD}_raw_test.jsonl"
  raw_cmd=(
    sbatch --parsable
    --job-name="${model_name}_${DATASET}_${METHOD}_raw"
    --export=ALL,MODEL="$model_path",INPUT="$INPUT",OUTPUT="$raw_output",LIMIT="$LIMIT",MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
    skill_aware_rag/slurm/submit_generic_qa_eval.sbatch
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf "DRY_RUN raw: %q " "${raw_cmd[@]}"
    printf "\n"
    raw_job="dry_raw_${model_name}"
  else
    raw_job=$("${raw_cmd[@]}")
  fi

  lora_job="missing_adapter"
  if [[ -d "$adapter_dir" ]]; then
    lora_output="outputs/qa_skill_predictions/${model_name}_${DATASET}_${METHOD}_lora_test.jsonl"
    lora_cmd=(
      sbatch --parsable
      --job-name="${model_name}_${DATASET}_${METHOD}_lora"
      --export=ALL,MODEL="$model_path",INPUT="$INPUT",OUTPUT="$lora_output",ADAPTER="$adapter_dir",LIMIT="$LIMIT",MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
      skill_aware_rag/slurm/submit_generic_qa_eval.sbatch
    )

    if [[ "$DRY_RUN" == "1" ]]; then
      printf "DRY_RUN lora: %q " "${lora_cmd[@]}"
      printf "\n"
      lora_job="dry_lora_${model_name}"
    else
      lora_job=$("${lora_cmd[@]}")
    fi
  fi

  printf "%s\t%s\t%s\traw_eval=%s\tlora_eval=%s\n" \
    "$DATASET" "$METHOD" "$model_name" "$raw_job" "$lora_job"
done
