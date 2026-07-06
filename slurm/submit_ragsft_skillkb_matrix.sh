#!/bin/bash
# Bring the sparse/graph/skill retrieval methods up to full RAG-SFT parity with
# the dense/fusion family: retrieve for the TRAIN split and retrain a per-method
# LoRA on it (not just inject evidence at eval time).
#
# For each (dataset, method): augment the corrected train split over the unified
# KB (CPU, shared across models). Then for each model: train a LoRA on the
# RAG-augmented train, and eval it on the ALREADY-augmented test split reused
# from outputs/qa_skill_data_skillkb (byte-identical to the existing eval-only
# runs, so only the training differs). Outputs land under outputs/baseline_rag/
# with the same naming as submit_baseline_matrix.sh, and aggregate_results.py
# labels them variant=raglora alongside the dense methods.
#
# Env knobs (same conventions as submit_baseline_matrix.sh):
#   MODELS   default "qwen2_5_7b qwen2_5_3b llama3_2_3b"
#   DATASETS default "mol_instructions fdarxbench"
#   METHODS  default "bm25 kg hybrid skill_hybrid skill_schema_v1"
#   LIMIT / MAX_NEW_TOKENS / MAX_STEPS / SAVE_STEPS / MAX_LENGTH
#   FORCE_AUGMENT=1 to rebuild augmented train; DRY_RUN=1 to print only.
set -euo pipefail

cd /playpen-jfs/jesse/drug_microbiome

MODELS="${MODELS:-qwen2_5_7b qwen2_5_3b llama3_2_3b}"
DATASETS="${DATASETS:-mol_instructions fdarxbench}"
METHODS="${METHODS:-bm25 kg hybrid skill_hybrid skill_schema_v1}"
LIMIT="${LIMIT:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
MAX_STEPS="${MAX_STEPS:-1125}"
SAVE_STEPS="${SAVE_STEPS:-375}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
FORCE_AUGMENT="${FORCE_AUGMENT:-0}"
DRY_RUN="${DRY_RUN:-0}"
ACCOUNT="${ACCOUNT:-jesseliu}"
QOS="${QOS:-normal}"
SB=(--account="$ACCOUNT" --qos="$QOS")

declare -A MODEL_PATHS=(
  [qwen2_5_7b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
  [qwen2_5_3b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
  [llama3_2_3b]="/playpen-shared/jesse/cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/main"
)

mkdir -p outputs/baseline_rag/slurm outputs/baseline_rag/data
LOG="outputs/baseline_rag/slurm/ragsft_skillkb_submissions_$(date +%Y%m%d_%H%M%S).tsv"
printf "dataset\tmethod\tmodel\taugment_train\ttrain\traglora_eval\n" > "$LOG"

submit() {  # echoes a job id (or a dry token); prints the command under DRY_RUN
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "DRY_RUN: %q " "$@" >&2; printf "\n" >&2
    echo "dry_$RANDOM"
  else
    "$@"
  fi
}

for dataset in $DATASETS; do
  corrected_train="outputs/qa_skill_data_corrected/${dataset}/train.jsonl"
  for method in $METHODS; do
    aug_train="outputs/baseline_rag/data/${dataset}/train_${method}.jsonl"
    # Test split already augmented by the earlier eval-only runs; reuse verbatim.
    aug_test="outputs/qa_skill_data_skillkb/${dataset}/test_${method}.jsonl"

    if [[ ! -f "$aug_test" ]]; then
      echo "Missing augmented test: $aug_test (build with augment_with_skill_kb.py first)" >&2
      exit 1
    fi

    # ---- augment train (CPU, shared across models) ----
    aug_train_job=""
    if [[ "$FORCE_AUGMENT" == "1" || ! -f "$aug_train" ]]; then
      aug_train_job=$(submit sbatch --parsable "${SB[@]}" \
        --job-name="augkb_${dataset}_${method}_train" \
        --export=ALL,INPUT="$corrected_train",OUTPUT="$aug_train",METHOD="$method",LIMIT=0 \
        skill_aware_rag/slurm/submit_augment_skillkb.sbatch)
    fi
    dep_train=(); [[ -n "$aug_train_job" ]] && dep_train=(--dependency=afterok:"$aug_train_job")

    for model in $MODELS; do
      model_path="${MODEL_PATHS[$model]}"
      adapter="outputs/baseline_rag/adapters/${model}_${dataset}_${method}_raglora"
      lora_pred="outputs/baseline_rag/predictions/${model}_${dataset}_${method}_raglora_test.jsonl"

      # ---- retrain LoRA on RAG-augmented train ----
      train_job=$(submit sbatch --parsable "${SB[@]}" "${dep_train[@]}" \
        --job-name="${model}_${dataset}_${method}_ragtrain" \
        --export=ALL,MODEL="$model_path",TRAIN="$aug_train",DEV="",OUTPUT_DIR="$adapter",MAX_STEPS="$MAX_STEPS",SAVE_STEPS="$SAVE_STEPS",MAX_LENGTH="$MAX_LENGTH" \
        skill_aware_rag/slurm/submit_generic_qa_lora_train.sbatch)

      # ---- RAG-trained LoRA eval on the (already augmented) test split ----
      lora_job=$(submit sbatch --parsable "${SB[@]}" --dependency=afterok:"$train_job" \
        --job-name="${model}_${dataset}_${method}_raglora_eval" \
        --export=ALL,MODEL="$model_path",INPUT="$aug_test",OUTPUT="$lora_pred",ADAPTER="$adapter",LIMIT="$LIMIT",MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
        skill_aware_rag/slurm/submit_generic_qa_eval.sbatch)

      printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$dataset" "$method" "$model" "${aug_train_job:-cached}" "$train_job" "$lora_job" | tee -a "$LOG"
    done
  done
done

echo "Wrote submission log: $LOG"
