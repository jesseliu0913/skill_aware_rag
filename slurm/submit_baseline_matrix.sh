#!/bin/bash
# Orchestrate the RAG-baseline matrix consistently with the existing pipeline.
#
# For each (dataset, method): augment train+test over the unified KB (GPU, shared
# across models). Then for each model: retrain a LoRA on the RAG-augmented train
# split, eval the raw model on the RAG test split, and eval the RAG-trained LoRA
# on the RAG test split. Same splits, models, train hyperparams, and metrics as
# skill_aware_rag/slurm/submit_model_matrix.sh.
#
# Env knobs:
#   MODELS   default "qwen2_5_7b"            (phase 1; add qwen2_5_3b llama3_2_3b to fan out)
#   DATASETS default "mol_instructions fdarxbench"
#   METHODS  default "dense_bge dense_medcpt rrf rerank"
#   LIMIT / MAX_NEW_TOKENS / MAX_STEPS / SAVE_STEPS / MAX_LENGTH
#   FORCE_AUGMENT=1 to rebuild augmented data; DRY_RUN=1 to print only.
set -euo pipefail

cd /playpen-jfs/jesse/drug_microbiome

MODELS="${MODELS:-qwen2_5_7b}"
DATASETS="${DATASETS:-mol_instructions fdarxbench}"
METHODS="${METHODS:-dense_bge dense_medcpt rrf rerank}"
LIMIT="${LIMIT:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
MAX_STEPS="${MAX_STEPS:-1125}"
SAVE_STEPS="${SAVE_STEPS:-375}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
FORCE_AUGMENT="${FORCE_AUGMENT:-0}"
DRY_RUN="${DRY_RUN:-0}"
ACCOUNT="${ACCOUNT:-jesseliu}"
QOS="${QOS:-normal}"
# CLI flags override #SBATCH directives, so this also fixes the reused scripts/ sbatches.
SB=(--account="$ACCOUNT" --qos="$QOS")

declare -A MODEL_PATHS=(
  [qwen2_5_7b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
  [qwen2_5_3b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
  [llama3_2_3b]="/playpen-shared/jesse/cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/main"
)

mkdir -p outputs/baseline_rag/slurm
LOG="outputs/baseline_rag/slurm/matrix_submissions_$(date +%Y%m%d_%H%M%S).tsv"
printf "dataset\tmethod\tmodel\taugment_train\taugment_test\ttrain\traw_eval\traglora_eval\n" > "$LOG"

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
  corrected_test="outputs/qa_skill_data_corrected/${dataset}/test.jsonl"
  for method in $METHODS; do
    aug_train="outputs/baseline_rag/data/${dataset}/train_${method}.jsonl"
    aug_test="outputs/baseline_rag/data/${dataset}/test_${method}.jsonl"

    # ---- augment (shared across models) ----
    aug_train_job=""
    if [[ "$FORCE_AUGMENT" == "1" || ! -f "$aug_train" ]]; then
      aug_train_job=$(submit sbatch --parsable "${SB[@]}" \
        --job-name="aug_${dataset}_${method}_train" \
        --export=ALL,INPUT="$corrected_train",OUTPUT="$aug_train",METHOD="$method",LIMIT=0 \
        skill_aware_rag/slurm/submit_augment.sbatch)
    fi
    aug_test_job=""
    if [[ "$FORCE_AUGMENT" == "1" || ! -f "$aug_test" ]]; then
      aug_test_job=$(submit sbatch --parsable "${SB[@]}" \
        --job-name="aug_${dataset}_${method}_test" \
        --export=ALL,INPUT="$corrected_test",OUTPUT="$aug_test",METHOD="$method",LIMIT=0 \
        skill_aware_rag/slurm/submit_augment.sbatch)
    fi

    dep_train=(); [[ -n "$aug_train_job" ]] && dep_train=(--dependency=afterok:"$aug_train_job")
    dep_test=();  [[ -n "$aug_test_job"  ]] && dep_test=(--dependency=afterok:"$aug_test_job")

    for model in $MODELS; do
      model_path="${MODEL_PATHS[$model]}"
      adapter="outputs/baseline_rag/adapters/${model}_${dataset}_${method}_raglora"
      raw_pred="outputs/baseline_rag/predictions/${model}_${dataset}_${method}_raw_test.jsonl"
      lora_pred="outputs/baseline_rag/predictions/${model}_${dataset}_${method}_raglora_test.jsonl"

      # ---- retrain LoRA on RAG-augmented train ----
      train_job=$(submit sbatch --parsable "${SB[@]}" "${dep_train[@]}" \
        --job-name="${model}_${dataset}_${method}_train" \
        --export=ALL,MODEL="$model_path",TRAIN="$aug_train",DEV="",OUTPUT_DIR="$adapter",MAX_STEPS="$MAX_STEPS",SAVE_STEPS="$SAVE_STEPS",MAX_LENGTH="$MAX_LENGTH" \
        skill_aware_rag/slurm/submit_generic_qa_lora_train.sbatch)

      # ---- raw-model eval on RAG test ----
      raw_job=$(submit sbatch --parsable "${SB[@]}" "${dep_test[@]}" \
        --job-name="${model}_${dataset}_${method}_raw_eval" \
        --export=ALL,MODEL="$model_path",INPUT="$aug_test",OUTPUT="$raw_pred",LIMIT="$LIMIT",MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
        skill_aware_rag/slurm/submit_generic_qa_eval.sbatch)

      # ---- RAG-trained LoRA eval on RAG test (needs both train + test augment) ----
      lora_dep="afterok:${train_job}"
      [[ -n "$aug_test_job" ]] && lora_dep="${lora_dep}:${aug_test_job}"
      lora_job=$(submit sbatch --parsable "${SB[@]}" --dependency="$lora_dep" \
        --job-name="${model}_${dataset}_${method}_raglora_eval" \
        --export=ALL,MODEL="$model_path",INPUT="$aug_test",OUTPUT="$lora_pred",ADAPTER="$adapter",LIMIT="$LIMIT",MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
        skill_aware_rag/slurm/submit_generic_qa_eval.sbatch)

      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$dataset" "$method" "$model" "${aug_train_job:-cached}" "${aug_test_job:-cached}" \
        "$train_job" "$raw_job" "$lora_job" | tee -a "$LOG"
    done
  done
done

echo "Wrote submission log: $LOG"
