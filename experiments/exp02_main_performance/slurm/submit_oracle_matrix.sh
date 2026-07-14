#!/bin/bash
# Exp02 main-performance + oracle_source matrix (RQ4).
#
# Full RAG-SFT pipeline per (model x dataset x method): augment TRAIN + TEST with
# the method's retriever, train a LoRA on the augmented train split, evaluate on the
# augmented test split -> *_eval.json. This adds the `oracle_source` UPPER BOUND
# alongside its comparison twins (`dense_bge` uniform, `dense_bge_skillrouted`
# deployed-style routing) so collate_main_table.py can test the ordering
# hypothesis uniform <= skill <= oracle on matched adapters.
#
# oracle_source is augmented by the standalone experiments/.../submit_augment_oracle.sbatch
# (dense_bge restricted to the GOLD-skill allowed sources); every other method reuses
# the shared augment sbatches. Nothing here is baseline-mutating.
#
# MODE=corrected (default): labeled corrected splits, both datasets, variant=raglora.
# MODE=nolabel           : FDARxBench no-label probe (label blanked), variant=nolabel.
#
# Env: MODELS DATASETS METHODS MODE LIMIT MAX_STEPS SAVE_STEPS MAX_LENGTH
#      MAX_NEW_TOKENS SCHEMA PARTITION FORCE_AUGMENT DRY_RUN=1.
# DRY_RUN=1 prints the sbatch commands instead of submitting. NOT auto-submitted.
set -euo pipefail
cd /playpen-jfs/jesse/drug_microbiome

MODE="${MODE:-corrected}"
MODELS="${MODELS:-qwen2_5_7b qwen2_5_3b llama3_2_3b}"
METHODS="${METHODS:-dense_bge dense_bge_skillrouted oracle_source}"
if [[ "$MODE" == "nolabel" ]]; then
  DATASETS="${DATASETS:-fdarxbench}"; VARIANT="nolabel"
  DATA_ROOT="outputs/qa_skill_data_nolabel"
else
  DATASETS="${DATASETS:-fdarxbench mol_instructions}"; VARIANT="raglora"
  DATA_ROOT="outputs/qa_skill_data_corrected"
fi
LIMIT="${LIMIT:-1000}"; MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
MAX_STEPS="${MAX_STEPS:-1125}"; SAVE_STEPS="${SAVE_STEPS:-375}"; MAX_LENGTH="${MAX_LENGTH:-2048}"
SCHEMA="${SCHEMA:-legacy}"
DRY_RUN="${DRY_RUN:-0}"; FORCE_AUGMENT="${FORCE_AUGMENT:-0}"
ACCOUNT="${ACCOUNT:-jesseliu}"; QOS="${QOS:-normal}"; PARTITION="${PARTITION:-a100}"
SB=(--account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION")

declare -A MODEL_PATHS=(
  [qwen2_5_7b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
  [qwen2_5_3b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
  [llama3_2_3b]="/playpen-shared/jesse/cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/main"
)
is_dense() { case "$1" in dense_bge|dense_bge_locked|dense_bge_skillrouted|dense_medcpt|rrf|rerank) return 0;; *) return 1;; esac; }
# Pick the augment sbatch: oracle -> standalone oracle augmenter; dense -> baseline dense; else sparse/skill.
augsb() {
  case "$1" in
    oracle_source) echo skill_aware_rag/experiments/exp02_main_performance/slurm/submit_augment_oracle.sbatch;;
    *) is_dense "$1" && echo skill_aware_rag/slurm/submit_augment.sbatch || echo skill_aware_rag/slurm/submit_augment_skillkb.sbatch;;
  esac
}
# oracle sbatch is METHOD-agnostic (fixed to oracle_source); others need METHOD=.
augexport() { if [[ "$1" == "oracle_source" ]]; then echo "SCHEMA=$SCHEMA"; else echo "METHOD=$1,SCHEMA=$SCHEMA"; fi; }

mkdir -p outputs/baseline_rag/predictions_oracle outputs/baseline_rag/data outputs/baseline_rag/slurm
LOG="outputs/baseline_rag/slurm/oracle_${MODE}_$(date +%Y%m%d_%H%M%S).tsv"
printf "dataset\tmethod\tmodel\taugtrain\taugtest\ttrain\teval\n" > "$LOG"
submit() { if [[ "$DRY_RUN" == "1" ]]; then printf "DRY_RUN: %q " "$@" >&2; printf "\n" >&2; echo "dry_$RANDOM"; else "$@"; fi; }

for dataset in $DATASETS; do
  base_train="$DATA_ROOT/${dataset}/train.jsonl"
  base_test="$DATA_ROOT/${dataset}/test.jsonl"
  for method in $METHODS; do
    train_src="outputs/baseline_rag/data/${dataset}/train_${method}_${VARIANT}.jsonl"
    eval_src="outputs/baseline_rag/data/${dataset}/test_${method}_${VARIANT}.jsonl"
    augtrain_job=""; augtest_job=""
    if [[ "$FORCE_AUGMENT" == "1" || ! -s "$train_src" ]]; then
      augtrain_job=$(submit sbatch --parsable "${SB[@]}" --job-name="augtr_${dataset}_${method}" \
        --export=ALL,INPUT="$base_train",OUTPUT="$train_src",LIMIT=0,"$(augexport "$method")" "$(augsb "$method")")
    fi
    if [[ "$FORCE_AUGMENT" == "1" || ! -s "$eval_src" ]]; then
      augtest_job=$(submit sbatch --parsable "${SB[@]}" --job-name="augte_${dataset}_${method}" \
        --export=ALL,INPUT="$base_test",OUTPUT="$eval_src",LIMIT=0,"$(augexport "$method")" "$(augsb "$method")")
    fi
    dtr=(); [[ -n "$augtrain_job" ]] && dtr=(--dependency=afterok:"$augtrain_job")
    for model in $MODELS; do
      adapter="outputs/baseline_rag/adapters/${model}_${dataset}_${method}_oracle${VARIANT}"
      pred="outputs/baseline_rag/predictions_oracle/${model}_${dataset}_${method}_${VARIANT}_test.jsonl"
      train_job=$(submit sbatch --parsable "${SB[@]}" "${dtr[@]}" --job-name="${model}_${method}_otr" \
        --export=ALL,MODEL="${MODEL_PATHS[$model]}",TRAIN="$train_src",DEV="",OUTPUT_DIR="$adapter",MAX_STEPS="$MAX_STEPS",SAVE_STEPS="$SAVE_STEPS",MAX_LENGTH="$MAX_LENGTH" \
        skill_aware_rag/slurm/submit_generic_qa_lora_train.sbatch)
      edep="afterok:$train_job"; [[ -n "$augtest_job" ]] && edep="afterok:$train_job:$augtest_job"
      eval_job=$(submit sbatch --parsable "${SB[@]}" --dependency="$edep" --job-name="${model}_${method}_oev" \
        --export=ALL,MODEL="${MODEL_PATHS[$model]}",INPUT="$eval_src",OUTPUT="$pred",ADAPTER="$adapter",LIMIT="$LIMIT",MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
        skill_aware_rag/slurm/submit_generic_qa_eval.sbatch)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$dataset" "$method" "$model" "${augtrain_job:-cached}" "${augtest_job:-cached}" "$train_job" "$eval_job" | tee -a "$LOG"
    done
  done
done
echo "Wrote submission log: $LOG"
