#!/bin/bash
# NO-LABEL probe (FDARxBench only). The given FDA label context is REMOVED from
# the prompt, so the answer is no longer handed to the model. This turns the task
# into a real retrieval test:
#   none   : closed-book (question only, no label, no retrieval).
#   retrieval methods: must RETRIEVE the label from the KB using the question
#            alone -- retrieval present at BOTH train and test (standard RAG).
#
# Data: outputs/qa_skill_data_nolabel/fdarxbench/{train,test}.jsonl (label blanked).
# Per method: augment no-label TRAIN + no-label TEST, train a LoRA, eval on the
# augmented no-label test. `none` trains/evals on the closed-book prompt directly.
# Outputs: adapters <model>_fdarxbench_<method>_nolabel
#          preds    predictions_nolabel/<model>_fdarxbench_<method>_nolabel_test.jsonl
#
# Env: MODELS METHODS LIMIT MAX_STEPS ... PARTITION FORCE_AUGMENT DRY_RUN=1.
set -euo pipefail
cd /playpen-jfs/jesse/drug_microbiome

MODELS="${MODELS:-qwen2_5_7b qwen2_5_3b llama3_2_3b}"
METHODS="${METHODS:-none bm25 kg hybrid skill_hybrid dense_bge dense_medcpt rrf rerank skill_schema_v1}"
LIMIT="${LIMIT:-1000}"; MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
MAX_STEPS="${MAX_STEPS:-1125}"; SAVE_STEPS="${SAVE_STEPS:-375}"; MAX_LENGTH="${MAX_LENGTH:-2048}"
DRY_RUN="${DRY_RUN:-0}"; FORCE_AUGMENT="${FORCE_AUGMENT:-0}"
ACCOUNT="${ACCOUNT:-jesseliu}"; QOS="${QOS:-normal}"; PARTITION="${PARTITION:-a100}"
SB=(--account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION")
DS=fdarxbench
NL="outputs/qa_skill_data_nolabel/${DS}"

declare -A MODEL_PATHS=(
  [qwen2_5_7b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
  [qwen2_5_3b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
  [llama3_2_3b]="/playpen-shared/jesse/cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/main"
)
is_dense() { case "$1" in dense_bge|dense_bge_locked|dense_medcpt|rrf|rerank) return 0;; *) return 1;; esac; }
augsb() { is_dense "$1" && echo skill_aware_rag/slurm/submit_augment.sbatch || echo skill_aware_rag/slurm/submit_augment_skillkb.sbatch; }

mkdir -p outputs/baseline_rag/predictions_nolabel outputs/baseline_rag/slurm
LOG="outputs/baseline_rag/slurm/nolabel_$(date +%Y%m%d_%H%M%S).tsv"
printf "method\tmodel\ttrain\teval\n" > "$LOG"
submit() { if [[ "$DRY_RUN" == "1" ]]; then printf "DRY_RUN: %q " "$@" >&2; printf "\n" >&2; echo "dry_$RANDOM"; else "$@"; fi; }

for method in $METHODS; do
  augtrain_job=""; augtest_job=""
  if [[ "$method" == "none" ]]; then
    train_src="$NL/train.jsonl"; eval_src="$NL/test.jsonl"
  else
    train_src="$NL/train_${method}.jsonl"; eval_src="$NL/test_${method}.jsonl"
    if [[ "$FORCE_AUGMENT" == "1" || ! -s "$train_src" ]]; then
      augtrain_job=$(submit sbatch --parsable "${SB[@]}" --job-name="nlaugtr_${method}" \
        --export=ALL,INPUT="$NL/train.jsonl",OUTPUT="$train_src",METHOD="$method",LIMIT=0 "$(augsb "$method")")
    fi
    if [[ "$FORCE_AUGMENT" == "1" || ! -s "$eval_src" ]]; then
      augtest_job=$(submit sbatch --parsable "${SB[@]}" --job-name="nlaugte_${method}" \
        --export=ALL,INPUT="$NL/test.jsonl",OUTPUT="$eval_src",METHOD="$method",LIMIT=0 "$(augsb "$method")")
    fi
  fi
  dtr=(); [[ -n "$augtrain_job" ]] && dtr=(--dependency=afterok:"$augtrain_job")
  for model in $MODELS; do
    adapter="outputs/baseline_rag/adapters/${model}_${DS}_${method}_nolabel"
    pred="outputs/baseline_rag/predictions_nolabel/${model}_${DS}_${method}_nolabel_test.jsonl"
    train_job=$(submit sbatch --parsable "${SB[@]}" "${dtr[@]}" --job-name="${model}_${method}_nltrain" \
      --export=ALL,MODEL="${MODEL_PATHS[$model]}",TRAIN="$train_src",DEV="",OUTPUT_DIR="$adapter",MAX_STEPS="$MAX_STEPS",SAVE_STEPS="$SAVE_STEPS",MAX_LENGTH="$MAX_LENGTH" \
      skill_aware_rag/slurm/submit_generic_qa_lora_train.sbatch)
    edep="afterok:$train_job"; [[ -n "$augtest_job" ]] && edep="afterok:$train_job:$augtest_job"
    eval_job=$(submit sbatch --parsable "${SB[@]}" --dependency="$edep" --job-name="${model}_${method}_nleval" \
      --export=ALL,MODEL="${MODEL_PATHS[$model]}",INPUT="$eval_src",OUTPUT="$pred",ADAPTER="$adapter",LIMIT="$LIMIT",MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
      skill_aware_rag/slurm/submit_generic_qa_eval.sbatch)
    printf "%s\t%s\t%s\t%s\n" "$method" "$model" "$train_job" "$eval_job" | tee -a "$LOG"
  done
done
echo "Wrote submission log: $LOG"
