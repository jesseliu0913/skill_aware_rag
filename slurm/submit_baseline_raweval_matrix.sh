#!/bin/bash
# BASELINE methods, new protocol:
#   TRAIN : retrieval-augmented (RAG-SFT) with the *new* minimal baseline prompt
#           (build_prompt): instruction + question's own context + retrieved
#           evidence + Q/A. Each method augments the TRAIN split with its own
#           retriever, then trains a per-method LoRA on it.
#   EVAL  : the RAW dataset question only -- outputs/qa_skill_data_corrected/<ds>/
#           test.jsonl's plain prompt, no retrieval, no skill, nothing added.
# So the test input is identical across all methods and the only thing that
# differs is what each LoRA was trained on. `none` = no-retrieval reference
# (existing base LoRA, eval-only).
#
# Outputs: adapters  <model>_<ds>_<method>_bsl
#          preds     outputs/baseline_rag/predictions_bsl/<model>_<ds>_<method>_bsl_test.jsonl
#
# Env: MODELS DATASETS METHODS LIMIT MAX_NEW_TOKENS MAX_STEPS SAVE_STEPS MAX_LENGTH
#      FORCE_AUGMENT=1 (default here, since the prompt changed) DRY_RUN=1.
set -euo pipefail
cd /playpen-jfs/jesse/drug_microbiome

MODELS="${MODELS:-qwen2_5_7b qwen2_5_3b llama3_2_3b}"
DATASETS="${DATASETS:-fdarxbench mol_instructions}"
METHODS="${METHODS:-none bm25 kg hybrid skill_hybrid dense_bge dense_medcpt rrf rerank}"
LIMIT="${LIMIT:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
MAX_STEPS="${MAX_STEPS:-1125}"; SAVE_STEPS="${SAVE_STEPS:-375}"; MAX_LENGTH="${MAX_LENGTH:-2048}"
FORCE_AUGMENT="${FORCE_AUGMENT:-1}"
DRY_RUN="${DRY_RUN:-0}"
ACCOUNT="${ACCOUNT:-jesseliu}"; QOS="${QOS:-normal}"
SB=(--account="$ACCOUNT" --qos="$QOS")

declare -A MODEL_PATHS=(
  [qwen2_5_7b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
  [qwen2_5_3b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
  [llama3_2_3b]="/playpen-shared/jesse/cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/main"
)
is_dense() { case "$1" in dense_bge|dense_medcpt|rrf|rerank) return 0;; *) return 1;; esac; }

mkdir -p outputs/baseline_rag/predictions_bsl outputs/baseline_rag/data outputs/baseline_rag/slurm
LOG="outputs/baseline_rag/slurm/baseline_raweval_$(date +%Y%m%d_%H%M%S).tsv"
printf "dataset\tmethod\tmodel\taugment\ttrain\teval\n" > "$LOG"
submit() { if [[ "$DRY_RUN" == "1" ]]; then printf "DRY_RUN: %q " "$@" >&2; printf "\n" >&2; echo "dry_$RANDOM"; else "$@"; fi; }

for dataset in $DATASETS; do
  corrected_train="outputs/qa_skill_data_corrected/${dataset}/train.jsonl"
  raw_test="outputs/qa_skill_data_corrected/${dataset}/test.jsonl"   # RAW question eval, same for all
  for method in $METHODS; do
    # ---- augment TRAIN with the new prompt (shared across models) ----
    # `none` = no retrieval: train directly on the base prompt (corrected_train),
    # so it is on the same footing as the retrieval methods (which add only a
    # retrieved-evidence block to that same base prompt).
    aug_job=""
    if [[ "$method" == "none" ]]; then
      aug_train="$corrected_train"
    else
      aug_train="outputs/baseline_rag/data/${dataset}/train_${method}.jsonl"
      if [[ "$FORCE_AUGMENT" == "1" || ! -f "$aug_train" ]]; then
        if is_dense "$method"; then AUG_SBATCH="skill_aware_rag/slurm/submit_augment.sbatch"; else AUG_SBATCH="skill_aware_rag/slurm/submit_augment_skillkb.sbatch"; fi
        aug_job=$(submit sbatch --parsable "${SB[@]}" \
          --job-name="aug_${dataset}_${method}" \
          --export=ALL,INPUT="$corrected_train",OUTPUT="$aug_train",METHOD="$method",LIMIT=0 \
          "$AUG_SBATCH")
      fi
    fi
    dep_aug=(); [[ -n "$aug_job" ]] && dep_aug=(--dependency=afterok:"$aug_job")
    for model in $MODELS; do
      adapter="outputs/baseline_rag/adapters/${model}_${dataset}_${method}_bsl"
      pred="outputs/baseline_rag/predictions_bsl/${model}_${dataset}_${method}_bsl_test.jsonl"
      train_job=$(submit sbatch --parsable "${SB[@]}" "${dep_aug[@]}" \
        --job-name="${model}_${dataset}_${method}_bsltrain" \
        --export=ALL,MODEL="${MODEL_PATHS[$model]}",TRAIN="$aug_train",DEV="",OUTPUT_DIR="$adapter",MAX_STEPS="$MAX_STEPS",SAVE_STEPS="$SAVE_STEPS",MAX_LENGTH="$MAX_LENGTH" \
        skill_aware_rag/slurm/submit_generic_qa_lora_train.sbatch)
      eval_job=$(submit sbatch --parsable "${SB[@]}" --dependency=afterok:"$train_job" \
        --job-name="${model}_${dataset}_${method}_bsleval" \
        --export=ALL,MODEL="${MODEL_PATHS[$model]}",INPUT="$raw_test",OUTPUT="$pred",ADAPTER="$adapter",LIMIT="$LIMIT",MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
        skill_aware_rag/slurm/submit_generic_qa_eval.sbatch)
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$dataset" "$method" "$model" "${aug_job:-cached}" "$train_job" "$eval_job" | tee -a "$LOG"
    done
  done
done
echo "Wrote submission log: $LOG"
