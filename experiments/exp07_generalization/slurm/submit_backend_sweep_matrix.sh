#!/bin/bash
# Exp07 — Retriever-backend sweep (RQ7: "does the schema help bm25/dense/hybrid/
# rerank alike?"). For each retriever BACKEND we run the SAME RAG-SFT pipeline
# twice — once with the uniform (whole-KB) retriever and once with its skill-
# routed counterpart — and compare. The only variable across a pair is whether
# the skill schema scopes retrieval; the backend is held fixed. This reuses the
# existing augment/train/eval sbatch scripts unchanged (additive; nothing here
# mutates the baseline pipeline).
#
#   augment TRAIN + TEST (method)  ->  LoRA train (RAG-SFT)  ->  eval on aug TEST
#
# Skill routing is expressed through existing method names (no new retriever):
#   bm25      : uniform=bm25             skill=bm25_skillrouted        (sparse sbatch)
#   dense_bge : uniform=dense_bge        skill=dense_bge_skillrouted   (dense  sbatch)
#   hybrid    : uniform=hybrid           skill=skill_hybrid            (sparse sbatch)
#   rerank    : uniform=rerank           skill=rerank                  (dense  sbatch) *
# * rerank has no dedicated skill-routed method yet (see backend_sweep.md, 🔴):
#   until a `rerank_skillrouted` augment method is added, its skill cell reuses
#   `rerank` so the driver stays runnable; that pair is a NO-OP comparison and is
#   skipped unless RERANK_SKILL=1 forces it.
#
# Inputs default to the drug/skill holdout splits from make_splits.py (set
# TRAIN/TEST to point at whichever split you built).
#
# Env: MODELS BACKENDS TRAIN TEST OUTROOT LIMIT MAX_NEW_TOKENS MAX_STEPS
#      RERANK_SKILL DRY_RUN=1 (default) ACCOUNT QOS.
# DRY_RUN defaults to 1 here: this driver is committed UNSUBMITTED.
set -euo pipefail
cd /playpen-jfs/jesse/drug_microbiome

MODELS="${MODELS:-qwen2_5_7b qwen2_5_3b llama3_2_3b}"
BACKENDS="${BACKENDS:-bm25 dense_bge hybrid rerank}"
# Holdout split produced by experiments/exp07_generalization/make_splits.py:
TRAIN="${TRAIN:-outputs/exp07_generalization/drug_holdout/train.jsonl}"
TEST="${TEST:-outputs/exp07_generalization/drug_holdout/test.jsonl}"
OUTROOT="${OUTROOT:-outputs/exp07_generalization/backend_sweep}"
LIMIT="${LIMIT:-1000}"; MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"; MAX_STEPS="${MAX_STEPS:-100}"
RERANK_SKILL="${RERANK_SKILL:-0}"; DRY_RUN="${DRY_RUN:-1}"
ACCOUNT="${ACCOUNT:-jesseliu}"; QOS="${QOS:-normal}"; SB=(--account="$ACCOUNT" --qos="$QOS")

declare -A MODEL_PATHS=(
  [qwen2_5_7b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
  [qwen2_5_3b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
  [llama3_2_3b]="/playpen-shared/jesse/cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/main"
)
# backend -> "uniform_method skill_method sbatch_family"
declare -A UNIFORM=( [bm25]=bm25 [dense_bge]=dense_bge [hybrid]=hybrid [rerank]=rerank )
declare -A SKILL=(   [bm25]=bm25_skillrouted [dense_bge]=dense_bge_skillrouted [hybrid]=skill_hybrid [rerank]=rerank )
is_dense() { case "$1" in dense_bge*|dense_medcpt|rrf|rerank) return 0;; *) return 1;; esac; }

mkdir -p "$OUTROOT"/{adapters,data,predictions,slurm}
LOG="$OUTROOT/slurm/backend_sweep_$(date +%Y%m%d_%H%M%S).tsv"
printf "backend\trouting\tmethod\tmodel\taug_train\taug_test\ttrain\teval\n" > "$LOG"
submit() { if [[ "$DRY_RUN" == "1" ]]; then printf "DRY_RUN: %q " "$@" >&2; printf "\n" >&2; echo "dry_$RANDOM"; else "$@"; fi; }

augment_job() {  # $1=input $2=output $3=method -> echoes job id
  local input="$1" output="$2" method="$3" sbatch
  if is_dense "$method"; then sbatch="skill_aware_rag/slurm/submit_augment.sbatch"; else sbatch="skill_aware_rag/slurm/submit_augment_skillkb.sbatch"; fi
  submit sbatch --parsable "${SB[@]}" --job-name="aug_${method}" \
    --export=ALL,INPUT="$input",OUTPUT="$output",METHOD="$method",LIMIT=0,SCHEMA=v1 "$sbatch"
}

for backend in $BACKENDS; do
  for routing in uniform skill; do
    if [[ "$routing" == "uniform" ]]; then method="${UNIFORM[$backend]}"; else method="${SKILL[$backend]}"; fi
    # Skip the degenerate rerank skill==uniform pair unless explicitly forced.
    if [[ "$backend" == "rerank" && "$routing" == "skill" && "$RERANK_SKILL" != "1" ]]; then
      echo "SKIP (no rerank_skillrouted method yet; set RERANK_SKILL=1 to force): $backend" >&2; continue
    fi
    aug_tr="$OUTROOT/data/train_${method}.jsonl"
    aug_te="$OUTROOT/data/test_${method}.jsonl"
    jtr=$(augment_job "$TRAIN" "$aug_tr" "$method")
    jte=$(augment_job "$TEST"  "$aug_te" "$method")
    for model in $MODELS; do
      adapter="$OUTROOT/adapters/${model}_${backend}_${routing}"
      train_job=$(submit sbatch --parsable "${SB[@]}" --dependency=afterok:"$jtr" \
        --job-name="tr_${model}_${backend}_${routing}" \
        --export=ALL,MODEL="${MODEL_PATHS[$model]}",TRAIN="$aug_tr",OUTPUT_DIR="$adapter",MAX_STEPS="$MAX_STEPS" \
        skill_aware_rag/slurm/submit_generic_qa_lora_train.sbatch)
      pred="$OUTROOT/predictions/${model}_${backend}_${routing}_eval.json"
      eval_job=$(submit sbatch --parsable "${SB[@]}" \
        --dependency=afterok:"$train_job":"$jte" \
        --job-name="ev_${model}_${backend}_${routing}" \
        --export=ALL,MODEL="${MODEL_PATHS[$model]}",INPUT="$aug_te",OUTPUT="${pred%_eval.json}.jsonl",ADAPTER="$adapter",LIMIT="$LIMIT",MAX_NEW_TOKENS="$MAX_NEW_TOKENS",EVAL_OUTPUT="$pred" \
        skill_aware_rag/slurm/submit_generic_qa_eval.sbatch)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$backend" "$routing" "$method" "$model" "$jtr" "$jte" "$train_job" "$eval_job" | tee -a "$LOG"
    done
  done
done
echo "Wrote submission log: $LOG"
[[ "$DRY_RUN" == "1" ]] && echo "DRY_RUN=1 (nothing submitted). Set DRY_RUN=0 to launch."
