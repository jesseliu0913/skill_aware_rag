#!/bin/bash
# Retrieval-at-TEST evaluation (standard RAG deployment) for the already-trained
# _bsl adapters. No retraining -- reuse the adapters from
# submit_baseline_raweval_matrix.sh (trained on base prompt + retrieval block) and
# evaluate them on a TEST set augmented by the SAME retriever, so retrieval is now
# present at both train and test.
#
#   augment TEST (current prompt) -> outputs/baseline_rag/data/<ds>/test_<method>.jsonl
#   eval adapter <model>_<ds>_<method>_bsl on it -> predictions_bslret/..._bslret_test.jsonl
#
# Only retrieving methods run here (none has nothing to add at test; carry over its
# raw-eval number). skill_schema_v1 is evaluated with its skill schema at test time.
#
# Env: MODELS DATASETS METHODS LIMIT MAX_NEW_TOKENS FORCE_AUGMENT(=1) DRY_RUN=1.
set -euo pipefail
cd /playpen-jfs/jesse/drug_microbiome

MODELS="${MODELS:-qwen2_5_7b qwen2_5_3b llama3_2_3b}"
DATASETS="${DATASETS:-fdarxbench mol_instructions}"
METHODS="${METHODS:-bm25 kg hybrid skill_hybrid dense_bge dense_medcpt rrf rerank skill_schema_v1}"
LIMIT="${LIMIT:-1000}"; MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
FORCE_AUGMENT="${FORCE_AUGMENT:-1}"; DRY_RUN="${DRY_RUN:-0}"
ACCOUNT="${ACCOUNT:-jesseliu}"; QOS="${QOS:-normal}"; SB=(--account="$ACCOUNT" --qos="$QOS")

declare -A MODEL_PATHS=(
  [qwen2_5_7b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
  [qwen2_5_3b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
  [llama3_2_3b]="/playpen-shared/jesse/cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/main"
)
is_dense() { case "$1" in dense_bge|dense_medcpt|rrf|rerank) return 0;; *) return 1;; esac; }

mkdir -p outputs/baseline_rag/predictions_bslret outputs/baseline_rag/data outputs/baseline_rag/slurm
LOG="outputs/baseline_rag/slurm/reteval_$(date +%Y%m%d_%H%M%S).tsv"
printf "dataset\tmethod\tmodel\taugment_test\teval\n" > "$LOG"
submit() { if [[ "$DRY_RUN" == "1" ]]; then printf "DRY_RUN: %q " "$@" >&2; printf "\n" >&2; echo "dry_$RANDOM"; else "$@"; fi; }

for dataset in $DATASETS; do
  corrected_test="outputs/qa_skill_data_corrected/${dataset}/test.jsonl"
  for method in $METHODS; do
    aug_test="outputs/baseline_rag/data/${dataset}/test_${method}.jsonl"
    aug_job=""
    if [[ "$FORCE_AUGMENT" == "1" || ! -f "$aug_test" ]]; then
      if is_dense "$method"; then AUG_SBATCH="skill_aware_rag/slurm/submit_augment.sbatch"; else AUG_SBATCH="skill_aware_rag/slurm/submit_augment_skillkb.sbatch"; fi
      aug_job=$(submit sbatch --parsable "${SB[@]}" \
        --job-name="augtest_${dataset}_${method}" \
        --export=ALL,INPUT="$corrected_test",OUTPUT="$aug_test",METHOD="$method",LIMIT=0 \
        "$AUG_SBATCH")
    fi
    dep=(); [[ -n "$aug_job" ]] && dep=(--dependency=afterok:"$aug_job")
    for model in $MODELS; do
      adapter="outputs/baseline_rag/adapters/${model}_${dataset}_${method}_bsl"
      if [[ ! -f "$adapter/adapter_config.json" ]]; then echo "SKIP (no adapter): $adapter" >&2; continue; fi
      pred="outputs/baseline_rag/predictions_bslret/${model}_${dataset}_${method}_bslret_test.jsonl"
      eval_job=$(submit sbatch --parsable "${SB[@]}" "${dep[@]}" \
        --job-name="${model}_${dataset}_${method}_bslreteval" \
        --export=ALL,MODEL="${MODEL_PATHS[$model]}",INPUT="$aug_test",OUTPUT="$pred",ADAPTER="$adapter",LIMIT="$LIMIT",MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
        skill_aware_rag/slurm/submit_generic_qa_eval.sbatch)
      printf "%s\t%s\t%s\t%s\t%s\n" "$dataset" "$method" "$model" "${aug_job:-cached}" "$eval_job" | tee -a "$LOG"
    done
  done
done
echo "Wrote submission log: $LOG"
