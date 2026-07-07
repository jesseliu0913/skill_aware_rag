#!/bin/bash
# Train-only retrieval, RAW-QUESTION evaluation.
#
# Protocol: every method's retrieval/schema is applied ONLY during training
# (the per-method LoRA is already trained on its augmented train split). At
# EVALUATION every method is fed the *identical raw dataset question*
# (outputs/qa_skill_data_corrected/<ds>/test.jsonl -> record["prompt"], no
# retrieved evidence, no skill schema). So the test input is byte-identical
# across all methods and the ONLY thing that differs is what each LoRA trained
# on. This removes the "different test prompt" confound entirely.
#
# Reuses the already-trained adapters:
#   retrieval methods -> outputs/baseline_rag/adapters/<model>_<ds>_<method>_raglora
#   none (reference)  -> outputs/qa_skill_lora/<model>_<ds>_lora
# Predictions land in outputs/baseline_rag/predictions_rawq/ with variant
# "rawqeval", aggregated by aggregate_results.py into results_summary_rawq.*.
#
# Env knobs: MODELS, DATASETS, METHODS, LIMIT, MAX_NEW_TOKENS, DRY_RUN=1.
set -euo pipefail
cd /playpen-jfs/jesse/drug_microbiome

MODELS="${MODELS:-qwen2_5_7b qwen2_5_3b llama3_2_3b}"
DATASETS="${DATASETS:-fdarxbench mol_instructions}"
METHODS="${METHODS:-none bm25 kg hybrid dense_bge dense_medcpt rerank rrf skill_hybrid skill_schema_v1}"
LIMIT="${LIMIT:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
DRY_RUN="${DRY_RUN:-0}"
ACCOUNT="${ACCOUNT:-jesseliu}"; QOS="${QOS:-normal}"
SB=(--account="$ACCOUNT" --qos="$QOS")

declare -A MODEL_PATHS=(
  [qwen2_5_7b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
  [qwen2_5_3b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
  [llama3_2_3b]="/playpen-shared/jesse/cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/main"
)

mkdir -p outputs/baseline_rag/predictions_rawq outputs/baseline_rag/slurm
LOG="outputs/baseline_rag/slurm/rawq_eval_submissions_$(date +%Y%m%d_%H%M%S).tsv"
printf "dataset\tmethod\tmodel\tjob\n" > "$LOG"

submit() { if [[ "$DRY_RUN" == "1" ]]; then printf "DRY_RUN: %q " "$@" >&2; printf "\n" >&2; echo "dry_$RANDOM"; else "$@"; fi; }

for dataset in $DATASETS; do
  raw_test="outputs/qa_skill_data_corrected/${dataset}/test.jsonl"   # RAW question, same for all methods
  [[ -f "$raw_test" ]] || { echo "Missing raw test: $raw_test" >&2; exit 1; }
  for method in $METHODS; do
    for model in $MODELS; do
      if [[ "$method" == "none" ]]; then
        adapter="outputs/qa_skill_lora/${model}_${dataset}_lora"
      else
        adapter="outputs/baseline_rag/adapters/${model}_${dataset}_${method}_raglora"
      fi
      if [[ ! -f "$adapter/adapter_config.json" ]]; then
        echo "SKIP (no adapter): $adapter" >&2; continue
      fi
      pred="outputs/baseline_rag/predictions_rawq/${model}_${dataset}_${method}_rawqeval_test.jsonl"
      job=$(submit sbatch --parsable "${SB[@]}" \
        --job-name="${model}_${dataset}_${method}_rawqeval" \
        --export=ALL,MODEL="${MODEL_PATHS[$model]}",INPUT="$raw_test",OUTPUT="$pred",ADAPTER="$adapter",LIMIT="$LIMIT",MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
        skill_aware_rag/slurm/submit_generic_qa_eval.sbatch)
      printf "%s\t%s\t%s\t%s\n" "$dataset" "$method" "$model" "$job" | tee -a "$LOG"
    done
  done
done
echo "Wrote submission log: $LOG"
