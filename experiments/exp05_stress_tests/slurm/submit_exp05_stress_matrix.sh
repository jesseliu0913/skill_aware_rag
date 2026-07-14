#!/bin/bash
# Exp05 stress-test matrix driver (RQ6). Mirrors slurm/submit_nolabel_matrix.sh:
# env-var driven, DRY_RUN=1 prints instead of submitting, chained sbatch deps.
#
# For each perturbed KB variant x method x top-k value:
#   0) (CPU, local) build the perturbed KB with perturb_kb.py if missing.
#   1) augment the TEST split against that KB at that top-k (submit_augment_skillkb
#      sbatch, reused via KB_DIR / TOP_K env vars added additively).
#   2) evaluate (submit_generic_qa_eval sbatch) -> *_eval.json.
#
# This is a RAW-eval stress sweep (no per-cell LoRA): it isolates the retrieval
# stressor. Reuse settings (no-label FDA, weak-retriever, multi-source open-QA)
# are covered by slurm/submit_nolabel_matrix.sh and the baseline matrices.
#
# NOTE: decoy KB variants are 🔴 HEURISTIC (synthetic near-dup + source relabel).
#
# Env: MODELS METHODS TOPKS KB_VARIANTS DATASET LIMIT ... PARTITION DRY_RUN=1.
set -euo pipefail
cd /playpen-jfs/jesse/drug_microbiome

REPO=skill_aware_rag
PERTURB="$REPO/experiments/exp05_stress_tests/perturb_kb.py"
PYTHON=/playpen-shared/jesse/miniconda3/envs/agentskill/bin/python

MODELS="${MODELS:-qwen2_5_7b}"
METHODS="${METHODS:-bm25 bm25_skillrouted skill_hybrid hybrid}"
TOPKS="${TOPKS:-1 2 3 8}"
# KB variants: <name>:<mode>:<arg>. clean has no perturbation. See topk_sweep.md.
KB_VARIANTS="${KB_VARIANTS:-clean:none:0 noisy_f0.5:noisy:0.5 noisy_f1.0:noisy:1.0 decoy_n1:decoy:1}"
DATASET="${DATASET:-fdarxbench}"
LIMIT="${LIMIT:-1000}"; MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
SEED="${SEED:-0}"; DRY_RUN="${DRY_RUN:-0}"; FORCE_AUGMENT="${FORCE_AUGMENT:-0}"
ACCOUNT="${ACCOUNT:-jesseliu}"; QOS="${QOS:-normal}"; PARTITION="${PARTITION:-ada}"
SB=(--account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION")

CLEAN_KB="outputs/knowledge_bank/unified"
PERT_ROOT="outputs/knowledge_bank/perturbed"
# Augmented test split lives per method/dataset (built by baselines); we re-augment
# per (variant,k) so the KB perturbation + budget actually take effect.
TEST_SRC="${TEST_SRC:-outputs/qa_skill_data_corrected/${DATASET}/test.jsonl}"

declare -A MODEL_PATHS=(
  [qwen2_5_7b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
  [qwen2_5_3b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
  [llama3_2_3b]="/playpen-shared/jesse/cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/main"
)

mkdir -p outputs/baseline_rag/predictions_bslret outputs/baseline_rag/slurm "$PERT_ROOT"
LOG="outputs/baseline_rag/slurm/exp05_stress_$(date +%Y%m%d_%H%M%S).tsv"
printf "variant\tk\tmethod\tmodel\taugment\teval\n" > "$LOG"
submit() { if [[ "$DRY_RUN" == "1" ]]; then printf "DRY_RUN: %q " "$@" >&2; printf "\n" >&2; echo "dry_$RANDOM"; else "$@"; fi; }

for variant_spec in $KB_VARIANTS; do
  IFS=: read -r vname vmode varg <<<"$variant_spec"
  if [[ "$vmode" == "none" ]]; then
    kb_dir="$CLEAN_KB"
  else
    kb_dir="$PERT_ROOT/$vname"
    if [[ "$FORCE_AUGMENT" == "1" || ! -s "$kb_dir/facts.jsonl" ]]; then
      if [[ "$vmode" == "noisy" ]]; then
        pcmd=("$PYTHON" "$PERTURB" --kb-dir "$CLEAN_KB" --out-dir "$kb_dir" --mode noisy --noise-frac "$varg" --seed "$SEED")
      else
        pcmd=("$PYTHON" "$PERTURB" --kb-dir "$CLEAN_KB" --out-dir "$kb_dir" --mode decoy --decoy-per "$varg" --seed "$SEED")
      fi
      if [[ "$DRY_RUN" == "1" ]]; then printf "DRY_RUN: %q " "${pcmd[@]}" >&2; printf "\n" >&2; else "${pcmd[@]}"; fi
    fi
  fi

  for k in $TOPKS; do
    for method in $METHODS; do
      aug="outputs/baseline_rag/data/${DATASET}_${method}_${vname}_k${k}_test.jsonl"
      aug_job=""
      if [[ "$FORCE_AUGMENT" == "1" || ! -s "$aug" ]]; then
        aug_job=$(submit sbatch --parsable "${SB[@]}" --job-name="e5aug_${vname}_k${k}_${method}" \
          --export=ALL,INPUT="$TEST_SRC",OUTPUT="$aug",METHOD="$method",KB_DIR="$kb_dir",TOP_K="$k",LIMIT=0 \
          "$REPO/slurm/submit_augment_skillkb.sbatch")
      fi
      dep=(); [[ -n "$aug_job" ]] && dep=(--dependency=afterok:"$aug_job")
      for model in $MODELS; do
        pred="outputs/baseline_rag/predictions_bslret/${model}_${DATASET}_${method}_${vname}_k${k}_test.jsonl"
        eval_job=$(submit sbatch --parsable "${SB[@]}" --gres=gpu:1 "${dep[@]}" \
          --job-name="e5ev_${vname}_k${k}_${method}_${model}" \
          --export=ALL,MODEL="${MODEL_PATHS[$model]}",INPUT="$aug",OUTPUT="$pred",LIMIT="$LIMIT",MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
          "$REPO/slurm/submit_generic_qa_eval.sbatch")
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$vname" "$k" "$method" "$model" "${aug_job:-cached}" "$eval_job" | tee -a "$LOG"
      done
    done
  done
done
echo "Wrote submission log: $LOG"
