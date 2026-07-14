#!/bin/bash
# Exp04 (RQ5) — skill-schema COMPONENT ABLATION matrix.
#
# Isolates which part of the skill schema matters ("is this just prompt
# engineering?"). Every variant is the full-schema skill method (skill_schema_v1
# = skill_hybrid + schema v1) with a subset of components REMOVED via the additive
# `--ablate` flag on retrieval/augment_with_skill_kb.py. The additive grid runs
# from uniform RAG (all components removed) up to full SkillRAG (none removed).
#
# 7 variants (component present = kept, absent = ablated):
#   variant           skill_id  source_routing  solving_steps  answer_format  evidence_org
#   v0_uniform          -            -               -              -              -
#   v1_skill_label      +            -               -              -              -
#   v2_source_routing   +            +               -              -              -
#   v3_solving_steps    +            -               +              -              -
#   v4_answer_format    +            -               -              +              -
#   v5_evidence_org     +            +               -              +              +
#   v6_full             +            +               +              +              +
# (v6_full ablate="" is byte-identical to the existing skill_schema_v1 runs.)
#
# Per variant x dataset x model: augment corrected TRAIN + TEST with that ablation,
# train a LoRA (RAG-SFT), eval on the augmented test. Same task-level metrics as
# Exp02. Mirrors slurm/submit_nolabel_matrix.sh (afterok dependency chaining).
#
# Env: MODELS DATASETS VARIANTS LIMIT MAX_STEPS ... PARTITION FORCE_AUGMENT DRY_RUN=1.
#   DRY_RUN=1 prints the sbatch commands instead of submitting (default here).
set -euo pipefail
cd /playpen-jfs/jesse/drug_microbiome

MODELS="${MODELS:-qwen2_5_7b qwen2_5_3b llama3_2_3b}"
DATASETS="${DATASETS:-fdarxbench mol_instructions}"
VARIANTS="${VARIANTS:-v0_uniform v1_skill_label v2_source_routing v3_solving_steps v4_answer_format v5_evidence_org v6_full}"
LIMIT="${LIMIT:-1000}"; MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
MAX_STEPS="${MAX_STEPS:-1125}"; SAVE_STEPS="${SAVE_STEPS:-375}"; MAX_LENGTH="${MAX_LENGTH:-2048}"
DRY_RUN="${DRY_RUN:-1}"; FORCE_AUGMENT="${FORCE_AUGMENT:-0}"
ACCOUNT="${ACCOUNT:-jesseliu}"; QOS="${QOS:-normal}"; PARTITION="${PARTITION:-a100}"
SB=(--account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION")
CORR="outputs/qa_skill_data_corrected"
AUGSB="skill_aware_rag/experiments/exp04_schema_ablations/slurm/submit_augment_ablate.sbatch"

# variant -> components to REMOVE (empty => full SkillRAG). Joined with '+' (NOT
# commas): sbatch --export is itself comma-delimited, so the augment sbatch
# translates '+' back to the comma list the augmenter's --ablate expects.
declare -A ABLATE=(
  [v0_uniform]="skill_id+source_routing+solving_steps+answer_format+evidence_org"
  [v1_skill_label]="source_routing+solving_steps+answer_format+evidence_org"
  [v2_source_routing]="solving_steps+answer_format+evidence_org"
  [v3_solving_steps]="source_routing+answer_format+evidence_org"
  [v4_answer_format]="source_routing+solving_steps+evidence_org"
  [v5_evidence_org]="solving_steps"
  [v6_full]=""
)

declare -A MODEL_PATHS=(
  [qwen2_5_7b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
  [qwen2_5_3b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
  [llama3_2_3b]="/playpen-shared/jesse/cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/main"
)

mkdir -p outputs/baseline_rag/predictions_ablate outputs/baseline_rag/data outputs/baseline_rag/slurm
LOG="outputs/baseline_rag/slurm/ablate_$(date +%Y%m%d_%H%M%S).tsv"
printf "variant\tdataset\tmodel\ttrain\teval\n" > "$LOG"
submit() { if [[ "$DRY_RUN" == "1" ]]; then printf "DRY_RUN: %q " "$@" >&2; printf "\n" >&2; echo "dry_$RANDOM"; else "$@"; fi; }

for ds in $DATASETS; do
  for variant in $VARIANTS; do
    abl="${ABLATE[$variant]}"
    train_src="outputs/baseline_rag/data/${ds}_${variant}_train.jsonl"
    eval_src="outputs/baseline_rag/data/${ds}_${variant}_test.jsonl"
    augtrain_job=""; augtest_job=""
    if [[ "$FORCE_AUGMENT" == "1" || ! -s "$train_src" ]]; then
      augtrain_job=$(submit sbatch --parsable "${SB[@]}" --job-name="ablaugtr_${ds}_${variant}" \
        --export=ALL,INPUT="$CORR/${ds}/train.jsonl",OUTPUT="$train_src",ABLATE="$abl",LIMIT=0 "$AUGSB")
    fi
    if [[ "$FORCE_AUGMENT" == "1" || ! -s "$eval_src" ]]; then
      augtest_job=$(submit sbatch --parsable "${SB[@]}" --job-name="ablaugte_${ds}_${variant}" \
        --export=ALL,INPUT="$CORR/${ds}/test.jsonl",OUTPUT="$eval_src",ABLATE="$abl",LIMIT=0 "$AUGSB")
    fi
    dtr=(); [[ -n "$augtrain_job" ]] && dtr=(--dependency=afterok:"$augtrain_job")
    for model in $MODELS; do
      adapter="outputs/baseline_rag/adapters/${model}_${ds}_${variant}_ablate"
      pred="outputs/baseline_rag/predictions_ablate/${model}_${ds}_${variant}_ablate_test.jsonl"
      train_job=$(submit sbatch --parsable "${SB[@]}" "${dtr[@]}" --job-name="${model}_${ds}_${variant}_abltrain" \
        --export=ALL,MODEL="${MODEL_PATHS[$model]}",TRAIN="$train_src",DEV="",OUTPUT_DIR="$adapter",MAX_STEPS="$MAX_STEPS",SAVE_STEPS="$SAVE_STEPS",MAX_LENGTH="$MAX_LENGTH" \
        skill_aware_rag/slurm/submit_generic_qa_lora_train.sbatch)
      edep="afterok:$train_job"; [[ -n "$augtest_job" ]] && edep="afterok:$train_job:$augtest_job"
      eval_job=$(submit sbatch --parsable "${SB[@]}" --dependency="$edep" --job-name="${model}_${ds}_${variant}_ableval" \
        --export=ALL,MODEL="${MODEL_PATHS[$model]}",INPUT="$eval_src",OUTPUT="$pred",ADAPTER="$adapter",LIMIT="$LIMIT",MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
        skill_aware_rag/slurm/submit_generic_qa_eval.sbatch)
      printf "%s\t%s\t%s\t%s\t%s\n" "$variant" "$ds" "$model" "$train_job" "$eval_job" | tee -a "$LOG"
    done
  done
done
echo "Wrote submission log: $LOG"
