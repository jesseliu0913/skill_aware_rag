#!/bin/bash
# Exp06 -- wrong-skill / oracle-skill controls (RQ3, RQ6).
#
# Full RAG-SFT parity: for each (dataset, override) we augment BOTH the train and
# test splits with the skill-override forced router, retrain a per-condition LoRA
# on the augmented train, and eval on the augmented test. The ONLY thing that
# changes across the five conditions is which skill the augmenter routes on
# (--skill-override), so downstream deltas isolate the value of correct skill
# identification. Mirrors slurm/submit_ragsft_skillkb_matrix.sh conventions.
#
# The five conditions (override -> role):
#   none       uniform-ish current route (gold-field router; == oracle here)  [reference]
#   oracle     gold-field skill                                   [upper bound]
#   predicted  question-only deployable router                    [deployed system]
#   random     sha1(id)-hashed skill                              [negative control]
#   wrong      fixed confusion-map skill                          [sensitivity]
#
# We fix METHOD=skill_hybrid + SCHEMA=v1 so the skill route is actually consumed.
#
# Env knobs (same conventions as submit_ragsft_skillkb_matrix.sh):
#   MODELS     default "qwen2_5_7b qwen2_5_3b llama3_2_3b"
#   DATASETS   default "mol_instructions fdarxbench"
#   OVERRIDES  default "none oracle predicted random wrong"
#   METHOD     default skill_hybrid ; SCHEMA default v1
#   LIMIT / MAX_NEW_TOKENS / MAX_STEPS / SAVE_STEPS / MAX_LENGTH
#   FORCE_AUGMENT=1 to rebuild augmented splits ; DRY_RUN=1 to print only.
set -euo pipefail

cd /playpen-jfs/jesse/drug_microbiome

MODELS="${MODELS:-qwen2_5_7b qwen2_5_3b llama3_2_3b}"
DATASETS="${DATASETS:-mol_instructions fdarxbench}"
OVERRIDES="${OVERRIDES:-none oracle predicted random wrong}"
METHOD="${METHOD:-skill_hybrid}"
SCHEMA="${SCHEMA:-v1}"
LIMIT="${LIMIT:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
MAX_STEPS="${MAX_STEPS:-1125}"
SAVE_STEPS="${SAVE_STEPS:-375}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
FORCE_AUGMENT="${FORCE_AUGMENT:-0}"
DRY_RUN="${DRY_RUN:-1}"   # Exp06 default: DRY_RUN on. Set DRY_RUN=0 to submit.
ACCOUNT="${ACCOUNT:-jesseliu}"
QOS="${QOS:-normal}"
SB=(--account="$ACCOUNT" --qos="$QOS")

declare -A MODEL_PATHS=(
  [qwen2_5_7b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
  [qwen2_5_3b]="/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
  [llama3_2_3b]="/playpen-shared/jesse/cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/main"
)

mkdir -p outputs/exp06_skill_controls/slurm outputs/exp06_skill_controls/data
LOG="outputs/exp06_skill_controls/slurm/submissions_$(date +%Y%m%d_%H%M%S).tsv"
printf "dataset\toverride\tmodel\taugment_train\taugment_test\ttrain\teval\n" > "$LOG"

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
  for override in $OVERRIDES; do
    aug_train="outputs/exp06_skill_controls/data/${dataset}/train_${METHOD}_ovr-${override}.jsonl"
    aug_test="outputs/exp06_skill_controls/data/${dataset}/test_${METHOD}_ovr-${override}.jsonl"

    # ---- augment train + test with the forced skill route (CPU, shared across models) ----
    aug_train_job=""
    if [[ "$FORCE_AUGMENT" == "1" || ! -f "$aug_train" ]]; then
      aug_train_job=$(submit sbatch --parsable "${SB[@]}" \
        --job-name="augkb_${dataset}_ovr-${override}_train" \
        --export=ALL,INPUT="$corrected_train",OUTPUT="$aug_train",METHOD="$METHOD",SCHEMA="$SCHEMA",SKILL_OVERRIDE="$override",LIMIT=0 \
        skill_aware_rag/experiments/exp06_skill_controls/slurm/submit_augment_override.sbatch)
    fi
    aug_test_job=""
    if [[ "$FORCE_AUGMENT" == "1" || ! -f "$aug_test" ]]; then
      aug_test_job=$(submit sbatch --parsable "${SB[@]}" \
        --job-name="augkb_${dataset}_ovr-${override}_test" \
        --export=ALL,INPUT="$corrected_test",OUTPUT="$aug_test",METHOD="$METHOD",SCHEMA="$SCHEMA",SKILL_OVERRIDE="$override",LIMIT=0 \
        skill_aware_rag/experiments/exp06_skill_controls/slurm/submit_augment_override.sbatch)
    fi
    dep_train=(); [[ -n "$aug_train_job" ]] && dep_train=(--dependency=afterok:"$aug_train_job")

    for model in $MODELS; do
      model_path="${MODEL_PATHS[$model]}"
      adapter="outputs/exp06_skill_controls/adapters/${model}_${dataset}_${METHOD}_ovr-${override}"
      pred="outputs/exp06_skill_controls/predictions/${model}_${dataset}_${METHOD}_ovr-${override}_test.jsonl"

      train_job=$(submit sbatch --parsable "${SB[@]}" "${dep_train[@]}" \
        --job-name="${model}_${dataset}_ovr-${override}_train" \
        --export=ALL,MODEL="$model_path",TRAIN="$aug_train",DEV="",OUTPUT_DIR="$adapter",MAX_STEPS="$MAX_STEPS",SAVE_STEPS="$SAVE_STEPS",MAX_LENGTH="$MAX_LENGTH" \
        skill_aware_rag/slurm/submit_generic_qa_lora_train.sbatch)

      dep_eval=(--dependency=afterok:"$train_job")
      [[ -n "$aug_test_job" ]] && dep_eval=(--dependency="afterok:${train_job}:${aug_test_job}")
      eval_job=$(submit sbatch --parsable "${SB[@]}" "${dep_eval[@]}" \
        --job-name="${model}_${dataset}_ovr-${override}_eval" \
        --export=ALL,MODEL="$model_path",INPUT="$aug_test",OUTPUT="$pred",ADAPTER="$adapter",LIMIT="$LIMIT",MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
        skill_aware_rag/slurm/submit_generic_qa_eval.sbatch)

      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$dataset" "$override" "$model" "${aug_train_job:-cached}" "${aug_test_job:-cached}" "$train_job" "$eval_job" | tee -a "$LOG"
    done
  done
done

echo "Wrote submission log: $LOG"
echo "NOTE: DRY_RUN=${DRY_RUN}. This driver was authored with DRY_RUN=1 default; set DRY_RUN=0 to actually submit."
