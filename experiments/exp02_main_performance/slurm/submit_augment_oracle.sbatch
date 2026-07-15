#!/bin/bash
#SBATCH --job-name=oracle_augment
#SBATCH --partition=ada
#SBATCH --account=jesseliu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/baseline_rag/slurm/%x-%j.out
#SBATCH --error=outputs/baseline_rag/slurm/%x-%j.err

# Augment a split with the Exp02 oracle_source upper bound (dense_bge restricted to
# the GOLD-skill's allowed KB sources). Mirrors slurm/submit_augment.sbatch env
# contract (INPUT/OUTPUT/LIMIT) but runs the standalone oracle augmenter; METHOD is
# fixed to oracle_source.
set -euo pipefail

cd /playpen-jfs/jesse/drug_microbiome
mkdir -p outputs/baseline_rag/slurm outputs/baseline_rag/data outputs/triton_cache
export TRITON_CACHE_DIR=/playpen-jfs/jesse/drug_microbiome/outputs/triton_cache
export HF_HOME=/playpen-shared/jesse/cache
export HF_HUB_OFFLINE=1

: "${INPUT:?Set INPUT to a corrected-split JSONL}"
: "${OUTPUT:?Set OUTPUT to the augmented JSONL path}"

LIMIT="${LIMIT:-0}"
SCHEMA="${SCHEMA:-legacy}"

/playpen-shared/jesse/miniconda3/envs/agentskill/bin/python \
  skill_aware_rag/experiments/exp02_main_performance/oracle_source_augment.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --schema-version "$SCHEMA" \
  --limit "$LIMIT"
