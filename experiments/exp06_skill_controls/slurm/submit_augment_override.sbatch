#!/bin/bash
# Exp06 CPU-only augmentation that forwards --skill-override to the shared skill
# augmenter. Self-contained (does not touch slurm/submit_augment_skillkb.sbatch)
# so it can pass the extra flag. With SKILL_OVERRIDE=none this is byte-identical
# to slurm/submit_augment_skillkb.sbatch output.
#
# Env: INPUT, OUTPUT, METHOD, SCHEMA(default v1), SKILL_OVERRIDE(default none), LIMIT(default 0)
#SBATCH --job-name=exp06_augment_override
#SBATCH --partition=ada
#SBATCH --account=jesseliu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/exp06_skill_controls/slurm/%x-%j.out
#SBATCH --error=outputs/exp06_skill_controls/slurm/%x-%j.err

set -euo pipefail

cd /playpen-jfs/jesse/drug_microbiome
mkdir -p outputs/exp06_skill_controls/slurm outputs/exp06_skill_controls/data

: "${INPUT:?Set INPUT to a corrected-split JSONL}"
: "${OUTPUT:?Set OUTPUT to the augmented JSONL path}"
: "${METHOD:?Set METHOD to bm25|kg|hybrid|skill_hybrid|bm25_skillrouted}"

SCHEMA="${SCHEMA:-v1}"
SKILL_OVERRIDE="${SKILL_OVERRIDE:-none}"
LIMIT="${LIMIT:-0}"
PYTHON=/playpen-shared/jesse/miniconda3/envs/agentskill/bin/python

"$PYTHON" skill_aware_rag/retrieval/augment_with_skill_kb.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --method "$METHOD" \
  --schema-version "$SCHEMA" \
  --skill-override "$SKILL_OVERRIDE" \
  --limit "$LIMIT"
