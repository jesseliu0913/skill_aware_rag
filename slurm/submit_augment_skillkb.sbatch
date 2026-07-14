#!/bin/bash
# CPU-only augmentation for the sparse/graph/skill retrieval family, so those
# methods can be retrieval-augmented at TRAIN time (RAG-SFT), not just at eval.
# Reuses skill_aware_rag/retrieval/augment_with_skill_kb.py (the proposed method's own augmenter),
# so prompts/records are byte-identical to the existing test_<method>.jsonl runs.
#
# METHOD is a display label; skill_schema_v1 maps to skill_hybrid + schema v1,
# matching how the existing test augmentations were built.
#SBATCH --job-name=rag_augment_skillkb
#SBATCH --partition=ada
#SBATCH --account=jesseliu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/baseline_rag/slurm/%x-%j.out
#SBATCH --error=outputs/baseline_rag/slurm/%x-%j.err

set -euo pipefail

cd /playpen-jfs/jesse/drug_microbiome
mkdir -p outputs/baseline_rag/slurm outputs/baseline_rag/data

: "${INPUT:?Set INPUT to a corrected-split JSONL}"
: "${OUTPUT:?Set OUTPUT to the augmented JSONL path}"
: "${METHOD:?Set METHOD to bm25|kg|hybrid|skill_hybrid|skill_schema_v1}"

LIMIT="${LIMIT:-0}"
PYTHON=/playpen-shared/jesse/miniconda3/envs/agentskill/bin/python

# Map the display label to augmenter flags, matching the existing test runs.
case "$METHOD" in
  skill_schema_v1) AUG_METHOD=skill_hybrid; SCHEMA=v1 ;;
  bm25_skillrouted) AUG_METHOD=bm25_skillrouted; SCHEMA=v1 ;;
  bm25|kg|hybrid|skill_hybrid) AUG_METHOD="$METHOD"; SCHEMA=legacy ;;
  *) echo "Unknown METHOD=$METHOD" >&2; exit 2 ;;
esac

"$PYTHON" skill_aware_rag/retrieval/augment_with_skill_kb.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --method "$AUG_METHOD" \
  --schema-version "$SCHEMA" \
  --limit "$LIMIT"
