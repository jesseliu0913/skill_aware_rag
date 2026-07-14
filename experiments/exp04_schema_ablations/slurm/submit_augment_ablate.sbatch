#!/bin/bash
# CPU-only augmentation for one Exp04 (RQ5) schema-ablation variant. Reuses the
# proposed method's own augmenter (retrieval/augment_with_skill_kb.py) at the
# full-schema base (method=skill_hybrid + schema v1 = "skill_schema_v1"), passing
# --ablate to REMOVE the components for this variant. ABLATE="" reproduces the
# full SkillRAG prompts byte-for-byte, so the un-ablated cell matches the
# existing skill_schema_v1 augmentation exactly.
#
# Env: INPUT OUTPUT ABLATE (comma-separated components to remove) LIMIT.
#SBATCH --job-name=rag_augment_ablate
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
# The matrix driver passes ABLATE with '+' separators (sbatch --export is itself
# comma-delimited); restore the comma list the augmenter's --ablate expects.
ABLATE="${ABLATE:-}"; ABLATE="${ABLATE//+/,}"
LIMIT="${LIMIT:-0}"
PYTHON=/playpen-shared/jesse/miniconda3/envs/agentskill/bin/python

"$PYTHON" skill_aware_rag/retrieval/augment_with_skill_kb.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --method skill_hybrid \
  --schema-version v1 \
  --ablate "$ABLATE" \
  --limit "$LIMIT"
