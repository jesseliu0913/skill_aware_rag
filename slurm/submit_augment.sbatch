#!/bin/bash
#SBATCH --job-name=rag_augment
#SBATCH --partition=ada
#SBATCH --account=jesseliu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/baseline_rag/slurm/%x-%j.out
#SBATCH --error=outputs/baseline_rag/slurm/%x-%j.err

set -euo pipefail

cd /playpen-jfs/jesse/drug_microbiome
mkdir -p outputs/baseline_rag/slurm outputs/baseline_rag/data outputs/triton_cache
export TRITON_CACHE_DIR=/playpen-jfs/jesse/drug_microbiome/outputs/triton_cache
export HF_HOME=/playpen-shared/jesse/cache
export HF_HUB_OFFLINE=1

: "${INPUT:?Set INPUT to a corrected-split JSONL}"
: "${OUTPUT:?Set OUTPUT to the augmented JSONL path}"
: "${METHOD:?Set METHOD to dense_bge|dense_medcpt|rrf|rerank}"

LIMIT="${LIMIT:-0}"

/playpen-shared/jesse/miniconda3/envs/agentskill/bin/python skill_aware_rag/retrieval/augment_baseline_rag.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --method "$METHOD" \
  --limit "$LIMIT"
