#!/bin/bash
#SBATCH --job-name=kb_embed
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
mkdir -p outputs/baseline_rag/slurm outputs/knowledge_bank/embeddings outputs/triton_cache
export TRITON_CACHE_DIR=/playpen-jfs/jesse/drug_microbiome/outputs/triton_cache
export HF_HOME=/playpen-shared/jesse/cache
export HF_HUB_OFFLINE=1

ENCODER="${ENCODER:?Set ENCODER to bge or medcpt}"
BATCH_SIZE="${BATCH_SIZE:-256}"

/playpen-shared/jesse/miniconda3/envs/agentskill/bin/python skill_aware_rag/kb/build_kb_embeddings.py \
  --encoder "$ENCODER" \
  --batch-size "$BATCH_SIZE"
