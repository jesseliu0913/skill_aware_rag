#!/bin/bash
# PROPOSED METHOD (skill_schema_v1) under the exact same protocol as the baselines
# (submit_baseline_raweval_matrix.sh):
#   TRAIN : augment the train split with the skill schema -- routed skill + its
#           solving steps + skill-based (source-constrained) retrieved knowledge,
#           all built on the same base prompt as the baselines -- then train a LoRA.
#   EVAL  : the RAW dataset question only (base prompt, no skill, no retrieval),
#           byte-identical to what every baseline is evaluated on.
# So our method vs the baselines differ ONLY in what the LoRA was trained on; the
# test input is the same. Thin wrapper so the proposed run is a one-liner and stays
# in lockstep with the baseline orchestrator (no duplicated logic to drift).
#
# Env knobs pass straight through (MODELS, DATASETS, LIMIT, DRY_RUN=1, ...).
set -euo pipefail
cd /playpen-jfs/jesse/drug_microbiome
METHODS="skill_schema_v1" exec bash skill_aware_rag/slurm/submit_baseline_raweval_matrix.sh "$@"
