#!/bin/bash
set -euo pipefail

cd /playpen-jfs/jesse/drug_microbiome

PYTHON="${PYTHON:-/playpen-shared/jesse/miniconda3/envs/agentskill/bin/python}"
DATASETS="${DATASETS:-mol_instructions fdarxbench}"
METHODS="${METHODS:-bm25 kg hybrid skill_schema_v1}"
LIMIT="${LIMIT:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_AUGMENT="${FORCE_AUGMENT:-0}"

KB_DIR="outputs/knowledge_bank/unified"
SUBMIT_LOG="outputs/slurm/retrieval_comparison_submissions_$(date +%Y%m%d_%H%M%S).tsv"

if [[ ! -f "$KB_DIR/facts.jsonl" || ! -f "$KB_DIR/alias_index.json" ]]; then
  echo "Missing unified KB under $KB_DIR. Run skill_aware_rag/kb/build_unified_kb.py first." >&2
  exit 1
fi

mkdir -p outputs/slurm
printf "dataset\tmethod\tmodel\traw_eval\tlora_eval\n" > "$SUBMIT_LOG"

input_path_for_dataset() {
  case "$1" in
    mol_instructions)
      printf "outputs/qa_skill_data_corrected/mol_instructions/test.jsonl"
      ;;
    fdarxbench)
      printf "outputs/qa_skill_data_corrected/fdarxbench/test.jsonl"
      ;;
    *)
      echo "Unknown dataset: $1" >&2
      return 2
      ;;
  esac
}

output_path_for_method() {
  local dataset="$1"
  local method="$2"
  printf "outputs/qa_skill_data_skillkb/%s/test_%s.jsonl" "$dataset" "$method"
}

build_augmented_input() {
  local dataset="$1"
  local method="$2"
  local input
  local output
  input="$(input_path_for_dataset "$dataset")"
  output="$(output_path_for_method "$dataset" "$method")"

  if [[ "$FORCE_AUGMENT" != "1" && -f "$output" ]]; then
    echo "exists: $output"
    return 0
  fi

  if [[ "$method" == "skill_schema_v1" ]]; then
    "$PYTHON" skill_aware_rag/retrieval/augment_with_skill_kb.py \
      --input "$input" \
      --output "$output" \
      --method skill_hybrid \
      --schema-version v1
  else
    "$PYTHON" skill_aware_rag/retrieval/augment_with_skill_kb.py \
      --input "$input" \
      --output "$output" \
      --method "$method"
  fi
}

for dataset in $DATASETS; do
  for method in $METHODS; do
    build_augmented_input "$dataset" "$method"
    submit_output="$(
      DATASET="$dataset" \
      METHOD="$method" \
      LIMIT="$LIMIT" \
      MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
      DRY_RUN="$DRY_RUN" \
      bash skill_aware_rag/slurm/submit_skillkb_eval_matrix.sh
    )"
    printf "%s\n" "$submit_output"
    printf "%s\n" "$submit_output" >> "$SUBMIT_LOG"
  done
done

echo "Wrote submission log: $SUBMIT_LOG"
