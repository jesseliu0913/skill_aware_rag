# Exp03 — Retrieval-planning quality diagnostics (RQ2, RQ3)

**Serves:** RQ2 — *Does uniform retrieval pull from wrong/unnecessary sources?* and
RQ3 — *Does the skill schema improve retrieval planning?*
**Paper:** §5.3 (often more important than the downstream score table).
**Status:** 🟡 build — new post-hoc script over existing augmented JSONL.

## Claim under test
Uniform whole-KB retrievers (bm25, dense, kg, hybrid, rrf, rerank) frequently
retrieve evidence from KB sources that are wrong for the task, while skill-routed
retrievers concentrate retrieval on the skill's allowed sources — measurable
directly on the `retrieved_kb_evidence` already attached to each record, with no
retraining.

## What already exists (reuse — imported, not duplicated)
- `retrieval/augment_with_skill_kb.py`: `infer_schema_v1_skill(record)` (the
  gold-skill router) and `SKILL_SCHEMA_V1[skill]["sources"]` (the allowed-source
  sets). Imported so "allowed sources" here never drift from the retriever's own
  definition.
- `eval/evaluate_instruction_outputs.py`: `normalize_text` — the exact token
  normalization used by the answer metrics (lowercase, strip punctuation, collapse
  whitespace). **Imported** (not replicated) and reused for `evidence_recall@k`
  and `evidence_utilization`.

## What this experiment adds (build)
`retrieval_planning_metrics.py` — pure post-hoc analysis over a glob of augmented
`*.jsonl`. Per (dataset = record `source`, `task_type`, `method`) it computes:

| metric | definition |
|---|---|
| `source_precision@k` | fraction of retrieved facts whose `source` ∈ gold-skill allowed sources |
| `wrong_source_rate` | `1 − source_precision@k` |
| `evidence_recall@k` | fraction of normalized gold-answer tokens present in the concatenated retrieved evidence text |
| `source_entropy_bits` | Shannon entropy (bits) over the retrieved facts' sources |
| `skill_source_agreement` | Jaccard(set(retrieved sources), schema allowed sources) |
| `evidence_utilization` | *(optional, needs `--predictions-glob)*` fraction of normalized `prediction` tokens present in the retrieved evidence text |

`method` is inferred from each filename via `--method-from-filename-regex`
(default recognizes the method roster, longest token first so `bm25_skillrouted`
beats `bm25`); falls back to the file stem on no match. Predictions are matched to
augmented records by `(method, id)`.

## Run
```bash
# core diagnostics over all augmented splits
python experiments/exp03_retrieval_planning/retrieval_planning_metrics.py \
  --augmented-glob 'outputs/baseline_rag/data/*_*.jsonl'

# add evidence_utilization by supplying matching prediction JSONL
python experiments/exp03_retrieval_planning/retrieval_planning_metrics.py \
  --augmented-glob   'outputs/baseline_rag/data/*_*.jsonl' \
  --predictions-glob 'outputs/baseline_rag/predictions_bslret/*_*.jsonl'

# custom method-token parsing
python experiments/exp03_retrieval_planning/retrieval_planning_metrics.py \
  --augmented-glob 'mydir/*.jsonl' \
  --method-from-filename-regex '__(?P<method>[a-z0-9_]+)__'

# tests (offline, stdlib unittest)
cd experiments/exp03_retrieval_planning && python -m unittest test_retrieval_planning_metrics -v
```

Outputs → `experiments/exp03_retrieval_planning/results/` (gitignored path is
`outputs/`; these small CSV/report files are committed once real runs land):
`retrieval_planning_metrics.csv`, `REPORT.md`.

## Notes / extensions
- Metric components with no retrieved evidence are dropped from that group's mean
  (they are undefined, not zero); `evidence_recall@k` is still defined (0.0) when
  evidence is empty but gold tokens exist.
- `evidence_utilization` is a grounding proxy (how much of the answer is lexically
  in the evidence), not a causal usage measure.
- Extension: per-source confusion (which wrong source is over-retrieved per skill).
