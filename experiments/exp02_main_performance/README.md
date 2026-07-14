# Exp02 — Main task performance + oracle-source upper bound (RQ4)

**Claim.** Skill-aware retrieval planning improves downstream answers, and it does
so by moving toward — but never past — an *oracle-source* upper bound that is
allowed to route on gold task labels. Concretely we test, per task, the ordering

```
uniform-RAG   ≤   SkillRAG   ≤   oracle_source
```

**RQ served.** RQ4 (does better planning improve downstream answers, and how close
does it get to an oracle-source bound?). Paper §5.2, main results table.

---

## What this experiment adds

| Piece | File | Reuse vs build |
|---|---|---|
| (A) `oracle_source` augmenter | `oracle_source_augment.py` | **build** — thin retriever; imports `retrieval.augment_with_skill_kb` (KB loader, `augment_record`, gold-skill router, `SKILL_SCHEMA_V1`) + `retrieval.augment_baseline_rag.load_embeddings` + `retrieval.retrievers` (`DenseIndex`, `build_encoder`, `fact_to_evidence`). Nothing duplicated. |
| (B) Main-table collator | `collate_main_table.py` | **build** — post-hoc, offline; parses the same `*_eval.json` schema as `eval/aggregate_results.py` and folds in `results_summary*.csv`. |
| Matrix driver | `slurm/submit_oracle_matrix.sh` + `slurm/submit_augment_oracle.sbatch` | **build** — mirrors `slurm/submit_nolabel_matrix.sh` (env-var + `DRY_RUN`); reuses the shared generic train/eval sbatches. |
| Tests | `test_oracle_source_augment.py`, `test_collate_main_table.py` | stdlib `unittest`, offline. |

**Additive only.** No shared/baseline file is edited. `oracle_source` lives in a
standalone augmenter (not a new `--method` in `augment_baseline_rag.py`) precisely
because its exhaustive per-record subset search does not fit that script's single
global-index control flow; keeping it standalone leaves the default dense path
byte-identical.

---

## (A) What `oracle_source` is — and how it differs from `dense_bge_skillrouted`

`oracle_source` = the strong dense retriever **dense_bge** (`BAAI/bge-large-en-v1.5`)
run over a candidate pool **hard-restricted, per record, to the KB sources allowed
for that record's GOLD-labelled skill**. The gold skill comes from
`infer_schema_v1_skill(record)`, which reads the gold `source` / `task_type` fields;
the allowed sources are `SKILL_SCHEMA_V1[gold_skill]["sources"]`. The dense top-k is
taken **exhaustively over the entire allowed-source subset of the KB**.

This is an **upper bound**: it consumes gold task labels, so it is *not deployable*.
The deployed SkillRAG must instead route on a **question-only predicted** skill
(Exp06 studies predicted-vs-oracle skill). `oracle_source` isolates the ceiling that
perfect source routing on top of a strong dense retriever could reach.

### Relationship to the existing `dense_bge_skillrouted`

Both `oracle_source` and `dense_bge_skillrouted`
(`retrieval/augment_baseline_rag.py`) route on the **same gold-field rule**
(`infer_schema_v1_skill`) with the **same retriever and evidence budget** — so
neither is deployable, and both are oracle-flavoured. The difference is *how* the
source restriction is applied:

| | `dense_bge_skillrouted` | `oracle_source` (this exp) |
|---|---|---|
| Routing signal | gold skill (`infer_schema_v1_skill`) | gold skill (`infer_schema_v1_skill`) |
| Retriever / budget | dense_bge, top-k | dense_bge, top-k |
| Candidate pool | dense-search whole KB for a **fixed 300-deep pool**, then **post-filter** by allowed sources | **restrict to allowed sources first**, then dense-search that subset **exhaustively** |
| Failure mode | allowed-source facts outside the top-300 are lost | none — sees every allowed-source fact |
| Status | approximation of the routing ceiling | **true routing ceiling (upper bound)** |

By construction `dense_bge_skillrouted ≤ oracle_source` (same retriever, strictly
larger effective candidate set for the allowed sources). That is exactly why
`oracle_source` sits at the top of the ordering hypothesis and
`dense_bge_skillrouted` is the deployed-style comparison beneath it.

Everything except which facts land in `retrieved_kb_evidence` is delegated to the
shared `augment_with_skill_kb.augment_record`, so emitted
`prompt` / `messages` / `skill_schema` are byte-identical in shape to every other
method — the retriever is the only independent variable.

---

## (B) Ordering check

`collate_main_table.py` globs the eval artifacts that exist, builds the tidy
per-task table, and for every `(model, dataset, subtask, variant)` cell where all
three roles are present tests

```
uniform-method  ≤  skill-method  ≤  oracle-method     (within --tol)
```

reporting where it **HOLDS** and where it is **VIOLATED** (upper vs lower), with the
per-cell gaps. Roles default to `dense_bge` / `dense_bge_skillrouted` /
`oracle_source` and are overridable (`--uniform-method`, `--skill-method`,
`--oracle-method`). Headline metric per task: `numeric_acc` when present (molecule
property), else `token_f1`. `--tol` (default 0) absorbs measurement noise
(SE ≈ 0.012 at n=782).

---

## How to run

Offline verification (no GPU, no outputs/ needed):

```bash
# unit tests
python3 -m unittest experiments.exp02_main_performance.test_collate_main_table -v
python3 experiments/exp02_main_performance/test_oracle_source_augment.py -v   # needs numpy+retrievers

# CLIs
python3 experiments/exp02_main_performance/oracle_source_augment.py --help
python3 experiments/exp02_main_performance/collate_main_table.py --help
```

Collate whatever eval files exist (safe even before any run):

```bash
python3 experiments/exp02_main_performance/collate_main_table.py \
  --out-csv outputs/baseline_rag/exp02_main_table.csv \
  --out-md  outputs/baseline_rag/exp02_main_table.md
```

Augment one split with `oracle_source` (GPU; needs the KB + bge embeddings):

```bash
python3 experiments/exp02_main_performance/oracle_source_augment.py \
  --input  outputs/qa_skill_data_corrected/fdarxbench/test.jsonl \
  --output outputs/baseline_rag/data/fdarxbench/test_oracle_source_raglora.jsonl
```

Submit the matrix (dry-run first — **not** auto-submitted):

```bash
DRY_RUN=1 bash experiments/exp02_main_performance/slurm/submit_oracle_matrix.sh          # corrected, both datasets
DRY_RUN=1 MODE=nolabel bash experiments/exp02_main_performance/slurm/submit_oracle_matrix.sh
```

Run paths mirror the consolidated layout (`cd /playpen-jfs/jesse/drug_microbiome`,
scripts referenced as `skill_aware_rag/...`), matching the other slurm drivers.

See `RESULTS.md` for the (empty) results template.
