# Exp07 — Retriever-backend sweep (retriever-agnostic claim)

**Question (RQ7).** Is the skill schema a *planning layer that helps regardless of
the underlying retriever*, or does it only pay off with one particular backend?

**Design.** Hold everything fixed except the retriever backend and whether skill
routing is on. For each backend we run the standard RAG-SFT pipeline twice —
*uniform* (whole-KB retrieval) vs *skill-routed* (retrieval scoped to the routed
skill's allowed sources) — and read the routing gain. A positive gain on *every*
backend is the retriever-agnostic result.

The sweep **reuses existing augment methods** (no new retriever code): skill
routing is already implemented per backend as a distinct method name.

| Backend | Family | Uniform method | Skill-routed method | Augment sbatch |
|---|---|---|---|---|
| `bm25` | sparse | `bm25` | `bm25_skillrouted` | `submit_augment_skillkb.sbatch` |
| `dense_bge` | dense | `dense_bge` | `dense_bge_skillrouted` | `submit_augment.sbatch` |
| `hybrid` | sparse+graph | `hybrid` | `skill_hybrid` | `submit_augment_skillkb.sbatch` |
| `rerank` | fusion+cross-encoder | `rerank` | *(none yet — 🔴)* | `submit_augment.sbatch` |

**rerank skill-routed gap (🔴).** There is no shipped `rerank_skillrouted` augment
method. Adding it is a small additive change mirroring `dense_bge_skillrouted`:
scope the first-stage candidate pool to `SKILL_SCHEMA_V1[skill]["sources"]` before
the cross-encoder rerank (add a `LOCK_SOURCE_FOR_METHOD`-style entry, or a
`skillrouted` flag, in `retrieval/augment_baseline_rag.py`). Until then the driver
skips the degenerate `rerank/skill == rerank/uniform` pair unless `RERANK_SKILL=1`.
This branch does **not** modify the shared augmenter (additive-only), so the rerank
skill cell is documented as future work rather than built here.

## Pipeline per (backend, routing, model)

```
augment TRAIN (method)  ─┐
augment TEST  (method)  ─┼─►  LoRA train (RAG-SFT on aug TRAIN)  ─►  eval on aug TEST
                         │        (submit_generic_qa_lora_train)     (submit_generic_qa_eval → *_eval.json)
```

- **Inputs.** Point `TRAIN`/`TEST` at a holdout split from `make_splits.py` (default
  `outputs/exp07_generalization/drug_holdout/{train,test}.jsonl`) to combine the
  backend sweep with the beyond-memorization test, or at the standard corrected
  splits for an in-distribution sweep.
- **RAG-SFT invariant.** Both TRAIN and TEST are augmented by the same method, per
  the plan's methodological invariant (no train/test prompt shift).
- **Aggregation.** `*_eval.json` files are consumable by
  `eval/aggregate_results.py`; compare the `uniform` vs `skill` variant per
  backend × model.

## Run (DRY_RUN — nothing submitted)

```bash
# Prints every sbatch line; submits nothing. DRY_RUN defaults to 1.
DRY_RUN=1 bash experiments/exp07_generalization/slurm/submit_backend_sweep_matrix.sh

# Narrow the grid while dry-running:
DRY_RUN=1 MODELS=qwen2_5_7b BACKENDS="bm25 dense_bge" \
  bash experiments/exp07_generalization/slurm/submit_backend_sweep_matrix.sh

# Real submission (only when ready): DRY_RUN=0, splits must already exist.
```

**Env knobs.** `MODELS BACKENDS TRAIN TEST OUTROOT LIMIT MAX_NEW_TOKENS MAX_STEPS
RERANK_SKILL DRY_RUN ACCOUNT QOS`.

## Expected reading

Retriever-agnostic ⇔ `skill − uniform` routing gain is `> 0` for bm25, dense_bge,
and hybrid (and rerank once its skill-routed method lands), across models. A gain
concentrated on one backend would instead indicate the schema is entangled with
that retriever rather than acting as a general planning layer.
