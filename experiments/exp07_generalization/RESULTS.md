# Exp07 — Generalization results (RQ7)

> Template. Filled only when runs land. Nothing here is a claim yet.

## 1. Held-out drug / molecule splits (beyond memorized identity)

Split provenance (from `make_splits.py … --mode drug_holdout`, disjointness proof
in each split's `split_summary.json`).

| Dataset | Seed | test-frac | #train | #test | #test identities | disjoint |
|---|---|---|---|---|---|---|
| FDARxBench | | | | | | |
| Mol-Instructions | | | | | | |

Downstream (routing gain under the disjoint-identity split, per model):

| Dataset | Model | Uniform RAG | SkillRAG | Δ |
|---|---|---|---|---|

Not yet run.

## 2. Held-out skill-family splits (declarative skill addition)

| Held-out skill | #train | #test | holdout_skill_in_train | Model | SkillRAG (held-out skill) |
|---|---|---|---|---|---|

Not yet run.

## 3. Held-out source splits (optional)

| Held-out source | #train | #test | Model | SkillRAG |
|---|---|---|---|---|

Not yet run.

## 4. Retriever-backend sweep (retriever-agnostic)

Routing gain (`skill − uniform`) per backend × model; positive on every backend ⇒
retriever-agnostic. See `backend_sweep.md`.

| Backend | Model | Uniform | Skill-routed | Δ |
|---|---|---|---|---|
| bm25 | | | | |
| dense_bge | | | | |
| hybrid | | | | |
| rerank | | | | |

Not yet run.

## 5. New-KB-source extensibility (🔴 new-data)

Deferred: requires a genuinely new biomedical KB source (design note in README).

Not yet run.
