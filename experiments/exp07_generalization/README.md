# Exp07 — Generalization to shifted / unseen settings (RQ7)

**Serves:** RQ7 — *Does skill-aware planning generalize to shifted / unseen tasks
and retrieval backends?*
**Paper:** §6 discussion.
**Status:** 🟡 build (splits + backend sweep, this dir) · 🔴 new-data (a genuinely
new KB source — design note below, not built).

## Claims under test

1. **Retriever-agnostic.** The skill schema is a *planning layer over* retrieval,
   not a retriever. Skill routing should help on top of any backend — sparse
   (`bm25`), dense (`dense_bge`), hybrid, and fusion+rerank alike. Tested by the
   backend sweep (`slurm/submit_backend_sweep_matrix.sh`, see `backend_sweep.md`).
2. **Beyond memorized identity.** Gains must survive when TEST drugs / molecules
   were never seen in TRAIN — otherwise we are measuring name memorization, not
   planning. Tested by `make_splits.py --mode drug_holdout`.
3. **Source-extensible / skill-extensible.** A new skill can be added
   *declaratively* (a schema entry + router rule), and evaluated on records the
   training set never contained. Tested by `make_splits.py --mode skill_holdout`
   (and the coarser `--mode source_holdout`).

## Which tests are "build" vs "new-data"

| Test | Kind | Note |
|---|---|---|
| Held-out drug / molecule split | 🟡 build | runs on current corrected QA JSONL |
| Held-out skill-family split | 🟡 build | routed via shipped `infer_schema_v1_skill` |
| Held-out source split | 🟡 build | optional, coarser than skill holdout |
| Retriever-backend sweep | 🟡 build | reuses existing augment/train/eval methods |
| **New KB source (source-extensible w/o full retrain)** | 🔴 **new-data** | design note only; no new corpus built here |

## Core deliverable — `make_splits.py`

Reads a normalized QA JSONL (schema per `INFRA_REF.md`) and writes deterministic
held-out `train.jsonl` / `test.jsonl` + a `split_summary.json` carrying a
disjointness proof (`"disjoint": true` iff `|train_keys ∩ test_keys| == 0`).

- `--mode drug_holdout` — a held-out fraction of *distinct* drug/molecule
  identities goes to TEST, so no TEST identity appears in TRAIN. Identity key:
  `drug_name` (FDA) or `decoded_smiles` → `input_molecule_or_context` (Mol).
- `--mode skill_holdout --holdout-skill S` — every record whose **routed** skill
  (`retrieval.augment_with_skill_kb.infer_schema_v1_skill`) equals `S` is removed
  from TRAIN and placed in TEST; `split_summary.json` records
  `holdout_skill_in_train == false`.
- `--mode source_holdout --holdout-source SRC` — hold out one `source` value.

All modes seed-deterministic (`--seed`); `drug_holdout` shuffles distinct
identities with `random.Random(seed)`.

## Run

```bash
# 1) Held-out DRUGS (FDA) — no test drug seen in training
python experiments/exp07_generalization/make_splits.py \
  --input outputs/qa_skill_data_corrected/fdarxbench/train.jsonl \
  --output-dir outputs/exp07_generalization/drug_holdout \
  --mode drug_holdout --test-frac 0.2 --seed 13

# 2) Held-out MOLECULE identity (Mol) — same flag, mol fallback key
python experiments/exp07_generalization/make_splits.py \
  --input outputs/qa_skill_data_corrected/mol_instructions/train.jsonl \
  --output-dir outputs/exp07_generalization/mol_holdout \
  --mode drug_holdout --test-frac 0.2 --seed 13

# 3) Held-out SKILL family — add a skill declaratively
python experiments/exp07_generalization/make_splits.py \
  --input outputs/qa_skill_data_corrected/mol_instructions/train.jsonl \
  --output-dir outputs/exp07_generalization/skill_holdout_property \
  --mode skill_holdout --holdout-skill molecule_property_numeric

# 4) (optional) Held-out SOURCE
python experiments/exp07_generalization/make_splits.py \
  --input outputs/qa_skill_data_corrected/mol_instructions/train.jsonl \
  --output-dir outputs/exp07_generalization/source_holdout_openqa \
  --mode source_holdout --holdout-source molinst_open_qa

# Offline unit tests (no KB / GPU)
python -m unittest experiments/exp07_generalization/test_make_splits.py

# Retriever-backend sweep — DRY RUN (prints sbatch, submits nothing)
DRY_RUN=1 bash experiments/exp07_generalization/slurm/submit_backend_sweep_matrix.sh
```

Skill names for `--holdout-skill` (from `SKILL_SCHEMA_V1`): `fda_label_factual`,
`fda_label_multihop`, `molecule_property_numeric`, `molecule_description`,
`molecule_design`, `biomedical_open_qa`.

## New-KB-source extensibility (🔴 future — design note, not built)

The schema is *source-addressable*: each skill declares an allowed-source set
(`SKILL_SCHEMA_V1[skill]["sources"]`) rather than hard-coding retriever indices.
Adding a KB source should therefore be additive and require **no retraining of
the router**:

1. Ingest the new corpus through `kb/build_unified_kb.py` so its facts land in
   `facts.jsonl` with a new `source` tag and get aliased into `alias_index.json`;
   embed it via the embeddings builder for dense methods.
2. Add that `source` tag to the `sources` set of whichever skills may draw on it
   in `SKILL_SCHEMA_V1` — a one-line declarative edit; the router (`infer_*_skill`)
   is unchanged because it keys on the *question*, not on KB contents.
3. Re-run augment for the affected skills only; unaffected skills' evidence and
   prompts are byte-identical, so their adapters need no retrain.

To *demonstrate* this we would need a genuinely new biomedical KB source not
already among {`primekg`, `drugchat_chembl`, `drugchat_pubchem`,
`fdarxbench_label`}. Curating and validating that corpus is the 🔴 new-data step
and is intentionally out of scope for this branch.

## Layout

```
experiments/exp07_generalization/
  make_splits.py                         # core: deterministic holdout splitter
  test_make_splits.py                    # offline stdlib unittest
  backend_sweep.md                       # retriever-agnostic sweep design
  slurm/submit_backend_sweep_matrix.sh   # DRY_RUN driver (not submitted)
  README.md  RESULTS.md
```

Outputs (`outputs/exp07_generalization/…`, adapters, predictions) are gitignored.
