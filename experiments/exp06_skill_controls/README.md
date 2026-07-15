# Exp06 — Wrong-skill / oracle-skill controls

**Serves:** RQ3 (does the skill schema improve retrieval *planning*?), RQ6 (when
does the schema help or not help?). **Paper:** §5.4 / §5.6.

## Claim

The value of the skill schema is the *skill identification* step, not just the
per-skill machinery. If we hold the retriever, KB, prompt assembly, and training
protocol fixed and vary **only which skill the router picks**, downstream answer
quality should track the correctness of that pick:

```
random skill  <  wrong-but-plausible skill  <  predicted (question-only)  ≤  oracle (gold-field)
```

- If **predicted ≈ oracle**, the deployable question-only router is good enough
  and the schema is safe to deploy without gold labels (supports RQ3/RQ6-positive).
- If **random/wrong collapse** relative to oracle, skill identity is causally
  load-bearing — the schema is not just prompt dressing (supports RQ3).
- If **oracle barely beats none**, the schema does not help for that stratum — an
  honest negative for RQ6.

We also report the **skill confusion matrix** (question-only prediction vs
gold-field skill) to localize *which* skills the deployable router confuses.

## The five conditions (`--skill-override`)

The routed skill is the only thing that changes across conditions; everything
downstream follows from it.

| override    | role                | how the skill is chosen |
|-------------|---------------------|-------------------------|
| `none`      | reference / current | current gold-field router — **byte-identical to the pre-Exp06 pipeline** |
| `oracle`    | upper bound         | gold-field skill (`infer_schema_v1_skill`, reads gold task fields) |
| `predicted` | deployed system     | `predict_skill_question_only` — uses ONLY `record["question"]` text |
| `random`    | negative control    | deterministic `sha1(id) mod n_skills` |
| `wrong`     | sensitivity         | deterministic fixed confusion map (each gold skill → one other skill) |

> **Why `none == oracle` here.** The *current* routers read gold task fields
> (`source`, `task_type`, `input_type`), so the deployed-as-is route already IS
> the gold-field (oracle) skill. `none` is kept as the byte-identical reference;
> `oracle` is the same route reached through the override path. The genuinely
> *deployable* router is `predicted` — question-only, no gold-field leakage.

## What was built (additive only)

- **Shared, additive edit** to `retrieval/augment_with_skill_kb.py`:
  a new `--skill-override {none,oracle,random,wrong,predicted}` flag (default
  `none`), threaded as an optional `skill_override="none"` param into
  `infer_record_skill` / `retrieve` / `augment_record`. Default `none` leaves the
  routed skill and produced records byte-identical. New helpers:
  `predict_skill_question_only`, `random_skill`, `wrong_skill`,
  `gold_field_skill`, `apply_skill_override`, and the `CONFUSION_MAP_*` tables.
- `router_report.py` — router accuracy + skill×skill confusion matrix of the
  question-only router vs the gold-field skill over a QA JSONL → CSV + `REPORT.md`.
- `slurm/submit_skill_controls_matrix.sh` — RAG-SFT driver enumerating the five
  override conditions (augment train+test → LoRA train → eval). **`DRY_RUN=1` by
  default; nothing is submitted until `DRY_RUN=0`.**
- `slurm/submit_augment_override.sbatch` — self-contained CPU augment step that
  forwards `--skill-override` (leaves the shared `submit_augment_skillkb.sbatch`
  untouched).
- `test_skill_controls.py` — offline stdlib unittests.

## How to run

Offline verification (no GPU / KB / network):

```bash
# unit tests
python experiments/exp06_skill_controls/test_skill_controls.py
# augmenter help shows the new flag
python retrieval/augment_with_skill_kb.py --help | grep skill-override
```

Router accuracy + confusion matrix over a split (CPU, no KB load):

```bash
python experiments/exp06_skill_controls/router_report.py \
  --input outputs/qa_skill_data_corrected/mol_instructions/test.jsonl \
  --schema-version v1 \
  --out-csv experiments/exp06_skill_controls/results/router_confusion.csv \
  --out-report experiments/exp06_skill_controls/results/REPORT.md
```

Augment one split under a forced skill route (CPU, needs the unified KB):

```bash
python retrieval/augment_with_skill_kb.py \
  --input  outputs/qa_skill_data_corrected/mol_instructions/test.jsonl \
  --output outputs/exp06_skill_controls/data/mol_instructions/test_skill_hybrid_ovr-predicted.jsonl \
  --method skill_hybrid --schema-version v1 --skill-override predicted
```

Full RAG-SFT matrix over all five conditions (GPU; **preview first**):

```bash
# dry run (default): prints every sbatch it would submit
bash experiments/exp06_skill_controls/slurm/submit_skill_controls_matrix.sh
# actually submit
DRY_RUN=0 bash experiments/exp06_skill_controls/slurm/submit_skill_controls_matrix.sh
```

Env knobs: `MODELS`, `DATASETS`, `OVERRIDES`, `METHOD` (default `skill_hybrid`),
`SCHEMA` (default `v1`), `LIMIT`, `MAX_STEPS`, `FORCE_AUGMENT`, `DRY_RUN`.

## Metrics

- **Task-level answer metrics** per override (same hooks as Exp02): FDA
  factual/multihop token-F1 & exact; property numeric accuracy; description
  token-F1; design validity/uniqueness; refusal abstention accuracy — always
  broken out per task, never aggregate-only.
- **Router accuracy** and **skill confusion matrix** from `router_report.py`.
