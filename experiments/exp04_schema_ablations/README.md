# Exp04 — Skill-schema component ablations (RQ5)

**Serves:** RQ5 — *Which part of the schema matters?*
**Paper:** §5.4 (+ §5.6 with Exp06). *Answers "is this just prompt engineering?"*
**Status:** 🟡 build — flag-driven augmenter + matrix on current data.

## Claim under test
The skill schema is not a single monolithic prompt trick. It is five separable
mechanisms — **skill identity, source routing, solving steps, answer format, and
evidence organization** — and removing each one degrades a *different* slice of
the task distribution. If any single component reproduced the full method's gains,
the schema would be redundant; the ablation shows that it does not.

## Design — additive ablation grid
Every variant is the full skill method (`skill_schema_v1` = `skill_hybrid` +
`--schema-version v1`) with a subset of components **removed** via the additive
`--ablate` flag on `retrieval/augment_with_skill_kb.py`. The grid runs from
uniform RAG (all removed) up to full SkillRAG (none removed):

| variant | skill id | source routing | solving steps | answer format | evidence org | `--ablate` (removed) |
|---|:-:|:-:|:-:|:-:|:-:|---|
| `v0_uniform` | ✗ | ✗ | ✗ | ✗ | ✗ | `skill_id,source_routing,solving_steps,answer_format,evidence_org` |
| `v1_skill_label` | ✓ | ✗ | ✗ | ✗ | ✗ | `source_routing,solving_steps,answer_format,evidence_org` |
| `v2_source_routing` | ✓ | ✓ | ✗ | ✗ | ✗ | `solving_steps,answer_format,evidence_org` |
| `v3_solving_steps` | ✓ | ✗ | ✓ | ✗ | ✗ | `source_routing,answer_format,evidence_org` |
| `v4_answer_format` | ✓ | ✗ | ✗ | ✓ | ✗ | `source_routing,solving_steps,evidence_org` |
| `v5_evidence_org` | ✓ | ✓ | ✗ | ✓ | ✓ | `solving_steps` |
| `v6_full` (SkillRAG) | ✓ | ✓ | ✓ | ✓ | ✓ | *(empty)* |

`v6_full` with an empty `--ablate` is **byte-identical** to the existing
`skill_schema_v1` augmentation, so the top of the grid is the deployed method
with no reimplementation.

## Component semantics (when listed, the component is REMOVED)
Realized in `retrieval/augment_with_skill_kb.py` (additive `ablate` param on
`retrieve` / `build_completed_skill_prompt` / `augment_record`):

| component | effect when ablated |
|---|---|
| `skill_id` | omit the `Skill: <name>` line |
| `source_routing` | drop the skill source filter — retrieve unconstrained over the whole KB, like the uniform methods |
| `solving_steps` | omit the `How to solve:` steps block |
| `answer_format` | omit the `Answer format:` line |
| `evidence_org` | use the plain baseline evidence dump (`Retrieved KB evidence:`) instead of the organized `Skill-based retrieved knowledge:` block |

**Invariant.** `--ablate ""` (default / `ablate=None`) leaves prompts *and*
records unchanged, byte-for-byte. The un-ablated pipeline is untouched.

## The core edit (additive, backward-compatible)
`retrieval/augment_with_skill_kb.py` only:
- new `--ablate` CLI flag + `ABLATABLE_COMPONENTS` + `parse_ablate()` helper;
- `ablate: set | None = None` threaded (trailing, default-`None`) into
  `retrieve()` (source routing), `build_completed_skill_prompt()` (prompt
  sections), and `augment_record()` (records + prompt);
- `skill_schema["ablate"]` and `summary["ablate"]` are written **only when
  non-empty**, so the default output is unchanged.

No other module signature changes; `augment_baseline_rag.py`'s positional
`augment_record(...)` call is unaffected.

## Run
Metrics use the corrected splits and the shared RAG-SFT pipeline
(augment → LoRA → infer → metrics), same task-level metrics as Exp02.

```bash
# Offline: unit tests + CLI surface (no GPU / no real KB needed)
python -m unittest experiments/exp04_schema_ablations/test_schema_ablations.py
python retrieval/augment_with_skill_kb.py --help    # shows --ablate

# Single-variant augmentation (example: source-routing ablated)
python retrieval/augment_with_skill_kb.py \
  --input outputs/qa_skill_data_corrected/fdarxbench/test.jsonl \
  --output outputs/baseline_rag/data/fdarxbench_v3_solving_steps_test.jsonl \
  --method skill_hybrid --schema-version v1 \
  --ablate source_routing,answer_format,evidence_org

# Full 7-variant x dataset x model matrix (DRY_RUN=1 by default: prints only)
DRY_RUN=1 bash experiments/exp04_schema_ablations/slurm/submit_ablation_matrix.sh
# To actually submit:  DRY_RUN=0 bash experiments/exp04_schema_ablations/slurm/submit_ablation_matrix.sh
```

Matrix env knobs: `MODELS DATASETS VARIANTS LIMIT MAX_STEPS SAVE_STEPS MAX_LENGTH
PARTITION FORCE_AUGMENT DRY_RUN`. Predictions land in
`outputs/baseline_rag/predictions_ablate/`; augmented splits in
`outputs/baseline_rag/data/<dataset>_<variant>_{train,test}.jsonl`.

## Files
- `slurm/submit_augment_ablate.sbatch` — CPU augment for one ablation variant.
- `slurm/submit_ablation_matrix.sh` — enumerates the 7 variants × datasets ×
  models, chaining augment → train → eval (mirrors `slurm/submit_nolabel_matrix.sh`).
- `test_schema_ablations.py` — offline stdlib `unittest`.
- `RESULTS.md` — results template (filled only once runs land).
