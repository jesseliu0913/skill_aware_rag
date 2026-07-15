# Exp05 — Skill routing under retrieval difficulty (stress tests, RQ6)

**Serves:** RQ6 — *When does the skill schema help or not help?*
**Paper:** §5.5 (stress tests: noisy KB, weak retriever, no-label).
**Status:** 🟡 build (KB perturbation + top-k sweep) + 🟢 reuse (no-label, bm25) + 🔴 new-data proxy (decoy corpus).

## Claim under test
Skill-aware routing should help *most* exactly when retrieval is hard — a weak
retriever, a tiny top-k budget, a noisier KB, or lexically-similar wrong-source
distractors. Where retrieval is trivially easy the schema should be roughly
neutral. This experiment builds the stressors and sweeps that expose that curve.

## The six stress settings

| # | Setting | Category | How it is produced |
|---|---------|----------|--------------------|
| 1 | **Weak retriever** | 🟢 reuse | run existing `bm25` / `bm25_skillrouted` methods instead of dense — no new code |
| 2 | **Small top-k budget** | 🟡 build (config) | `topk_sweep.md` sweeps `--top-k k ∈ {1,2,3,8}` through the existing augmenters |
| 3 | **Noisy heterogeneous KB** | 🟡 build (this module) | `perturb_kb.py --mode noisy` injects off-task synthetic distractor facts |
| 4 | **Decoy-source injection** | 🔴 new-data (this module, heuristic) | `perturb_kb.py --mode decoy` emits wrong-source near-duplicates |
| 5 | **No-label FDA** | 🟢 reuse | `outputs/qa_skill_data_nolabel/fdarxbench/*` + `slurm/submit_nolabel_matrix.sh` (answer leakage removed) |
| 6 | **Multi-source open-QA** | 🟢 reuse | existing `molinst_open_qa` records evaluated with the standard matrix |

Settings 1, 5, 6 are pure **reuse** of existing data/methods. Setting 2 is a
**build** (a sweep config, no new code path). Settings 3–4 are the **build /
new-data** contribution of this experiment: `perturb_kb.py`.

## Core deliverable: `perturb_kb.py`
Reads a unified KB (`facts.jsonl` + `alias_index.json`; fields
`id, source, source_id, title, text, aliases, entities, relation, metadata,
tokens`) and writes a perturbed KB under `--out-dir` (perturbed `facts.jsonl` +
a verbatim copy of `alias_index.json`). The **exact fact schema is preserved**
so `retrieval/augment_with_skill_kb.py` and `retrieval/augment_baseline_rag.py`
consume it with `--kb-dir <out-dir>` **unchanged**.

- `--mode noisy --noise-frac F` — keep all originals, then inject `round(F·N)`
  random off-task synthetic distractor facts (new `noise_*` ids, plausible-
  looking text, `relation=distractor`, `metadata.synthetic_noise=true`, source
  drawn from the existing sources). Retrieval must sift through more candidates.
- `--mode decoy --decoy-per N [--decoy-frac P]` — sample fraction `P` of the
  facts and, per sampled fact, emit `N` **decoy** copies: text near-duplicated
  (original text kept as a prefix, light suffix appended) but `source` relabeled
  to a **different, wrong** source. Flagged `metadata.decoy=true` (plus
  `decoy_of` / `decoy_true_source`). A lexically near-identical wrong-source
  fact now competes with the correct one, stressing source routing.

🔴 **Decoy construction is a HEURISTIC** — a synthetic near-duplicate + source
relabel, not a hand-curated paraphrase corpus. It is a lower-cost proxy for the
paper's "new-data" decoy corpus and is flagged as such here and in RESULTS.md.

**Determinism:** all randomness flows through a single `random.Random(--seed)`;
there is **no** time-based randomness, so a fixed seed reproduces byte-identical
output. Injected/decoy facts carry no aliases, so the copied `alias_index.json`
stays valid for the original facts.

## Run
```bash
cd /playpen-jfs/jesse/drug_microbiome/skill_aware_rag

# noisy KB: 0.5x extra distractor facts
python experiments/exp05_stress_tests/perturb_kb.py \
  --kb-dir outputs/knowledge_bank/unified \
  --out-dir outputs/knowledge_bank/perturbed/noisy_f0.5 \
  --mode noisy --noise-frac 0.5 --seed 0

# decoy-source injection: 1 wrong-source near-dup per fact
python experiments/exp05_stress_tests/perturb_kb.py \
  --kb-dir outputs/knowledge_bank/unified \
  --out-dir outputs/knowledge_bank/perturbed/decoy_n1 \
  --mode decoy --decoy-per 1 --decoy-frac 1.0 --seed 0

# offline unit tests (stdlib only, no outputs/ needed)
cd experiments/exp05_stress_tests && python -m unittest test_perturb_kb.py -v
```

See **`topk_sweep.md`** for the k∈{1,2,3,8} budget sweep and
**`slurm/submit_exp05_stress_matrix.sh`** for the augment+evaluate driver over
perturbed KBs × k values (`DRY_RUN=1` capable; not submitted here).

Outputs (gitignored): perturbed KBs under `outputs/knowledge_bank/perturbed/…`,
predictions/eval under the standard `outputs/baseline_rag/…` tree.

## Extensions (not yet built)
- Curated (non-heuristic) decoy paraphrases to replace the 🔴 synthetic decoys.
- Noise-fraction ablation curve (F ∈ {0.25, 0.5, 1, 2}) vs. token-F1.
