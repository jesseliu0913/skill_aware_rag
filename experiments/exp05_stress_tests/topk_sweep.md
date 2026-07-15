# Exp05 — Top-k budget sweep config

Small top-k budget is stress setting #2. It reuses the existing augmenters
(`retrieval/augment_with_skill_kb.py` and `retrieval/augment_baseline_rag.py`)
with a smaller `--top-k`; **no new code path** — only a swept CLI value.

## Swept values
```
TOPK  = {1, 2, 3, 8}     # 8 = the deployed default; 1..3 = tight budgets
```

## Methods (weak vs. skill-routed)
```
bm25                # weak sparse retriever (baseline)
bm25_skillrouted    # 🟢 reuse: skill-routed sparse
skill_hybrid        # skill-routed hybrid
hybrid              # non-routed hybrid control
```

## KB variants (from perturb_kb.py; the driver iterates these)
```
unified                     # clean baseline KB (outputs/knowledge_bank/unified)
perturbed/noisy_f0.5        # --mode noisy --noise-frac 0.5
perturbed/noisy_f1.0        # --mode noisy --noise-frac 1.0
perturbed/decoy_n1          # --mode decoy --decoy-per 1   (🔴 heuristic)
```

## Interpretation
At **k=8** on the clean KB the schema is expected to be near-neutral; the gap in
favour of skill-routing should **widen** as k shrinks and as the KB is perturbed
(noisy / decoy). That widening curve is the RQ6 result.

## How the driver consumes this
`slurm/submit_exp05_stress_matrix.sh` loops `TOPK × METHODS × KB_VARIANTS`,
augmenting each `test` split with `--top-k $k --kb-dir <variant>` then evaluating.
It mirrors `slurm/submit_nolabel_matrix.sh` (env-var driven, `DRY_RUN=1`).
