# Exp05 — Results (stress tests, RQ6)

Not yet run.

All tables below are placeholders; populate after running the top-k sweep and
the perturbed-KB augment+eval matrix (see `slurm/submit_exp05_stress_matrix.sh`).

## 1. Top-k budget sweep (weak vs. skill-routed retriever)
Method × k ∈ {1, 2, 3, 8}, on the unperturbed KB.

| model | method | k=1 | k=2 | k=3 | k=8 |
|-------|--------|-----|-----|-----|-----|
| _tbd_ | bm25 | Not yet run. | Not yet run. | Not yet run. | Not yet run. |
| _tbd_ | bm25_skillrouted | Not yet run. | Not yet run. | Not yet run. | Not yet run. |
| _tbd_ | skill_hybrid | Not yet run. | Not yet run. | Not yet run. | Not yet run. |

## 2. Noisy KB (off-task distractor injection)
Method × noise fraction F.

| model | method | F=0 (clean) | F=0.5 | F=1.0 | F=2.0 |
|-------|--------|-------------|-------|-------|-------|
| _tbd_ | bm25 | Not yet run. | Not yet run. | Not yet run. | Not yet run. |
| _tbd_ | skill_hybrid | Not yet run. | Not yet run. | Not yet run. | Not yet run. |

## 3. Decoy-source injection 🔴 (heuristic near-duplicate corpus)
Wrong-source lexical near-duplicates compete with the correct fact.

| model | method | clean | decoy-per=1 | decoy-per=2 |
|-------|--------|-------|-------------|-------------|
| _tbd_ | bm25 | Not yet run. | Not yet run. | Not yet run. |
| _tbd_ | skill_hybrid | Not yet run. | Not yet run. | Not yet run. |

🔴 Decoys are synthetic (near-dup + source relabel), a proxy for a curated
paraphrase corpus — read the numbers as directional, not final.

## 4. Reuse settings (no-label FDA, multi-source open-QA, weak retriever)
Pulled from the existing no-label / baseline matrices.

| setting | method | metric | value |
|---------|--------|--------|-------|
| no-label FDA | (matrix) | token_f1 | Not yet run. |
| multi-source open-QA | (matrix) | token_f1 / entity_f1 | Not yet run. |
