# Exp06 — Results

> Template. Filled only when runs land. **Not yet run.**

## 1. Router quality (question-only vs gold-field skill)

Produced by `router_report.py`.

| dataset | schema | n | router accuracy |
|---|---|---:|---:|
| mol_instructions | v1 | — | Not yet run. |
| fdarxbench | v1 | — | Not yet run. |

### Skill confusion matrix (rows = gold-field skill, cols = question-only prediction)

Not yet run. See `results/router_confusion.csv` and `results/REPORT.md` once generated.

## 2. Downstream answer quality under each skill override

Full RAG-SFT (augment train+test → LoRA train → eval), per task stratum.
Expected ordering to test: `random < wrong < predicted ≤ oracle`, with `none`
as the byte-identical reference (== oracle route).

| model | dataset | task | none | oracle | predicted | random | wrong |
|---|---|---|---|---|---|---|---|
| qwen2_5_7b | fdarxbench | factual (token-F1) | — | — | — | — | — |
| qwen2_5_7b | fdarxbench | multihop (token-F1) | — | — | — | — | — |
| qwen2_5_7b | fdarxbench | refusal (abstention) | — | — | — | — | — |
| qwen2_5_7b | mol_instructions | property (numeric acc) | — | — | — | — | — |
| qwen2_5_7b | mol_instructions | description (token-F1) | — | — | — | — | — |
| qwen2_5_7b | mol_instructions | design (validity/uniq) | — | — | — | — | — |
| qwen2_5_7b | mol_instructions | open-QA (token-F1) | — | — | — | — | — |

Not yet run.

## 3. Takeaways

Not yet run.
