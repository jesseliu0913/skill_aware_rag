# Exp01 — Knowledge-source heterogeneity (RQ1)

Headline retriever: `hybrid`. Full per-method values are in the CSVs.

## Per-task source concentration

| Dataset | Question type | Dominant source | Dominant share | Eff. #sources | Oracle-source share | Oracle support share |
|---|---|---|---:|---:|---:|---:|
| fdarxbench | fda_factual | fdarxbench_label | 0.581 | 3.05 | 0.581 | 0.978 |
| fdarxbench | fda_multihop | fdarxbench_label | 0.558 | 3.10 | 0.558 | 0.991 |
| fdarxbench | fda_refusal | fdarxbench_label | 0.444 | 3.20 | 0.444 | 0.989 |
| mol_instructions | biomedical_open_qa | primekg | 0.333 | 3.92 | 1.000 | 1.000 |
| mol_instructions | molecule_description | primekg | 0.333 | 3.67 | 0.397 | 0.864 |
| mol_instructions | molecule_design | drugchat_pubchem | 0.455 | 3.54 | 0.575 | 0.000 |
| mol_instructions | molecule_property | primekg | 0.507 | 2.77 | 0.307 | 0.000 |

## Cross-task source divergence (mean pairwise JS)

| Dataset | Method | Mean pairwise JS divergence |
|---|---|---:|
| fdarxbench | bm25 | 0.016 |
| fdarxbench | hybrid | 0.019 |
| fdarxbench | kg | 0.021 |
| fdarxbench | skill_hybrid | 0.105 |
| fdarxbench | skill_schema_v1 | 0.000 |
| mol_instructions | bm25 | 0.137 |
| mol_instructions | hybrid | 0.111 |
| mol_instructions | kg | 0.000 |
| mol_instructions | skill_hybrid | 0.218 |
| mol_instructions | skill_schema_v1 | 0.229 |

Reading: a high dominant-share with low effective-sources means the task is
single-source; a high mean pairwise JS divergence means different question
types draw on different sources — the necessity argument for the skill schema.
Oracle-source / support share near 1.0 means the schema's allowed-source set
covers where the evidence (and its answer support) actually lives.
