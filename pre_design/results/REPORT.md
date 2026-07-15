# Preliminary Study: Question-Type-Conditioned Retrieval

This diagnostic uses only completed generic RAG-SFT runs. Example IDs are
deterministically split into discovery and evaluation halves. Retrieval
policies are selected on discovery and scored on evaluation.

## Held-Out Policy Comparison

| Model | Dataset | Fixed policy | Fixed utility | Personalized utility | Gain |
|---|---|---|---:|---:|---:|
| llama3_2_3b | fdarxbench | rrf | 0.778 | 0.779 | +0.002 |
| llama3_2_3b | mol_instructions | kg | 0.320 | 0.320 | +0.000 |
| qwen2_5_3b | fdarxbench | dense_bge | 0.761 | 0.761 | +0.000 |
| qwen2_5_3b | mol_instructions | kg | 0.323 | 0.324 | +0.001 |
| qwen2_5_7b | fdarxbench | kg | 0.782 | 0.782 | +0.000 |
| qwen2_5_7b | mol_instructions | kg | 0.323 | 0.323 | -0.000 |

## Policies Selected on the Discovery Half

| Model | Dataset | Question type | Selected policy | Discovery utility |
|---|---|---|---|---:|
| llama3_2_3b | fdarxbench | fda_factual | bm25 | 0.757 |
| llama3_2_3b | fdarxbench | fda_multihop | rrf | 0.638 |
| llama3_2_3b | fdarxbench | fda_refusal | rrf | 1.000 |
| llama3_2_3b | mol_instructions | biomedical_open_qa | kg | 0.434 |
| llama3_2_3b | mol_instructions | molecule_description | kg | 0.604 |
| llama3_2_3b | mol_instructions | molecule_design | rrf | 0.000 |
| llama3_2_3b | mol_instructions | molecule_property | kg | 0.296 |
| qwen2_5_3b | fdarxbench | fda_factual | dense_bge | 0.739 |
| qwen2_5_3b | fdarxbench | fda_multihop | bm25 | 0.619 |
| qwen2_5_3b | fdarxbench | fda_refusal | rrf | 1.000 |
| qwen2_5_3b | mol_instructions | biomedical_open_qa | bm25 | 0.428 |
| qwen2_5_3b | mol_instructions | molecule_description | hybrid | 0.610 |
| qwen2_5_3b | mol_instructions | molecule_design | rrf | 0.000 |
| qwen2_5_3b | mol_instructions | molecule_property | kg | 0.272 |
| qwen2_5_7b | fdarxbench | fda_factual | kg | 0.754 |
| qwen2_5_7b | fdarxbench | fda_multihop | kg | 0.657 |
| qwen2_5_7b | fdarxbench | fda_refusal | rrf | 1.000 |
| qwen2_5_7b | mol_instructions | biomedical_open_qa | dense_medcpt | 0.432 |
| qwen2_5_7b | mol_instructions | molecule_description | kg | 0.601 |
| qwen2_5_7b | mol_instructions | molecule_design | kg | 0.007 |
| qwen2_5_7b | mol_instructions | molecule_property | kg | 0.304 |

## Source Support Profile (Hybrid Retrieval)

Gold-token recall is an analysis-only proxy for whether retrieved evidence
contains answer information; it is never used by the router.

| Dataset | Question type | Best observed source | Gold-token recall | N |
|---|---|---|---:|---:|
| fdarxbench | fda_factual | fdarxbench_label | 0.896 | 578 |
| fdarxbench | fda_multihop | fdarxbench_label | 0.807 | 204 |
| fdarxbench | fda_refusal | fdarxbench_label | 0.271 | 218 |
| mol_instructions | biomedical_open_qa | drugchat_pubchem | 0.115 | 250 |
| mol_instructions | molecule_description | drugchat_pubchem | 0.006 | 250 |
| mol_instructions | molecule_design | drugchat_chembl | 0.000 | 250 |
| mol_instructions | molecule_property | drugchat_chembl | 0.000 | 250 |

## Interpretation

The selected-policy table is the direct preliminary test of the paper's
motivation: if one retrieval policy were uniformly best, every row within a
model/dataset would select the same method. Variation across question types
supports the existence of retrieval-policy heterogeneity.

The largest held-out gain is +0.0018. These completed runs therefore
do not yet show a practically meaningful answer-quality gain from selecting a
different generic retriever per task. The stronger preliminary evidence should
come from controlled source/relation ablations under a fixed retriever and
evidence budget; this analysis identifies which ablations to run.

This is a preliminary diagnostic, not a final benchmark result. The router is
represented by benchmark task strata here; the deployed method must infer its
question type from inference-time inputs only.

Analyzed 294 task/method/partition aggregates.
