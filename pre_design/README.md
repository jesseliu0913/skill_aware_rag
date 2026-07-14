# Preliminary Design: Personalized Retrieval

## Research question

Does the best retrieval policy depend on the question type?

This folder contains a lightweight diagnostic for that claim. It does not train
new models and does not implement the final skill router. It reuses completed
generic RAG-SFT predictions and asks whether question-type-specific policy
selection transfers from a discovery split to a held-out evaluation split.

## Protocol

1. Keep only generic retrieval methods: BM25, KG, hybrid, BGE, MedCPT, RRF, and
   rerank.
2. Assign benchmark examples to analysis strata using public task fields. These
   strata are for preliminary analysis only, not inputs to the final router.
3. Deterministically split IDs 50/50 into discovery and evaluation halves.
4. On discovery, select one global method and one method per question type.
5. Compare both policies on evaluation.
6. Summarize which KB sources and relations each task retrieves in the existing
   augmented datasets.

For numeric references the utility is tolerance-based numeric accuracy; for
other references it is token F1. Molecular design requires domain-specific
metrics in the final paper and should not be interpreted from token F1 alone.

## Run

```bash
python skill_aware_rag/pre_design/analyze_preliminary.py
python skill_aware_rag/pre_design/plot_preliminary.py
python -m unittest discover -s skill_aware_rag/pre_design -p 'test_*.py'
```

Outputs are written to `skill_aware_rag/pre_design/results/`:

- `REPORT.md`: concise preliminary argument and held-out comparison
- `selected_policies.csv`: method selected for each question type
- `policy_comparison.csv`: fixed versus personalized held-out utility
- `performance_by_question_type.csv`: complete grouped results
- `evidence_by_source.csv`: task/method/source composition
- `evidence_by_relation.csv`: task/method/relation composition
- `source_answer_support.csv`: analysis-only answer-token coverage by source
- `task_inventory.csv`: analysis strata and input types

Paper-ready PDF and PNG figures are written to `skill_aware_rag/pre_design/figures/`.
Figure 1 uses source-composition bars rather than answer-token recall because
the molecular generation and numeric tasks do not admit meaningful lexical
answer-support scores.

## Interpretation boundary

This study can support the motivation that retrieval should be conditioned on
question type. It cannot establish the final method's effectiveness because it
uses benchmark task strata in place of a deployed question-only router. The
final method must infer its routing decision using inference-time inputs only.
