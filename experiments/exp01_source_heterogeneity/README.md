# Exp01 — Knowledge-source heterogeneity analysis (RQ1)

**Serves:** RQ1 — *Are drug-discovery tasks knowledge-heterogeneous?*
**Paper:** §5.1, Figure 1 + heterogeneity table.
**Status:** 🟢 reuse (mixtures + Figure 1 already exist) + 🟡 build (this metrics layer).

## Claim under test
Different question types induce sharply different knowledge-source mixtures, so a
single uniform retrieval policy over one heterogeneous KB is under-specified.

## What already exists (reuse)
- `pre_design/analyze_preliminary.py` → `pre_design/results/evidence_by_source.csv`
  (per-task-type KB source mixtures) and `source_answer_support.csv`.
- `pre_design/plot_preliminary.py` → `pre_design/figures/figure1_source_composition.*`
  (the stacked-bar Figure 1: "Question types induce different knowledge-source mixtures").

## What this experiment adds (build)
`source_heterogeneity.py` turns the mixture CSVs into the quantitative metrics the
paper argues from:
- **dominant-source share** and **effective #sources** (2^entropy) — is a task single- or multi-source?
- **source entropy** (bits + normalized to [0,1]).
- **cross-task JS divergence** — do different question types draw on different sources?
- **oracle-source share** — does the schema's allowed-source set cover where the
  evidence lands? (allowed sources imported from `retrieval.augment_with_skill_kb.SKILL_SCHEMA_V1`)
- **oracle support share** — does it cover where the *answer support* lands (gold-token recall)?

## Run
```bash
python experiments/exp01_source_heterogeneity/source_heterogeneity.py \
  --evidence-csv pre_design/results/evidence_by_source.csv \
  --support-csv  pre_design/results/source_answer_support.csv \
  --method hybrid

python -m unittest experiments/exp01_source_heterogeneity/test_source_heterogeneity.py
```

Outputs → `experiments/exp01_source_heterogeneity/results/`:
`heterogeneity_metrics.csv`, `cross_task_divergence.csv`, `REPORT.md`.

## Extensions (not yet built)
- Recompute mixtures over dense/rrf/rerank methods (current figure headlines `hybrid`).
- Add a heterogeneity heatmap figure alongside the existing stacked bar.
