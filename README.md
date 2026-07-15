# Skill-Aware RAG for Drug Discovery QA

This is the active AAAI project on personalized retrieval for drug and
molecular questions. The hypothesis is that different question types need
different knowledge sources and relations; the system should infer that need
from the question and constrain retrieval accordingly.

The baseline study is complete. The final question-only skill router and its
ablations are not yet implemented.

## Start Here

- [`results/baseline.md`](results/baseline.md): complete per-method results
- [`pre_design/README.md`](pre_design/README.md): preliminary routing protocol
- [`pre_design/results/REPORT.md`](pre_design/results/REPORT.md): held-out analysis
- [`pre_design/figures/`](pre_design/figures/): current figures

## Experiment Setup

### Datasets and models

| Dataset | Train / test | Task groups |
|---|---:|---|
| FDARxBench | 3,000 / 1,000 | factual, multihop, refusal |
| Mol-Instructions | 3,000 / 1,000 | design, description, open QA, property prediction |

Models: Qwen2.5-7B-Instruct, Qwen2.5-3B-Instruct, and
Llama-3.2-3B-Instruct.

### Baseline methods

| Family | Methods |
|---|---|
| Closed book | `none` |
| Sparse / graph | `bm25`, `kg` |
| Dense | `dense_bge`, `dense_medcpt` |
| Fusion / reranking | `hybrid`, `rrf`, `rerank` |
| Previous skill diagnostics | `skill_hybrid`, `skill_schema_v1` |

The previous skill variants are baselines, not the final method.

### Protocols

| Protocol | Train input | Test input | Purpose |
|---|---|---|---|
| A: no answer context | question + retrieval | question + retrieval | Test whether retrieval supplies missing knowledge |
| B: raw test | method-specific RAG training | original test record | Isolate retrieval-aware training |
| C: retrieval at test | method-specific RAG training | original record + retrieval | Standard RAG-SFT deployment |

FDARxBench Protocol A removes FDA `label_context` at train and test time.
Mol-Instructions has no supplied answer context, so its Protocol A is identical
to Protocol C; its closed-book `none` result is inherited from Protocol B.

## Current Results

All formal baseline matrices are complete.

| Protocol | Coverage | Evaluation files | Status |
|---|---:|---:|---|
| A: FDARxBench no-label | 3 models x 10 methods | 30 | Complete |
| A: Mol-Instructions | 3 models x 10 methods | 27 C + 3 B/`none` | Complete |
| B: raw test | 3 models x 2 datasets x 10 methods | 60 | Complete |
| C: retrieval at test | 3 models x 2 datasets x 9 methods | 54 | Complete |

Protocol C does not duplicate `none`, because it has no test retrieval and is
identical to Protocol B.

### Protocol A: FDARxBench no-label

Token F1, `n=1000`:

| Method | Qwen-7B | Qwen-3B | Llama-3B |
|---|---:|---:|---:|
| `none` | 0.545 | 0.525 | 0.545 |
| `bm25` | 0.680 | 0.663 | 0.678 |
| `dense_medcpt` | 0.679 | 0.658 | 0.669 |
| `hybrid` | 0.703 | 0.680 | 0.704 |
| `kg` | 0.710 | 0.688 | 0.705 |
| `rerank` | 0.713 | 0.691 | 0.706 |
| `rrf` | 0.728 | 0.702 | 0.719 |
| **`dense_bge`** | **0.729** | **0.714** | **0.732** |
| `skill_hybrid` | 0.701 | 0.682 | 0.701 |
| `skill_schema_v1` | 0.700 | 0.684 | 0.698 |

Dense BGE improves over closed book by 0.184--0.189 F1 on every model.
Refusal is already near saturation; factual and multihop questions account for
the retrieval gain. Neither previous skill method beats Dense BGE.

### Protocol A: Mol-Instructions

Macro token F1 and property numeric accuracy, `n=1000`:

| Model | Closed-book F1 | Best F1 method | Best F1 | Best numeric method | Numeric accuracy |
|---|---:|---|---:|---|---:|
| Qwen-7B | 0.182 | `hybrid` | 0.258 | `kg` | 0.280 |
| Qwen-3B | 0.179 | `bm25` | 0.259 | `kg` | 0.260 |
| Llama-3B | 0.185 | `kg` | 0.257 | `skill_schema_v1` | 0.284 |

No retriever dominates across models and subtasks. Report molecular design,
description, open QA, and property prediction separately: token F1 is not a
valid molecule-design metric, and numeric accuracy applies only to property
prediction.

### Protocol B: raw test

| Dataset | Qwen-7B best macro/F1 | Qwen-3B best macro/F1 | Llama-3B best macro/F1 |
|---|---|---|---|
| FDARxBench | `none` | `none` | `none` |
| Mol-Instructions | `skill_schema_v1`: 0.252 | `kg`: 0.250 | `skill_schema_v1`: 0.238 |

When the FDA label is already supplied, retrieval-aware training alone does not
improve FDARxBench. This is a control, not evidence against retrieval in the
clean no-label setting.

### Protocol C: retrieval at test

| Dataset/model | Best F1 method | F1 | Best exact/numeric method | Exact/numeric |
|---|---|---:|---|---:|
| FDARx Qwen-7B | `skill_schema_v1` | 0.789 | `skill_schema_v1` | 0.362 |
| FDARx Qwen-3B | `dense_bge` | 0.768 | `kg` | 0.337 |
| FDARx Llama-3B | `hybrid` / `rrf` | 0.783 | `rrf` | 0.350 |
| Mol Qwen-7B | `hybrid` | 0.258 | `kg` | 0.280 |
| Mol Qwen-3B | `bm25` | 0.259 | `kg` | 0.260 |
| Mol Llama-3B | `kg` | 0.257 | `skill_schema_v1` | 0.284 |

FDARx Protocol C retains the FDA label, so its higher absolute scores do not
measure retrieval from missing knowledge. Mol Protocol C is also Protocol A.

### Main conclusions

1. Retrieval is necessary in the clean FDARxBench no-label setting.
2. Dense BGE is the strongest fixed FDARx retriever across all three models.
3. Mol retrieval is heterogeneous across models, tasks, and metrics.
4. Existing skill variants do not establish the proposed method.
5. The paper must compare predicted routing against the strongest fixed
   retriever, not only against closed book.

An extra `dense_bge_locked` diagnostic is outside the formal ten-method matrix.
Its augmentation jobs `43066` and `43067` completed, but training job `43068`
failed with exit code 120 during JFS I/O errors. Evaluation `43069` was
cancelled, so no score is reported.

## Proposed Method

Keep the schema compact:

```text
skill = {
  task,
  sources,
  relations,
  strategy,
  budget
}
```

At inference time:

1. Infer `task` from question-only inputs.
2. Map the task to source and relation constraints.
3. Execute sparse, dense, graph, or fused retrieval within the evidence budget.
4. Generate from retrieved evidence and abstain when required support is absent.

Gold task labels may define an oracle upper bound but must not be inputs to the
deployed router.

## Run

Run commands from `/playpen-jfs/jesse/drug_microbiome`.

```bash
# Protocol A: separate no-label matrix is needed only for FDARxBench
bash skill_aware_rag/slurm/submit_nolabel_matrix.sh

# Protocol B
bash skill_aware_rag/slurm/submit_baseline_raweval_matrix.sh

# Protocol C, reusing Protocol-B adapters
bash skill_aware_rag/slurm/submit_reteval_matrix.sh
```

Regenerate the preliminary analysis and figures:

```bash
python skill_aware_rag/pre_design/analyze_preliminary.py
python skill_aware_rag/pre_design/plot_preliminary.py
python -m unittest discover -s skill_aware_rag/pre_design -p 'test_*.py'
```

## Remaining Experiments

1. Oracle question-type routing upper bound.
2. Predicted question-only routing and router accuracy.
3. Source, relation, strategy, and budget ablations.
4. Evidence recall/precision and source-coverage metrics.
5. Molecular validity/similarity and numeric task metrics.
6. Multiple seeds, confidence intervals, latency, and token-budget comparisons.

## Output Layout

```text
outputs/baseline_rag/
  predictions_nolabel/   Protocol A FDARxBench
  predictions_bsl/       Protocol B and inherited `none`
  predictions_bslret/    Protocol C and Mol Protocol A retrieval
  results_summary_nolabel.csv
  results_summary_bsl.csv
  results_summary_bslret.csv
  adapters/
  data/
  slurm/
```

Large generated data, adapters, predictions, checkpoints, and Slurm logs are
local artifacts and should not be committed.
