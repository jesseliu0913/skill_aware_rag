# Protocol A: Complete Results

Protocol A evaluates retrieval when no answer-bearing context is given.
This report contains all 3 models x 2 datasets x 10 baseline methods.

| Dataset | Models | Methods | Evaluations | Status |
|---|---:|---:|---:|---|
| FDARxBench | 3 | 10 | 30 separate no-label runs | Complete |
| Mol-Instructions | 3 | 10 | 27 retrieval runs + 3 inherited `none` runs | Complete |

Metrics are computed over the fixed 1,000-example test split. Values are
rounded to three decimals; source CSVs retain full precision.

## FDARxBench

The FDA `label_context` is removed at train and test time. Retrieval
methods retrieve from the shared KB using the question; `none` is
question-only closed book. Each cell reports token F1 / exact match.

### Overall (n=1000)

| Method | Qwen2.5-7B | Qwen2.5-3B | Llama-3.2-3B |
|---|---:|---:|---:|
| `none` | 0.545 / 0.230 | 0.525 / 0.226 | 0.545 / 0.227 |
| `bm25` | 0.680 / 0.298 | 0.663 / 0.292 | 0.678 / 0.297 |
| `kg` | 0.710 / 0.311 | 0.688 / 0.301 | 0.705 / 0.302 |
| `hybrid` | 0.703 / 0.305 | 0.680 / 0.295 | 0.704 / 0.313 |
| `dense_bge` | 0.729 / 0.309 | 0.714 / 0.292 | 0.732 / 0.315 |
| `dense_medcpt` | 0.679 / 0.295 | 0.658 / 0.284 | 0.669 / 0.285 |
| `rrf` | 0.728 / 0.311 | 0.702 / 0.295 | 0.719 / 0.292 |
| `rerank` | 0.713 / 0.301 | 0.691 / 0.294 | 0.706 / 0.292 |
| `skill_hybrid` | 0.701 / 0.304 | 0.682 / 0.298 | 0.701 / 0.303 |
| `skill_schema_v1` | 0.700 / 0.303 | 0.684 / 0.297 | 0.698 / 0.305 |

### By question type: Qwen2.5-7B

| Method | Factual (n=578) | Multihop (n=204) | Refusal (n=218) | Overall |
|---|---:|---:|---:|---:|
| `none` | 0.409 / 0.021 | 0.446 / 0.000 | 1.000 / 1.000 | 0.545 / 0.230 |
| `bm25` | 0.606 / 0.135 | 0.547 / 0.010 | 1.000 / 1.000 | 0.680 / 0.298 |
| `kg` | 0.653 / 0.159 | 0.564 / 0.010 | 0.995 / 0.995 | 0.710 / 0.311 |
| `hybrid` | 0.642 / 0.147 | 0.558 / 0.010 | 1.000 / 1.000 | 0.703 / 0.305 |
| `dense_bge` | 0.675 / 0.154 | 0.594 / 0.015 | 0.995 / 0.995 | 0.729 / 0.309 |
| `dense_medcpt` | 0.613 / 0.133 | 0.523 / 0.000 | 1.000 / 1.000 | 0.679 / 0.295 |
| `rrf` | 0.676 / 0.159 | 0.586 / 0.005 | 1.000 / 1.000 | 0.728 / 0.311 |
| `rerank` | 0.648 / 0.144 | 0.591 / 0.000 | 1.000 / 1.000 | 0.713 / 0.301 |
| `skill_hybrid` | 0.642 / 0.149 | 0.554 / 0.005 | 0.996 / 0.995 | 0.701 / 0.304 |
| `skill_schema_v1` | 0.637 / 0.145 | 0.559 / 0.010 | 0.995 / 0.995 | 0.700 / 0.303 |

### By question type: Qwen2.5-3B

| Method | Factual (n=578) | Multihop (n=204) | Refusal (n=218) | Overall |
|---|---:|---:|---:|---:|
| `none` | 0.378 / 0.014 | 0.432 / 0.000 | 1.000 / 1.000 | 0.525 / 0.226 |
| `bm25` | 0.585 / 0.130 | 0.526 / 0.000 | 0.995 / 0.995 | 0.663 / 0.292 |
| `kg` | 0.624 / 0.144 | 0.535 / 0.000 | 1.000 / 1.000 | 0.688 / 0.301 |
| `hybrid` | 0.608 / 0.131 | 0.543 / 0.005 | 1.000 / 1.000 | 0.680 / 0.295 |
| `dense_bge` | 0.658 / 0.126 | 0.566 / 0.005 | 1.000 / 1.000 | 0.714 / 0.292 |
| `dense_medcpt` | 0.583 / 0.111 | 0.506 / 0.010 | 1.000 / 1.000 | 0.658 / 0.284 |
| `rrf` | 0.638 / 0.128 | 0.562 / 0.015 | 1.000 / 1.000 | 0.702 / 0.295 |
| `rerank` | 0.619 / 0.126 | 0.566 / 0.015 | 1.000 / 1.000 | 0.691 / 0.294 |
| `skill_hybrid` | 0.611 / 0.135 | 0.544 / 0.010 | 1.000 / 1.000 | 0.682 / 0.298 |
| `skill_schema_v1` | 0.611 / 0.135 | 0.551 / 0.005 | 1.000 / 1.000 | 0.684 / 0.297 |

### By question type: Llama-3.2-3B

| Method | Factual (n=578) | Multihop (n=204) | Refusal (n=218) | Overall |
|---|---:|---:|---:|---:|
| `none` | 0.407 / 0.016 | 0.452 / 0.000 | 1.000 / 1.000 | 0.545 / 0.227 |
| `bm25` | 0.601 / 0.135 | 0.550 / 0.005 | 1.000 / 1.000 | 0.678 / 0.297 |
| `kg` | 0.637 / 0.144 | 0.581 / 0.005 | 1.000 / 1.000 | 0.705 / 0.302 |
| `hybrid` | 0.639 / 0.161 | 0.571 / 0.010 | 1.000 / 1.000 | 0.704 / 0.313 |
| `dense_bge` | 0.680 / 0.166 | 0.597 / 0.010 | 0.995 / 0.995 | 0.732 / 0.315 |
| `dense_medcpt` | 0.595 / 0.116 | 0.523 / 0.000 | 1.000 / 1.000 | 0.669 / 0.285 |
| `rrf` | 0.657 / 0.126 | 0.596 / 0.005 | 1.000 / 1.000 | 0.719 / 0.292 |
| `rerank` | 0.641 / 0.131 | 0.584 / 0.000 | 0.991 / 0.991 | 0.706 / 0.292 |
| `skill_hybrid` | 0.637 / 0.147 | 0.563 / 0.000 | 1.000 / 1.000 | 0.701 / 0.303 |
| `skill_schema_v1` | 0.631 / 0.151 | 0.566 / 0.000 | 1.000 / 1.000 | 0.698 / 0.305 |

## Mol-Instructions

Mol-Instructions has no supplied answer context, so Protocol A is the
existing retrieval-at-test condition. Retrieval methods use Protocol-C
evaluations; `none` is the identical closed-book Protocol-B evaluation.
Each subtask has 250 examples.

Macro values average the four equally sized subtasks. Property numeric
accuracy is tolerance based. Token F1/exact for molecular design are
listed for completeness but are not chemically valid design metrics.

### Overall macro F1 / property numeric accuracy (n=1000)

| Method | Qwen2.5-7B | Qwen2.5-3B | Llama-3.2-3B |
|---|---:|---:|---:|
| `none` | 0.182 / 0.276 | 0.179 / 0.296 | 0.185 / 0.328 |
| `bm25` | 0.256 / 0.240 | 0.259 / 0.224 | 0.252 / 0.260 |
| `kg` | 0.257 / 0.280 | 0.256 / 0.260 | 0.257 / 0.268 |
| `hybrid` | 0.258 / 0.260 | 0.254 / 0.204 | 0.253 / 0.200 |
| `dense_bge` | 0.242 / 0.248 | 0.239 / 0.216 | 0.250 / 0.208 |
| `dense_medcpt` | 0.257 / 0.240 | 0.252 / 0.208 | 0.253 / 0.220 |
| `rrf` | 0.256 / 0.252 | 0.252 / 0.204 | 0.254 / 0.208 |
| `rerank` | 0.249 / 0.212 | 0.232 / 0.208 | 0.248 / 0.260 |
| `skill_hybrid` | 0.255 / 0.260 | 0.252 / 0.204 | 0.255 / 0.228 |
| `skill_schema_v1` | 0.254 / 0.176 | 0.254 / 0.236 | 0.255 / 0.284 |

### By subtask: Qwen2.5-7B

| Method | Design F1/exact | Description F1/exact | Open QA F1/exact | Property F1/exact | Property numeric | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| `none` | 0.000 / 0.000 | 0.604 / 0.032 | 0.125 / 0.012 | 0.000 / 0.000 | 0.276 | 0.182 |
| `bm25` | 0.000 / 0.000 | 0.588 / 0.012 | 0.430 / 0.000 | 0.008 / 0.008 | 0.240 | 0.256 |
| `kg` | 0.004 / 0.000 | 0.590 / 0.024 | 0.431 / 0.004 | 0.004 / 0.004 | 0.280 | 0.257 |
| `hybrid` | 0.008 / 0.000 | 0.585 / 0.016 | 0.432 / 0.000 | 0.008 / 0.004 | 0.260 | 0.258 |
| `dense_bge` | 0.000 / 0.000 | 0.554 / 0.004 | 0.416 / 0.004 | 0.000 / 0.000 | 0.248 | 0.242 |
| `dense_medcpt` | 0.000 / 0.000 | 0.593 / 0.028 | 0.435 / 0.000 | 0.000 / 0.000 | 0.240 | 0.257 |
| `rrf` | 0.000 / 0.000 | 0.585 / 0.024 | 0.434 / 0.000 | 0.004 / 0.004 | 0.252 | 0.256 |
| `rerank` | 0.000 / 0.000 | 0.578 / 0.020 | 0.419 / 0.004 | 0.000 / 0.000 | 0.212 | 0.249 |
| `skill_hybrid` | 0.000 / 0.000 | 0.588 / 0.028 | 0.434 / 0.004 | 0.000 / 0.000 | 0.260 | 0.255 |
| `skill_schema_v1` | 0.000 / 0.000 | 0.586 / 0.020 | 0.428 / 0.000 | 0.000 / 0.000 | 0.176 | 0.254 |

### By subtask: Qwen2.5-3B

| Method | Design F1/exact | Description F1/exact | Open QA F1/exact | Property F1/exact | Property numeric | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| `none` | 0.000 / 0.000 | 0.603 / 0.032 | 0.107 / 0.000 | 0.004 / 0.004 | 0.296 | 0.179 |
| `bm25` | 0.000 / 0.000 | 0.600 / 0.028 | 0.434 / 0.000 | 0.004 / 0.004 | 0.224 | 0.259 |
| `kg` | 0.000 / 0.000 | 0.595 / 0.020 | 0.426 / 0.000 | 0.004 / 0.004 | 0.260 | 0.256 |
| `hybrid` | 0.000 / 0.000 | 0.599 / 0.028 | 0.419 / 0.000 | 0.000 / 0.000 | 0.204 | 0.254 |
| `dense_bge` | 0.000 / 0.000 | 0.559 / 0.004 | 0.396 / 0.000 | 0.000 / 0.000 | 0.216 | 0.239 |
| `dense_medcpt` | 0.000 / 0.000 | 0.588 / 0.028 | 0.418 / 0.000 | 0.000 / 0.000 | 0.208 | 0.252 |
| `rrf` | 0.000 / 0.000 | 0.587 / 0.016 | 0.416 / 0.000 | 0.004 / 0.004 | 0.204 | 0.252 |
| `rerank` | 0.000 / 0.000 | 0.530 / 0.012 | 0.399 / 0.000 | 0.000 / 0.000 | 0.208 | 0.232 |
| `skill_hybrid` | 0.000 / 0.000 | 0.583 / 0.012 | 0.426 / 0.000 | 0.000 / 0.000 | 0.204 | 0.252 |
| `skill_schema_v1` | 0.000 / 0.000 | 0.572 / 0.012 | 0.438 / 0.000 | 0.004 / 0.004 | 0.236 | 0.254 |

### By subtask: Llama-3.2-3B

| Method | Design F1/exact | Description F1/exact | Open QA F1/exact | Property F1/exact | Property numeric | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| `none` | 0.000 / 0.000 | 0.609 / 0.040 | 0.130 / 0.000 | 0.000 / 0.000 | 0.328 | 0.185 |
| `bm25` | 0.000 / 0.000 | 0.575 / 0.020 | 0.434 / 0.000 | 0.000 / 0.000 | 0.260 | 0.252 |
| `kg` | 0.000 / 0.000 | 0.592 / 0.020 | 0.438 / 0.000 | 0.000 / 0.000 | 0.268 | 0.257 |
| `hybrid` | 0.000 / 0.000 | 0.577 / 0.012 | 0.434 / 0.000 | 0.000 / 0.000 | 0.200 | 0.253 |
| `dense_bge` | 0.000 / 0.000 | 0.577 / 0.020 | 0.422 / 0.000 | 0.000 / 0.000 | 0.208 | 0.250 |
| `dense_medcpt` | 0.000 / 0.000 | 0.583 / 0.016 | 0.430 / 0.000 | 0.000 / 0.000 | 0.220 | 0.253 |
| `rrf` | 0.000 / 0.000 | 0.573 / 0.016 | 0.438 / 0.000 | 0.004 / 0.004 | 0.208 | 0.254 |
| `rerank` | 0.000 / 0.000 | 0.557 / 0.012 | 0.430 / 0.000 | 0.004 / 0.004 | 0.260 | 0.248 |
| `skill_hybrid` | 0.000 / 0.000 | 0.583 / 0.016 | 0.436 / 0.000 | 0.000 / 0.000 | 0.228 | 0.255 |
| `skill_schema_v1` | 0.000 / 0.000 | 0.578 / 0.012 | 0.436 / 0.000 | 0.008 / 0.008 | 0.284 | 0.255 |

## Findings

1. Dense BGE is the strongest overall FDARxBench retriever for all three models.
2. Retrieval improves FDARxBench token F1 by 0.184--0.189 over closed book.
3. FDARxBench refusal is near saturation; factual and multihop drive the gain.
4. Mol-Instructions has no universal best retriever across models and metrics.
5. Existing skill variants do not beat Dense BGE on clean FDARxBench Protocol A.

## Provenance

- FDARx predictions: `outputs/baseline_rag/predictions_nolabel/`
- FDARx aggregate: `outputs/baseline_rag/results_summary_nolabel.csv`
- Mol retrieval predictions: `outputs/baseline_rag/predictions_bslret/`
- Mol retrieval aggregate: `outputs/baseline_rag/results_summary_bslret.csv`
- Mol closed-book predictions: `outputs/baseline_rag/predictions_bsl/`
- Mol closed-book aggregate: `outputs/baseline_rag/results_summary_bsl.csv`

Regenerate this report from the project root:

```bash
python skill_aware_rag/eval/build_protocol_a_report.py
```
