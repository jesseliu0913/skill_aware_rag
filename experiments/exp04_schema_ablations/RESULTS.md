# Exp04 — Schema-component ablation results (RQ5)

Fill this in only when the ablation matrix has run. Metrics mirror Exp02
(task-level, not just aggregate). Each variant is `skill_schema_v1` with the
listed components removed via `--ablate`; `v6_full` = deployed SkillRAG.

## Ablation grid — task-level metrics

### FDARxBench
| variant | model | factual token-F1 | multihop token-F1 | refusal abstention | Δ vs v6_full |
|---|---|---|---|---|---|
| v0_uniform | | | | | |
| v1_skill_label | | | | | |
| v2_source_routing | | | | | |
| v3_solving_steps | | | | | |
| v4_answer_format | | | | | |
| v5_evidence_org | | | | | |
| v6_full (SkillRAG) | | | | | |

Not yet run.

### Mol-Instructions
| variant | model | property numeric acc | description token-F1 | design validity | open-QA token-F1 | Δ vs v6_full |
|---|---|---|---|---|---|---|
| v0_uniform | | | | | | |
| v1_skill_label | | | | | | |
| v2_source_routing | | | | | | |
| v3_solving_steps | | | | | | |
| v4_answer_format | | | | | | |
| v5_evidence_org | | | | | | |
| v6_full (SkillRAG) | | | | | | |

Not yet run.

## Per-component contribution (marginal effect of removing one mechanism)
| removed component | most-affected task(s) | mean Δ token-F1 | notes |
|---|---|---|---|
| skill_id | | | |
| source_routing | | | |
| solving_steps | | | |
| answer_format | | | |
| evidence_org | | | |

Not yet run.

## Takeaways
Not yet run.
