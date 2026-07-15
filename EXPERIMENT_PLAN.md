# Skill-Aware RAG — Master Experiment Plan (AAAI)

This document organizes the full experimental program behind the paper. It maps
the storyline (skill schema as a **knowledge-planning layer**) onto concrete,
runnable experiments, records what infrastructure already exists versus what must
be built, and defines the branch/record conventions used to track each one.

> **Scope note.** This is a *plan*, not a results report. Every experiment is a
> planned slot with a build status. We deliberately do **not** bake current
> result numbers in here as conclusions; per-experiment result writeups live in
> `results/` and each experiment's own `RESULTS.md`.

---

## 0. Thesis and contributions

**Thesis.** Drug-discovery queries are not a homogeneous retrieval problem. They
are different *skills*, and each skill requires a different knowledge source,
retrieval behavior, reasoning pattern, and answer format. Existing RAG retrieves
from a heterogeneous KB with a uniform strategy, implicitly assuming every
question needs the same kind of evidence. We make this structure explicit through
a **skill schema** that plans retrieval and generation before ranking begins:

```
q  →  skill identification  →  skill-conditioned retrieval / evidence organization  →  skill-conditioned generation
```

**Contributions.**

1. **Skill schema for drug-discovery knowledge planning** — a typed task
   interface mapping task types to required sources, retrieval behavior, solving
   procedures, and answer formats.
2. **Skill-aware retrieval over a heterogeneous biomedical + molecular KB** —
   instantiated over regulatory labels, biomedical KG facts, and molecular
   databases, enabling task-conditioned evidence retrieval and organization.
3. **Diagnostic analysis of task-specific knowledge requirements** — showing that
   drug-discovery question types induce distinct knowledge-source mixtures, so
   uniform RAG is structurally mismatched.
4. **Controlled evaluation of schema mechanisms** — source routing, solving steps,
   answer-format control, and evidence organization, isolated via ablations and
   stress tests, establishing when planning helps and when uniform retrieval
   suffices.

The method placeholder name in code/docs is **`SkillRAG`** (rename freely; used
as `\methodname` in the paper).

---

## 1. Research-question map

| RQ | Question | Answered by | Status |
|---|---|---|---|
| **RQ1** | Are drug-discovery tasks knowledge-heterogeneous? | Exp01 source-heterogeneity | 🟢🟡 |
| **RQ2** | Does uniform retrieval pull from wrong/unnecessary sources? | Exp03 retrieval-planning metrics | 🟡 |
| **RQ3** | Does the skill schema improve retrieval *planning*? | Exp03 + Exp06 controls | 🟡 |
| **RQ4** | Does better planning improve downstream answers? | Exp02 main performance (+ oracle-source bound) | 🟢🟡 |
| **RQ5** | Which part of the schema matters? | Exp04 schema ablations | 🟡 |
| **RQ6** | When does the schema help or not help? | Exp05 stress tests + Exp06 controls | 🟡🔴 |
| **RQ7** | Does it generalize to shifted/unseen tasks and backends? | Exp07 generalization | 🟡🔴 |
| (necessity) | Does source choice *causally* matter when it is ambiguous? | Exp08 necessity benchmark | 🔴 |

**Status legend.** 🟢 `reuse` — infrastructure + runs already exist. 🟡 `build` —
runnable on current data, needs new code/config. 🔴 `new-data` — needs a
benchmark/probe/KB that does not exist yet. Mixed tags mean the experiment spans
categories.

---

## 2. Method recap — the skill schema as a typed task interface

Each skill is a declarative record (not a prompt template). Current definition:
`SKILL_SCHEMA_V1` in `retrieval/augment_with_skill_kb.py:103`.

| Schema field | Role | Realized in code |
|---|---|---|
| **Skill identity** | which task this is | `infer_schema_v1_skill()` (rule router) `augment_with_skill_kb.py:278` |
| **Allowed / preferred sources** | where evidence comes from | `SKILL_SCHEMA_V1[skill]["sources"]`, `relations` |
| **Solving procedure** | how evidence is used | `["steps"]` shown in `build_completed_skill_prompt()` |
| **Answer format** | expected output type | `["answer_format"]` |
| (budget) | evidence budget | `["top_k"]`, `["required_slots"]` |

Six skills: `fda_label_factual`, `fda_label_multihop`, `molecule_property_numeric`,
`molecule_description`, `molecule_design`, `biomedical_open_qa`.

**Deployment rule.** The deployed router infers the skill from **question-only**
inputs. Gold task labels define the *oracle* upper bound (Exp06) but must never be
an input to the deployed system.

---

## 3. Shared experiment substrate (what every experiment builds on)

### Datasets
| Dataset | `source` value(s) | Task strata | Train / test |
|---|---|---|---|
| FDARxBench | `fdarxbench` | `factual`, `multihop`, `refusal` | 3,000 / 1,000 |
| Mol-Instructions | `molinst_description_guided_design`, `molinst_molecular_description`, `molinst_property`, `molinst_open_qa` | design, description, property, open-QA | 3,000 / 1,000 (4×250 test) |

Builders: `data/build_corrected_qa_datasets.py` → `outputs/qa_skill_data_corrected/{dataset}/{train,test}.jsonl`.
No-label FDA probe: `outputs/qa_skill_data_nolabel/fdarxbench/`.

### Models
`qwen2_5_7b` (headline), `qwen2_5_3b`, `llama3_2_3b`. Paths in each matrix's
`MODEL_PATHS` map.

### Knowledge base
`outputs/knowledge_bank/unified/facts.jsonl` — **160,083 facts** over 4 sources:
`primekg` 83,075 · `drugchat_chembl` 50,000 · `drugchat_pubchem` 13,735 ·
`fdarxbench_label` 13,273. Plus `alias_index.json`. Fact fields: `id, source,
source_id, title, text, aliases, entities, relation, metadata, tokens`.
Builder: `kb/build_unified_kb.py`. Dense embeddings: `outputs/knowledge_bank/embeddings/{bge,medcpt}/`.

### Pipeline (identical across all methods — retriever is the only variable)
```
augment (retrieve + build prompt)  →  LoRA train (RAG-SFT)  →  inference  →  metrics
```
- **augment**: `retrieval/augment_baseline_rag.py` (dense family) and
  `retrieval/augment_with_skill_kb.py` (sparse/graph/skill). Both emit records
  with `skill_schema`, `retrieved_kb_evidence`, `prompt`, `messages`, and (v1)
  `evidence_slots`.
- **train**: `train/train_qa_skill_lora.py` (LoRA r=16, reads `messages`).
- **infer**: `eval/run_qa_jsonl_inference.py` (reads `record["prompt"]`).
- **metrics**: `eval/evaluate_instruction_outputs.py` (`token_f1`, `exact`,
  `numeric_match`), aggregated by `eval/aggregate_results.py`.
- **orchestration**: `slurm/*_matrix.sh` drivers chain generic sbatch jobs
  (`submit_augment*.sbatch`, `submit_generic_qa_lora_train.sbatch`,
  `submit_generic_qa_eval.sbatch`) with `afterok` dependencies.

### RAG-SFT protocol (methodological invariant)
Every retrieval method augments the **training** split too, not only the test
prompt. Comparing eval-only retrieval against a no-retrieval LoRA is confounded by
train/test prompt shift. All method comparisons use RAG-SFT.

### Method roster (retrieval conditions)
| method | family | source scope | code |
|---|---|---|---|
| `none` | control (closed book) | — | raw split, no augment |
| `bm25` | sparse | whole KB | `augment_with_skill_kb.py` |
| `kg` | graph (alias-linked) | whole KB | `augment_with_skill_kb.py` |
| `hybrid` | sparse+graph | whole KB | `augment_with_skill_kb.py` |
| `dense_bge` | dense | whole KB | `augment_baseline_rag.py` |
| `dense_medcpt` | dense (biomedical) | whole KB | `augment_baseline_rag.py` |
| `rrf` | fusion | whole KB | `augment_baseline_rag.py` |
| `rerank` | fusion + cross-encoder | whole KB | `augment_baseline_rag.py` |
| `skill_hybrid` | skill (coarse route) | skill-allowed | `augment_with_skill_kb.py` |
| `skill_schema_v1` | **skill (ours)** | per-skill | `augment_with_skill_kb.py` |
| `bm25_skillrouted` | skill route on bm25 | per-skill | `augment_with_skill_kb.py` |
| `dense_bge_skillrouted` | skill route on dense | per-skill | `augment_baseline_rag.py` |
| **`oracle_source`** | **upper bound (build)** | gold-skill source | Exp02 (new) |

---

## 4. Experiment registry

Each experiment is developed on its own branch `exp/NN-<name>` off `jesse_baseline`,
with code under `experiments/expNN_<name>/` and a per-experiment `RESULTS.md`
(empty template until run). Reuse of existing scripts is preferred; new code is
additive and never mutates the baseline pipeline.

### Exp01 — Knowledge-source heterogeneity analysis 🟢🟡
- **Serves:** RQ1. **Paper:** §5.1, Figure 1.
- **Question:** do question types induce distinct knowledge-source mixtures?
- **Conditions:** for each question-type stratum, over retrieved evidence:
  dominant-source share, source entropy, cross-task source divergence
  (JS-divergence between per-task source distributions), oracle-source coverage
  (does the schema's allowed-sources set cover the answer-supporting source?).
- **Metrics:** distributional (share, entropy, divergence, coverage).
- **Reuse (🟢):** `pre_design/analyze_preliminary.py` already computes per-task-type
  KB-source mixtures (`evidence_by_source.csv`) and `pre_design/plot_preliminary.py`
  renders the stacked-bar **Figure 1** (`figure1_source_composition`).
- **Build (🟡):** add entropy / cross-task JS-divergence / oracle-source-coverage
  metrics and their table; extend the figure to more retrievers than `hybrid`.
- **Deps:** completed `predictions_bslret/` + `qa_skill_data_skillkb/` augmented files.

### Exp02 — Main task performance + oracle-source upper bound 🟢🟡
- **Serves:** RQ4. **Paper:** §5.2, main results table.
- **Question:** does skill-aware planning improve downstream answers, and how close
  does it get to an oracle-source bound?
- **Conditions:** full method roster (§3) + `oracle_source` upper bound (retrieve
  only from the gold-skill's allowed sources using the **gold** task label).
  Expected ordering to test: `uniform RAG < SkillRAG ≤ oracle_source`.
- **Metrics (task-level, not only aggregate):** FDA factual/multihop → token-F1,
  exact; refusal → abstention accuracy; open-QA → token-F1/entity-F1; property →
  numeric accuracy; description → token-F1; design → validity/uniqueness (Exp reuses
  metric hooks, see §5 catalog).
- **Reuse (🟢):** baseline matrices (`submit_nolabel_matrix.sh`,
  `submit_reteval_matrix.sh`) + `build_protocol_a_report.py`.
- **Build (🟡):** `oracle_source` augment method (a thin variant that hard-filters
  candidates to `SKILL_SCHEMA_V1[gold_skill]["sources"]`); one matrix driver adding it.

### Exp03 — Retrieval-planning quality 🟡
- **Serves:** RQ2, RQ3. **Paper:** §5.3. *(Often more important than the score table.)*
- **Question:** does uniform retrieval retrieve wrong/unnecessary sources, and does
  the schema fix source selection?
- **Metrics (over `retrieved_kb_evidence`, no retraining needed):**
  source precision@k, evidence recall@k (gold-token support proxy), wrong-source
  rate, source entropy, skill–source agreement (retrieved sources vs schema's
  allowed set), evidence utilization (answer↔evidence token overlap).
- **Build (🟡):** `experiments/exp03_retrieval_planning/retrieval_planning_metrics.py`
  consuming existing augmented/prediction JSONL. Pure post-hoc analysis.
- **Deps:** augmented splits per method + predictions for utilization.

### Exp04 — Skill-schema component ablations 🟡
- **Serves:** RQ5. **Paper:** §5.4. *(Answers "is this just prompt engineering?")*
- **Conditions (additive grid):**
  | variant | skill id | source routing | solving steps | answer format | evidence org |
  |---|:-:|:-:|:-:|:-:|:-:|
  | uniform RAG | ✗ | ✗ | ✗ | ✗ | ✗ |
  | + skill label only | ✓ | ✗ | ✗ | ✗ | ✗ |
  | + source routing | ✓ | ✓ | ✗ | ✗ | ✗ |
  | + solving steps | ✓ | ✗ | ✓ | ✗ | ✗ |
  | + answer format | ✓ | ✗ | ✗ | ✓ | ✗ |
  | + evidence org | ✓ | ✓ | ✗ | ✓ | ✓ |
  | full SkillRAG | ✓ | ✓ | ✓ | ✓ | ✓ |
- **Build (🟡):** parameterize prompt/route assembly in the skill augmenter behind
  flags (`--ablate source_routing,steps,answer_format,evidence_org`); one matrix.
- **Metrics:** same task-level metrics as Exp02.

### Exp05 — Skill routing under retrieval difficulty (stress tests) 🟡🔴
- **Serves:** RQ6. **Paper:** §5.5.
- **Stress settings:** weak retriever (bm25); small top-k budget (k∈{1,2,3});
  noisy heterogeneous KB (inject off-task facts); **decoy-source injection**
  (lexically similar wrong-source facts); no-label FDA (removes answer leakage);
  multi-source biomedical QA.
- **Reuse (🟢):** no-label matrix (`submit_nolabel_matrix.sh`), `bm25_skillrouted`.
- **Build (🟡):** top-k sweep config; a KB-perturbation module
  (`experiments/exp05_stress/perturb_kb.py`) producing noisy/decoy KB variants.
- **New-data (🔴):** decoy corpus construction (paraphrased wrong-source facts).

### Exp06 — Wrong-skill / oracle-skill controls 🟡
- **Serves:** RQ3, RQ6. **Paper:** §5.4/§5.6.
- **Conditions:** predicted skill (deployed), oracle skill (upper bound), random
  skill (negative control), wrong-but-plausible skill (sensitivity), no skill
  (uniform RAG). Plus a **skill confusion matrix** (predicted vs gold).
- **Build (🟡):** a skill-override hook (`--skill-override {predicted,oracle,random,wrong}`)
  in the augmenter that forces the routed skill; router-accuracy report.
- **Metrics:** task-level accuracy under each override + router accuracy/confusion.

### Exp07 — Generalization to shifted / unseen settings 🟡🔴
- **Serves:** RQ7. **Paper:** §6 discussion.
- **Tests:** held-out drug/source split (generalize beyond memorized drugs);
  held-out skill family (add a skill declaratively); new KB source (source-extensible
  without full retrain); new retriever backend (schema helps bm25/dense/hybrid/rerank alike).
- **Build (🟡):** split scripts (`experiments/exp07_generalization/make_splits.py`)
  for drug/source/skill holdouts; retriever-backend sweep reusing existing methods.
- **New-data (🔴):** a genuinely new KB source to demonstrate extensibility.

### Exp08 — Necessity benchmark (source-ambiguous, retrieval-sensitive) 🔴
- **Serves:** the necessity argument (§10 of the storyline). **Paper:** §5.5 core.
- **Requirement:** items where (1) the answer is *not* already in the input,
  (2) multiple KB sources contain plausible-but-not-equally-appropriate evidence,
  (3) choosing the wrong source hurts, (4) the schema picks the right source more
  often than uniform RAG.
- **Comparison:** uniform RAG, dense RAG, oracle-source RAG, wrong-skill RAG,
  SkillRAG — expected: SkillRAG closest to oracle without gold evidence.
- **New-data (🔴):** construction script
  (`experiments/exp08_necessity/build_necessity_probe.py`) + a written spec of the
  ambiguity criterion and decoy design. Neither FDARxBench nor Mol satisfies this,
  so this is net-new curation.

---

## 5. Metrics catalog

### Task-level answer metrics
| Task | Metric | Source |
|---|---|---|
| FDA factual | token-F1, exact | `evaluate_instruction_outputs.token_f1`, `exact` |
| FDA multihop | token-F1, exact, evidence coverage | + Exp03 recall@k |
| FDA refusal | abstention accuracy | derived from prediction vs "Information not found!" |
| Biomedical open QA | token-F1, entity-F1 | token_f1 + Exp03 entity overlap |
| Molecule property | numeric accuracy (tol), MAE | `numeric_match`; MAE new |
| Molecule description | token-F1 / semantic sim | token_f1 (+ optional embedding sim) |
| Molecule design | validity, uniqueness, constraint satisfaction | **new** (RDKit/SELFIES validity) |

> Note: token-F1 is **not** a valid molecule-design metric; report design under
> validity/uniqueness only.

### Retrieval-planning diagnostics (Exp03)
source precision@k · evidence recall@k · wrong-source rate · source entropy ·
skill–source agreement · evidence utilization.

### Reporting
Per-task breakdown always (aggregate hides skill-differential effects). SE ≈ 0.012
on 782 items, ≈ 0.008 on 2000; report multiple seeds + CIs where feasible.

---

## 6. Coverage / gap matrix

| Experiment | FDARxBench | Mol-Instructions | 3 models | Build status |
|---|:-:|:-:|:-:|---|
| Exp01 heterogeneity | 🟢 mixtures + fig | 🟢 mixtures + fig | 🟡 (metrics 7B-first) | extend metrics |
| Exp02 main + oracle | 🟢 baseline / 🟡 oracle | 🟢 baseline / 🟡 oracle | 🟢 | add `oracle_source` |
| Exp03 planning metrics | 🟡 | 🟡 | 🟡 | new post-hoc script |
| Exp04 ablations | 🟡 | 🟡 | 🟡 | flag-driven augmenter |
| Exp05 stress | 🟢 no-label / 🟡🔴 | 🟡🔴 | 🟡 | perturb + decoy |
| Exp06 skill controls | 🟡 | 🟡 | 🟡 | override hook |
| Exp07 generalization | 🟡🔴 | 🟡🔴 | 🟡 | split scripts |
| Exp08 necessity | 🔴 | 🔴 | — | new curation |

**Not yet in place, in priority order:** (1) `oracle_source` upper bound (Exp02),
(2) retrieval-planning metrics (Exp03), (3) skill-override controls (Exp06),
(4) ablation flags (Exp04), (5) KB perturbation + decoy (Exp05), (6) generalization
splits (Exp07), (7) necessity benchmark (Exp08).

---

## 7. Paper-section crosswalk

| Paper section | Experiment(s) |
|---|---|
| §5.1 Question types require different sources | Exp01 |
| §5.2 Main downstream performance | Exp02 |
| §5.3 Retrieval-planning quality | Exp03 |
| §5.4 Ablation of schema components | Exp04 (+ Exp06) |
| §5.5 Stress tests: noisy KB, weak retriever, no-label | Exp05 (+ Exp08) |
| §5.6 Failure analysis | Exp06 + honest negatives from Exp03/05 |

---

## 8. Branch and record conventions

- **Base:** every experiment branches from `jesse_baseline` (contains the full
  baseline pipeline + this plan). `git checkout jesse_baseline && git checkout -b exp/NN-<name>`.
- **Layout:** `experiments/expNN_<name>/` holds the experiment's code, its
  `README.md` (what/why/how-to-run), a `slurm/` driver if it submits jobs, and an
  empty `RESULTS.md` template (filled only when runs land).
- **Additive only:** experiment branches add code; shared-pipeline edits (e.g. an
  `--ablate` flag) are made backward-compatibly so `jesse_baseline` behavior is
  unchanged by default.
- **Push cadence:** push each `exp/NN-*` branch when its code is complete; open a
  PR per experiment. This plan doc lists status but does not track PR state.
- **No large artifacts:** `outputs/`, adapters, predictions, embeddings, and slurm
  logs stay local (see `.gitignore`). Only code + small CSV/figure/report outputs
  are committed.
