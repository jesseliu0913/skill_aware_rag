# Middle Check — All Results So Far

Honest synthesis of every experiment run, with numbers, mechanism, and what each
does / does not support for the paper. Headline model = **Qwen2.5-7B** unless
noted; metric = token-F1 (exact / numeric where relevant).

---

## 0. Setup

- **FDARxBench**: 3,000 train / 1,000 test (factual 578, multihop 204, refusal 218).
- **Mol-Instructions**: 3,000 train / 1,000 test (4 subtasks × 250: design, description, property, open-QA).
- **Models**: Qwen2.5-7B, Qwen2.5-3B, Llama-3.2-3B (3-model coverage only for §2; later probes are 7B-only).
- **Base prompt** (shared): instruction (+ abstention line) + question's own context + Q/A. Every method's train prompt = base + a `Retrieved KB evidence` block; skill method adds `Skill / How to solve / Answer format`.
- **KB**: 160,083 facts (PrimeKG, DrugChat ChEMBL/PubChem, FDARxBench label).

---

## 1. Training-protocol confound (foundational, solid)

Comparing retrieval methods is only fair if each retriever augments the **training**
split too (RAG-SFT), not just the test prompt (eval-only). Under **eval-only**,
sparse/graph/skill methods appear to fall *below* a no-retrieval LoRA; under
**RAG-SFT** they recover to parity. → We use RAG-SFT throughout. *(Solid, reusable
methodological point.)*

---

## 2. Two eval protocols on the trained adapters (FDARxBench, 3 models)

Same trained adapters, two ways to test:
- **raw-eval**: test on the raw question (no retrieval at test).
- **ret@test**: retrieval also present at test (standard RAG).

**Qwen2.5-7B, FDARxBench, token-F1 (all-1000 / exact):**

| method | raw-eval | ret@test |
|---|--:|--:|
| none (no retrieval) | **0.782** / 0.344 | — |
| bm25 | 0.770 | 0.784 |
| kg | 0.768 | 0.787 |
| dense_bge | 0.768 | 0.782 |
| rrf | 0.770 | **0.788** |
| skill_schema_v1 | 0.759 / 0.321 | **0.789 / 0.362** |

**Findings:**
- **raw-eval**: `none` wins; retrieval methods dip *below* it (train/test format shift; skill drops most — heaviest prompt).
- **ret@test**: retrieval recovers, all cluster 0.78–0.79; **skill_schema_v1 tops token-F1 (0.789) and exact (0.362)** — but margin over rrf is 0.001 (within noise).
- Consistent across 3 models (qwen3B / llama3B same ordering).

**Read:** retrieval's value is at **test time**, not baked into weights. On FDA the
gap between methods is tiny (≤0.007). Not enough alone for a "skill" claim.

---

## 3. Why FDA barely separates methods (diagnosis, important)

- **The answer is already given.** 83% of *factual* gold tokens and 73% of *multihop*
  gold tokens are already in the question's own label context → extractive reading,
  not retrieval.
- **Refusal (22%) is trivially perfect** (F1 = 1.000 for everyone once the abstention
  line is in) → inflates the aggregate. Excluding refusal, `none` (7B) = 0.727, not 0.782.

→ **FDARxBench in its standard form does not test retrieval.** This motivates the
no-label probe.

---

## 4. No-label probe — retrieval value (decisive, strong)

Remove the given FDA label from the prompt (FDA-only). `none` = closed-book;
retrieval methods must retrieve the label from the KB using the question alone.

**Qwen2.5-7B, FDA, factual+multihop (782), token-F1, WITH → NO label:**

| method | WITH label | NO label | drop |
|---|--:|--:|--:|
| **none (closed-book)** | 0.727 | **0.422** | **−0.305** |
| dense_bge | 0.725 | **0.659** | −0.066 |
| rrf | 0.734 | 0.657 | −0.077 |
| rerank | 0.725 | 0.638 | −0.087 |
| kg | 0.732 | 0.635 | −0.096 |
| hybrid | 0.728 | 0.624 | −0.104 |
| skill_hybrid | 0.727 | 0.624 | −0.103 |
| skill_schema_v1 | 0.736 | 0.622 | −0.114 |
| bm25 | 0.729 | 0.595 | −0.134 |
| dense_medcpt | 0.726 | 0.594 | −0.132 |

**Findings:**
- **Closed-book `none` collapses −0.305 (0.727→0.422)** — the model has no parametric knowledge of specific FDA facts.
- Retrieval methods **hold at 0.59–0.66**. Best `dense_bge` 0.659 vs closed-book `none` 0.422 = **+0.237**.
- **skill_schema_v1's source-constrained retrieval (0.622) trails free dense_bge (0.659)** — the constraint *hurts* a strong retriever.

**Read:** the paper's **strongest, cleanest result**. Retrieval has large value once
the answer is not handed over — exactly what the standard FDA setup hides.

---

## 5. Skill template as a pure add-on (fixed dense_bge + skill), 7B FDA no-label

Isolate the skill **template** (steps + answer_format) on top of a fixed retriever;
retrieval is identical, only the prompt wrapper differs.

| | fact+multi F1 | by task |
|---|--:|---|
| dense_bge | 0.659 | — |
| dense_bge + skill | 0.665 (**+0.005**) | factual +0.004, multihop +0.009, refusal +0.005 |

**Read:** template is **≈ neutral** (+0.005 < 1 SE ≈ 0.012). Never hurts (unlike the
constrained version), helps most on multihop (needs the "combine sections" step), but
within noise. FDA factual is too simple to need solving steps.

---

## 6. Mixed experiment — does skill *routing* help? (the skill-based claim)

Combined **FDA no-label + Mol** (2,000 test, 6 task types). Same retriever, k=8, one
LoRA on the mixed data; only difference = routing retrieval to each task's source
(FDA→label, molecule→drugchat, open→primekg+drugchat).

### 6a. Strong retriever (dense_bge) — routing is redundant
Evidence sources for MOL questions are **near-identical** with/without routing
(dense already ~91% on-source: a molecule query is semantically close to molecule
facts). Scores: A (no route) overall 0.581 / MOL 0.434; B (routed+template) 0.589 /
0.446 → **Δ = +0.008 / +0.012 (noise)**. Mechanism does **not** fire for dense.

### 6b. Weak retriever (bm25) — routing **rescues** it (mechanism fires)
Evidence sources for MOL questions:

| | drugchat (right) | primekg (wrong) | fda_label (wrong) |
|---|--:|--:|--:|
| **bm25 (no route)** | 31% | **60%** | 9% |
| **bm25 + skill route** | **80%** | 13% | 7% |

**Plain bm25 pulls 60% PrimeKG + 9% FDA label for molecule questions** — lexical
matching can't tell a molecule question should query the compound DB. **Skill routing
corrects it to 80% on-source.** (Residual primekg/fda is from open-QA, whose skill
legitimately allows multiple sources.)

**Scores (Qwen7B, 2000):**

| | overall | FDA | MOL |
|---|--:|--:|--:|
| A′ bm25 (no route) | 0.563 | 0.681 | 0.445 |
| B′ bm25 + skill route | 0.570 | 0.692 | 0.447 |
| Δ | +0.007 | +0.011 | **+0.003** |

**Read (the decisive negative):** the mechanism fires (60%→80% on-source) but **does
not convert to accuracy** — MOL moves only +0.003 (SE≈0.018, pure noise). The small
overall +0.007 comes from FDA (+0.011), *not* from the hypothesized molecule
source-correction. **Reason: Mol tasks don't benefit from retrieval at all**
(answers come from the molecule structure + reasoning, not from retrieved similar
molecules), so fixing the source fixes something the score doesn't depend on.
→ "routing rescues weak retrievers" is **not supported** on these benchmarks.

---

## 7. Honest scorecard for a "skill-aware" claim

| skill ingredient | result | verdict |
|---|---|---|
| source-**constrained** retrieval | FDA no-label: 0.622 vs free dense 0.659 | **hurts** strong retriever |
| skill **template** (steps/format) | +0.005 (noise), multihop +0.009 | **≈ neutral** |
| skill **routing** on **dense** | +0.008 (noise), sources unchanged | **redundant** (dense self-routes) |
| skill **routing** on **bm25** | 60%→80% source correction, but MOL +0.003 (noise) | **fires but no accuracy effect** (Mol doesn't use retrieval) |

---

## 8. Paper-narrative options

**Strong & already-solid (retrieval-centric):**
1. Training-protocol confound (RAG-SFT vs eval-only).
2. Benchmark critique: FDARxBench hands over the answer (83%/73% overlap; refusal inflates) → doesn't test retrieval.
3. No-label probe: retrieval value = **+0.237** once the answer is withheld.

**Skill-specific (what makes it *your* paper):**
4. On strong dense retrievers, explicit skill routing/constraint is **redundant or harmful** (dense self-routes). — honest negative.
5. On **weak retrievers**, skill routing **recovers the source selection** strong retrievers get for free (bm25 60%→80% on-source). — *if scores confirm*, this is the mechanistic skill claim: **"skill routing makes cheap retrievers competitive."**

**Recommended framing (§6b came back negative — decided):**

All four skill ingredients are neutral-to-negative on FDA+Mol, because **neither
benchmark rewards skill-awareness**: FDA hands over the answer (or needs one source);
Mol doesn't use retrieval at all. So the paper **cannot** claim a skill accuracy win
here honestly. Two viable directions:

- **(i) Honest-scope paper on these benchmarks.** Lead with the three solid results
  (protocol confound, benchmark critique, no-label retrieval value +0.237). Position
  the skill schema as an **interpretable, non-harmful control layer** whose *mechanism*
  is demonstrable (task-adaptive source routing: bm25 60%→80% on-source) even though
  accuracy is retrieval-bound on these datasets. Frame the neutral skill result as a
  **finding**: "on benchmarks where one source suffices or retrieval is inert, explicit
  skill structure neither helps nor hurts — its value needs a heterogeneous,
  retrieval-sensitive benchmark."
- **(ii) Change the testbed.** To make skill-awareness pay off in accuracy, we need a
  benchmark that is (a) retrieval-sensitive per task AND (b) genuinely multi-source,
  where picking the wrong source hurts. Neither FDA nor Mol is. This is the honest
  requirement, but it is more work (new/curated eval).

**My recommendation: (i)** — it is defensible today and the retrieval-value + benchmark-
critique story is strong on its own; the skill schema rides along as the interpretable
control layer with a demonstrated routing mechanism, not as an accuracy claim.

---

## 9. Open items
- **§6b bm25 mixed scores** (A′ vs B′) — running; decides the main skill claim.
- **3-model coverage** for §4–6 (currently 7B only) — for robustness once direction is fixed.
- Optional: bm25 vs dense routing across seeds to pin the noise floor (~±0.012 on 782, ~±0.008 on 2000).

---

## Appendix A — All methods (hands-off reference)

Every method queries the same unified KB (160,083 facts) and, except `none`, is
trained under RAG-SFT. Default budget k = 8.

| method | family | how it retrieves | source scope | prompt |
|---|---|---|---|---|
| `none` | control | no retrieval | — | base prompt |
| `bm25` | sparse | Okapi BM25 (k1=1.2, b=0.75) over fact tokens | whole KB | base + evidence |
| `kg` | graph | alias/entity linking to graph facts; keeps only alias-matched facts (BM25 + alias boost +8, PrimeKG +1) | whole KB (alias-linked) | base + evidence |
| `hybrid` | sparse+graph | BM25 candidates ∪ alias-linked facts, alias hits boosted +8 | whole KB | base + evidence |
| `skill_hybrid` | skill (coarse) | hybrid retrieval + coarse skill source filter (generic legacy skill) | skill-allowed | base + evidence |
| `dense_bge` | dense | BAAI/bge-large-en-v1.5 bi-encoder, cosine top-k | whole KB | base + evidence |
| `dense_medcpt` | dense | NCBI/MedCPT bi-encoder (biomedical) | whole KB | base + evidence |
| `rrf` | fusion | reciprocal-rank fusion of BM25 + dense(BGE) ranks | whole KB | base + evidence |
| `rerank` | fusion+CE | RRF first stage (50) → MedCPT cross-encoder rerank → top-k | whole KB | base + evidence |
| **`skill_schema_v1`** (ours, orig.) | skill | skill-routed, source-constrained hybrid; per-skill top_k (3–4) | per-skill | **skill schema** |
| **`dense_bge_skillrouted`** (ours, new) | skill | dense_bge retrieval, then filter candidates to routed-skill sources; k=8 | per-skill | **skill schema** |
| **`bm25_skillrouted`** (ours, new) | skill | pure BM25 scoring, then filter to routed-skill sources; k=8 | per-skill | **skill schema** |

Encoders/rerankers: BGE = `BAAI/bge-large-en-v1.5`; MedCPT = `ncbi/MedCPT-{Query,Article}-Encoder` + `ncbi/MedCPT-Cross-Encoder`.

---

## Appendix B — Skill schema (v1), as designed

**Routing** (rule-based, no learned classifier), by dataset/task fields:

```
if source == fdarxbench (or input_type == "FDA label context"):
    multihop -> fda_label_multihop   else -> fda_label_factual
elif task is property / "molecular weight" / "logp"      -> molecule_property_numeric
elif task is "description guided" / "design" / "synthesize" -> molecule_design
elif task is "molecular description" / "describe"        -> molecule_description
else                                                     -> biomedical_open_qa
```

**Six skills** — each a typed record `{allowed sources, answer_format, solving steps}`.
(For the fixed-retriever variants the `sources` field drives **routing**; the
retriever and k come from the fixed retriever. `steps` + `answer_format` form the
prompt template.)

| skill | allowed sources | answer_format |
|---|---|---|
| `fda_label_factual` | fdarxbench_label | short_label_grounded_answer |
| `fda_label_multihop` | fdarxbench_label | synthesized_label_answer |
| `molecule_property_numeric` | drugchat_chembl, drugchat_pubchem | numeric_or_short_property_value |
| `molecule_description` | drugchat_chembl, drugchat_pubchem | natural_language_description |
| `molecule_design` | drugchat_chembl, drugchat_pubchem | molecule_string |
| `biomedical_open_qa` | primekg, drugchat_chembl, drugchat_pubchem, fdarxbench_label | concise_biomedical_answer |

**Solving steps (`How to solve`) per skill:**

- **fda_label_factual**: (1) locate the relevant label section; (2) extract the exact fact asked (event/dose/contraindication/population); (3) answer concisely, grounded in label text.
- **fda_label_multihop**: (1) identify the 2+ label sections the question connects; (2) extract the key fact from each; (3) combine into one coherent grounded answer.
- **molecule_property_numeric**: (1) read structure from SELFIES/SMILES; (2) identify the requested property (e.g. HOMO-LUMO gap, logP); (3) use similar retrieved molecules as reference, give the numeric value only.
- **molecule_description**: (1) parse structure — class / functional groups / natural-product source; (2) use similar retrieved compound facts as reference; (3) write a concise natural-language description.
- **molecule_design**: (1) read the design requirement (role/class/scaffold/target property); (2) recall similar molecules from retrieved compound facts; (3) output a SELFIES string satisfying it.
- **biomedical_open_qa**: (1) identify entities (drug/disease/protein/pathway); (2) retrieve relevant relations/facts; (3) answer concisely from retrieved facts.

**Prompt layout** (skill methods): instruction (+abstention) → `Skill:` → `How to solve:` (steps) → `Answer format:` → `Known information:` (given context; absent in no-label) → `Skill-based retrieved knowledge:` (evidence) → `Question:` / `Answer:`.

Dropped from the original design (unused after redesign): `fda_label_refusal`,
`drug_relation_qa`, `unsupported_or_low_evidence` skills; and the `required_slots` /
per-skill `top_k` fields (retrieval budget now comes from the fixed retriever).

---

## Appendix C — KB, data, and file map

- **KB sources**: PrimeKG 83,075 (graph relations), DrugChat-ChEMBL 50,000, DrugChat-PubChem 13,735, FDARxBench label 13,273 → `outputs/knowledge_bank/unified/`.
- **Data splits**: `outputs/qa_skill_data_corrected/{fdarxbench,mol_instructions}/` (with-label); `outputs/qa_skill_data_nolabel/fdarxbench/` (label removed); `outputs/qa_skill_data_mixed/` (FDA no-label + Mol).
- **Predictions**: `outputs/baseline_rag/predictions_bsl/` (raw-eval), `predictions_bslret/` (ret@test), `predictions_nolabel/` (no-label), `predictions_mixed/` (mixed).
- **Metric**: token-F1 (whitespace/punctuation-normalized), exact match, numeric accuracy (Mol property). SE ≈ 0.012 on 782 items, ≈ 0.008 on 2000.
- **Code**: retrieval `skill_aware_rag/retrieval/{augment_with_skill_kb,augment_baseline_rag,retrievers}.py`; skill schema `SKILL_SCHEMA_V1` + `infer_schema_v1_skill` in `augment_with_skill_kb.py`; orchestrators in `skill_aware_rag/slurm/`.
