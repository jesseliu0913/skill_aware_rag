# Exp08 — Necessity benchmark: design specification

**Serves:** the *necessity* argument (§10 of the storyline). **Paper:** §5.5 core.
**Status:** 🔴 **new-data** — this benchmark does not exist yet. This document is
the design; `build_necessity_probe.py` is the mechanical construction scaffold.
**Curation and human validation of gold answers are REQUIRED and NOT YET DONE.**

---

## 1. Why a new probe

Neither FDARxBench nor Mol-Instructions isolates the claim we most need to
defend: that *choosing the right knowledge source is causally load-bearing*. In
both existing datasets the question's surface form already implies its source
(an FDA-label question wants the label; a SELFIES property question wants the
molecular DBs), and — worse — the answer is often recoverable from the input
context itself, so retrieval is not strictly necessary. A reviewer can therefore
attribute SkillRAG's gains to prompt formatting rather than to *source
selection*.

The necessity probe removes those escape hatches by construction: the answer is
absent from the input, and at least two KB sources present lexically plausible
evidence for the same surface query, but only one source is *appropriate*.

---

## 2. The four requirements (precise)

An item qualifies for the necessity probe **iff** all four hold:

- **(R1) Answer not in the input.** The gold answer is not derivable from the
  record's `input_molecule_or_context` / `question` alone. Operationally the
  input is held empty (`input_type = "none"`) or is a neutral stem that shares no
  answer-bearing token with the gold answer. A closed-book model must therefore
  *retrieve* to answer. (Enforced at construction; re-checked at validation.)

- **(R2) ≥2 sources with plausible-but-unequal evidence.** At least two distinct
  KB `source` values contain a fact whose surface mentions the query's key
  entity (the *collision alias*), so both are retrievable for the query's surface
  form — yet they are **not equally appropriate**: exactly one (source **A**,
  *gold source*) actually answers the question; the other(s) (source **B**,
  *decoy source*) mention the entity under a different relation/claim.

- **(R3) Wrong source ⇒ wrong/worse answer.** Answering from the decoy fact in
  source B yields a wrong or materially worse answer than answering from source
  A. The gold and decoy facts must carry *different* relations / answer objects
  (construction prefers `relation(A) ≠ relation(B)`); validation confirms the
  decoy-grounded answer is actually incorrect.

- **(R4) Schema beats uniform at source selection.** The skill schema's
  `allowed_sources` for the item's intended skill must select source A **more
  often** than an unconditioned (uniform) retriever would. Construction favors
  *discriminative* pairs where the decoy source is **not** in
  `SKILL_SCHEMA_V1[intended_skill]["sources"]`, so a skill-routed retriever
  filters the decoy out while a uniform retriever does not.

R1–R3 are properties of a single item; R4 is a property of the *item set under
the schema* and is the quantity the experiment measures.

---

## 3. Comparison arms

All arms share the identical pipeline (augment → RAG-SFT → infer → metrics); the
**only** variable is the retriever's source scope over the same unified KB. No
arm receives gold evidence.

| Arm | Source scope on the probe | Purpose |
|---|---|---|
| **uniform RAG** (`bm25` / `hybrid`) | whole KB, unconditioned | baseline; sees decoys |
| **dense RAG** (`dense_bge` / `rrf`) | whole KB, embedding-ranked | stronger uniform baseline; still sees decoys |
| **oracle-source RAG** | hard-filtered to item's **gold source A** (from provenance) | upper bound: correct source given for free |
| **wrong-skill RAG** | filtered to the **decoy source B** (negative control) | demonstrates R3: forcing the wrong source hurts |
| **SkillRAG** (`skill_schema_v1`) | filtered to `allowed_sources` of the **question-inferred** skill | ours: picks the source from the question, no gold label |

**Expected result.** `wrong-skill < uniform ≈ dense < SkillRAG ≤ oracle-source`,
with **SkillRAG the closest arm to the oracle-source bound while using no gold
evidence and no gold source label** — that gap-closure is the necessity result.
If uniform RAG already matches SkillRAG, the probe failed to be source-sensitive
(see §5 acceptance). If SkillRAG ≉ oracle, routing is the bottleneck (a finding,
not a bug).

Router note: SkillRAG infers the skill from the **question only** (deployment
rule, Plan §2). `intended_skill` in provenance is the construction-time target
used to build discriminative pairs and to define the oracle/wrong-skill scopes;
it is **not** fed to SkillRAG at eval time.

---

## 4. Decoy design

- **Collision alias.** A normalized entity string (from a fact's `aliases` /
  `entities`) that occurs in facts from **≥2 distinct sources**. This is the
  shared surface form that makes both sources retrievable for one query.
- **Gold fact (source A).** The fact whose source is highest-priority in the
  intended answer framing (`--gold-sources` order) and whose relation defines the
  question. Its object/text is the *provisional* gold answer.
- **Decoy fact (source B).** A fact from a **different** source that (i) shares
  the collision alias, (ii) is **lexically similar** to the gold fact (token
  Jaccard ≥ `--min-decoy-similarity`), and (iii) preferably carries a **different
  relation** so it implies a different answer. Discriminative pairs additionally
  require B ∉ `allowed_sources(intended_skill)` so the schema excludes it (R4).
- **Surface-ambiguity score.** Token Jaccard between gold and decoy fact text,
  recorded per item; higher = more confusable = a harder, better probe item.
- **Input neutrality.** The emitted record's input is empty so no answer leaks
  (R1). The *decoy is planted only in the KB*, never in the prompt — it enters an
  arm's context solely if that arm's retriever chooses source B.

---

## 5. Acceptance / validation criteria

The scaffold does **mechanical assembly only**. Before any run, a curator MUST
complete the validation pass (marked `TODO(validation)` in code / provenance).
An item is accepted into the benchmark iff:

1. **Gold verified.** A human confirms the provisional `gold_answer` is correct
   and answerable from source A's fact (fix or drop otherwise). *No fabricated or
   auto-accepted gold ships.*
2. **R1 holds.** The gold answer is not present in / inferable from the input;
   the question does not leak it.
3. **R2 holds.** Both source A and source B facts are genuinely retrievable for
   the question's surface form (the alias is a real shared mention, not a
   tokenizer artifact).
4. **R3 holds.** The decoy-grounded answer is judged wrong/worse than the
   gold-grounded answer (relations genuinely differ in meaning).
5. **Question well-formed & source-ambiguous.** The question reads naturally,
   mentions the collision entity, and does **not** telegraph source A in a way
   that trivializes routing; the intended skill is inferable from the question
   alone (else R4 is untestable at deploy time).
6. **Schema-discriminative.** B ∉ `allowed_sources(intended_skill)` (or the item
   is explicitly retained as a hard non-discriminative case and labeled so).

**Set-level acceptance.** Target ≥ N validated items per intended skill with a
mean surface-ambiguity score above a floor, and a measurable oracle−uniform gap
on a pilot model (if uniform already equals oracle, the set is not
source-sensitive and must be regenerated with stricter decoys).

**Metrics.** Task-level answer metrics per arm (`token_f1`, `exact`,
`numeric_match` as applicable; `eval/evaluate_instruction_outputs.py`) plus a
**source-selection accuracy** per arm = fraction of items whose retrieved
evidence includes source A (proxy for R4), computed from `retrieved_kb_evidence`.

---

## 6. Record schema emitted

Standard normalized QA schema (pipeline-consumable): `id, source,
input_molecule_or_context, input_type, question, gold_answer, task_type,
metadata, prompt, messages` (+ `source_path, source_index, decoded_smiles,
drug_name, retrieved_kg_evidence, retrieved_molecule_evidence` for parity). The
`gold_answer` is **provisional/unvalidated** and `metadata.validation.validated =
false` until curation. Per-item A/B provenance is written separately to
`provenance.jsonl` (never fed to any arm).
