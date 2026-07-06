# Skill-Aware Knowledge Retrieval for Drug Discovery QA — Project Code

Self-contained code for the AAAI-2026 paper *Skill-Aware Knowledge Retrieval for
Drug Discovery Question Answering* (`../AAAI26_personalized_drug_discovery/`).
This folder is the **sole code location** for this project's pipeline. The scripts
previously lived scattered across `../scripts/` and `../baseline/`; `../baseline/`
has been removed (fully consolidated here) and `../scripts/` now retains only
legacy one-off / debug sbatches from earlier iterations (history, not part of this
pipeline). All data/results still live under `../outputs/`, and the raw KB source
`../knowledge_bank/primekg.csv` stays put (it is data, not code) — nothing large
was moved.

Every script anchors to the repo root `/playpen-jfs/jesse/drug_microbiome` and
reads/writes `outputs/...` there, so this folder is runnable in place. Verified:
`eval/aggregate_results.py --include-existing` reproduces
`outputs/baseline_rag/results_summary.csv` byte-for-byte, and the SLURM
orchestrators submit the sbatch files in this folder (dry-run checked).

Python env: `/playpen-shared/jesse/miniconda3/envs/agentskill/bin/python`.
HF cache / models: `/playpen-shared/jesse/cache`.

---

## 1. Pipeline (end to end)

```
                     scripts in this folder                         data under ../outputs/
  ┌──────────────────────────────────────────────────┐   ┌──────────────────────────────────────┐
  data/build_corrected_qa_datasets.py  ───────────────────►  qa_skill_data_corrected/<ds>/{train,test}.jsonl
  kb/build_unified_kb.py               ───────────────────►  knowledge_bank/unified/{facts.jsonl,alias_index.json}
  kb/build_kb_embeddings.py            ───────────────────►  knowledge_bank/embeddings/   (dense: BGE, MedCPT)
                                                              │
  retrieval/augment_with_skill_kb.py  (bm25|kg|hybrid|skill_hybrid|skill_schema_v1)
  retrieval/augment_baseline_rag.py   (dense_bge|dense_medcpt|rrf|rerank)
        │  augment TRAIN split  ──────────────────────────►  baseline_rag/data/<ds>/train_<method>.jsonl
        │  augment TEST  split  ──────────────────────────►  qa_skill_data_skillkb/<ds>/test_<method>.jsonl
        ▼
  train/train_qa_skill_lora.py  (LoRA r16, 1125 steps)  ──►  baseline_rag/adapters/<model>_<ds>_<method>_raglora/
        ▼
  eval/run_qa_jsonl_inference.py  (load base+adapter, gen) ►  baseline_rag/predictions/<...>_raglora_test.jsonl
  eval/evaluate_instruction_outputs.py  (F1/EM/numeric)  ──►  ..._raglora_test_eval.json
        ▼
  eval/aggregate_results.py  ──────────────────────────────►  baseline_rag/results_summary.{csv,md}
```

**RAG-SFT is the key protocol.** Each retriever augments the *training* split, a
per-method LoRA is trained on that augmented train, and it is evaluated on the
test split augmented by the *same* retriever. The retriever is therefore the only
independent variable — datasets, splits, decoding, LoRA hyperparameters, and the
eval script are held fixed. (An earlier *eval-only* protocol injected evidence at
test time on top of one shared no-retrieval LoRA; that understated every
retrieval method and is reported only as a fairness contrast.)

---

## 2. File index

| Path | Role |
|---|---|
| `data/build_corrected_qa_datasets.py` | Build the corrected FDARxBench / Mol-Instructions train/test splits |
| `data/build_qa_skill_dataset.py` | Earlier base QA-skill dataset builder (upstream of the corrected splits) |
| `kb/build_unified_kb.py` | Serialize PrimeKG + DrugChat(ChEMBL,PubChem) + FDARxBench into the unified KB (160,083 facts, 19,953 aliases) |
| `kb/build_kb_embeddings.py` | Encode KB facts into dense indices (BGE, MedCPT) for dense retrieval |
| `kb/download_models.py` | Fetch base models / encoders into the HF cache |
| `retrieval/retrievers.py` | Dense bi-encoders (BGE, MedCPT), cross-encoder reranker, FAISS/numpy index, RRF |
| `retrieval/augment_with_skill_kb.py` | **Proposed method + sparse/graph baselines.** BM25/KG/hybrid retrieval, the 8-skill schema (`SKILL_SCHEMA_V1`), skill routing, slot filling, completed-prompt builder |
| `retrieval/augment_baseline_rag.py` | Dense-family augmentation (`dense_bge`, `dense_medcpt`, `rrf`, `rerank`); reuses the skill-KB module so prompts stay byte-identical across methods |
| `train/train_qa_skill_lora.py` | PEFT LoRA SFT (rank 16, q/k/v/o/gate/up/down_proj). Masks the prompt tokens (`labels=-100`) and trains only on the answer |
| `eval/run_qa_jsonl_inference.py` | Load base model + adapter, generate predictions for a JSONL test file |
| `eval/evaluate_instruction_outputs.py` | Metrics: exact, normalized exact, token F1, numeric accuracy, grouped by (file, dataset/subtask) |
| `eval/aggregate_results.py` | Roll all `*_eval.json` into `results_summary.{csv,md}` |
| `analysis/analyze_skill_schema.py` | Skill routing / slot-coverage analysis and plots |
| `slurm/submit_ragsft_skillkb_matrix.sh` | **Main orchestrator** for the sparse/graph/skill family under RAG-SFT (5 methods × 2 datasets × 3 models) |
| `slurm/submit_baseline_matrix.sh` | Orchestrator for the dense/fusion family under RAG-SFT |
| `slurm/submit_skillkb_eval_matrix.sh` | Eval-only matrix (historical fairness contrast) |
| `slurm/submit_model_matrix.sh` | Base no-retrieval LoRA + raw baselines |
| `slurm/submit_generic_qa_lora_train.sbatch` | Generic GPU LoRA-train job (called by orchestrators) |
| `slurm/submit_generic_qa_eval.sbatch` | Generic GPU inference+eval job (called by orchestrators) |
| `slurm/submit_augment_skillkb.sbatch` | CPU augment for bm25/kg/hybrid/skill/skill_schema_v1 |
| `slurm/submit_augment.sbatch` | CPU/GPU augment for the dense family |
| `slurm/submit_build_embeddings.sbatch` | Build the dense KB indices |
| `slurm/build_and_submit_retrieval_comparison.sh` | Convenience wrapper: augment then submit the eval matrix |

---

## 3. Reproduce

```bash
cd /playpen-jfs/jesse/drug_microbiome        # every script anchors here

# (data + KB already built under outputs/; rebuild only if needed)
# python skill_aware_rag/data/build_corrected_qa_datasets.py
# python skill_aware_rag/kb/build_unified_kb.py

# RAG-SFT for the sparse/graph/skill family (all 5 methods × 2 datasets × 3 models)
bash skill_aware_rag/slurm/submit_ragsft_skillkb_matrix.sh
# dense/fusion family
bash skill_aware_rag/slurm/submit_baseline_matrix.sh

# (env knobs: MODELS, DATASETS, METHODS, LIMIT, MAX_STEPS, FORCE_AUGMENT=1, DRY_RUN=1)

# collect results
python skill_aware_rag/eval/aggregate_results.py --include-existing
```

Status: all 54 RAG-SFT adapters (9 methods × 2 datasets × 3 models) are trained
and evaluated; `outputs/baseline_rag/results_summary.{csv,md}` is current.

---

## 4. Results snapshot (Qwen2.5-7B, RAG-SFT)

FDARxBench is token-F1 / exact over 1000 test items; Mol-Instructions is average
token-F1 over 4 subtasks and numeric accuracy on property prediction.

| System | FDA F1 | FDA Exact | Mol F1 | Mol Num. |
|---|--:|--:|--:|--:|
| No retrieval (raw)  | 0.612 | 0.237 | 0.103 | 0.000 |
| No retrieval (LoRA) | 0.789 | 0.342 | 0.252 | 0.280 |
| BM25          | 0.789 | 0.353 | 0.258 | 0.248 |
| KG            | 0.788 | 0.351 | **0.261** | 0.260 |
| Hybrid        | 0.786 | 0.358 | 0.259 | 0.292 |
| Dense-BGE     | 0.786 | 0.348 | 0.246 | 0.296 |
| Dense-MedCPT  | 0.784 | 0.345 | 0.259 | 0.252 |
| Rerank        | 0.786 | 0.345 | 0.249 | **0.312** |
| RRF           | **0.793** | 0.350 | 0.250 | 0.216 |
| Skill-hybrid  | 0.786 | 0.355 | 0.258 | 0.236 |
| **Skill-schema (ours)** | 0.792 | **0.359** | 0.255 | 0.292 |

Under RAG-SFT the retrieval methods converge to 0.784–0.793 F1 on FDARxBench;
the skill schema ties the best RAG (RRF 0.793) within 0.001 and takes the best
exact match. Under the eval-only protocol the same skill/sparse methods fell
*below* the no-retrieval LoRA (e.g. BM25 0.733, skill-hybrid 0.738) — the whole
point of retraining on retrieved evidence.

---

## 5. What the model actually sees (input / output / ground truth)

This is exactly the train and test flow. **Training** builds a `messages` pair;
`train/train_qa_skill_lora.py` masks the user turn and computes loss only on the
assistant turn (the gold answer). **Testing** feeds the same-shaped prompt and
compares the generated string to `reference_output` (which equals the corpus gold
for that id).

### 5a. Training example — FDARxBench factual (`train_skill_schema_v1.jsonl`)

`messages[0]` (USER) — the completed skill prompt the LoRA reads (masked in loss):

```
You are answering a drug and molecular QA task using a completed skill schema.
Use only the filled evidence slots that are relevant to the question.
If a required slot is unfilled and the answer cannot be supported, say: Information not found!
Return only the final answer in the requested answer format.

Skill: fda_label_factual
Answer format: short_label_grounded_answer
Abstention policy: abstain_if_label_missing
Retrieval method: skill_hybrid
Task: factual
Input type: FDA label context
Drug: Rabeprazole Sodium

Label/context evidence:
5.9 Hypomagnesemia and Mineral Metabolism: Hypomagnesemia, symptomatic and
asymptomatic, has been reported rarely in patients treated with PPIs ...
Serious adverse events include tetany, arrhythmias, and seizures. ...
```

`messages[1]` (ASSISTANT) — the **only** tokens the loss is computed on (gold target):

```
Serious adverse events include tetany, arrhythmias, and seizures.
```

### 5b. Test example — FDARxBench multihop (prediction file, Qwen2.5-7B skill-schema RAG-SFT)

`prompt` (model input):

```
Skill: fda_label_multihop
Answer format: synthesized_label_answer
Abstention policy: abstain_if_any_required_label_missing
Task: multihop   Input type: FDA label context
Drug: Galantamine Hydrobromide

Label/context evidence:
5.2 Anesthesia: Galantamine, as a cholinesterase inhibitor, is likely to
exaggerate the neuromuscular blocking effects of succinylcholine-type ...
5.6 Neurological Conditions: Seizures: Cholinesterase inhibitors are believed
to have some potential to cause generalized convulsions ...
```

`prediction` (model output):

```
Galantamine can exaggerate the neuromuscular blocking effects of anesthesia, and
it may also exacerbate symptoms ... particularly in patients with Alzheimer's
disease who are at risk for seizures.
```

`reference_output` (ground truth = corpus gold for this id):

```
Galantamine may exaggerate the neuromuscular blocking effects of
succinylcholine-type agents, and patients with neurological conditions,
particularly Alzheimer's disease, are at risk for seizures which could
complicate anesthesia management.
```

### 5c. Test example — Mol-Instructions property prediction (numeric)

`prompt` (model input): skill `molecule_property_numeric`, no external free text,
input is a SELFIES string:

```
Skill: molecule_property_numeric
Answer format: numeric_or_short_property_value
Abstention policy: no_external_free_text_by_default
Task: property prediction   Input type: SELFIES

Input:
[C][N][C][C][Branch1][C][O][C][O][C][C][Ring1][Branch2][Ring1][=Branch1]
Required evidence slots:
- molecule_structure: filled
```

`prediction`: `0.2873`  |  `reference_output` (gold): `0.2934`
(scored by numeric accuracy within tolerance; token F1 also applies.)

### Fairness of the eval (verified)

For every method family and both datasets, the test files carry the same 1000
ids with identical `reference_output` per id, and each prediction file's
`reference_output` equals the corpus gold (1000/1000). Same N per subtask
(FDA=1000; each Mol subtask=250), same decoding (`max_new_tokens=160`), same
`eval/evaluate_instruction_outputs.py`. The only thing that varies across systems
is the retrieved evidence in the prompt and the LoRA trained on it.

---

## 6. How the knowledge base is built

`kb/build_unified_kb.py` builds a **compact, benchmark-scoped** KB slice (not the
full upstream graphs) and writes plain JSONL so retrieval runs with no database
service. It fuses four sources and keys everything to an alias index.

**Inputs**

| Source | File | What it contributes |
|---|---|---|
| PrimeKG | `knowledge_bank/primekg.csv` (≈982 MB raw edge table) | drug/disease/protein graph relations |
| DrugChat ChEMBL | `DrugChat/data/ChEMBL_Drug_Instructions/ChEMBL_Drug_Instructions.json` | compound instruction facts |
| DrugChat PubChem | `DrugChat/data/PubChem_Drug_Instructions/PubChem_Drug_Instructions.json` | compound instruction facts |
| FDARxBench | `FDARxBench/data/qa/qa.jsonl` | FDA label snippets |
| Benchmark records | `outputs/qa_skill_data_corrected/{mol_instructions,fdarxbench}/{train,test}.jsonl` | define the in-scope entity/mention set (candidates only) |

**Steps**

1. **Candidate set.** Scan the benchmark train/test records and extract 1–5-word
   phrases (length ≥ 4) plus normalized `drug_name`s from the question, context,
   task type, and drug-name fields. This is the whitelist of entities the KB is
   allowed to be *about* — it keeps the slice benchmark-scoped.
2. **PrimeKG facts.** Stream `primekg.csv`; keep a row only if its `relation` is in
   the useful set (`indication`, `contraindication`, `off-label use`,
   `drug_protein`, `disease_protein`, `drug_effect`, `drug_drug`,
   `phenotype_protein`, `disease_phenotype_{positive,negative}`,
   `exposure_disease`, `exposure_protein`, `pathway_protein`) **and** its `x_name`
   or `y_name` matches a candidate mention. Cap at `--max-primekg-edges-per-entity`
   (default 80) so hub entities don't dominate. Each kept edge → a fact with title
   `X --relation--> Y`, a text serialization, `aliases=[x_name, y_name]`, typed
   entities, and the relation label.
3. **DrugChat ChEMBL / PubChem facts.** Read up to `--max-source-facts` (default
   50,000) instruction rows each; every row → a fact whose text is
   `Question / Input / Answer`, with aliases including the decoded SELFIES, the raw
   SMILES, and any CID / `chembl_id`; relation `instruction_fact`.
4. **FDARxBench label facts.** Read up to 50,000 `qa.jsonl` rows; take up to 4
   context section snippets (`section_title: text`) per drug; relation
   `label_context`, alias = `drug_name`, source `fdarxbench_label`.
5. **Finalize.** Every fact gets a stable SHA-1 id and a sorted token set (used by
   BM25). Alias→fact-id maps from all sources are merged into one index. Outputs
   land in `outputs/knowledge_bank/unified/`: `facts.jsonl`, `alias_index.json`,
   `summary.json`.

**Result:** 160,083 facts (83,075 PrimeKG, 50,000 ChEMBL, 13,735 PubChem, 13,273
FDA label) and 19,953 aliases.

```bash
cd /playpen-jfs/jesse/drug_microbiome
python skill_aware_rag/kb/build_unified_kb.py \
  --primekg knowledge_bank/primekg.csv \
  --output-dir outputs/knowledge_bank/unified
# then build the dense retrieval indices (BGE + MedCPT) over the KB facts:
python skill_aware_rag/kb/build_kb_embeddings.py   # -> outputs/knowledge_bank/embeddings/
```

---

## 7. Data & external dependencies (not committed to this repo)

This repo is **code only**. The following live outside it (under the parent
workspace `/playpen-jfs/jesse/drug_microbiome/`) and are `.gitignore`d / external,
because they are large or third-party:

- `outputs/` — all built data, LoRA adapters (~69 GB), predictions, eval JSON,
  and `results_summary.{csv,md}`. Regenerate with the SLURM matrices in `slurm/`.
- `knowledge_bank/primekg.csv` — raw PrimeKG source (≈982 MB).
- `FDARxBench/`, `Mol-Instructions/`, `DrugChat/` — third-party benchmark repos
  used as raw inputs to dataset/KB construction.

Env: `/playpen-shared/jesse/miniconda3/envs/agentskill/bin/python` (transformers,
peft, datasets, faiss, numpy, selfies). Models/encoders cached under
`/playpen-shared/jesse/cache`. Not used by this project: `DrugBench/`,
`DrugEHRQA/`.
