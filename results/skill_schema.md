# Skill Schema

Method = fixed retriever (**dense_bge**, top-8, unconstrained) + skill schema.
The skill schema is only a **template layer**: `skill` name + `steps` + `answer_format`.
(Retrieval is dense_bge's job, so per-skill sources/relations/top_k/required_slots are dropped.)

## Skill definitions

### `fda_label_factual`
```
answer_format: short_label_grounded_answer
steps:
  1. Locate the label section relevant to the question.
  2. Extract the exact fact asked for (event, dose, contraindication, population, ...).
  3. Answer concisely, grounded only in the label text.
```

### `fda_label_multihop`
```
answer_format: synthesized_label_answer
steps:
  1. Identify the two or more label sections the question connects.
  2. Extract the key fact from each section.
  3. Combine them into one coherent, grounded answer.
```

### `molecule_property_numeric`
```
answer_format: numeric_or_short_property_value
steps:
  1. Read the molecular structure from the SELFIES/SMILES string.
  2. Identify the requested property (e.g., HOMO-LUMO gap, logP).
  3. Use similar retrieved molecules as reference, then give the numeric value only.
```

### `molecule_description`
```
answer_format: natural_language_description
steps:
  1. Parse the structure: identify its class, functional groups, or natural-product source.
  2. Use similar retrieved compound facts as reference.
  3. Write a concise natural-language description.
```

### `molecule_design`
```
answer_format: molecule_string
steps:
  1. Read the design requirement (role, class, scaffold, target property).
  2. Recall similar molecules from the retrieved compound facts.
  3. Output a molecule string (SELFIES) that satisfies the requirement.
```

### `biomedical_open_qa`
```
answer_format: concise_biomedical_answer
steps:
  1. Identify the biomedical entities in the question (drug, disease, protein, pathway).
  2. Retrieve the relevant relations/facts for those entities.
  3. Answer concisely from the retrieved facts.
```

## Example prompt (dense_bge + skill schema, NO-LABEL)

```
You are answering a drug and molecular QA task with external knowledge.

Use the provided information to answer the question. If the answer is not available, respond: Information not found!

Answer in the same style as the gold answer.

Skill: fda_label_factual
How to solve:
  1. Locate the label section relevant to the question.
  2. Extract the exact fact asked for (event, dose, contraindication, population, ...).
  3. Answer concisely, grounded only in the label text.
Answer format: short_label_grounded_answer

Skill-based retrieved knowledge:
1. [fdarxbench_label:label_context; score=0.8851] HOW SUPPLIED SECTION: 8.5 Geriatric Use In a controlled clinical study for the reduction in the combined risk of cardiovascular death, stroke and myocardial infarction in hypertensive patients with left ventricular hypertrophy, 2857 patients (62%) were 65 years and over, while 808 patients (18%) were 75 years and over. In an effort to control blood pressure in this study, patients were coadministered losartan and hydrochlorothiazide 74% of the total time they were on study drug. No overall differences in effectiveness were observed between these patients and younger patients. Adverse events were somewhat more frequent in the elderly compared to non-elderly patients for both the losartan-hydrochlorothiazide and the control groups .
2. [fdarxbench_label:label_context; score=0.7697] Geriatric Use: Clinical studies of amiloride HCl did not include sufficient numbers of subjects aged 65 and over to determine whether they respond differently from younger subjects. Other reported clinical experience has not identified differences in responses between the elderly and younger patients. In general, dose selection for an elderly patient should be cautious, usually starting at the low end of the dosing range, reflecting the greater frequency of decreased hepatic, renal or cardiac function, and of concomitant disease or other drug therapy. This drug is known to be substantially excreted by the kidney, and the risk of toxic reactions to this drug may be greater in patients with impaired renal function. Because elderly patients are more likely to have decreased renal function, care should be taken in dose selection, and it may be useful to monitor renal function.  Hyperkalemia: Amiloride HCl should not be used in the presence of elevated serum potassium levels (greater than 5.5 mEq per liter).  Like other potassium-conserving agents, amiloride may cause hyperkalemia (serum potassium levels greater than 5.5 mEq per liter) which, if uncorrected, is potentially fatal. Hyperkalemia occurs commonly (about 10%) when amiloride is used without a kaliuretic diuretic. This incidence is greater in patients with renal impairment, diabetes mellitus (with or without recognized renal insufficiency), and in the elderly. When amiloride HCl is used concomitantly with a thiazide diuretic in patients without these complications, the risk of hyperkalemia is reduced to about 1-2 percent. It is thus essential to monitor serum potassium levels carefully in any patient receiving amiloride, particularly when it is first introduced, at the time of diuretic dosage adjustments, and during any illness that could affect renal function.
3. [fdarxbench_label:label_context; score=0.7691] INDICATIONS & USAGE SECTION: Hypertensive Patients with Left Ventricular Hypertrophy Losartan Potassium and Hydrochlorothiazide Tablets are indicated to reduce the risk of stroke in patients with hypertension and left ventricular hypertrophy, but there is evidence that this benefit does not apply to Black patients.

Question: What percentage of patients in the geriatric use study for losartan and hydrochlorothiazide were 65 years and over?
Answer:
```

gold: '62% of the patients were 65 years and over.'
