#!/usr/bin/env python
"""Analyze candidate skill schemas and evidence slots for QA/KG prompting.

This script does not train or retrieve new evidence. It audits existing
corrected datasets, skill-hybrid augmented datasets, and prediction files to
produce heatmap-ready CSVs for deciding the next schema iteration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import string
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MODELS = ("qwen2_5_7b", "qwen2_5_3b", "llama3_2_3b")
DATASETS = ("mol_instructions", "fdarxbench")
VARIANTS = ("raw", "lora")
NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class SkillSpec:
    skill: str
    required_slots: tuple[str, ...]
    allowed_sources: tuple[str, ...]
    default_top_k: int
    answer_format: str
    abstention_policy: str


SKILL_SPECS: dict[str, SkillSpec] = {
    "fda_label_factual": SkillSpec(
        skill="fda_label_factual",
        required_slots=("label_context",),
        allowed_sources=("fdarxbench_label",),
        default_top_k=3,
        answer_format="short_label_grounded_answer",
        abstention_policy="abstain_if_label_missing",
    ),
    "fda_label_multihop": SkillSpec(
        skill="fda_label_multihop",
        required_slots=("label_context_primary", "label_context_secondary"),
        allowed_sources=("fdarxbench_label",),
        default_top_k=4,
        answer_format="synthesized_label_answer",
        abstention_policy="abstain_if_any_required_label_missing",
    ),
    "fda_label_refusal": SkillSpec(
        skill="fda_label_refusal",
        required_slots=("label_context_check", "unsupported_flag"),
        allowed_sources=("fdarxbench_label",),
        default_top_k=2,
        answer_format="information_not_found_or_label_answer",
        abstention_policy="prefer_abstention_when_unsupported",
    ),
    "molecule_property_numeric": SkillSpec(
        skill="molecule_property_numeric",
        required_slots=("molecule_structure", "property_definition"),
        allowed_sources=(),
        default_top_k=0,
        answer_format="numeric_or_short_property_value",
        abstention_policy="no_external_free_text_by_default",
    ),
    "molecule_description": SkillSpec(
        skill="molecule_description",
        required_slots=("molecule_structure",),
        allowed_sources=("drugchat_pubchem", "drugchat_chembl"),
        default_top_k=2,
        answer_format="natural_language_description",
        abstention_policy="use_structure_first",
    ),
    "molecule_design": SkillSpec(
        skill="molecule_design",
        required_slots=("design_requirement",),
        allowed_sources=("drugchat_pubchem", "drugchat_chembl"),
        default_top_k=1,
        answer_format="molecule_string",
        abstention_policy="avoid_unrelated_facts",
    ),
    "biomedical_open_qa": SkillSpec(
        skill="biomedical_open_qa",
        required_slots=("question_entities", "biomedical_fact"),
        allowed_sources=("primekg", "drugchat_pubchem", "drugchat_chembl"),
        default_top_k=3,
        answer_format="concise_biomedical_answer",
        abstention_policy="abstain_if_no_relevant_fact",
    ),
    "drug_relation_qa": SkillSpec(
        skill="drug_relation_qa",
        required_slots=("drug_entity", "relation_fact"),
        allowed_sources=("primekg", "fdarxbench_label"),
        default_top_k=3,
        answer_format="short_relation_answer",
        abstention_policy="abstain_if_no_relation_fact",
    ),
    "unsupported_or_low_evidence": SkillSpec(
        skill="unsupported_or_low_evidence",
        required_slots=("evidence_absence_reason",),
        allowed_sources=(),
        default_top_k=0,
        answer_format="information_not_found",
        abstention_policy="always_abstain",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/skill_schema_analysis")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    return parser.parse_args()


def normalize_text(text: Any) -> str:
    text = str(text).strip().lower().replace("\n", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(prediction: Any, reference: Any) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def first_number(text: Any) -> float | None:
    match = NUMBER_RE.search(str(text))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def numeric_match(prediction: Any, reference: Any, abs_tol: float = 1e-3, rel_tol: float = 0.05) -> bool | None:
    ref_num = first_number(reference)
    if ref_num is None or not NUMBER_RE.fullmatch(str(reference).strip()):
        return None
    pred_num = first_number(prediction)
    if pred_num is None:
        return False
    return math.isclose(pred_num, ref_num, abs_tol=max(abs_tol, abs(ref_num) * rel_tol), rel_tol=0.0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def corrected_dataset_path(dataset: str) -> Path:
    return Path(f"outputs/qa_skill_data_corrected/{dataset}/test.jsonl")


def infer_schema_skill(record: dict[str, Any]) -> str:
    source = str(record.get("source") or record.get("dataset") or "")
    task_type = normalize_text(record.get("task_type", ""))
    question = normalize_text(record.get("question", ""))
    if source == "fdarxbench" or record.get("input_type") == "FDA label context":
        if task_type == "multihop":
            return "fda_label_multihop"
        if task_type == "refusal" or record.get("input_type") == "none":
            return "fda_label_refusal"
        return "fda_label_factual"
    if "property" in task_type or "molecular weight" in question or "logp" in question:
        return "molecule_property_numeric"
    if "description guided" in task_type or "design" in task_type or "synthesize" in question:
        return "molecule_design"
    if "molecular description" in task_type or "describe" in question:
        return "molecule_description"
    relation_terms = ("contraindication", "indication", "interact", "target", "protein", "side effect")
    if any(term in question for term in relation_terms):
        return "drug_relation_qa"
    if not question:
        return "unsupported_or_low_evidence"
    return "biomedical_open_qa"


def answer_shape(record: dict[str, Any]) -> str:
    answer = str(record.get("gold_answer") or record.get("reference_output") or "").strip()
    if not answer:
        return "empty"
    if NUMBER_RE.fullmatch(answer):
        return "numeric"
    if answer.startswith("[") and "]" in answer:
        return "molecule_string"
    if answer.lower() in {"information not found!", "information not found"}:
        return "abstention"
    if "\n" in answer or answer.startswith("-"):
        return "list_or_multiline"
    if len(answer.split()) <= 8:
        return "short_text"
    return "long_text"


def preliminary_slots_from_dataset(record: dict[str, Any], skill: str) -> tuple[str, ...]:
    if skill == "fda_label_factual":
        return ("label_context",)
    if skill == "fda_label_multihop":
        return ("label_context_primary", "label_context_secondary")
    if skill == "fda_label_refusal":
        return ("label_context_check", "unsupported_flag")
    if skill == "molecule_property_numeric":
        return ("molecule_structure", "property_definition")
    if skill == "molecule_description":
        return ("molecule_structure",)
    if skill == "molecule_design":
        return ("design_requirement",)
    if skill == "drug_relation_qa":
        return ("question_entities", "relation_fact")
    if skill == "biomedical_open_qa":
        return ("question_entities", "biomedical_fact")
    return ("evidence_absence_reason",)


def dataset_slot_availability(record: dict[str, Any]) -> dict[str, bool]:
    task_type = normalize_text(record.get("task_type", ""))
    question = normalize_text(record.get("question", ""))
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    citations = metadata.get("citations") if isinstance(metadata.get("citations"), list) else []
    return {
        "label_context": record.get("input_type") == "FDA label context" and bool(record.get("input_molecule_or_context")),
        "label_context_primary": record.get("input_type") == "FDA label context" and bool(record.get("input_molecule_or_context")),
        "label_context_secondary": len(citations) >= 2 or task_type == "multihop",
        "label_context_check": record.get("source") == "fdarxbench",
        "unsupported_flag": task_type == "refusal" or record.get("input_type") == "none",
        "molecule_structure": bool(record.get("decoded_smiles") or record.get("input_type") in {"SELFIES", "SMILES"}),
        "property_definition": "property" in task_type or "molecular weight" in question or "logp" in question,
        "design_requirement": bool(record.get("input_molecule_or_context")) and "design" in task_type,
        "question_entities": bool(record.get("question")),
        "biomedical_fact": bool(record.get("retrieved_kg_evidence") or record.get("retrieved_molecule_evidence")),
        "relation_fact": bool(record.get("retrieved_kg_evidence")),
        "evidence_absence_reason": not bool(record.get("retrieved_kg_evidence") or record.get("retrieved_molecule_evidence")),
    }


def analyze_corrected_datasets(datasets: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inventory_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        records = read_jsonl(corrected_dataset_path(dataset))
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        field_counts: Counter[str] = Counter()
        for record in records:
            grouped[
                (
                    str(record.get("source", "")),
                    str(record.get("task_type", "")),
                    str(record.get("input_type", "")),
                )
            ].append(record)
            for key, value in record.items():
                if value not in (None, "", [], {}):
                    field_counts[key] += 1

        for field, count in sorted(field_counts.items()):
            field_rows.append(
                {
                    "dataset": dataset,
                    "field": field,
                    "present": count,
                    "total": len(records),
                    "coverage": round(count / max(len(records), 1), 6),
                }
            )

        for (source, task_type, input_type), rows in sorted(grouped.items()):
            shape_counts = Counter(answer_shape(row) for row in rows)
            skill_counts = Counter(infer_schema_skill(row) for row in rows)
            question_lengths = [len(str(row.get("question", "")).split()) for row in rows]
            answer_lengths = [len(str(row.get("gold_answer", "")).split()) for row in rows]
            inventory_rows.append(
                {
                    "dataset": dataset,
                    "source": source,
                    "task_type": task_type,
                    "input_type": input_type,
                    "n": len(rows),
                    "candidate_skills": "|".join(f"{k}:{v}" for k, v in skill_counts.most_common()),
                    "answer_shapes": "|".join(f"{k}:{v}" for k, v in shape_counts.most_common()),
                    "decoded_smiles_coverage": round(
                        sum(bool(row.get("decoded_smiles")) for row in rows) / len(rows), 6
                    ),
                    "drug_name_coverage": round(sum(bool(row.get("drug_name")) for row in rows) / len(rows), 6),
                    "kg_evidence_coverage": round(
                        sum(bool(row.get("retrieved_kg_evidence")) for row in rows) / len(rows), 6
                    ),
                    "molecule_evidence_coverage": round(
                        sum(bool(row.get("retrieved_molecule_evidence")) for row in rows) / len(rows), 6
                    ),
                    "avg_question_words": round(sum(question_lengths) / len(question_lengths), 2),
                    "avg_answer_words": round(sum(answer_lengths) / len(answer_lengths), 2),
                }
            )

            for skill in sorted(skill_counts):
                skill_rows = [row for row in rows if infer_schema_skill(row) == skill]
                required_slots = preliminary_slots_from_dataset(skill_rows[0], skill)
                for slot in required_slots:
                    available = sum(dataset_slot_availability(row).get(slot, False) for row in skill_rows)
                    slot_rows.append(
                        {
                            "dataset": dataset,
                            "source": source,
                            "task_type": task_type,
                            "input_type": input_type,
                            "candidate_skill": skill,
                            "slot": slot,
                            "available": available,
                            "n": len(skill_rows),
                            "coverage": round(available / len(skill_rows), 6),
                        }
                    )

    return inventory_rows, field_rows, slot_rows


def evidence_sources(record: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in record.get("retrieved_kb_evidence") or []:
        counts[str(item.get("source") or "unknown")] += 1
    for item in record.get("retrieved_kg_evidence") or []:
        counts[str(item.get("source") or "corrected_primekg")] += 1
    for item in record.get("retrieved_molecule_evidence") or []:
        counts[str(item.get("source") or "corrected_molecule")] += 1
    return counts


def evidence_relations(record: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in record.get("retrieved_kb_evidence") or []:
        counts[str(item.get("relation") or "unknown")] += 1
    return counts


def slot_fills(record: dict[str, Any], skill: str) -> dict[str, bool]:
    sources = evidence_sources(record)
    relations = evidence_relations(record)
    has_structure = bool(record.get("decoded_smiles") or record.get("input_type") in {"SELFIES", "SMILES"})
    has_label_input = record.get("input_type") == "FDA label context" and bool(record.get("input_molecule_or_context"))
    has_label_retrieval = sources["fdarxbench_label"] > 0 or relations["label_context"] > 0
    has_relation_fact = sources["primekg"] > 0
    has_drugchat = sources["drugchat_pubchem"] > 0 or sources["drugchat_chembl"] > 0
    has_question = bool(str(record.get("question", "")).strip())
    has_input = bool(str(record.get("input_molecule_or_context", "")).strip())

    return {
        "label_context": has_label_input or has_label_retrieval,
        "label_context_primary": has_label_input or has_label_retrieval,
        "label_context_secondary": has_label_retrieval and len(record.get("retrieved_kb_evidence") or []) >= 2,
        "label_context_check": has_label_input or has_label_retrieval,
        "unsupported_flag": record.get("task_type") == "refusal" or record.get("input_type") == "none",
        "molecule_structure": has_structure,
        "property_definition": "property" in normalize_text(record.get("task_type", "")) or "property" in normalize_text(record.get("question", "")),
        "design_requirement": has_input,
        "similar_example": has_drugchat or bool(record.get("retrieved_molecule_evidence")),
        "question_entities": has_question,
        "biomedical_fact": has_relation_fact or has_drugchat,
        "drug_entity": bool(record.get("drug_name")) or has_question,
        "relation_fact": has_relation_fact,
        "evidence_absence_reason": not any(sources.values()),
    }


def clustering_text(record: dict[str, Any]) -> str:
    availability = dataset_slot_availability(record)
    categorical = [
        f"source={record.get('source', '')}",
        f"task={record.get('task_type', '')}",
        f"input_type={record.get('input_type', '')}",
        f"answer_shape={answer_shape(record)}",
    ]
    binary_features = [f"has_{key}" for key, value in availability.items() if value]
    text_fields = [
        record.get("question", ""),
        record.get("task_type", ""),
        record.get("input_type", ""),
        record.get("input_molecule_or_context", "")[:500],
    ]
    return " ".join(str(part) for part in categorical + binary_features + text_fields if part)


def analyze_dataset_clusters(datasets: list[str], output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.preprocessing import normalize

    cluster_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    plot_dir = output_dir / "plots"

    for dataset in datasets:
        records = read_jsonl(corrected_dataset_path(dataset))
        if not records:
            continue
        skills = [infer_schema_skill(record) for record in records]
        unique_skills = sorted(set(skills))
        n_clusters = len(unique_skills)
        texts = [clustering_text(record) for record in records]
        vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2)
        features = vectorizer.fit_transform(texts)
        features = normalize(features, norm="l2", copy=False)
        svd_components = min(20, max(2, features.shape[1] - 1))
        svd = TruncatedSVD(n_components=svd_components, random_state=13)
        embedded = svd.fit_transform(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=13, n_init=20)
        clusters = kmeans.fit_predict(embedded)
        skill_to_int = {skill: idx for idx, skill in enumerate(unique_skills)}
        skill_ids = [skill_to_int[skill] for skill in skills]
        nmi = normalized_mutual_info_score(skill_ids, clusters)
        ari = adjusted_rand_score(skill_ids, clusters)

        cluster_skill_counts: dict[int, Counter[str]] = defaultdict(Counter)
        for cluster_id, skill in zip(clusters, skills):
            cluster_skill_counts[int(cluster_id)][skill] += 1
        for cluster_id in sorted(cluster_skill_counts):
            counts = cluster_skill_counts[cluster_id]
            total = sum(counts.values())
            top_skill, top_count = counts.most_common(1)[0]
            summary_rows.append(
                {
                    "dataset": dataset,
                    "cluster": cluster_id,
                    "n": total,
                    "top_skill": top_skill,
                    "purity": round(top_count / total, 6),
                    "skill_counts": "|".join(f"{skill}:{count}" for skill, count in counts.most_common()),
                    "nmi": round(nmi, 6),
                    "ari": round(ari, 6),
                    "svd_explained_variance_2d": round(float(sum(svd.explained_variance_ratio_[:2])), 6),
                }
            )

        for idx, (record, skill, cluster_id, xy) in enumerate(zip(records, skills, clusters, embedded[:, :2])):
            cluster_rows.append(
                {
                    "dataset": dataset,
                    "record_index": idx,
                    "id": record.get("id", ""),
                    "source": record.get("source", ""),
                    "task_type": record.get("task_type", ""),
                    "input_type": record.get("input_type", ""),
                    "answer_shape": answer_shape(record),
                    "candidate_skill": skill,
                    "cluster": int(cluster_id),
                    "pc1": round(float(xy[0]), 8),
                    "pc2": round(float(xy[1]), 8),
                }
            )

        plot_cluster_scatter(
            plot_dir / f"{dataset}_pca_by_skill.png",
            embedded[:, :2],
            skills,
            f"{dataset}: dataset-only PCA/SVD by candidate skill",
        )
        plot_cluster_scatter(
            plot_dir / f"{dataset}_pca_by_cluster.png",
            embedded[:, :2],
            [f"cluster_{int(cluster_id)}" for cluster_id in clusters],
            f"{dataset}: dataset-only KMeans clusters",
        )

    return cluster_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def parse_counter_cell(text: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in str(text).split("|"):
        if not item or ":" not in item:
            continue
        key, value = item.rsplit(":", 1)
        try:
            counts[key] = int(value)
        except ValueError:
            continue
    return counts


def plot_heatmap(
    path: Path,
    row_labels: list[str],
    col_labels: list[str],
    matrix: list[list[float]],
    title: str,
    value_format: str = "{:.2f}",
) -> None:
    if not row_labels or not col_labels:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width = max(8.0, 0.85 * len(col_labels) + 3.0)
    height = max(4.0, 0.45 * len(row_labels) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0)
    ax.set_title(title)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    max_value = max((value for row in matrix for value in row), default=0)
    threshold = max_value * 0.55
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            color = "white" if value > threshold and max_value > 0 else "black"
            ax.text(col_idx, row_idx, value_format.format(value), ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cluster_scatter(path: Path, xy: Any, labels: list[str], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique_labels = list(dict.fromkeys(labels))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, label in enumerate(unique_labels):
        xs = [float(point[0]) for point, point_label in zip(xy, labels) if point_label == label]
        ys = [float(point[1]) for point, point_label in zip(xy, labels) if point_label == label]
        ax.scatter(xs, ys, s=18, alpha=0.75, label=label, color=cmap(idx % 10), edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_preliminary_report(
    path: Path,
    inventory_rows: list[dict[str, Any]],
    field_rows: list[dict[str, Any]],
    preliminary_slot_rows: list[dict[str, Any]],
    cluster_summary_rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Dataset-Only Preliminary Skill Analysis\n\n")
        f.write("This report intentionally ignores model prediction results. It derives candidate skills from dataset source, task type, input type, answer shape, and fields available before new retrieval.\n\n")
        f.write("## Task Inventory\n\n")
        f.write("| Dataset | Source | Task | Input | N | Answer shapes | Candidate skills |\n")
        f.write("|---|---|---|---|---:|---|---|\n")
        for row in inventory_rows:
            f.write(
                f"| {row['dataset']} | {row['source']} | {row['task_type']} | {row['input_type']} | "
                f"{row['n']} | {row['answer_shapes']} | {row['candidate_skills']} |\n"
            )
        f.write("\n## Field Coverage\n\n")
        f.write("| Dataset | Field | Coverage |\n")
        f.write("|---|---|---:|\n")
        important = {
            "question",
            "input_molecule_or_context",
            "input_type",
            "decoded_smiles",
            "gold_answer",
            "task_type",
            "drug_name",
            "retrieved_kg_evidence",
            "retrieved_molecule_evidence",
        }
        for row in field_rows:
            if row["field"] in important:
                f.write(f"| {row['dataset']} | {row['field']} | {row['coverage']} |\n")
        f.write("\n## Preliminary Evidence Slots\n\n")
        f.write("| Dataset | Candidate skill | Slot | N | Dataset-only availability |\n")
        f.write("|---|---|---|---:|---:|\n")
        for row in preliminary_slot_rows:
            f.write(
                f"| {row['dataset']} | {row['candidate_skill']} | {row['slot']} | "
                f"{row['n']} | {row['coverage']} |\n"
            )
        f.write("\n## Design Takeaways\n\n")
        f.write("- FDARxBench naturally splits into factual, multihop, and refusal skills because task labels, input availability, and answer shapes differ.\n")
        f.write("- Mol-Instructions naturally splits into molecule design, molecule description, numeric property prediction, and open QA.\n")
        f.write("- Open QA has weak dataset-only evidence coverage, so relation/open biomedical skills need retrieval slots rather than relying only on existing prompt fields.\n")
        f.write("- Numeric property prediction has complete structure/property slots and should not require free-text KB evidence by default.\n")
        f.write("\n## PCA/Clustering Check\n\n")
        f.write("The clustering check uses only dataset fields: task type, input type, answer shape, slot availability, question text, and input/context text. It does not use model predictions.\n\n")
        f.write("| Dataset | Cluster | N | Top skill | Purity | Skill counts | NMI | ARI |\n")
        f.write("|---|---:|---:|---|---:|---|---:|---:|\n")
        for row in cluster_summary_rows:
            f.write(
                f"| {row['dataset']} | {row['cluster']} | {row['n']} | {row['top_skill']} | "
                f"{row['purity']} | {row['skill_counts']} | {row['nmi']} | {row['ari']} |\n"
            )


def write_preliminary_plots(
    output_dir: Path,
    inventory_rows: list[dict[str, Any]],
    field_rows: list[dict[str, Any]],
    preliminary_slot_rows: list[dict[str, Any]],
) -> None:
    plot_dir = output_dir / "plots"
    task_labels = [
        f"{row['dataset']}::{row['task_type']}"
        for row in inventory_rows
    ]
    answer_shapes = sorted({shape for row in inventory_rows for shape in parse_counter_cell(row["answer_shapes"])})
    answer_matrix = []
    for row in inventory_rows:
        counts = parse_counter_cell(row["answer_shapes"])
        total = max(int(row["n"]), 1)
        answer_matrix.append([counts.get(shape, 0) / total for shape in answer_shapes])
    plot_heatmap(
        plot_dir / "dataset_answer_shape_heatmap.png",
        task_labels,
        answer_shapes,
        answer_matrix,
        "Dataset-only answer shape by task",
    )

    slot_labels = sorted({str(row["slot"]) for row in preliminary_slot_rows})
    skill_labels = [
        f"{row['dataset']}::{row['candidate_skill']}"
        for row in preliminary_slot_rows
    ]
    skill_labels = list(dict.fromkeys(skill_labels))
    slot_lookup = {
        (f"{row['dataset']}::{row['candidate_skill']}", row["slot"]): float(row["coverage"])
        for row in preliminary_slot_rows
    }
    slot_matrix = [[slot_lookup.get((skill, slot), 0.0) for slot in slot_labels] for skill in skill_labels]
    plot_heatmap(
        plot_dir / "dataset_slot_coverage_heatmap.png",
        skill_labels,
        slot_labels,
        slot_matrix,
        "Dataset-only preliminary slot coverage",
    )

    important_fields = [
        "question",
        "input_molecule_or_context",
        "input_type",
        "decoded_smiles",
        "gold_answer",
        "task_type",
        "drug_name",
        "retrieved_kg_evidence",
        "retrieved_molecule_evidence",
    ]
    datasets = sorted({str(row["dataset"]) for row in field_rows})
    field_lookup = {(row["dataset"], row["field"]): float(row["coverage"]) for row in field_rows}
    field_matrix = [[field_lookup.get((dataset, field), 0.0) for field in important_fields] for dataset in datasets]
    plot_heatmap(
        plot_dir / "dataset_field_coverage_heatmap.png",
        datasets,
        important_fields,
        field_matrix,
        "Dataset-only field coverage",
    )


def prediction_paths(dataset: str, model: str, variant: str) -> tuple[Path, Path]:
    base = Path(f"outputs/qa_skill_predictions/{model}_{dataset}_{variant}_test.jsonl")
    kb = Path(f"outputs/qa_skill_predictions/{model}_{dataset}_skill_hybrid_{variant}_test.jsonl")
    return base, kb


def analyze_prediction_deltas(
    datasets: list[str],
    models: list[str],
    records_by_dataset: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        reference_records = records_by_dataset[dataset]
        for model in models:
            for variant in VARIANTS:
                base_path, kb_path = prediction_paths(dataset, model, variant)
                base_rows = read_jsonl(base_path)
                kb_rows = read_jsonl(kb_path)
                if not base_rows or not kb_rows:
                    continue
                grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
                for idx, (base_row, kb_row) in enumerate(zip(base_rows, kb_rows)):
                    record = reference_records[idx] if idx < len(reference_records) else base_row
                    skill = infer_schema_skill(record)
                    base_f1 = token_f1(base_row.get("prediction", ""), base_row.get("reference_output", ""))
                    kb_f1 = token_f1(kb_row.get("prediction", ""), kb_row.get("reference_output", ""))
                    base_num = numeric_match(base_row.get("prediction", ""), base_row.get("reference_output", ""))
                    kb_num = numeric_match(kb_row.get("prediction", ""), kb_row.get("reference_output", ""))
                    grouped[skill].append(
                        {
                            "base_f1": base_f1,
                            "kb_f1": kb_f1,
                            "base_num": float(base_num) if base_num is not None else math.nan,
                            "kb_num": float(kb_num) if kb_num is not None else math.nan,
                        }
                    )
                for skill, values in sorted(grouped.items()):
                    base_f1 = sum(v["base_f1"] for v in values) / len(values)
                    kb_f1 = sum(v["kb_f1"] for v in values) / len(values)
                    base_nums = [v["base_num"] for v in values if not math.isnan(v["base_num"])]
                    kb_nums = [v["kb_num"] for v in values if not math.isnan(v["kb_num"])]
                    rows.append(
                        {
                            "dataset": dataset,
                            "skill": skill,
                            "model": model,
                            "variant": variant,
                            "n": len(values),
                            "no_kb_token_f1": round(base_f1, 6),
                            "skill_hybrid_token_f1": round(kb_f1, 6),
                            "delta_token_f1": round(kb_f1 - base_f1, 6),
                            "no_kb_numeric_acc": (
                                round(sum(base_nums) / len(base_nums), 6) if base_nums else ""
                            ),
                            "skill_hybrid_numeric_acc": (
                                round(sum(kb_nums) / len(kb_nums), 6) if kb_nums else ""
                            ),
                        }
                    )
    return rows


def analyze_datasets(datasets: list[str], records_by_dataset: dict[str, list[dict[str, Any]]]) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    skill_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    source_totals: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    relation_totals: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    slot_totals: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    slot_denoms: Counter[tuple[str, str, str]] = Counter()
    prompt_totals: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    skill_counts: Counter[tuple[str, str, str]] = Counter()

    for dataset in datasets:
        for record in records_by_dataset[dataset]:
            skill = infer_schema_skill(record)
            task_type = str(record.get("task_type", ""))
            key = (dataset, skill, task_type)
            spec = SKILL_SPECS[skill]
            skill_counts[key] += 1
            prompt_totals[key].append(len(str(record.get("prompt", ""))))
            for source, count in evidence_sources(record).items():
                source_totals[key][source] += count
            for relation, count in evidence_relations(record).items():
                relation_totals[key][relation] += count
            fills = slot_fills(record, skill)
            slot_denoms[key] += 1
            for slot in spec.required_slots:
                if fills.get(slot, False):
                    slot_totals[key][slot] += 1

    for (dataset, skill, task_type), n in sorted(skill_counts.items()):
        spec = SKILL_SPECS[skill]
        filled_all = 0
        for record in records_by_dataset[dataset]:
            if infer_schema_skill(record) != skill or str(record.get("task_type", "")) != task_type:
                continue
            fills = slot_fills(record, skill)
            if all(fills.get(slot, False) for slot in spec.required_slots):
                filled_all += 1
        skill_rows.append(
            {
                "dataset": dataset,
                "skill": skill,
                "task_type": task_type,
                "n": n,
                "required_slots": "|".join(spec.required_slots),
                "allowed_sources": "|".join(spec.allowed_sources) if spec.allowed_sources else "none",
                "default_top_k": spec.default_top_k,
                "all_required_slots_coverage": round(filled_all / n, 6),
                "answer_format": spec.answer_format,
                "abstention_policy": spec.abstention_policy,
            }
        )
        prompt_values = prompt_totals[(dataset, skill, task_type)]
        prompt_rows.append(
            {
                "dataset": dataset,
                "skill": skill,
                "task_type": task_type,
                "n": len(prompt_values),
                "avg_prompt_chars": round(sum(prompt_values) / len(prompt_values), 2),
                "max_prompt_chars": max(prompt_values),
                "default_top_k": spec.default_top_k,
            }
        )
        for slot in spec.required_slots:
            slot_rows.append(
                {
                    "dataset": dataset,
                    "skill": skill,
                    "task_type": task_type,
                    "slot": slot,
                    "n": n,
                    "filled": slot_totals[(dataset, skill, task_type)][slot],
                    "coverage": round(slot_totals[(dataset, skill, task_type)][slot] / n, 6),
                }
            )
        sources = source_totals[(dataset, skill, task_type)]
        for source, count in sorted(sources.items()):
            source_rows.append(
                {
                    "dataset": dataset,
                    "skill": skill,
                    "task_type": task_type,
                    "source": source,
                    "retrieved_facts": count,
                    "facts_per_example": round(count / n, 6),
                    "allowed_by_schema": source in spec.allowed_sources,
                }
            )
        if not sources:
            source_rows.append(
                {
                    "dataset": dataset,
                    "skill": skill,
                    "task_type": task_type,
                    "source": "none",
                    "retrieved_facts": 0,
                    "facts_per_example": 0,
                    "allowed_by_schema": True,
                }
            )
    return skill_rows, source_rows, slot_rows, prompt_rows


def write_report(
    path: Path,
    inventory_rows: list[dict[str, Any]],
    preliminary_slot_rows: list[dict[str, Any]],
    skill_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> None:
    worst = sorted(delta_rows, key=lambda row: float(row["delta_token_f1"]))[:8]
    best = sorted(delta_rows, key=lambda row: float(row["delta_token_f1"]), reverse=True)[:8]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Skill Schema Analysis\n\n")
        f.write("## Dataset-Only Preliminary Analysis\n\n")
        f.write("| Dataset | Source | Task | Input | N | Candidate skills | Answer shapes |\n")
        f.write("|---|---|---|---|---:|---|---|\n")
        for row in inventory_rows:
            f.write(
                f"| {row['dataset']} | {row['source']} | {row['task_type']} | {row['input_type']} | "
                f"{row['n']} | {row['candidate_skills']} | {row['answer_shapes']} |\n"
            )
        f.write("\n## Preliminary Slot Availability\n\n")
        f.write("| Dataset | Candidate skill | Slot | N | Coverage |\n")
        f.write("|---|---|---|---:|---:|\n")
        for row in preliminary_slot_rows:
            f.write(
                f"| {row['dataset']} | {row['candidate_skill']} | {row['slot']} | "
                f"{row['n']} | {row['coverage']} |\n"
            )
        f.write("\n## Skill Coverage After KB Augmentation\n\n")
        f.write("| Dataset | Skill | Task | N | Required slots | Coverage |\n")
        f.write("|---|---|---|---:|---|---:|\n")
        for row in skill_rows:
            f.write(
                f"| {row['dataset']} | {row['skill']} | {row['task_type']} | {row['n']} | "
                f"{row['required_slots']} | {row['all_required_slots_coverage']} |\n"
            )
        f.write("\n## Largest KB Gains\n\n")
        f.write("| Dataset | Skill | Model | Variant | No-KB F1 | KB F1 | Delta |\n")
        f.write("|---|---|---|---|---:|---:|---:|\n")
        for row in best:
            f.write(
                f"| {row['dataset']} | {row['skill']} | {row['model']} | {row['variant']} | "
                f"{row['no_kb_token_f1']} | {row['skill_hybrid_token_f1']} | {row['delta_token_f1']} |\n"
            )
        f.write("\n## Largest KB Drops\n\n")
        f.write("| Dataset | Skill | Model | Variant | No-KB F1 | KB F1 | Delta |\n")
        f.write("|---|---|---|---|---:|---:|---:|\n")
        for row in worst:
            f.write(
                f"| {row['dataset']} | {row['skill']} | {row['model']} | {row['variant']} | "
                f"{row['no_kb_token_f1']} | {row['skill_hybrid_token_f1']} | {row['delta_token_f1']} |\n"
            )
        f.write("\n## Outputs\n\n")
        f.write("- `skill_schema_v1.json`: executable schema spec.\n")
        f.write("- `dataset_task_inventory.csv`: dataset-only task/input/answer inventory.\n")
        f.write("- `dataset_field_coverage.csv`: dataset-only field coverage.\n")
        f.write("- `dataset_preliminary_slot_coverage.csv`: slot availability before new retrieval.\n")
        f.write("- `skill_model_prompt_f1.csv`: heatmap-ready per-skill F1 deltas.\n")
        f.write("- `skill_evidence_source_heatmap.csv`: evidence source counts by skill.\n")
        f.write("- `slot_coverage.csv`: required evidence slot coverage.\n")
        f.write("- `prompt_budget.csv`: prompt length by skill.\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_dataset = {
        "mol_instructions": read_jsonl(Path("outputs/qa_skill_data_skillkb/mol_instructions/test_skill_hybrid.jsonl")),
        "fdarxbench": read_jsonl(Path("outputs/qa_skill_data_skillkb/fdarxbench/test_skill_hybrid.jsonl")),
    }
    records_by_dataset = {key: value for key, value in records_by_dataset.items() if key in args.datasets}

    schema_path = output_dir / "skill_schema_v1.json"
    with schema_path.open("w", encoding="utf-8") as f:
        json.dump({name: asdict(spec) for name, spec in SKILL_SPECS.items()}, f, indent=2)

    inventory_rows, field_rows, preliminary_slot_rows = analyze_corrected_datasets(args.datasets)
    skill_rows, source_rows, slot_rows, prompt_rows = analyze_datasets(args.datasets, records_by_dataset)
    delta_rows = analyze_prediction_deltas(args.datasets, args.models, records_by_dataset)

    write_csv(
        output_dir / "dataset_task_inventory.csv",
        inventory_rows,
        [
            "dataset",
            "source",
            "task_type",
            "input_type",
            "n",
            "candidate_skills",
            "answer_shapes",
            "decoded_smiles_coverage",
            "drug_name_coverage",
            "kg_evidence_coverage",
            "molecule_evidence_coverage",
            "avg_question_words",
            "avg_answer_words",
        ],
    )
    write_csv(
        output_dir / "dataset_field_coverage.csv",
        field_rows,
        ["dataset", "field", "present", "total", "coverage"],
    )
    write_csv(
        output_dir / "dataset_preliminary_slot_coverage.csv",
        preliminary_slot_rows,
        ["dataset", "source", "task_type", "input_type", "candidate_skill", "slot", "available", "n", "coverage"],
    )
    cluster_rows, cluster_summary_rows = analyze_dataset_clusters(args.datasets, output_dir)
    write_csv(
        output_dir / "dataset_cluster_projection.csv",
        cluster_rows,
        [
            "dataset",
            "record_index",
            "id",
            "source",
            "task_type",
            "input_type",
            "answer_shape",
            "candidate_skill",
            "cluster",
            "pc1",
            "pc2",
        ],
    )
    write_csv(
        output_dir / "dataset_cluster_summary.csv",
        cluster_summary_rows,
        [
            "dataset",
            "cluster",
            "n",
            "top_skill",
            "purity",
            "skill_counts",
            "nmi",
            "ari",
            "svd_explained_variance_2d",
        ],
    )
    write_preliminary_report(
        output_dir / "PRELIMINARY_DATASET_REPORT.md",
        inventory_rows,
        field_rows,
        preliminary_slot_rows,
        cluster_summary_rows,
    )
    write_preliminary_plots(output_dir, inventory_rows, field_rows, preliminary_slot_rows)
    write_csv(
        output_dir / "skill_schema_coverage.csv",
        skill_rows,
        [
            "dataset",
            "skill",
            "task_type",
            "n",
            "required_slots",
            "allowed_sources",
            "default_top_k",
            "all_required_slots_coverage",
            "answer_format",
            "abstention_policy",
        ],
    )
    write_csv(
        output_dir / "skill_evidence_source_heatmap.csv",
        source_rows,
        ["dataset", "skill", "task_type", "source", "retrieved_facts", "facts_per_example", "allowed_by_schema"],
    )
    write_csv(
        output_dir / "slot_coverage.csv",
        slot_rows,
        ["dataset", "skill", "task_type", "slot", "n", "filled", "coverage"],
    )
    write_csv(
        output_dir / "prompt_budget.csv",
        prompt_rows,
        ["dataset", "skill", "task_type", "n", "avg_prompt_chars", "max_prompt_chars", "default_top_k"],
    )
    write_csv(
        output_dir / "skill_model_prompt_f1.csv",
        delta_rows,
        [
            "dataset",
            "skill",
            "model",
            "variant",
            "n",
            "no_kb_token_f1",
            "skill_hybrid_token_f1",
            "delta_token_f1",
            "no_kb_numeric_acc",
            "skill_hybrid_numeric_acc",
        ],
    )
    write_report(output_dir / "REPORT.md", inventory_rows, preliminary_slot_rows, skill_rows, delta_rows, source_rows)

    summary = {
        "output_dir": str(output_dir),
        "skills": len(SKILL_SPECS),
        "datasets": args.datasets,
        "models": args.models,
        "rows": {
            "dataset_task_inventory": len(inventory_rows),
            "dataset_field_coverage": len(field_rows),
            "dataset_preliminary_slot_coverage": len(preliminary_slot_rows),
            "dataset_cluster_projection": len(cluster_rows),
            "dataset_cluster_summary": len(cluster_summary_rows),
            "skill_schema_coverage": len(skill_rows),
            "skill_evidence_source_heatmap": len(source_rows),
            "slot_coverage": len(slot_rows),
            "prompt_budget": len(prompt_rows),
            "skill_model_prompt_f1": len(delta_rows),
        },
        "preliminary_outputs": {
            "report": str(output_dir / "PRELIMINARY_DATASET_REPORT.md"),
            "plots": str(output_dir / "plots"),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
