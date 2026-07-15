#!/usr/bin/env python
"""Preliminary evidence for question-type-conditioned retrieval.

The script reuses completed RAG-SFT prediction files.  A deterministic split of
example IDs is used to select retrieval policies on one half and evaluate them
on the other half, keeping this diagnostic separate from the final test claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PREDICTION_RE = re.compile(
    r"(?P<model>qwen2_5_7b|qwen2_5_3b|llama3_2_3b)_"
    r"(?P<dataset>fdarxbench|mol_instructions)_"
    r"(?P<method>bm25|kg|hybrid|dense_bge|dense_medcpt|rrf|rerank)_bslret_test\.jsonl$"
)
GENERIC_METHODS = ("bm25", "kg", "hybrid", "dense_bge", "dense_medcpt", "rrf", "rerank")
KB_SOURCES = ("primekg", "drugchat_pubchem", "drugchat_chembl", "fdarxbench_label")
NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "may", "of", "on", "or", "that", "the", "their", "this", "to",
    "was", "were", "which", "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--augmented-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower().replace("\n", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(prediction: Any, reference: Any) -> float:
    pred = normalize_text(prediction).split()
    ref = normalize_text(reference).split()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def numeric_match(prediction: Any, reference: Any) -> float | None:
    ref_text = str(reference or "").strip()
    if not NUMBER_RE.fullmatch(ref_text):
        return None
    pred_match = NUMBER_RE.search(str(prediction or ""))
    if not pred_match:
        return 0.0
    pred_value = float(pred_match.group(0))
    ref_value = float(ref_text)
    tolerance = max(1e-3, abs(ref_value) * 0.05)
    return float(math.isclose(pred_value, ref_value, abs_tol=tolerance, rel_tol=0.0))


def record_utility(record: dict[str, Any]) -> float:
    numeric = numeric_match(record.get("prediction"), record.get("reference_output"))
    if numeric is not None:
        return numeric
    return token_f1(record.get("prediction"), record.get("reference_output"))


def question_type(record: dict[str, Any]) -> str:
    """Return analysis strata from public benchmark fields, never gold answers."""
    dataset = str(record.get("dataset") or record.get("source") or "")
    task = normalize_text(record.get("task_type"))
    if dataset == "fdarxbench" or str(record.get("source")) == "fdarxbench":
        return {"factual": "fda_factual", "multihop": "fda_multihop", "refusal": "fda_refusal"}.get(
            task, "fda_other"
        )
    if "description guided" in task or "design" in task:
        return "molecule_design"
    if "molecular description" in task:
        return "molecule_description"
    if "property" in task:
        return "molecule_property"
    return "biomedical_open_qa"


def discovery_partition(record_id: Any) -> str:
    digest = hashlib.sha256(str(record_id).encode("utf-8")).digest()
    return "discovery" if digest[0] < 128 else "evaluation"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_prediction_scores(predictions_dir: Path) -> list[dict[str, Any]]:
    scores = []
    for path in sorted(predictions_dir.glob("*_bslret_test.jsonl")):
        match = PREDICTION_RE.match(path.name)
        if not match:
            continue
        metadata = match.groupdict()
        for record in read_jsonl(path):
            scores.append(
                {
                    **metadata,
                    "id": str(record.get("id")),
                    "question_type": question_type(record),
                    "partition": discovery_partition(record.get("id")),
                    "utility": record_utility(record),
                }
            )
    return scores


def aggregate_performance(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in scores:
        key = (row["model"], row["dataset"], row["question_type"], row["method"], row["partition"])
        grouped[key].append(float(row["utility"]))
    result = []
    for key, values in sorted(grouped.items()):
        result.append(
            dict(
                zip(
                    ("model", "dataset", "question_type", "method", "partition"),
                    key,
                    strict=True,
                ),
                n=len(values),
                utility=round(mean(values), 6),
            )
        )
    return result


def evaluate_selected_policies(scores: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_run: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        by_run[(row["model"], row["dataset"])].append(row)

    selections = []
    comparisons = []
    for (model, dataset), rows in sorted(by_run.items()):
        discovery = [row for row in rows if row["partition"] == "discovery"]
        evaluation = [row for row in rows if row["partition"] == "evaluation"]
        method_scores = {
            method: mean(row["utility"] for row in discovery if row["method"] == method)
            for method in GENERIC_METHODS
        }
        fixed_method = max(method_scores, key=lambda method: (method_scores[method], method))
        skill_methods: dict[str, str] = {}
        for skill in sorted({row["question_type"] for row in discovery}):
            candidates = {
                method: mean(
                    row["utility"]
                    for row in discovery
                    if row["question_type"] == skill and row["method"] == method
                )
                for method in GENERIC_METHODS
            }
            chosen = max(candidates, key=lambda method: (candidates[method], method))
            skill_methods[skill] = chosen
            selections.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "question_type": skill,
                    "selected_method": chosen,
                    "discovery_utility": round(candidates[chosen], 6),
                    "fixed_method": fixed_method,
                }
            )

        fixed_values = [row["utility"] for row in evaluation if row["method"] == fixed_method]
        personalized_values = [
            row["utility"]
            for row in evaluation
            if row["method"] == skill_methods.get(row["question_type"])
        ]
        fixed_utility = mean(fixed_values)
        personalized_utility = mean(personalized_values)
        comparisons.append(
            {
                "model": model,
                "dataset": dataset,
                "evaluation_n": len(fixed_values),
                "fixed_method": fixed_method,
                "fixed_utility": round(fixed_utility, 6),
                "personalized_utility": round(personalized_utility, 6),
                "absolute_gain": round(personalized_utility - fixed_utility, 6),
            }
        )
    return selections, comparisons


def answer_token_recall(evidence_text: str, answer: Any) -> float:
    evidence_tokens = Counter(
        token for token in normalize_text(evidence_text).split() if len(token) > 2 and token not in STOPWORDS
    )
    answer_tokens = Counter(
        token for token in normalize_text(answer).split() if len(token) > 2 and token not in STOPWORDS
    )
    if not answer_tokens:
        return 0.0
    return sum((evidence_tokens & answer_tokens).values()) / sum(answer_tokens.values())


def analyze_evidence(
    augmented_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inventory: dict[tuple[str, str], dict[str, Any]] = {}
    source_groups: dict[tuple[str, str, str, str], list[tuple[int, int]]] = defaultdict(list)
    relation_groups: dict[tuple[str, str, str, str], list[tuple[int, int]]] = defaultdict(list)
    source_support: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for dataset_dir in sorted(path for path in augmented_dir.iterdir() if path.is_dir()):
        dataset = dataset_dir.name
        for path in sorted(dataset_dir.glob("test_*.jsonl")):
            method = path.stem.removeprefix("test_")
            if method not in {"bm25", "kg", "hybrid", "skill_hybrid", "skill_schema_v1"}:
                continue
            for record in read_jsonl(path):
                skill = question_type({**record, "dataset": dataset})
                key = (dataset, skill)
                inventory.setdefault(
                    key,
                    {
                        "dataset": dataset,
                        "question_type": skill,
                        "task_type": str(record.get("task_type", "")),
                        "input_type": str(record.get("input_type", "")),
                    },
                )
                evidence = record.get("retrieved_kb_evidence") or []
                source_counts = Counter(str(item.get("source") or "unknown") for item in evidence)
                relation_counts = Counter(str(item.get("relation") or "unknown") for item in evidence)
                for source, count in source_counts.items():
                    source_groups[(dataset, skill, method, source)].append((count, len(evidence)))
                for source in KB_SOURCES:
                    source_text = " ".join(
                        str(item.get("text") or "")
                        for item in evidence
                        if str(item.get("source") or "unknown") == source
                    )
                    source_support[(dataset, skill, method, source)].append(
                        answer_token_recall(source_text, record.get("gold_answer"))
                    )
                for relation, count in relation_counts.items():
                    relation_groups[(dataset, skill, method, relation)].append((count, len(evidence)))

    source_rows = _summarize_evidence_groups(source_groups, "source")
    relation_rows = _summarize_evidence_groups(relation_groups, "relation")
    support_rows = []
    for (dataset, skill, method, source), values in sorted(source_support.items()):
        support_rows.append(
            {
                "dataset": dataset,
                "question_type": skill,
                "method": method,
                "source": source,
                "n": len(values),
                "mean_gold_token_recall": round(mean(values), 6),
            }
        )
    return (
        list(sorted(inventory.values(), key=lambda row: (row["dataset"], row["question_type"]))),
        source_rows,
        relation_rows,
        support_rows,
    )


def _summarize_evidence_groups(
    groups: dict[tuple[str, str, str, str], list[tuple[int, int]]], label: str
) -> list[dict[str, Any]]:
    rows = []
    for (dataset, skill, method, value), counts in sorted(groups.items()):
        rows.append(
            {
                "dataset": dataset,
                "question_type": skill,
                "method": method,
                label: value,
                "examples_with_evidence": len(counts),
                "avg_facts": round(mean(count for count, _ in counts), 4),
                "mean_within_prompt_share": round(mean(count / total for count, total in counts if total), 4),
            }
        )
    return rows


def write_report(
    path: Path,
    selections: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    performance: list[dict[str, Any]],
    source_support: list[dict[str, Any]],
) -> None:
    lines = [
        "# Preliminary Study: Question-Type-Conditioned Retrieval",
        "",
        "This diagnostic uses only completed generic RAG-SFT runs. Example IDs are",
        "deterministically split into discovery and evaluation halves. Retrieval",
        "policies are selected on discovery and scored on evaluation.",
        "",
        "## Held-Out Policy Comparison",
        "",
        "| Model | Dataset | Fixed policy | Fixed utility | Personalized utility | Gain |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['fixed_method']} | "
            f"{row['fixed_utility']:.3f} | {row['personalized_utility']:.3f} | {row['absolute_gain']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Policies Selected on the Discovery Half",
            "",
            "| Model | Dataset | Question type | Selected policy | Discovery utility |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in selections:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['question_type']} | "
            f"{row['selected_method']} | {row['discovery_utility']:.3f} |"
        )
    hybrid_support = [row for row in source_support if row["method"] == "hybrid"]
    best_sources = []
    for dataset, skill in sorted({(row["dataset"], row["question_type"]) for row in hybrid_support}):
        candidates = [
            row for row in hybrid_support if row["dataset"] == dataset and row["question_type"] == skill
        ]
        if candidates:
            best_sources.append(max(candidates, key=lambda row: row["mean_gold_token_recall"]))
    lines.extend(
        [
            "",
            "## Source Support Profile (Hybrid Retrieval)",
            "",
            "Gold-token recall is an analysis-only proxy for whether retrieved evidence",
            "contains answer information; it is never used by the router.",
            "",
            "| Dataset | Question type | Best observed source | Gold-token recall | N |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in best_sources:
        lines.append(
            f"| {row['dataset']} | {row['question_type']} | {row['source']} | "
            f"{row['mean_gold_token_recall']:.3f} | {row['n']} |"
        )
    gains = [float(row["absolute_gain"]) for row in comparisons]
    max_gain = max(gains, default=0.0)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The selected-policy table is the direct preliminary test of the paper's",
            "motivation: if one retrieval policy were uniformly best, every row within a",
            "model/dataset would select the same method. Variation across question types",
            "supports the existence of retrieval-policy heterogeneity.",
            "",
            f"The largest held-out gain is {max_gain:+.4f}. These completed runs therefore",
            "do not yet show a practically meaningful answer-quality gain from selecting a",
            "different generic retriever per task. The stronger preliminary evidence should",
            "come from controlled source/relation ablations under a fixed retriever and",
            "evidence budget; this analysis identifies which ablations to run.",
            "",
            "This is a preliminary diagnostic, not a final benchmark result. The router is",
            "represented by benchmark task strata here; the deployed method must infer its",
            "question type from inference-time inputs only.",
            "",
            f"Analyzed {len(performance)} task/method/partition aggregates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    predictions_dir = args.predictions_dir or args.repo_root / "outputs/baseline_rag/predictions_bslret"
    augmented_dir = args.augmented_dir or args.repo_root / "outputs/qa_skill_data_skillkb"
    scores = load_prediction_scores(predictions_dir)
    if not scores:
        raise SystemExit(f"No completed generic bslret predictions found under {predictions_dir}")

    performance = aggregate_performance(scores)
    selections, comparisons = evaluate_selected_policies(scores)
    inventory, sources, relations, source_support = analyze_evidence(augmented_dir)
    output_dir = args.output_dir
    write_csv(
        output_dir / "performance_by_question_type.csv",
        performance,
        ["model", "dataset", "question_type", "method", "partition", "n", "utility"],
    )
    write_csv(
        output_dir / "selected_policies.csv",
        selections,
        ["model", "dataset", "question_type", "selected_method", "discovery_utility", "fixed_method"],
    )
    write_csv(
        output_dir / "policy_comparison.csv",
        comparisons,
        ["model", "dataset", "evaluation_n", "fixed_method", "fixed_utility", "personalized_utility", "absolute_gain"],
    )
    write_csv(output_dir / "task_inventory.csv", inventory, ["dataset", "question_type", "task_type", "input_type"])
    write_csv(
        output_dir / "evidence_by_source.csv",
        sources,
        ["dataset", "question_type", "method", "source", "examples_with_evidence", "avg_facts", "mean_within_prompt_share"],
    )
    write_csv(
        output_dir / "evidence_by_relation.csv",
        relations,
        ["dataset", "question_type", "method", "relation", "examples_with_evidence", "avg_facts", "mean_within_prompt_share"],
    )
    write_csv(
        output_dir / "source_answer_support.csv",
        source_support,
        ["dataset", "question_type", "method", "source", "n", "mean_gold_token_recall"],
    )
    write_report(output_dir / "REPORT.md", selections, comparisons, performance, source_support)
    print(f"Wrote preliminary analysis to {output_dir} ({len(scores)} scored predictions)")


if __name__ == "__main__":
    main()
