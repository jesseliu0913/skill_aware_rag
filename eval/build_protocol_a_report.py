#!/usr/bin/env python
"""Build the complete Protocol-A Markdown report from evaluation artifacts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from evaluate_instruction_outputs import score_record


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs" / "baseline_rag"
REPORT = ROOT / "skill_aware_rag" / "results" / "baseline.md"

MODELS = ["qwen2_5_7b", "qwen2_5_3b", "llama3_2_3b"]
MODEL_LABELS = {
    "qwen2_5_7b": "Qwen2.5-7B",
    "qwen2_5_3b": "Qwen2.5-3B",
    "llama3_2_3b": "Llama-3.2-3B",
}
METHODS = [
    "none",
    "bm25",
    "kg",
    "hybrid",
    "dense_bge",
    "dense_medcpt",
    "rrf",
    "rerank",
    "skill_hybrid",
    "skill_schema_v1",
]
TASKS = ["factual", "multihop", "refusal"]
MOL_TASKS = [
    "molinst_description_guided_design",
    "molinst_molecular_description",
    "molinst_open_qa",
    "molinst_property",
]
MOL_LABELS = {
    "molinst_description_guided_design": "Design",
    "molinst_molecular_description": "Description",
    "molinst_open_qa": "Open QA",
    "molinst_property": "Property",
}


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def pair(f1: float | None, exact: float | None) -> str:
    return f"{fmt(f1)} / {fmt(exact)}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_fdarx() -> tuple[dict, dict[str, int]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    task_counts: dict[str, int] = {}
    for model in MODELS:
        for method in METHODS:
            path = (
                OUTPUTS
                / "predictions_nolabel"
                / f"{model}_fdarxbench_{method}_nolabel_test.jsonl"
            )
            with path.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            counts: dict[str, int] = defaultdict(int)
            for row in rows:
                task = row["task_type"]
                counts[task] += 1
                grouped[(model, method, task)].append(score_record(row, 1e-3, 0.05))
                grouped[(model, method, "overall")].append(score_record(row, 1e-3, 0.05))
            if not task_counts:
                task_counts = dict(counts)

    metrics = {}
    for key, scores in grouped.items():
        metrics[key] = {
            "f1": sum(row["token_f1"] for row in scores) / len(scores),
            "exact": sum(row["exact"] for row in scores) / len(scores),
            "n": len(scores),
        }
    return metrics, task_counts


def load_mol() -> dict[tuple[str, str, str], dict[str, float | None]]:
    retrieval_rows = read_csv(OUTPUTS / "results_summary_bslret.csv")
    closed_rows = read_csv(OUTPUTS / "results_summary_bsl.csv")
    selected = [
        row
        for row in retrieval_rows
        if row["dataset"] == "mol_instructions" and row["method"] != "none"
    ]
    selected.extend(
        row
        for row in closed_rows
        if row["dataset"] == "mol_instructions" and row["method"] == "none"
    )

    metrics = {}
    for row in selected:
        metrics[(row["model"], row["method"], row["subtask"])] = {
            "f1": float(row["token_f1"]),
            "exact": float(row["exact"]),
            "numeric": float(row["numeric_acc"]) if row["numeric_acc"] else None,
            "n": int(row["n"]),
        }
    return metrics


def assert_complete(fdarx: dict, mol: dict) -> None:
    missing = []
    for model in MODELS:
        for method in METHODS:
            for task in [*TASKS, "overall"]:
                if (model, method, task) not in fdarx:
                    missing.append(f"FDARx:{model}:{method}:{task}")
            for task in MOL_TASKS:
                if (model, method, task) not in mol:
                    missing.append(f"Mol:{model}:{method}:{task}")
    if missing:
        raise RuntimeError("Missing Protocol-A results:\n" + "\n".join(missing))


def render_fdarx(fdarx: dict, counts: dict[str, int]) -> list[str]:
    lines = [
        "## FDARxBench",
        "",
        "The FDA `label_context` is removed at train and test time. Retrieval",
        "methods retrieve from the shared KB using the question; `none` is",
        "question-only closed book. Each cell reports token F1 / exact match.",
        "",
        "### Overall (n=1000)",
        "",
        "| Method | Qwen2.5-7B | Qwen2.5-3B | Llama-3.2-3B |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        values = []
        for model in MODELS:
            row = fdarx[(model, method, "overall")]
            values.append(pair(row["f1"], row["exact"]))
        lines.append(f"| `{method}` | " + " | ".join(values) + " |")

    for model in MODELS:
        lines.extend(
            [
                "",
                f"### By question type: {MODEL_LABELS[model]}",
                "",
                (
                    f"| Method | Factual (n={counts['factual']}) | "
                    f"Multihop (n={counts['multihop']}) | "
                    f"Refusal (n={counts['refusal']}) | Overall |"
                ),
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            values = []
            for task in [*TASKS, "overall"]:
                row = fdarx[(model, method, task)]
                values.append(pair(row["f1"], row["exact"]))
            lines.append(f"| `{method}` | " + " | ".join(values) + " |")
    return lines


def render_mol(mol: dict) -> list[str]:
    lines = [
        "",
        "## Mol-Instructions",
        "",
        "Mol-Instructions has no supplied answer context, so Protocol A is the",
        "existing retrieval-at-test condition. Retrieval methods use Protocol-C",
        "evaluations; `none` is the identical closed-book Protocol-B evaluation.",
        "Each subtask has 250 examples.",
        "",
        "Macro values average the four equally sized subtasks. Property numeric",
        "accuracy is tolerance based. Token F1/exact for molecular design are",
        "listed for completeness but are not chemically valid design metrics.",
        "",
        "### Overall macro F1 / property numeric accuracy (n=1000)",
        "",
        "| Method | Qwen2.5-7B | Qwen2.5-3B | Llama-3.2-3B |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        values = []
        for model in MODELS:
            rows = [mol[(model, method, task)] for task in MOL_TASKS]
            macro = sum(row["f1"] for row in rows) / len(rows)
            numeric = mol[(model, method, "molinst_property")]["numeric"]
            values.append(f"{fmt(macro)} / {fmt(numeric)}")
        lines.append(f"| `{method}` | " + " | ".join(values) + " |")

    for model in MODELS:
        lines.extend(
            [
                "",
                f"### By subtask: {MODEL_LABELS[model]}",
                "",
                "| Method | Design F1/exact | Description F1/exact | Open QA F1/exact | Property F1/exact | Property numeric | Macro F1 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            rows = [mol[(model, method, task)] for task in MOL_TASKS]
            values = [pair(row["f1"], row["exact"]) for row in rows]
            numeric = rows[-1]["numeric"]
            macro = sum(row["f1"] for row in rows) / len(rows)
            lines.append(
                f"| `{method}` | "
                + " | ".join(values)
                + f" | {fmt(numeric)} | {fmt(macro)} |"
            )
    return lines


def main() -> None:
    fdarx, counts = load_fdarx()
    mol = load_mol()
    assert_complete(fdarx, mol)

    lines = [
        "# Protocol A: Complete Results",
        "",
        "Protocol A evaluates retrieval when no answer-bearing context is given.",
        "This report contains all 3 models x 2 datasets x 10 baseline methods.",
        "",
        "| Dataset | Models | Methods | Evaluations | Status |",
        "|---|---:|---:|---:|---|",
        "| FDARxBench | 3 | 10 | 30 separate no-label runs | Complete |",
        "| Mol-Instructions | 3 | 10 | 27 retrieval runs + 3 inherited `none` runs | Complete |",
        "",
        "Metrics are computed over the fixed 1,000-example test split. Values are",
        "rounded to three decimals; source CSVs retain full precision.",
        "",
    ]
    lines.extend(render_fdarx(fdarx, counts))
    lines.extend(render_mol(mol))
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. Dense BGE is the strongest overall FDARxBench retriever for all three models.",
            "2. Retrieval improves FDARxBench token F1 by 0.184--0.189 over closed book.",
            "3. FDARxBench refusal is near saturation; factual and multihop drive the gain.",
            "4. Mol-Instructions has no universal best retriever across models and metrics.",
            "5. Existing skill variants do not beat Dense BGE on clean FDARxBench Protocol A.",
            "",
            "## Provenance",
            "",
            "- FDARx predictions: `outputs/baseline_rag/predictions_nolabel/`",
            "- FDARx aggregate: `outputs/baseline_rag/results_summary_nolabel.csv`",
            "- Mol retrieval predictions: `outputs/baseline_rag/predictions_bslret/`",
            "- Mol retrieval aggregate: `outputs/baseline_rag/results_summary_bslret.csv`",
            "- Mol closed-book predictions: `outputs/baseline_rag/predictions_bsl/`",
            "- Mol closed-book aggregate: `outputs/baseline_rag/results_summary_bsl.csv`",
            "",
            "Regenerate this report from the project root:",
            "",
            "```bash",
            "python skill_aware_rag/eval/build_protocol_a_report.py",
            "```",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(lines) + "\n"
    temporary_report = REPORT.with_suffix(".md.tmp")
    temporary_report.write_text(report_text, encoding="utf-8")
    temporary_report.replace(REPORT)
    print(f"Wrote {REPORT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
