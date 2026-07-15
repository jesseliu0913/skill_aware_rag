#!/usr/bin/env python
"""Exp06 router report: accuracy + confusion matrix of the deployable
question-only router against the gold-field skill, over a QA JSONL.

The gold-field skill (``gold_field_skill``) is what the current pipeline routes
on (it reads source/task_type/input_type). The deployable router
(``predict_skill_question_only``) reads ONLY the question text. This script
scores the latter against the former -- i.e. how much accuracy a real deployment
(question-only inputs) loses versus the gold-field/oracle route -- and dumps the
full skill x skill confusion matrix.

Pure stdlib, CPU-only. No KB, no model, no retrieval is loaded.

  python experiments/exp06_skill_controls/router_report.py \
    --input outputs/qa_skill_data_corrected/mol_instructions/test.jsonl \
    --schema-version v1 \
    --out-csv experiments/exp06_skill_controls/results/router_confusion.csv \
    --out-report experiments/exp06_skill_controls/results/REPORT.md
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

# Reuse the routers from the shared augmenter so this report can never drift from
# what the pipeline actually does.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "retrieval"))
from augment_with_skill_kb import (  # noqa: E402
    gold_field_skill,
    predict_skill_question_only,
    skill_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="QA JSONL (normalized records).")
    parser.add_argument("--schema-version", default="v1", choices=["legacy", "v1"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-csv", default="experiments/exp06_skill_controls/results/router_confusion.csv")
    parser.add_argument("--out-report", default="experiments/exp06_skill_controls/results/REPORT.md")
    return parser.parse_args()


def iter_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if limit and len(rows) >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_confusion(records: list[dict[str, Any]], schema_version: str) -> tuple[dict[tuple[str, str], int], int, int]:
    """Return (confusion counts keyed (gold, predicted), n_total, n_correct)."""
    confusion: Counter[tuple[str, str]] = Counter()
    n_correct = 0
    for record in records:
        gold = gold_field_skill(record, schema_version)
        pred = predict_skill_question_only(record, schema_version)
        confusion[(gold, pred)] += 1
        if gold == pred:
            n_correct += 1
    return dict(confusion), len(records), n_correct


def write_csv(path: Path, confusion: dict[tuple[str, str], int], names: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold_skill"] + list(names))
        for gold in names:
            row = [gold] + [confusion.get((gold, pred), 0) for pred in names]
            writer.writerow(row)


def write_report(
    path: Path,
    confusion: dict[tuple[str, str], int],
    names: tuple[str, ...],
    n_total: int,
    n_correct: int,
    input_path: str,
    schema_version: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    acc = (n_correct / n_total) if n_total else 0.0
    lines = [
        "# Exp06 Router Report",
        "",
        f"- Input: `{input_path}`",
        f"- Schema version: `{schema_version}`",
        f"- Records: {n_total}",
        f"- Question-only router accuracy vs gold-field skill: **{acc:.4f}** ({n_correct}/{n_total})",
        "",
        "## Per-skill router accuracy",
        "",
        "| gold skill | n | correct | accuracy |",
        "|---|---:|---:|---:|",
    ]
    for gold in names:
        n_gold = sum(confusion.get((gold, pred), 0) for pred in names)
        correct = confusion.get((gold, gold), 0)
        gold_acc = (correct / n_gold) if n_gold else 0.0
        lines.append(f"| {gold} | {n_gold} | {correct} | {gold_acc:.4f} |")
    lines += [
        "",
        "## Confusion matrix (rows = gold-field skill, cols = question-only prediction)",
        "",
        "| gold \\ predicted | " + " | ".join(names) + " |",
        "|---|" + "|".join(["---:"] * len(names)) + "|",
    ]
    for gold in names:
        row = [str(confusion.get((gold, pred), 0)) for pred in names]
        lines.append(f"| {gold} | " + " | ".join(row) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = iter_jsonl(Path(args.input), args.limit)
    names = skill_names(args.schema_version)
    confusion, n_total, n_correct = build_confusion(records, args.schema_version)
    write_csv(Path(args.out_csv), confusion, names)
    write_report(Path(args.out_report), confusion, names, n_total, n_correct, args.input, args.schema_version)
    acc = (n_correct / n_total) if n_total else 0.0
    print(json.dumps({
        "input": args.input,
        "schema_version": args.schema_version,
        "records": n_total,
        "router_accuracy": round(acc, 4),
        "out_csv": args.out_csv,
        "out_report": args.out_report,
    }, indent=2))


if __name__ == "__main__":
    main()
