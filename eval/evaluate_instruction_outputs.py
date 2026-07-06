#!/usr/bin/env python
"""Evaluate instruction-output JSONL files with task-aware lightweight metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("--numeric-abs-tol", type=float, default=1e-3)
    parser.add_argument("--numeric-rel-tol", type=float, default=0.05)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def normalize_text(text: Any) -> str:
    text = str(text).strip().lower()
    text = text.replace("\n", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def first_number(text: Any) -> float | None:
    match = NUMBER_RE.search(str(text))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def is_numeric_reference(value: Any) -> bool:
    if isinstance(value, int | float):
        return not isinstance(value, bool)
    text = str(value).strip()
    return bool(NUMBER_RE.fullmatch(text))


def numeric_match(prediction: Any, reference: Any, abs_tol: float, rel_tol: float) -> bool | None:
    if not is_numeric_reference(reference):
        return None
    pred_num = first_number(prediction)
    ref_num = first_number(reference)
    if pred_num is None or ref_num is None:
        return False
    tolerance = max(abs_tol, abs(ref_num) * rel_tol)
    return math.isclose(pred_num, ref_num, abs_tol=tolerance, rel_tol=0.0)


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


def score_record(record: dict[str, Any], abs_tol: float, rel_tol: float) -> dict[str, Any]:
    pred = record.get("prediction", "")
    ref = record.get("reference_output", "")
    num_match = numeric_match(pred, ref, abs_tol, rel_tol)
    return {
        "exact": str(pred).strip() == str(ref).strip(),
        "normalized_exact": normalize_text(pred) == normalize_text(ref),
        "numeric_match": num_match,
        "token_f1": token_f1(pred, ref),
    }


def summarize(records: list[dict[str, Any]], abs_tol: float, rel_tol: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record.get("prediction_file", ""), record.get("dataset", ""))].append(record)

    summaries = []
    for (prediction_file, dataset), rows in sorted(grouped.items()):
        scores = [score_record(row, abs_tol, rel_tol) for row in rows]
        numeric_scores = [s["numeric_match"] for s in scores if s["numeric_match"] is not None]
        summaries.append(
            {
                "prediction_file": prediction_file,
                "dataset": dataset,
                "n": len(rows),
                "exact": sum(s["exact"] for s in scores) / len(scores),
                "normalized_exact": sum(s["normalized_exact"] for s in scores) / len(scores),
                "numeric_n": len(numeric_scores),
                "numeric_accuracy": (
                    sum(bool(s) for s in numeric_scores) / len(numeric_scores)
                    if numeric_scores
                    else None
                ),
                "avg_token_f1": sum(s["token_f1"] for s in scores) / len(scores),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    records = []
    for prediction_path in args.predictions:
        path = Path(prediction_path)
        with path.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                record["prediction_file"] = str(path)
                records.append(record)

    summaries = summarize(records, args.numeric_abs_tol, args.numeric_rel_tol)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)

    for row in summaries:
        numeric = "NA" if row["numeric_accuracy"] is None else f"{row['numeric_accuracy']:.3f}"
        print(
            f"{row['prediction_file']}\t{row['dataset']}\t"
            f"n={row['n']}\texact={row['exact']:.3f}\t"
            f"norm_exact={row['normalized_exact']:.3f}\t"
            f"numeric_acc={numeric} ({row['numeric_n']})\t"
            f"token_f1={row['avg_token_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
