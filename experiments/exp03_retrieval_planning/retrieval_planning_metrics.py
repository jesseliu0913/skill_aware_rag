#!/usr/bin/env python
"""Exp03 — Retrieval-planning quality diagnostics (RQ2, RQ3, paper §5.3).

Pure POST-HOC analysis over augmented QA JSONL (the records emitted by
`retrieval/augment_with_skill_kb.py` / `augment_baseline_rag.py`, each carrying a
`retrieved_kb_evidence` list). It asks, per (dataset, task_type, method), whether
uniform retrieval pulls evidence from the *wrong* KB sources and whether the skill
schema fixes source selection — without any retraining or GPU.

Metrics per (dataset, task_type, method):
  - source_precision@k     fraction of retrieved facts whose `source` is in the
                           gold-skill's allowed source set.
  - wrong_source_rate      1 - source_precision@k.
  - evidence_recall@k      fraction of normalized gold-answer tokens present in the
                           concatenated retrieved evidence text.
  - source_entropy_bits    Shannon entropy (bits) over the retrieved facts' sources.
  - skill_source_agreement Jaccard(set(retrieved sources), schema allowed sources).
  - evidence_utilization   (optional; only when --predictions-glob is supplied)
                           fraction of normalized prediction tokens present in the
                           retrieved evidence text.

Reuse (imported, never duplicated):
  - `retrieval.augment_with_skill_kb.infer_schema_v1_skill` + `SKILL_SCHEMA_V1`
    define the gold-skill router and the allowed-source sets, so "allowed sources"
    here can never drift from the retriever's own definition.
  - `eval.evaluate_instruction_outputs.normalize_text` is the token normalization
    used for evidence_recall and evidence_utilization, identical to the answer
    metrics (lowercase, strip punctuation, collapse whitespace).

Inputs:
  --augmented-glob   glob of augmented *.jsonl files. `method` is inferred from each
                     filename (see --method-from-filename-regex).
  --predictions-glob (optional) glob of prediction *.jsonl (from
                     eval/run_qa_jsonl_inference.py). Matched to augmented records by
                     (method, id) to compute evidence_utilization.

Outputs (to --output-dir, default = this experiment's results/):
  - retrieval_planning_metrics.csv   one row per (dataset, task_type, method)
  - REPORT.md                        wrong-source rate + skill-source agreement per
                                     method, grouped uniform vs skill.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from retrieval.augment_with_skill_kb import (  # noqa: E402
    SKILL_SCHEMA_V1,
    infer_schema_v1_skill,
)
from eval.evaluate_instruction_outputs import normalize_text  # noqa: E402

# Skill-family methods (per-skill / skill-routed source scope). Everything else is a
# uniform whole-KB retriever. Used only for the REPORT.md grouping.
SKILL_METHODS = {
    "skill_schema_v1",
    "skill_hybrid",
    "bm25_skillrouted",
    "dense_bge_skillrouted",
    "dense_bge_locked",
    "oracle_source",
}

# Default method vocabulary, longest tokens first so e.g. `bm25_skillrouted` wins
# over `bm25` and `dense_bge_skillrouted` over `dense_bge`.
DEFAULT_METHOD_REGEX = (
    r"(?P<method>dense_bge_skillrouted|dense_bge_locked|bm25_skillrouted|"
    r"skill_schema_v1|skill_hybrid|oracle_source|dense_medcpt|dense_bge|"
    r"rerank|rrf|hybrid|bm25|kg|none)"
)

CSV_FIELDS = [
    "dataset",
    "task_type",
    "method",
    "method_family",
    "n_records",
    "source_precision_at_k",
    "wrong_source_rate",
    "evidence_recall_at_k",
    "source_entropy_bits",
    "skill_source_agreement",
    "evidence_utilization",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--augmented-glob",
        required=True,
        help="glob for augmented *.jsonl files (records with retrieved_kb_evidence).",
    )
    p.add_argument(
        "--predictions-glob",
        default="",
        help="optional glob for prediction *.jsonl (enables evidence_utilization).",
    )
    p.add_argument(
        "--method-from-filename-regex",
        default=DEFAULT_METHOD_REGEX,
        help="regex whose (named group 'method' or first group) captures the method "
        "token from a filename. Falls back to the file stem if no match.",
    )
    p.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "results"),
    )
    return p.parse_args()


def method_from_filename(path: str, pattern: re.Pattern[str]) -> str:
    """Infer the retrieval method from a filename via `pattern`."""
    name = Path(path).name
    match = pattern.search(name)
    if match:
        if match.groupdict().get("method"):
            return match.group("method")
        if match.groups():
            return match.group(1)
        return match.group(0)
    return Path(path).stem


def method_family(method: str) -> str:
    return "skill" if method in SKILL_METHODS else "uniform"


def allowed_sources(record: dict[str, Any]) -> set[str]:
    """Gold-skill allowed KB sources for a record (schema-v1 router)."""
    skill = infer_schema_v1_skill(record)
    config = SKILL_SCHEMA_V1.get(skill, SKILL_SCHEMA_V1["biomedical_open_qa"])
    return set(config.get("sources", ()))


def evidence_text(evidence: list[dict[str, Any]]) -> str:
    """Concatenate title + text over the retrieved facts."""
    parts: list[str] = []
    for item in evidence:
        title = item.get("title")
        text = item.get("text")
        if title:
            parts.append(str(title))
        if text:
            parts.append(str(text))
    return " ".join(parts)


def token_set(text: Any) -> set[str]:
    return set(normalize_text(text).split())


def entropy_bits(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def record_metrics(
    record: dict[str, Any],
    prediction: str | None,
) -> dict[str, float | None]:
    """Per-record planning metrics; components missing evidence return None so the
    group mean is taken only over records where the metric is defined."""
    evidence = record.get("retrieved_kb_evidence") or []
    allowed = allowed_sources(record)
    sources = [str(item.get("source")) for item in evidence]

    out: dict[str, float | None] = {
        "source_precision_at_k": None,
        "wrong_source_rate": None,
        "evidence_recall_at_k": None,
        "source_entropy_bits": None,
        "skill_source_agreement": None,
        "evidence_utilization": None,
    }

    if sources:
        on_source = sum(1 for s in sources if s in allowed)
        precision = on_source / len(sources)
        out["source_precision_at_k"] = precision
        out["wrong_source_rate"] = 1.0 - precision
        out["source_entropy_bits"] = entropy_bits(Counter(sources))
        out["skill_source_agreement"] = jaccard(set(sources), allowed)

    ev_tokens = token_set(evidence_text(evidence))
    gold_tokens = token_set(record.get("gold_answer", ""))
    if gold_tokens:
        out["evidence_recall_at_k"] = len(gold_tokens & ev_tokens) / len(gold_tokens)

    if prediction is not None:
        pred_tokens = token_set(prediction)
        if pred_tokens:
            out["evidence_utilization"] = len(pred_tokens & ev_tokens) / len(pred_tokens)

    return out


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_predictions(
    prediction_files: list[str], pattern: re.Pattern[str]
) -> dict[tuple[str, str], str]:
    """Map (method, id) -> prediction string over the prediction files."""
    preds: dict[tuple[str, str], str] = {}
    for path in prediction_files:
        method = method_from_filename(path, pattern)
        for row in load_jsonl(path):
            rid = row.get("id")
            if rid is None:
                continue
            preds[(method, str(rid))] = str(row.get("prediction", ""))
    return preds


METRIC_KEYS = [
    "source_precision_at_k",
    "wrong_source_rate",
    "evidence_recall_at_k",
    "source_entropy_bits",
    "skill_source_agreement",
    "evidence_utilization",
]


def compute(
    augmented_files: list[str],
    prediction_files: list[str],
    pattern: re.Pattern[str],
) -> list[dict[str, Any]]:
    """Return one aggregated row per (dataset, task_type, method)."""
    predictions = load_predictions(prediction_files, pattern) if prediction_files else {}
    # group -> metric -> list of per-record values (Nones dropped)
    accum: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    counts: dict[tuple[str, str, str], int] = defaultdict(int)

    for path in augmented_files:
        method = method_from_filename(path, pattern)
        for record in load_jsonl(path):
            dataset = str(record.get("source", ""))
            task_type = str(record.get("task_type", ""))
            key = (dataset, task_type, method)
            counts[key] += 1
            rid = record.get("id")
            prediction = None
            if predictions and rid is not None:
                prediction = predictions.get((method, str(rid)))
            metrics = record_metrics(record, prediction)
            for mkey, value in metrics.items():
                if value is not None:
                    accum[key][mkey].append(value)

    rows: list[dict[str, Any]] = []
    for key in sorted(counts):
        dataset, task_type, method = key
        row: dict[str, Any] = {
            "dataset": dataset,
            "task_type": task_type,
            "method": method,
            "method_family": method_family(method),
            "n_records": counts[key],
        }
        for mkey in METRIC_KEYS:
            values = accum[key][mkey]
            row[mkey] = round(sum(values) / len(values), 6) if values else ""
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _weighted_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    num = 0.0
    den = 0
    for row in rows:
        val = row.get(field)
        if val == "" or val is None:
            continue
        n = int(row["n_records"])
        num += float(val) * n
        den += n
    return num / den if den else None


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def write_report(rows: list[dict[str, Any]], path: Path) -> None:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    lines: list[str] = []
    lines.append("# Exp03 — Retrieval-planning quality (RQ2, RQ3)\n")
    lines.append(
        "Post-hoc diagnostics over augmented `retrieved_kb_evidence`. "
        "Wrong-source rate and skill-source agreement are record-weighted means "
        "across (dataset, task_type) strata, per method.\n"
    )
    lines.append("## Per-method summary (uniform vs skill)\n")
    lines.append(
        "| method | family | n | wrong_source_rate | skill_source_agreement | "
        "source_precision@k | evidence_recall@k |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for method in sorted(by_method, key=lambda m: (method_family(m), m)):
        mrows = by_method[method]
        n = sum(int(r["n_records"]) for r in mrows)
        lines.append(
            "| {method} | {fam} | {n} | {wsr} | {agr} | {prec} | {rec} |".format(
                method=method,
                fam=method_family(method),
                n=n,
                wsr=_fmt(_weighted_mean(mrows, "wrong_source_rate")),
                agr=_fmt(_weighted_mean(mrows, "skill_source_agreement")),
                prec=_fmt(_weighted_mean(mrows, "source_precision_at_k")),
                rec=_fmt(_weighted_mean(mrows, "evidence_recall_at_k")),
            )
        )
    lines.append("")

    uniform = [r for r in rows if r["method_family"] == "uniform"]
    skill = [r for r in rows if r["method_family"] == "skill"]
    lines.append("## Family aggregate\n")
    lines.append("| family | n | wrong_source_rate | skill_source_agreement |")
    lines.append("|---|---:|---:|---:|")
    for label, group in (("uniform", uniform), ("skill", skill)):
        n = sum(int(r["n_records"]) for r in group)
        lines.append(
            "| {label} | {n} | {wsr} | {agr} |".format(
                label=label,
                n=n,
                wsr=_fmt(_weighted_mean(group, "wrong_source_rate")),
                agr=_fmt(_weighted_mean(group, "skill_source_agreement")),
            )
        )
    lines.append("")
    lines.append(
        "Reading: skill-family methods should show lower wrong-source rate and higher "
        "skill-source agreement than uniform whole-KB retrievers (RQ2/RQ3). See "
        "`retrieval_planning_metrics.csv` for the full (dataset, task_type, method) grid.\n"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    pattern = re.compile(args.method_from_filename_regex)
    augmented_files = sorted(glob.glob(args.augmented_glob))
    if not augmented_files:
        raise SystemExit(f"no augmented files matched: {args.augmented_glob}")
    prediction_files = sorted(glob.glob(args.predictions_glob)) if args.predictions_glob else []

    rows = compute(augmented_files, prediction_files, pattern)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "retrieval_planning_metrics.csv")
    write_report(rows, out_dir / "REPORT.md")
    print(
        f"wrote {out_dir/'retrieval_planning_metrics.csv'} "
        f"({len(rows)} rows over {len(augmented_files)} augmented files, "
        f"{len(prediction_files)} prediction files)"
    )


if __name__ == "__main__":
    main()
