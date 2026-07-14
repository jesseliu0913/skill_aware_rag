#!/usr/bin/env python
"""Exp02 main-table collator + ordering-hypothesis checker (RQ4).

Assembles the per-task main comparison table from whatever eval artifacts already
exist on disk -- the per-run `*_eval.json` files (same schema as
`eval/evaluate_instruction_outputs` / `eval/aggregate_results`) and any
`results_summary*.csv` -- folding in the new `oracle_source` runs alongside the
standard method roster. It then EXPLICITLY tests, per task, the ordering
hypothesis behind RQ4:

    uniform-RAG  <=  SkillRAG  <=  oracle_source

reporting, for every (model, dataset, subtask, variant) cell where all three roles
are present, whether the ordering HOLDS or is VIOLATED (and by how much).

Roles are configurable (`--uniform-method`, `--skill-method`, `--oracle-method`);
defaults compare `dense_bge` (uniform dense over the whole KB) against
`dense_bge_skillrouted` (deployed-style skill routing) against `oracle_source`
(gold upper bound). A tolerance (`--tol`, default 0) absorbs measurement noise
(SE ~ 0.012 at n=782) so near-ties are not flagged as violations.

Runs fully offline over globbed files; writes nothing if given nowhere to read.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any, Iterable

REPO = Path("/playpen-jfs/jesse/drug_microbiome")

MODELS = ("qwen2_5_7b", "qwen2_5_3b", "llama3_2_3b")
# Full roster incl. the routed-dense twins and the new oracle_source upper bound.
METHODS = (
    "none", "bm25", "kg", "hybrid", "skill_hybrid", "skill_schema_v1",
    "bm25_skillrouted", "dense_bge", "dense_bge_locked", "dense_bge_skillrouted",
    "dense_medcpt", "rrf", "rerank", "oracle_source",
)
DATASETS = ("mol_instructions", "fdarxbench")
VARIANTS = ("raw", "lora", "raglora", "rawqeval", "bsl", "bslret", "nolabel")

_MODEL_RE = "|".join(MODELS)
_METHOD_RE = "|".join(sorted(METHODS, key=len, reverse=True))  # longest-first so dense_bge_skillrouted wins over dense_bge
_DATASET_RE = "|".join(DATASETS)
_VARIANT_RE = "|".join(VARIANTS)

# {model}_{dataset}_{method}_{variant}_test_eval.json
METHOD_FILE_RE = re.compile(
    rf"^(?P<model>{_MODEL_RE})_(?P<dataset>{_DATASET_RE})_"
    rf"(?P<method>{_METHOD_RE})_(?P<variant>{_VARIANT_RE})_test_eval\.json$"
)
# {model}_{dataset}_{variant}_test_eval.json  (no retrieval method -> "none")
PLAIN_FILE_RE = re.compile(
    rf"^(?P<model>{_MODEL_RE})_(?P<dataset>{_DATASET_RE})_(?P<variant>{_VARIANT_RE})_test_eval\.json$"
)


def parse_eval_filename(name: str) -> dict[str, str] | None:
    """(model, dataset, method, variant) from an eval filename, or None."""
    m = METHOD_FILE_RE.match(name)
    if m:
        return {**m.groupdict()}
    m = PLAIN_FILE_RE.match(name)
    if m:
        g = m.groupdict()
        g["method"] = "none"
        return g
    return None


def headline_metric(row: dict[str, Any]) -> tuple[str, float] | None:
    """Pick the task's headline metric value: numeric_acc when present, else token_f1."""
    for key in ("numeric_acc", "token_f1"):
        val = row.get(key)
        if isinstance(val, (int, float)):
            return key, float(val)
    return None


def _row_from_eval_record(meta: dict[str, str], rec: dict[str, Any], src: str) -> dict[str, Any]:
    return {
        "dataset": meta["dataset"],
        "method": meta["method"],
        "model": meta["model"],
        "variant": meta["variant"],
        "subtask": rec.get("dataset", meta["dataset"]),
        "n": rec.get("n"),
        "token_f1": rec.get("avg_token_f1"),
        "exact": rec.get("exact"),
        "numeric_acc": rec.get("numeric_accuracy"),
        "file": src,
    }


def collect_eval_jsons(pred_dirs: Iterable[str], rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for d in pred_dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in sorted(p.glob("*_eval.json")):
            if f.name in seen:
                continue
            meta = parse_eval_filename(f.name)
            if not meta:
                continue
            try:
                recs = json.loads(f.read_text())
            except Exception:
                continue
            seen.add(f.name)
            for rec in (recs if isinstance(recs, list) else [recs]):
                rows.append(_row_from_eval_record(meta, rec, f.name))


def collect_summary_csvs(csv_globs: Iterable[str], rows: list[dict[str, Any]]) -> None:
    """Fold in pre-aggregated results_summary*.csv (cols from eval/aggregate_results)."""
    for pattern in csv_globs:
        for path in sorted(glob.glob(pattern)):
            try:
                with open(path, newline="") as fh:
                    for r in csv.DictReader(fh):
                        rows.append({
                            "dataset": r.get("dataset"),
                            "method": r.get("method"),
                            "model": r.get("model"),
                            "variant": r.get("variant"),
                            "subtask": r.get("subtask") or r.get("dataset"),
                            "n": _num(r.get("n")),
                            "token_f1": _num(r.get("token_f1")),
                            "exact": _num(r.get("exact")),
                            "numeric_acc": _num(r.get("numeric_acc")),
                            "file": r.get("file") or Path(path).name,
                        })
            except Exception:
                continue


def _num(x: Any) -> Any:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


def check_ordering(
    rows: list[dict[str, Any]],
    uniform: str,
    skill: str,
    oracle: str,
    tol: float = 0.0,
) -> list[dict[str, Any]]:
    """For each (model, dataset, subtask, variant) cell with all three method-roles,
    test uniform <= skill <= oracle (within tol). Returns one result per cell."""
    by_cell: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for r in rows:
        metric = headline_metric(r)
        if metric is None:
            continue
        if r.get("method") not in (uniform, skill, oracle):
            continue
        cell = (str(r.get("model")), str(r.get("dataset")), str(r.get("subtask")), str(r.get("variant")))
        by_cell.setdefault(cell, {})[r["method"]] = metric[1]

    results = []
    for cell, vals in sorted(by_cell.items()):
        if not all(m in vals for m in (uniform, skill, oracle)):
            continue
        u, s, o = vals[uniform], vals[skill], vals[oracle]
        lower_ok = u <= s + tol
        upper_ok = s <= o + tol
        results.append({
            "model": cell[0], "dataset": cell[1], "subtask": cell[2], "variant": cell[3],
            "uniform": u, "skill": s, "oracle": o,
            "lower_holds": lower_ok, "upper_holds": upper_ok,
            "holds": lower_ok and upper_ok,
            "uniform_gap": round(s - u, 4), "oracle_gap": round(o - s, 4),
        })
    return results


def _fmt(x: Any) -> str:
    return "" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def render_markdown(rows: list[dict[str, Any]], checks: list[dict[str, Any]],
                    uniform: str, skill: str, oracle: str, tol: float) -> str:
    lines = ["# Exp02 main task-performance table\n"]
    for dataset in DATASETS:
        drows = [r for r in rows if r.get("dataset") == dataset]
        if not drows:
            continue
        lines.append(f"\n## {dataset}\n")
        lines.append("| method | model | variant | subtask | n | token_f1 | exact | numeric_acc |")
        lines.append("|---|---|---|---|--:|--:|--:|--:|")
        for r in sorted(drows, key=lambda r: (str(r["method"]), str(r["model"]), str(r["variant"]), str(r["subtask"]))):
            lines.append(
                f"| {r['method']} | {r['model']} | {r['variant']} | {r['subtask']} | "
                f"{_fmt(r['n'])} | {_fmt(r['token_f1'])} | {_fmt(r['exact'])} | {_fmt(r['numeric_acc'])} |"
            )

    lines.append(f"\n## Ordering hypothesis: `{uniform}` <= `{skill}` <= `{oracle}` (tol={tol})\n")
    if not checks:
        lines.append("_No cell has all three roles present yet._")
    else:
        held = sum(c["holds"] for c in checks)
        lines.append(f"Holds in {held}/{len(checks)} comparable cells.\n")
        lines.append("| model | dataset | subtask | variant | uniform | skill | oracle | uniform_gap | oracle_gap | status |")
        lines.append("|---|---|---|---|--:|--:|--:|--:|--:|---|")
        for c in checks:
            status = "OK" if c["holds"] else ("VIOLATION(upper)" if not c["upper_holds"] else "VIOLATION(lower)")
            lines.append(
                f"| {c['model']} | {c['dataset']} | {c['subtask']} | {c['variant']} | "
                f"{_fmt(c['uniform'])} | {_fmt(c['skill'])} | {_fmt(c['oracle'])} | "
                f"{_fmt(c['uniform_gap'])} | {_fmt(c['oracle_gap'])} | {status} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dirs", nargs="*", default=[
        str(REPO / "outputs/baseline_rag/predictions"),
        str(REPO / "outputs/baseline_rag/predictions_bslret"),
        str(REPO / "outputs/baseline_rag/predictions_nolabel"),
        str(REPO / "outputs/baseline_rag/predictions_oracle"),
    ], help="Directories globbed for *_eval.json.")
    p.add_argument("--summary-csv", nargs="*", default=[
        str(REPO / "outputs/baseline_rag/results_summary*.csv"),
    ], help="Glob(s) for pre-aggregated results_summary*.csv.")
    p.add_argument("--uniform-method", default="dense_bge")
    p.add_argument("--skill-method", default="dense_bge_skillrouted")
    p.add_argument("--oracle-method", default="oracle_source")
    p.add_argument("--tol", type=float, default=0.0)
    p.add_argument("--out-csv", default=str(REPO / "outputs/baseline_rag/exp02_main_table.csv"))
    p.add_argument("--out-md", default=str(REPO / "outputs/baseline_rag/exp02_main_table.md"))
    args = p.parse_args()

    rows: list[dict[str, Any]] = []
    collect_eval_jsons(args.pred_dirs, rows)
    collect_summary_csvs(args.summary_csv, rows)

    if not rows:
        print("No eval JSONs or summary CSVs found yet -- nothing to collate.")
        return

    checks = check_ordering(rows, args.uniform_method, args.skill_method, args.oracle_method, args.tol)

    cols = ["dataset", "method", "model", "variant", "subtask", "n", "token_f1", "exact", "numeric_acc", "file"]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r.get(c) for c in cols} for r in rows)

    md = render_markdown(rows, checks, args.uniform_method, args.skill_method, args.oracle_method, args.tol)
    Path(args.out_md).write_text(md)
    held = sum(c["holds"] for c in checks)
    print(f"Wrote {args.out_csv} ({len(rows)} rows) and {args.out_md}")
    print(f"Ordering {args.uniform_method} <= {args.skill_method} <= {args.oracle_method}: "
          f"{held}/{len(checks)} cells hold.")


if __name__ == "__main__":
    main()
