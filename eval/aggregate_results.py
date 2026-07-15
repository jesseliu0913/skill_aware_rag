#!/usr/bin/env python
"""Aggregate RAG-baseline eval JSONs into one comparison table.

Reads the per-run *_eval.json files (same schema as evaluate_instruction_outputs)
from the baseline predictions dir and, optionally, the existing proposed-method
predictions dir, then emits a tidy CSV + a markdown table keyed by
(dataset, method, model, variant) with the headline metric per dataset:

  fdarxbench     -> avg_token_f1, exact           (single aggregate row per file)
  mol_instructions -> per-subtask avg_token_f1 / numeric_accuracy
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO = Path("/playpen-jfs/jesse/drug_microbiome")  # consolidated-folder anchor (was: parent.parent)

# {model}_{dataset}_{method}_{variant}_test_eval.json
BASELINE_RE = re.compile(
    r"^(?P<model>qwen2_5_7b|qwen2_5_3b|llama3_2_3b)_"
    r"(?P<dataset>mol_instructions|fdarxbench)_"
    r"(?P<method>dense_bge|dense_medcpt|rrf|rerank)_"
    r"(?P<variant>raw|raglora)_test_eval\.json$"
)
# existing proposed/baseline runs: {model}_{dataset}_{method}_{raw|lora|raglora}_test_eval.json
# (raglora = the RAG-SFT completion from submit_ragsft_skillkb_matrix.sh, which
#  trains a per-method LoRA on the retrieval-augmented train split. The eval-only
#  "lora" rows share one plain adapter and inject evidence only at test time.)
EXISTING_RE = re.compile(
    r"^(?P<model>qwen2_5_7b|qwen2_5_3b|llama3_2_3b)_"
    r"(?P<dataset>mol_instructions|fdarxbench)_"
    r"(?P<method>bm25|kg|hybrid|skill_hybrid|skill_schema_v1)_"
    r"(?P<variant>raw|lora|raglora)_test_eval\.json$"
)
# plain LoRA / raw (no retrieval): {model}_{dataset}_{raw|lora}_test_eval.json
PLAIN_RE = re.compile(
    r"^(?P<model>qwen2_5_7b|qwen2_5_3b|llama3_2_3b)_"
    r"(?P<dataset>mol_instructions|fdarxbench)_"
    r"(?P<variant>raw|lora)_test_eval\.json$"
)
# train-only retrieval, RAW-question eval (submit_rawq_eval_matrix.sh):
# {model}_{dataset}_{method}_rawqeval_test_eval.json. Every method's LoRA is
# trained on its augmented train split but evaluated on the identical raw
# dataset question, so test input is constant and only training differs.
RAWQ_RE = re.compile(
    r"^(?P<model>qwen2_5_7b|qwen2_5_3b|llama3_2_3b)_"
    r"(?P<dataset>mol_instructions|fdarxbench)_"
    r"(?P<method>none|bm25|kg|hybrid|dense_bge|dense_medcpt|rrf|rerank|skill_hybrid|skill_schema_v1)_"
    r"(?P<variant>rawqeval)_test_eval\.json$"
)


# baseline RAG-SFT trained + RAW-question eval (submit_baseline_raweval_matrix.sh):
# {model}_{dataset}_{method}_bsl_test_eval.json. Each method trains on the base
# prompt + its retrieved-evidence block and is evaluated on the base prompt alone.
BSL_RE = re.compile(
    r"^(?P<model>qwen2_5_7b|qwen2_5_3b|llama3_2_3b)_"
    r"(?P<dataset>mol_instructions|fdarxbench)_"
    r"(?P<method>none|bm25|kg|hybrid|skill_hybrid|dense_bge|dense_medcpt|rrf|rerank|skill_schema_v1)_"
    r"(?P<variant>bsl)_test_eval\.json$"
)
# same trained adapters, but retrieval present at TEST too (submit_reteval_matrix.sh):
BSLRET_RE = re.compile(
    r"^(?P<model>qwen2_5_7b|qwen2_5_3b|llama3_2_3b)_"
    r"(?P<dataset>mol_instructions|fdarxbench)_"
    r"(?P<method>bm25|kg|hybrid|skill_hybrid|dense_bge|dense_medcpt|rrf|rerank|skill_schema_v1)_"
    r"(?P<variant>bslret)_test_eval\.json$"
)
# FDA label context removed at train and test; retrieval remains at test.
NOLABEL_RE = re.compile(
    r"^(?P<model>qwen2_5_7b|qwen2_5_3b|llama3_2_3b)_"
    r"(?P<dataset>fdarxbench)_"
    r"(?P<method>none|bm25|kg|hybrid|skill_hybrid|dense_bge|dense_medcpt|rrf|rerank|skill_schema_v1)_"
    r"(?P<variant>nolabel)_test_eval\.json$"
)


def collect(pred_dir: Path, rows: list[dict]) -> None:
    if not pred_dir.is_dir():
        return
    for f in sorted(pred_dir.glob("*_eval.json")):
        for regex, method_default in ((BASELINE_RE, None), (EXISTING_RE, None), (RAWQ_RE, None), (BSL_RE, None), (BSLRET_RE, None), (NOLABEL_RE, None), (PLAIN_RE, "none")):
            m = regex.match(f.name)
            if not m:
                continue
            g = m.groupdict()
            method = g.get("method", method_default) or method_default
            try:
                recs = json.loads(f.read_text())
            except Exception:
                break
            recs = recs if isinstance(recs, list) else [recs]
            for r in recs:
                rows.append(
                    {
                        "dataset": g["dataset"],
                        "method": method,
                        "model": g["model"],
                        "variant": g["variant"],
                        "subtask": r.get("dataset", g["dataset"]),
                        "n": r.get("n"),
                        "token_f1": r.get("avg_token_f1"),
                        "exact": r.get("exact"),
                        "numeric_acc": r.get("numeric_accuracy"),
                        "file": f.name,
                    }
                )
            break


def fmt(x) -> str:
    return "" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir", default=str(REPO / "outputs/baseline_rag/predictions"))
    p.add_argument("--include-existing", action="store_true", help="Also fold in outputs/qa_skill_predictions.")
    p.add_argument("--out-csv", default=str(REPO / "outputs/baseline_rag/results_summary.csv"))
    p.add_argument("--out-md", default=str(REPO / "outputs/baseline_rag/results_summary.md"))
    args = p.parse_args()

    rows: list[dict] = []
    collect(Path(args.baseline_dir), rows)
    if args.include_existing:
        collect(REPO / "outputs/qa_skill_predictions", rows)

    if not rows:
        print("No eval JSONs found yet.")
        return

    cols = ["dataset", "method", "model", "variant", "subtask", "n", "token_f1", "exact", "numeric_acc", "file"]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Markdown: fdarxbench aggregate rows only, mol per-subtask.
    lines = ["# RAG baseline results\n"]
    for dataset in ("fdarxbench", "mol_instructions"):
        drows = [r for r in rows if r["dataset"] == dataset]
        if not drows:
            continue
        lines.append(f"\n## {dataset}\n")
        lines.append("| method | model | variant | subtask | n | token_f1 | exact | numeric_acc |")
        lines.append("|---|---|---|---|--:|--:|--:|--:|")
        for r in sorted(drows, key=lambda r: (r["method"], r["model"], r["variant"], str(r["subtask"]))):
            lines.append(
                f"| {r['method']} | {r['model']} | {r['variant']} | {r['subtask']} | "
                f"{fmt(r['n'])} | {fmt(r['token_f1'])} | {fmt(r['exact'])} | {fmt(r['numeric_acc'])} |"
            )
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out_csv} ({len(rows)} rows) and {args.out_md}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
