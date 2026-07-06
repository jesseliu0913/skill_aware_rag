#!/usr/bin/env python
"""Build corrected task-specific QA JSONL datasets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from build_qa_skill_dataset import (
    attach_kg_evidence,
    build_kg_evidence,
    build_molecule_index,
    finalize_record,
    load_json_rows,
    mention_candidates,
    molecule_evidence,
    normalize_record,
)


MOL_INSTRUCTION_SOURCES = [
    (
        "molinst_description_guided_design",
        "Mol-Instructions/data/Molecule-oriented_Instructions/description_guided_molecule_design.json",
    ),
    (
        "molinst_molecular_description",
        "Mol-Instructions/data/Molecule-oriented_Instructions/molecular_description_generation.json",
    ),
    ("molinst_property", "Mol-Instructions/data/Molecule-oriented_Instructions/property_prediction.json"),
    ("molinst_open_qa", "Mol-Instructions/data/Biomolecular_Text_Instructions/open_question.json"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/qa_skill_data_corrected")
    parser.add_argument("--kg-file", default="knowledge_bank/primekg.csv")
    parser.add_argument("--source-read-limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-kg-evidence", type=int, default=6)
    parser.add_argument("--max-molecule-evidence", type=int, default=3)
    parser.add_argument("--mol-train-per-source", type=int, default=750)
    parser.add_argument("--mol-dev-per-source", type=int, default=0)
    parser.add_argument("--mol-test-per-source", type=int, default=250)
    parser.add_argument("--fdarx-train", type=int, default=3000)
    parser.add_argument("--fdarx-dev", type=int, default=0)
    parser.add_argument("--fdarx-test", type=int, default=1000)
    return parser.parse_args()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_records(
    records_by_source: dict[str, list[dict[str, Any]]],
    split_sizes: dict[str, int],
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {split: [] for split, size in split_sizes.items() if size > 0}
    for records in records_by_source.values():
        shuffled = list(records)
        rng.shuffle(shuffled)
        cursor = 0
        for split_name, size in split_sizes.items():
            if size <= 0:
                continue
            splits[split_name].extend(shuffled[cursor : cursor + size])
            cursor += size
    return splits


def build_mol_instructions(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    all_records: list[dict[str, Any]] = []
    for source_name, source_path in MOL_INSTRUCTION_SOURCES:
        path = Path(source_path)
        rows = load_json_rows(path, args.source_read_limit)
        records = [normalize_record(source_name, path, idx, row) for idx, row in enumerate(rows)]
        by_source[source_name] = records
        all_records.extend(records)

    kg_index = build_kg_evidence(
        Path(args.kg_file),
        mention_candidates(all_records),
        args.max_kg_evidence,
    )
    molecule_index = build_molecule_index(all_records)
    for record in all_records:
        record["retrieved_kg_evidence"] = attach_kg_evidence(record, kg_index, args.max_kg_evidence)
        record["retrieved_molecule_evidence"] = molecule_evidence(
            record,
            molecule_index,
            args.max_molecule_evidence,
        )
        finalize_record(record)

    split_sizes = {
        "train": args.mol_train_per_source,
        "test": args.mol_test_per_source,
    }
    if args.mol_dev_per_source > 0:
        split_sizes = {
            "train": args.mol_train_per_source,
            "dev": args.mol_dev_per_source,
            "test": args.mol_test_per_source,
        }
    splits = split_records(by_source, split_sizes, random.Random(args.seed))
    target_dir = output_dir / "mol_instructions"
    for split_name, records in splits.items():
        write_jsonl(target_dir / f"{split_name}.jsonl", records)

    return {
        "sources": {source: len(records) for source, records in by_source.items()},
        "splits": {split: len(records) for split, records in splits.items()},
        "with_kg_evidence": sum(bool(r["retrieved_kg_evidence"]) for r in all_records),
        "with_molecule_evidence": sum(bool(r["retrieved_molecule_evidence"]) for r in all_records),
    }


def fdarx_prompt(record: dict[str, Any]) -> str:
    lines = [
        "You are answering an FDA drug-label QA task.",
        "Use the provided label evidence when it is present.",
        "If the answer is not in the evidence, answer: Information not found!",
        "",
        f"Task: {record['task_type']}",
        f"Drug: {record['drug_name']}",
    ]
    if record["input_molecule_or_context"]:
        lines.extend(["", "Label evidence:", record["input_molecule_or_context"]])
    lines.extend(["", f"Question: {record['question']}", "Answer:"])
    return "\n".join(lines)


def normalize_fdarx_record(row: dict[str, Any]) -> dict[str, Any]:
    context_rows = row.get("context") if isinstance(row.get("context"), list) else []
    context = []
    for item in context_rows:
        if not isinstance(item, dict):
            continue
        passage = item.get("text", "")
        title = item.get("section_title", "")
        if title:
            context.append(f"{title}: {passage}")
        elif passage:
            context.append(str(passage))
    record = {
        "id": row["qid"],
        "source": "fdarxbench",
        "source_path": "FDARxBench/data/qa/qa.jsonl",
        "source_index": row.get("qid"),
        "question": row["question"],
        "input_molecule_or_context": "\n\n".join(context),
        "input_type": "FDA label context" if context else "none",
        "decoded_smiles": None,
        "gold_answer": row["answer"],
        "task_type": row.get("task") or row.get("question_type") or "fdarxbench",
        "drug_name": row.get("drug_name", ""),
        "metadata": {
            "set_id": row.get("set_id"),
            "question_type": row.get("question_type"),
            "citations": row.get("citations", []),
            "references": row.get("references", []),
        },
        "retrieved_kg_evidence": [],
        "retrieved_molecule_evidence": [],
    }
    prompt = fdarx_prompt(record)
    answer = str(record["gold_answer"]).strip()
    record["prompt"] = prompt
    record["messages"] = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    return record


def build_fdarxbench(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    rows = []
    with Path("FDARxBench/data/qa/qa.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    records = [normalize_fdarx_record(row) for row in rows]
    rng = random.Random(args.seed)
    rng.shuffle(records)

    split_sizes = {
        "train": args.fdarx_train,
        "test": args.fdarx_test,
    }
    if args.fdarx_dev > 0:
        split_sizes = {
            "train": args.fdarx_train,
            "dev": args.fdarx_dev,
            "test": args.fdarx_test,
        }
    splits: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for split_name, size in split_sizes.items():
        if size <= 0:
            continue
        splits[split_name] = records[cursor : cursor + size]
        cursor += size

    target_dir = output_dir / "fdarxbench"
    for split_name, split_records_ in splits.items():
        write_jsonl(target_dir / f"{split_name}.jsonl", split_records_)

    return {
        "sources": {"fdarxbench": len(records)},
        "splits": {split: len(split_records_) for split, split_records_ in splits.items()},
        "tasks": {
            task: sum(1 for record in records if record["task_type"] == task)
            for task in sorted({record["task_type"] for record in records})
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    summary = {
        "mol_instructions": build_mol_instructions(args, output_dir),
        "fdarxbench": build_fdarxbench(args, output_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
