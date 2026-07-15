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
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mol_instructions", "fdarxbench"],
        choices=["mol_instructions", "fdarxbench"],
        help="Which datasets to (re)build. Others are left untouched on disk.",
    )
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


def _official_split(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    return str(meta.get("split", "train"))


def build_mol_instructions(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    # Respect Mol-Instructions' OWN train/test split (metadata.split). The train
    # split is sampled from a bounded pool of official-train rows (source-read-limit,
    # matching the historical read window, which was all-train anyway because the
    # official test rows sit at the END of each file); the test split is sampled
    # exclusively from official-test rows. This replaces the old blind shuffle,
    # which drew the "test" set from the official-train pool and never touched the
    # official test set.
    by_source_train: dict[str, list[dict[str, Any]]] = {}
    by_source_test: dict[str, list[dict[str, Any]]] = {}
    all_records: list[dict[str, Any]] = []
    for source_name, source_path in MOL_INSTRUCTION_SOURCES:
        path = Path(source_path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise TypeError(f"{path} must contain a JSON list")
        train_pool, test_rows = [], []
        for idx, row in enumerate(data):
            if _official_split(row) == "test":
                test_rows.append((idx, row))
            elif len(train_pool) < args.source_read_limit:
                train_pool.append((idx, row))
        train_records = [normalize_record(source_name, path, idx, row) for idx, row in train_pool]
        test_records = [normalize_record(source_name, path, idx, row) for idx, row in test_rows]
        by_source_train[source_name] = train_records
        by_source_test[source_name] = test_records
        all_records.extend(train_records)
        all_records.extend(test_records)

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

    rng = random.Random(args.seed)

    def sample(by_source: dict[str, list[dict[str, Any]]], per_source: int) -> list[dict[str, Any]]:
        picked: list[dict[str, Any]] = []
        for records in by_source.values():
            shuffled = list(records)
            rng.shuffle(shuffled)
            picked.extend(shuffled if per_source <= 0 else shuffled[:per_source])
        return picked

    splits = {
        "train": sample(by_source_train, args.mol_train_per_source),
        "test": sample(by_source_test, args.mol_test_per_source),
    }
    target_dir = output_dir / "mol_instructions"
    for split_name, records in splits.items():
        write_jsonl(target_dir / f"{split_name}.jsonl", records)

    return {
        "sources": {
            source: {"train_pool": len(by_source_train[source]), "test_pool": len(by_source_test[source])}
            for source in by_source_train
        },
        "splits": {split: len(records) for split, records in splits.items()},
        "with_kg_evidence": sum(bool(r["retrieved_kg_evidence"]) for r in all_records),
        "with_molecule_evidence": sum(bool(r["retrieved_molecule_evidence"]) for r in all_records),
    }


def fdarx_prompt(record: dict[str, Any]) -> str:
    # Base prompt = augment_with_skill_kb.build_prompt WITHOUT the retrieved-evidence
    # block. Every method is evaluated on exactly this (raw dataset question); each
    # retrieval method's TRAIN prompt is this + a "Retrieved KB evidence" block.
    lines = [
        "You are answering a drug and molecular QA task with external knowledge.",
        "",
        "Use the provided information to answer the question. If the answer is not available, respond: Information not found!",
        "",
        "Answer in the same style as the gold answer.",
    ]
    if record["input_molecule_or_context"]:
        lines.extend(["", "Label/context evidence:", record["input_molecule_or_context"]])
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
    builders = {
        "mol_instructions": build_mol_instructions,
        "fdarxbench": build_fdarxbench,
    }
    summary = {name: builders[name](args, output_dir) for name in args.datasets}
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / ("summary.json" if set(args.datasets) == set(builders) else "summary_mol.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
