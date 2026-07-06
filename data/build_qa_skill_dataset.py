#!/usr/bin/env python
"""Build QA-centered skill data with optional PrimeKG and molecule evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import selfies as sf
except ImportError:
    sf = None


DEFAULT_SOURCES = [
    ("drugchat_chembl", "DrugChat/data/ChEMBL_Drug_Instructions/ChEMBL_Drug_Instructions.json"),
    ("drugchat_pubchem", "DrugChat/data/PubChem_Drug_Instructions/PubChem_Drug_Instructions.json"),
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

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*(?:\s+[A-Za-z][A-Za-z0-9+.-]*){0,4}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/qa_skill_data")
    parser.add_argument("--kg-file", default="knowledge_bank/primekg.csv")
    parser.add_argument("--train-per-source", type=int, default=100)
    parser.add_argument("--dev-per-source", type=int, default=20)
    parser.add_argument("--test-per-source", type=int, default=20)
    parser.add_argument("--source-read-limit", type=int, default=5000)
    parser.add_argument("--max-kg-evidence", type=int, default=6)
    parser.add_argument("--max-molecule-evidence", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def stable_id(*parts: Any) -> str:
    data = "||".join(str(part) for part in parts)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:16]


def decode_selfies(text: str) -> str | None:
    if sf is None or not text.startswith("["):
        return None
    try:
        return sf.decoder(text)
    except Exception:
        return None


def input_type(source_name: str, input_text: str) -> str:
    if not input_text:
        return "none"
    if source_name.startswith("drugchat"):
        return "SMILES"
    if source_name.startswith("molinst_") and input_text.startswith("["):
        return "SELFIES"
    return "text"


def task_type(source_name: str, row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if metadata.get("task"):
        return str(metadata["task"])
    if source_name == "drugchat_chembl":
        return "drug property QA"
    if source_name == "drugchat_pubchem":
        return "drug description QA"
    return source_name


def load_json_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list")
    return data[:limit]


def normalize_record(source_name: str, path: Path, idx: int, row: dict[str, Any]) -> dict[str, Any]:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    kind = input_type(source_name, input_text)
    decoded_smiles = decode_selfies(input_text) if kind == "SELFIES" else None
    return {
        "id": stable_id(source_name, idx, instruction, input_text, row.get("output", "")),
        "source": source_name,
        "source_path": str(path),
        "source_index": idx,
        "question": instruction,
        "input_molecule_or_context": input_text,
        "input_type": kind,
        "decoded_smiles": decoded_smiles,
        "gold_answer": row.get("output", ""),
        "task_type": task_type(source_name, row),
        "metadata": row.get("metadata"),
    }


def mention_candidates(records: list[dict[str, Any]]) -> set[str]:
    candidates: set[str] = set()
    for record in records:
        # Do not use gold answers for retrieval; that would leak the target.
        context = record["input_molecule_or_context"] if record["input_type"] == "text" else ""
        text = f"{record['question']} {context}"
        for match in WORD_RE.finditer(text):
            candidate = " ".join(match.group(0).lower().split())
            if len(candidate) >= 4:
                candidates.add(candidate)
    return candidates


def build_kg_evidence(
    kg_path: Path,
    candidates: set[str],
    max_evidence: int,
) -> dict[str, list[dict[str, str]]]:
    evidence: dict[str, list[dict[str, str]]] = defaultdict(list)
    if not kg_path.exists() or not candidates:
        return evidence
    useful_relations = {
        "indication",
        "off-label use",
        "contraindication",
        "drug_protein",
        "disease_protein",
        "drug_effect",
        "drug_drug",
        "disease_phenotype_positive",
        "disease_phenotype_negative",
    }
    with kg_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["relation"] not in useful_relations:
                continue
            names = [row["x_name"], row["y_name"]]
            for name in names:
                key = " ".join(name.lower().split())
                if key in candidates and len(evidence[key]) < max_evidence:
                    evidence[key].append(
                        {
                            "entity": name,
                            "x_name": row["x_name"],
                            "x_type": row["x_type"],
                            "relation": row["display_relation"],
                            "relation_type": row["relation"],
                            "y_name": row["y_name"],
                            "y_type": row["y_type"],
                        }
                    )
    return evidence


def attach_kg_evidence(
    record: dict[str, Any],
    kg_index: dict[str, list[dict[str, str]]],
    max_evidence: int,
) -> list[dict[str, str]]:
    context = record["input_molecule_or_context"] if record["input_type"] == "text" else ""
    text = f"{record['question']} {context}".lower()
    matched = []
    seen = set()
    for key, edges in kg_index.items():
        if key in text:
            for edge in edges:
                marker = (edge["x_name"], edge["relation_type"], edge["y_name"])
                if marker not in seen:
                    seen.add(marker)
                    matched.append(edge)
                if len(matched) >= max_evidence:
                    return matched
    return matched


def build_molecule_index(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = record["decoded_smiles"] or record["input_molecule_or_context"]
        if key:
            grouped[key].append(record)
    return grouped


def molecule_evidence(
    record: dict[str, Any],
    molecule_index: dict[str, list[dict[str, Any]]],
    max_evidence: int,
) -> list[dict[str, str]]:
    key = record["decoded_smiles"] or record["input_molecule_or_context"]
    if not key:
        return []
    items = []
    for other in molecule_index.get(key, []):
        if other["id"] == record["id"]:
            continue
        items.append(
            {
                "source": other["source"],
                "task_type": other["task_type"],
                "question": str(other["question"]),
                "answer": str(other["gold_answer"]),
            }
        )
        if len(items) >= max_evidence:
            break
    return items


def build_prompt(record: dict[str, Any]) -> str:
    lines = [
        "You are answering a drug and molecular QA task.",
        "Use the retrieved evidence when it is relevant, but do not invent facts not supported by the question or evidence.",
        "Return only the final answer in the same style as the gold answer.",
        "",
        f"Task: {record['task_type']}",
        f"Input type: {record['input_type']}",
    ]
    if record["input_molecule_or_context"]:
        lines.append(f"Input: {record['input_molecule_or_context']}")
    if record.get("decoded_smiles"):
        lines.append(f"Decoded SMILES: {record['decoded_smiles']}")
    lines.extend(["", "Retrieved molecule evidence:"])
    mol_evidence = record.get("retrieved_molecule_evidence") or []
    if mol_evidence:
        for idx, item in enumerate(mol_evidence, 1):
            lines.append(f"{idx}. {item['question']} -> {item['answer']}")
    else:
        lines.append("No matched molecule evidence found.")
    lines.extend(["", "Retrieved KG evidence:"])
    kg_evidence = record.get("retrieved_kg_evidence") or []
    if kg_evidence:
        for idx, edge in enumerate(kg_evidence, 1):
            lines.append(f"{idx}. {edge['x_name']} --{edge['relation']}--> {edge['y_name']}")
    else:
        lines.append("No matched KG evidence found.")
    lines.extend(["", f"Question: {record['question']}", "Answer:"])
    return "\n".join(lines)


def finalize_record(record: dict[str, Any]) -> dict[str, Any]:
    prompt = build_prompt(record)
    answer = str(record["gold_answer"]).strip()
    record["prompt"] = prompt
    record["messages"] = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    return record


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    by_source: dict[str, list[dict[str, Any]]] = {}
    all_records: list[dict[str, Any]] = []
    for source_name, source_path in DEFAULT_SOURCES:
        path = Path(source_path)
        rows = load_json_rows(path, args.source_read_limit)
        records = [normalize_record(source_name, path, idx, row) for idx, row in enumerate(rows)]
        by_source[source_name] = records
        all_records.extend(records)

    candidates = mention_candidates(all_records)
    kg_index = build_kg_evidence(Path(args.kg_file), candidates, args.max_kg_evidence)
    molecule_index = build_molecule_index(all_records)

    for record in all_records:
        record["retrieved_kg_evidence"] = attach_kg_evidence(record, kg_index, args.max_kg_evidence)
        record["retrieved_molecule_evidence"] = molecule_evidence(
            record, molecule_index, args.max_molecule_evidence
        )
        finalize_record(record)

    split_sizes = {
        "train": args.train_per_source,
        "dev": args.dev_per_source,
        "test": args.test_per_source,
    }
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in split_sizes}
    for source_name, records in by_source.items():
        shuffled = list(records)
        rng.shuffle(shuffled)
        cursor = 0
        for split_name, size in split_sizes.items():
            splits[split_name].extend(shuffled[cursor : cursor + size])
            cursor += size

    for split_name, records in splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", records)

    summary = {
        "sources": {source: len(records) for source, records in by_source.items()},
        "splits": {split: len(records) for split, records in splits.items()},
        "with_kg_evidence": sum(bool(r["retrieved_kg_evidence"]) for r in all_records),
        "with_molecule_evidence": sum(bool(r["retrieved_molecule_evidence"]) for r in all_records),
        "selfies_decoder_available": sf is not None,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
