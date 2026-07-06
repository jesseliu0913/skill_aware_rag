#!/usr/bin/env python
"""Build a benchmark-scoped drug/molecule knowledge bank.

The full upstream resources are large, so this script creates a compact,
reproducible KB slice for the current QA benchmarks:

* PrimeKG edges whose endpoints match benchmark mentions or FDA drug names.
* Local DrugChat PubChem and ChEMBL instruction facts as lightweight
  compound/source-specific evidence.
* FDA label snippets from FDARxBench as a label evidence source.

The output is intentionally plain JSONL so retrieval baselines can run without
database services.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import selfies as sf
except ImportError:  # pragma: no cover - optional local dependency
    sf = None


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.'-]*(?:\s+[A-Za-z][A-Za-z0-9+.'-]*){0,4}")
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.'-]*")
SMILES_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+$")

USEFUL_PRIMEKG_RELATIONS = {
    "indication",
    "off-label use",
    "contraindication",
    "drug_protein",
    "disease_protein",
    "drug_effect",
    "drug_drug",
    "phenotype_protein",
    "disease_phenotype_positive",
    "disease_phenotype_negative",
    "exposure_disease",
    "exposure_protein",
    "pathway_protein",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primekg", default="knowledge_bank/primekg.csv")
    parser.add_argument("--output-dir", default="outputs/knowledge_bank/unified")
    parser.add_argument(
        "--candidate-jsonl",
        nargs="*",
        default=[
            "outputs/qa_skill_data_corrected/mol_instructions/train.jsonl",
            "outputs/qa_skill_data_corrected/mol_instructions/test.jsonl",
            "outputs/qa_skill_data_corrected/fdarxbench/train.jsonl",
            "outputs/qa_skill_data_corrected/fdarxbench/test.jsonl",
        ],
        help="Benchmark records used only to define entity/query candidates.",
    )
    parser.add_argument("--drugchat-chembl", default="DrugChat/data/ChEMBL_Drug_Instructions/ChEMBL_Drug_Instructions.json")
    parser.add_argument("--drugchat-pubchem", default="DrugChat/data/PubChem_Drug_Instructions/PubChem_Drug_Instructions.json")
    parser.add_argument("--fdarxbench", default="FDARxBench/data/qa/qa.jsonl")
    parser.add_argument("--source-read-limit", type=int, default=20000)
    parser.add_argument("--max-primekg-edges-per-entity", type=int, default=80)
    parser.add_argument("--max-source-facts", type=int, default=50000)
    return parser.parse_args()


def stable_id(*parts: Any) -> str:
    data = "||".join(str(part) for part in parts)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:20]


def norm_text(text: Any) -> str:
    return " ".join(str(text).lower().split())


def tokens(text: Any) -> list[str]:
    return TOKEN_RE.findall(norm_text(text))


def is_probable_smiles(text: str) -> bool:
    text = text.strip()
    return 4 <= len(text) <= 300 and bool(SMILES_RE.fullmatch(text)) and any(ch in text for ch in "=#[]()")


def decode_selfies(text: str) -> str | None:
    if sf is None or not text.startswith("["):
        return None
    try:
        return sf.decoder(text)
    except Exception:
        return None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_json_list(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list")
    return data[:limit]


def mention_candidates_from_record(record: dict[str, Any]) -> set[str]:
    fields = [
        record.get("question", ""),
        record.get("input_molecule_or_context", ""),
        record.get("task_type", ""),
        record.get("drug_name", ""),
    ]
    candidates: set[str] = set()
    for field in fields:
        for match in WORD_RE.finditer(str(field)):
            candidate = norm_text(match.group(0))
            if len(candidate) >= 4:
                candidates.add(candidate)
    drug_name = norm_text(record.get("drug_name", ""))
    if drug_name:
        candidates.add(drug_name)
        candidates.add(drug_name.replace(" hydrochloride", "").replace(" hydrobromide", ""))
    return {c for c in candidates if c}


def build_candidate_set(paths: list[str]) -> set[str]:
    candidates: set[str] = set()
    for raw_path in paths:
        for record in iter_jsonl(Path(raw_path)):
            candidates.update(mention_candidates_from_record(record))
    return candidates


def fact_text(fact: dict[str, Any]) -> str:
    title = fact.get("title") or ""
    body = fact.get("text") or ""
    aliases = " ".join(fact.get("aliases") or [])
    return " ".join(str(x) for x in [title, body, aliases] if x)


def add_fact(
    facts: list[dict[str, Any]],
    alias_to_facts: dict[str, list[str]],
    fact: dict[str, Any],
) -> None:
    fact["id"] = fact.get("id") or stable_id(fact.get("source"), fact.get("title"), fact.get("text"))
    text = fact_text(fact)
    fact["tokens"] = sorted(set(tokens(text)))
    facts.append(fact)
    for alias in fact.get("aliases", []):
        key = norm_text(alias)
        if key:
            alias_to_facts[key].append(fact["id"])


def build_primekg_facts(
    path: Path,
    candidates: set[str],
    max_edges_per_entity: int,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    alias_to_facts: dict[str, list[str]] = defaultdict(list)
    per_entity_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    matched_rows = 0
    if not path.exists():
        return facts, alias_to_facts, {"missing": str(path)}

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            relation = row.get("relation", "")
            if relation not in USEFUL_PRIMEKG_RELATIONS:
                continue
            x_name = row.get("x_name", "")
            y_name = row.get("y_name", "")
            x_key = norm_text(x_name)
            y_key = norm_text(y_name)
            matched_keys = [key for key in (x_key, y_key) if key in candidates]
            if not matched_keys:
                continue
            if all(per_entity_counts[key] >= max_edges_per_entity for key in matched_keys):
                continue
            for key in matched_keys:
                per_entity_counts[key] += 1
            matched_rows += 1
            relation_counts[relation] += 1
            display_relation = row.get("display_relation") or relation
            add_fact(
                facts,
                alias_to_facts,
                {
                    "source": "primekg",
                    "source_id": f"{row.get('x_id')}|{relation}|{row.get('y_id')}",
                    "title": f"{x_name} --{display_relation}--> {y_name}",
                    "text": (
                        f"PrimeKG relation: {x_name} ({row.get('x_type')}) "
                        f"--{display_relation}--> {y_name} ({row.get('y_type')})."
                    ),
                    "aliases": [x_name, y_name],
                    "entities": [
                        {"name": x_name, "type": row.get("x_type"), "id": row.get("x_id"), "source": row.get("x_source")},
                        {"name": y_name, "type": row.get("y_type"), "id": row.get("y_id"), "source": row.get("y_source")},
                    ],
                    "relation": relation,
                    "metadata": {
                        "display_relation": display_relation,
                        "x_type": row.get("x_type"),
                        "y_type": row.get("y_type"),
                    },
                },
            )
    summary = {
        "matched_rows": matched_rows,
        "relation_counts": dict(relation_counts.most_common()),
        "matched_entities": len(per_entity_counts),
    }
    return facts, alias_to_facts, summary


def build_drugchat_facts(path: Path, source_name: str, limit: int) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    facts: list[dict[str, Any]] = []
    alias_to_facts: dict[str, list[str]] = defaultdict(list)
    rows = load_json_list(path, limit)
    for idx, row in enumerate(rows):
        instruction = str(row.get("instruction", "")).strip()
        input_text = str(row.get("input", "")).strip()
        output = str(row.get("output", "")).strip()
        aliases = []
        decoded = decode_selfies(input_text)
        if decoded:
            aliases.append(decoded)
        if is_probable_smiles(input_text):
            aliases.append(input_text)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        for key in ("cid", "CID", "chembl_id", "molecule_chembl_id", "drug_name", "name"):
            if metadata.get(key):
                aliases.append(str(metadata[key]))
        add_fact(
            facts,
            alias_to_facts,
            {
                "source": source_name,
                "source_id": f"{source_name}:{idx}",
                "title": instruction[:180] or f"{source_name} fact",
                "text": f"Question: {instruction}\nInput: {input_text}\nAnswer: {output}",
                "aliases": aliases,
                "entities": [{"name": alias, "type": "compound"} for alias in aliases],
                "relation": "instruction_fact",
                "metadata": {"source_index": idx, "task": metadata.get("task")},
            },
        )
    return facts, alias_to_facts


def build_fdarx_facts(path: Path, limit: int) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    facts: list[dict[str, Any]] = []
    alias_to_facts: dict[str, list[str]] = defaultdict(list)
    if not path.exists():
        return facts, alias_to_facts
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            row = json.loads(line)
            drug_name = str(row.get("drug_name", "")).strip()
            context_rows = row.get("context") if isinstance(row.get("context"), list) else []
            snippets = []
            for item in context_rows[:4]:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("section_title", "")).strip()
                text = str(item.get("text", "")).strip()
                if title or text:
                    snippets.append(f"{title}: {text}" if title else text)
            if not snippets:
                continue
            add_fact(
                facts,
                alias_to_facts,
                {
                    "source": "fdarxbench_label",
                    "source_id": str(row.get("qid") or idx),
                    "title": f"FDA label evidence for {drug_name}",
                    "text": "\n\n".join(snippets),
                    "aliases": [drug_name] if drug_name else [],
                    "entities": [{"name": drug_name, "type": "drug"}] if drug_name else [],
                    "relation": "label_context",
                    "metadata": {
                        "question_type": row.get("question_type"),
                        "citations": row.get("citations", []),
                    },
                },
            )
    return facts, alias_to_facts


def merge_alias_maps(*maps: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = defaultdict(list)
    for alias_map in maps:
        for alias, fact_ids in alias_map.items():
            merged[alias].extend(fact_ids)
    return {alias: sorted(set(ids)) for alias, ids in merged.items()}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_candidate_set(args.candidate_jsonl)
    primekg_facts, primekg_aliases, primekg_summary = build_primekg_facts(
        Path(args.primekg),
        candidates,
        args.max_primekg_edges_per_entity,
    )
    chembl_facts, chembl_aliases = build_drugchat_facts(
        Path(args.drugchat_chembl),
        "drugchat_chembl",
        args.max_source_facts,
    )
    pubchem_facts, pubchem_aliases = build_drugchat_facts(
        Path(args.drugchat_pubchem),
        "drugchat_pubchem",
        args.max_source_facts,
    )
    fdarx_facts, fdarx_aliases = build_fdarx_facts(Path(args.fdarxbench), args.max_source_facts)

    facts = primekg_facts + chembl_facts + pubchem_facts + fdarx_facts
    alias_index = merge_alias_maps(primekg_aliases, chembl_aliases, pubchem_aliases, fdarx_aliases)

    write_jsonl(output_dir / "facts.jsonl", facts)
    with (output_dir / "alias_index.json").open("w", encoding="utf-8") as f:
        json.dump(alias_index, f, indent=2, ensure_ascii=False)
    summary = {
        "candidate_mentions": len(candidates),
        "facts": {
            "total": len(facts),
            "primekg": len(primekg_facts),
            "drugchat_chembl": len(chembl_facts),
            "drugchat_pubchem": len(pubchem_facts),
            "fdarxbench_label": len(fdarx_facts),
        },
        "aliases": len(alias_index),
        "primekg": primekg_summary,
        "outputs": {
            "facts": str(output_dir / "facts.jsonl"),
            "alias_index": str(output_dir / "alias_index.json"),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
