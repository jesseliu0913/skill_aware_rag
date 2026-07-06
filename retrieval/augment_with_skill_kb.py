#!/usr/bin/env python
"""Attach skill-aware KB retrieval evidence to normalized QA JSONL records."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.'-]*")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.'-]*(?:\s+[A-Za-z][A-Za-z0-9+.'-]*){0,4}")

STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "are",
    "as",
    "be",
    "by",
    "can",
    "does",
    "for",
    "from",
    "given",
    "has",
    "have",
    "how",
    "in",
    "input",
    "is",
    "it",
    "its",
    "may",
    "molecule",
    "of",
    "on",
    "or",
    "patients",
    "question",
    "should",
    "task",
    "that",
    "the",
    "this",
    "to",
    "use",
    "used",
    "what",
    "when",
    "which",
    "why",
    "with",
}


LEGACY_SKILL_CONFIG: dict[str, dict[str, Any]] = {
    "fda_label_qa": {
        "sources": {"fdarxbench_label", "primekg", "drugchat_chembl", "drugchat_pubchem"},
        "relations": {"label_context", "drug_drug", "drug_effect", "contraindication", "indication", "off-label use"},
    },
    "molecule_property": {
        "sources": {"drugchat_pubchem", "drugchat_chembl", "primekg"},
        "relations": {"instruction_fact", "drug_protein", "drug_effect"},
    },
    "molecule_description": {
        "sources": {"drugchat_pubchem", "drugchat_chembl", "primekg"},
        "relations": {"instruction_fact", "drug_effect", "drug_protein"},
    },
    "molecule_design": {
        "sources": {"drugchat_pubchem", "drugchat_chembl", "primekg"},
        "relations": {"instruction_fact", "drug_effect", "drug_protein", "indication"},
    },
    "biomedical_open_qa": {
        "sources": {"primekg", "drugchat_chembl", "drugchat_pubchem", "fdarxbench_label"},
        "relations": {
            "indication",
            "off-label use",
            "contraindication",
            "drug_protein",
            "disease_protein",
            "drug_effect",
            "drug_drug",
            "disease_phenotype_positive",
            "disease_phenotype_negative",
            "instruction_fact",
            "label_context",
        },
    },
}


SKILL_SCHEMA_V1: dict[str, dict[str, Any]] = {
    "fda_label_factual": {
        "required_slots": ("label_context",),
        "sources": {"fdarxbench_label"},
        "relations": {"label_context"},
        "top_k": 3,
        "answer_format": "short_label_grounded_answer",
        "abstention_policy": "abstain_if_label_missing",
    },
    "fda_label_multihop": {
        "required_slots": ("label_context_primary", "label_context_secondary"),
        "sources": {"fdarxbench_label"},
        "relations": {"label_context"},
        "top_k": 4,
        "answer_format": "synthesized_label_answer",
        "abstention_policy": "abstain_if_any_required_label_missing",
    },
    "fda_label_refusal": {
        "required_slots": ("label_context_check", "unsupported_flag"),
        "sources": {"fdarxbench_label"},
        "relations": {"label_context"},
        "top_k": 2,
        "answer_format": "information_not_found_or_label_answer",
        "abstention_policy": "prefer_abstention_when_unsupported",
    },
    "molecule_property_numeric": {
        "required_slots": ("molecule_structure", "property_definition"),
        "sources": set(),
        "relations": set(),
        "top_k": 0,
        "answer_format": "numeric_or_short_property_value",
        "abstention_policy": "no_external_free_text_by_default",
    },
    "molecule_description": {
        "required_slots": ("molecule_structure",),
        "sources": {"drugchat_pubchem", "drugchat_chembl"},
        "relations": {"instruction_fact"},
        "top_k": 2,
        "answer_format": "natural_language_description",
        "abstention_policy": "use_structure_first",
    },
    "molecule_design": {
        "required_slots": ("design_requirement",),
        "sources": {"drugchat_pubchem", "drugchat_chembl"},
        "relations": {"instruction_fact"},
        "top_k": 1,
        "answer_format": "molecule_string",
        "abstention_policy": "avoid_unrelated_facts",
    },
    "biomedical_open_qa": {
        "required_slots": ("question_entities", "biomedical_fact"),
        "sources": {"primekg", "drugchat_pubchem", "drugchat_chembl"},
        "relations": {
            "indication",
            "off-label use",
            "contraindication",
            "drug_protein",
            "disease_protein",
            "drug_effect",
            "drug_drug",
            "disease_phenotype_positive",
            "disease_phenotype_negative",
            "instruction_fact",
        },
        "top_k": 3,
        "answer_format": "concise_biomedical_answer",
        "abstention_policy": "abstain_if_no_relevant_fact",
    },
    "drug_relation_qa": {
        "required_slots": ("drug_entity", "relation_fact"),
        "sources": {"primekg", "fdarxbench_label"},
        "relations": {"indication", "off-label use", "contraindication", "drug_protein", "drug_effect", "drug_drug", "label_context"},
        "top_k": 3,
        "answer_format": "short_relation_answer",
        "abstention_policy": "abstain_if_no_relation_fact",
    },
    "unsupported_or_low_evidence": {
        "required_slots": ("evidence_absence_reason",),
        "sources": set(),
        "relations": set(),
        "top_k": 0,
        "answer_format": "information_not_found",
        "abstention_policy": "always_abstain",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kb-dir", default="outputs/knowledge_bank/unified")
    parser.add_argument("--method", default="skill_hybrid", choices=["bm25", "kg", "hybrid", "skill_hybrid"])
    parser.add_argument("--schema-version", default="legacy", choices=["legacy", "v1"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-facts", type=int, default=0, help="Debug limit for KB facts.")
    parser.add_argument("--max-candidate-facts", type=int, default=20000)
    parser.add_argument("--max-prompt-evidence-chars", type=int, default=3500)
    return parser.parse_args()


def norm_text(text: Any) -> str:
    return " ".join(str(text).lower().split())


def tokens(text: Any) -> list[str]:
    return TOKEN_RE.findall(norm_text(text))


def informative_terms(query_terms: list[str], idf: dict[str, float], token_index: dict[str, list[int]]) -> list[str]:
    terms = []
    for term in set(query_terms):
        if len(term) < 3 or term in STOPWORDS:
            continue
        if term not in token_index:
            continue
        terms.append(term)
    terms.sort(key=lambda term: (idf.get(term, 0.0), -len(token_index.get(term, []))), reverse=True)
    return terms[:24]


def iter_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if limit and len(rows) >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_kb(
    kb_dir: Path,
    max_facts: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, float], dict[str, list[int]]]:
    facts = []
    doc_freq: Counter[str] = Counter()
    token_index: dict[str, list[int]] = {}
    with (kb_dir / "facts.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if max_facts and len(facts) >= max_facts:
                break
            fact = json.loads(line)
            fact["tokens"] = fact.get("tokens") or sorted(set(tokens(fact.get("text", ""))))
            fact_idx = len(facts)
            facts.append(fact)
            unique_tokens = set(fact["tokens"])
            doc_freq.update(unique_tokens)
            for token in unique_tokens:
                token_index.setdefault(token, []).append(fact_idx)
    with (kb_dir / "alias_index.json").open(encoding="utf-8") as f:
        alias_index = json.load(f)
    n_docs = max(len(facts), 1)
    idf = {term: math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0) for term, df in doc_freq.items()}
    return facts, alias_index, idf, token_index


def infer_skill(record: dict[str, Any]) -> str:
    source = str(record.get("source", ""))
    task_type = norm_text(record.get("task_type", ""))
    question = norm_text(record.get("question", ""))
    if source == "fdarxbench" or record.get("input_type") == "FDA label context":
        return "fda_label_qa"
    if "description guided" in task_type or "synthesize" in question or "design" in task_type:
        return "molecule_design"
    if "molecular description" in task_type or "describe" in question:
        return "molecule_description"
    if "property" in task_type or "molecular weight" in question or "logp" in question:
        return "molecule_property"
    return "biomedical_open_qa"


def infer_schema_v1_skill(record: dict[str, Any]) -> str:
    source = str(record.get("source", ""))
    task_type = norm_text(record.get("task_type", ""))
    question = norm_text(record.get("question", ""))
    if source == "fdarxbench" or record.get("input_type") == "FDA label context":
        if task_type == "multihop":
            return "fda_label_multihop"
        if task_type == "refusal" or record.get("input_type") == "none":
            return "fda_label_refusal"
        return "fda_label_factual"
    if "property" in task_type or "molecular weight" in question or "logp" in question:
        return "molecule_property_numeric"
    if "description guided" in task_type or "design" in task_type or "synthesize" in question:
        return "molecule_design"
    if "molecular description" in task_type or "describe" in question:
        return "molecule_description"
    relation_terms = ("contraindication", "indication", "interact", "target", "protein", "side effect")
    if any(term in question for term in relation_terms):
        return "drug_relation_qa"
    if not question:
        return "unsupported_or_low_evidence"
    return "biomedical_open_qa"


def infer_record_skill(record: dict[str, Any], schema_version: str) -> str:
    if schema_version == "v1":
        return infer_schema_v1_skill(record)
    return infer_skill(record)


def skill_config(skill: str, schema_version: str) -> dict[str, Any]:
    if schema_version == "v1":
        return SKILL_SCHEMA_V1.get(skill, SKILL_SCHEMA_V1["unsupported_or_low_evidence"])
    return LEGACY_SKILL_CONFIG.get(skill, LEGACY_SKILL_CONFIG["biomedical_open_qa"])


def query_text(record: dict[str, Any]) -> str:
    parts = [
        record.get("question", ""),
        record.get("drug_name", ""),
        record.get("input_molecule_or_context", ""),
        record.get("decoded_smiles", ""),
        record.get("task_type", ""),
    ]
    return " ".join(str(part) for part in parts if part)


def mention_candidates(record: dict[str, Any]) -> set[str]:
    text = query_text(record)
    candidates = set()
    for match in WORD_RE.finditer(text):
        candidate = norm_text(match.group(0))
        if len(candidate) >= 4:
            candidates.add(candidate)
    drug_name = norm_text(record.get("drug_name", ""))
    if drug_name:
        candidates.add(drug_name)
        candidates.add(drug_name.replace(" hydrochloride", "").replace(" hydrobromide", ""))
    if record.get("decoded_smiles"):
        candidates.add(norm_text(record["decoded_smiles"]))
    raw_input = str(record.get("input_molecule_or_context", "")).strip()
    if raw_input and len(raw_input) <= 300:
        candidates.add(norm_text(raw_input))
    return {candidate for candidate in candidates if candidate}


def bm25_score(query_terms: list[str], fact_terms: list[str], idf: dict[str, float], avgdl: float) -> float:
    if not query_terms or not fact_terms:
        return 0.0
    tf = Counter(fact_terms)
    dl = len(fact_terms)
    k1 = 1.2
    b = 0.75
    score = 0.0
    for term in set(query_terms):
        if term not in tf:
            continue
        denom = tf[term] + k1 * (1 - b + b * dl / max(avgdl, 1.0))
        score += idf.get(term, 0.0) * (tf[term] * (k1 + 1)) / denom
    return score


def allowed_by_skill(fact: dict[str, Any], skill: str, use_skill_filter: bool, schema_version: str) -> bool:
    if not use_skill_filter:
        return True
    config = skill_config(skill, schema_version)
    return fact.get("source") in config["sources"] and fact.get("relation") in config["relations"]


def retrieve(
    record: dict[str, Any],
    facts: list[dict[str, Any]],
    fact_id_to_idx: dict[str, int],
    alias_index: dict[str, list[str]],
    idf: dict[str, float],
    token_index: dict[str, list[int]],
    method: str,
    top_k: int,
    max_candidate_facts: int,
    schema_version: str,
) -> list[dict[str, Any]]:
    skill = infer_record_skill(record, schema_version)
    use_skill_filter = method == "skill_hybrid"
    if schema_version == "v1":
        top_k = min(top_k, int(skill_config(skill, schema_version)["top_k"]))
    if top_k <= 0:
        return []
    raw_query_terms = tokens(query_text(record))
    query_terms = informative_terms(raw_query_terms, idf, token_index)
    aliases = mention_candidates(record)
    alias_fact_ids = set()
    for alias in aliases:
        alias_fact_ids.update(alias_index.get(alias, []))
    avgdl = sum(len(f.get("tokens", [])) for f in facts) / max(len(facts), 1)
    candidate_idxs = set()
    for term in query_terms:
        candidate_idxs.update(token_index.get(term, []))
        if len(candidate_idxs) >= max_candidate_facts:
            break
    for fact_id in alias_fact_ids:
        if fact_id in fact_id_to_idx:
            candidate_idxs.add(fact_id_to_idx[fact_id])
    if method == "kg":
        candidate_idxs = {fact_id_to_idx[fact_id] for fact_id in alias_fact_ids if fact_id in fact_id_to_idx}

    scored = []
    for fact_idx in candidate_idxs:
        fact = facts[fact_idx]
        if not allowed_by_skill(fact, skill, use_skill_filter, schema_version):
            continue
        fact_id = fact["id"]
        alias_hit = fact_id in alias_fact_ids
        if method == "kg" and not alias_hit:
            continue
        score = bm25_score(query_terms, fact.get("tokens", []), idf, avgdl)
        if method in {"hybrid", "skill_hybrid", "kg"} and alias_hit:
            score += 8.0
        if fact.get("source") == "primekg" and alias_hit:
            score += 1.0
        if score <= 0:
            continue
        scored.append(
            {
                "id": fact_id,
                "score": round(score, 4),
                "source": fact.get("source"),
                "relation": fact.get("relation"),
                "title": fact.get("title"),
                "text": fact.get("text"),
                "entities": fact.get("entities", []),
                "metadata": fact.get("metadata", {}),
                "alias_hit": alias_hit,
            }
        )
    scored.sort(key=lambda row: (row["score"], row["alias_hit"]), reverse=True)
    return scored[:top_k]


def evidence_block(evidence: list[dict[str, Any]], max_chars: int) -> str:
    if not evidence:
        return "No KB evidence retrieved."
    lines = []
    used = 0
    for idx, item in enumerate(evidence, 1):
        text = str(item.get("text", "")).replace("\n", " ")
        line = f"{idx}. [{item.get('source')}:{item.get('relation')}; score={item.get('score')}] {text}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines) if lines else "No KB evidence retrieved within budget."


def build_prompt(record: dict[str, Any], skill: str, method: str, evidence: list[dict[str, Any]], max_chars: int) -> str:
    lines = [
        "You are answering a drug and molecular QA task with external knowledge.",
        "Use the provided label context and retrieved KB evidence only when relevant.",
        "If the evidence does not support an answer, say: Information not found!",
        "Return only the final answer in the same style as the gold answer.",
        "",
        f"Skill: {skill}",
        f"Retrieval method: {method}",
        f"Task: {record.get('task_type', '')}",
        f"Input type: {record.get('input_type', '')}",
    ]
    if record.get("drug_name"):
        lines.append(f"Drug: {record['drug_name']}")
    if record.get("input_molecule_or_context"):
        label = "Label/context evidence" if record.get("source") == "fdarxbench" else "Input"
        lines.extend(["", f"{label}:", str(record["input_molecule_or_context"])])
    if record.get("decoded_smiles"):
        lines.append(f"Decoded SMILES: {record['decoded_smiles']}")
    lines.extend(["", "Retrieved KB evidence:", evidence_block(evidence, max_chars)])
    lines.extend(["", f"Question: {record.get('question', '')}", "Answer:"])
    return "\n".join(lines)


def source_counts(evidence: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(item.get("source") or "unknown") for item in evidence)


def relation_counts(evidence: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(item.get("relation") or "unknown") for item in evidence)


def evidence_slot_fills(record: dict[str, Any], skill: str, evidence: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    config = skill_config(skill, "v1")
    sources = source_counts(evidence)
    relations = relation_counts(evidence)
    has_structure = bool(record.get("decoded_smiles") or record.get("input_type") in {"SELFIES", "SMILES"})
    has_label_input = record.get("input_type") == "FDA label context" and bool(record.get("input_molecule_or_context"))
    has_label_retrieval = sources["fdarxbench_label"] > 0 or relations["label_context"] > 0
    has_drugchat = sources["drugchat_pubchem"] > 0 or sources["drugchat_chembl"] > 0
    has_primekg = sources["primekg"] > 0
    checks = {
        "label_context": has_label_input or has_label_retrieval,
        "label_context_primary": has_label_input or has_label_retrieval,
        "label_context_secondary": has_label_retrieval and len(evidence) >= 2,
        "label_context_check": has_label_input or has_label_retrieval,
        "unsupported_flag": record.get("task_type") == "refusal" or record.get("input_type") == "none",
        "molecule_structure": has_structure,
        "property_definition": "property" in norm_text(record.get("task_type", "")) or "property" in norm_text(record.get("question", "")),
        "design_requirement": bool(str(record.get("input_molecule_or_context", "")).strip()),
        "question_entities": bool(str(record.get("question", "")).strip()),
        "biomedical_fact": has_primekg or has_drugchat,
        "drug_entity": bool(record.get("drug_name")) or bool(str(record.get("question", "")).strip()),
        "relation_fact": has_primekg,
        "evidence_absence_reason": not evidence,
    }
    slots = {}
    for slot in config["required_slots"]:
        slots[slot] = {
            "filled": bool(checks.get(slot, False)),
            "evidence_ids": [str(item.get("id")) for item in evidence[: config["top_k"]]],
        }
    return slots


def build_completed_skill_prompt(
    record: dict[str, Any],
    skill: str,
    method: str,
    evidence: list[dict[str, Any]],
    max_chars: int,
) -> str:
    config = skill_config(skill, "v1")
    slots = evidence_slot_fills(record, skill, evidence)
    lines = [
        "You are answering a drug and molecular QA task using a completed skill schema.",
        "Use only the filled evidence slots that are relevant to the question.",
        "If a required slot is unfilled and the answer cannot be supported, say: Information not found!",
        "Return only the final answer in the requested answer format.",
        "",
        f"Skill: {skill}",
        f"Answer format: {config['answer_format']}",
        f"Abstention policy: {config['abstention_policy']}",
        f"Retrieval method: {method}",
        f"Task: {record.get('task_type', '')}",
        f"Input type: {record.get('input_type', '')}",
    ]
    if record.get("drug_name"):
        lines.append(f"Drug: {record['drug_name']}")
    if record.get("input_molecule_or_context"):
        label = "Label/context evidence" if record.get("source") == "fdarxbench" else "Input"
        lines.extend(["", f"{label}:", str(record["input_molecule_or_context"])])
    if record.get("decoded_smiles"):
        lines.append(f"Decoded SMILES: {record['decoded_smiles']}")
    lines.extend(["", "Required evidence slots:"])
    for slot, status in slots.items():
        state = "filled" if status["filled"] else "missing"
        lines.append(f"- {slot}: {state}")
    lines.extend(["", "Retrieved KB evidence:", evidence_block(evidence, max_chars)])
    lines.extend(["", f"Question: {record.get('question', '')}", "Answer:"])
    return "\n".join(lines)


def augment_record(
    record: dict[str, Any],
    evidence: list[dict[str, Any]],
    method: str,
    max_chars: int,
    schema_version: str,
) -> dict[str, Any]:
    new_record = dict(record)
    skill = infer_record_skill(record, schema_version)
    config = skill_config(skill, schema_version)
    skill_schema = {
        "skill": skill,
        "entities": sorted(mention_candidates(record)),
        "retrieval_method": method,
        "schema_version": schema_version,
        "retrieved_sources": sorted({str(item.get("source")) for item in evidence}),
        "top_k": len(evidence),
        "required_slots": list(config.get("required_slots", ())),
        "allowed_sources": sorted(config.get("sources", ())),
        "answer_format": config.get("answer_format", ""),
        "abstention_policy": config.get("abstention_policy", ""),
        "answer_constraints": {
            "use_evidence_when_relevant": True,
            "abstain_if_missing": True,
        },
    }
    new_record["skill_schema"] = skill_schema
    new_record["retrieved_kb_evidence"] = evidence
    if schema_version == "v1":
        new_record["evidence_slots"] = evidence_slot_fills(new_record, skill, evidence)
        new_record["slot_fill_status"] = {
            "all_required_filled": all(item["filled"] for item in new_record["evidence_slots"].values()),
            "filled_slots": [slot for slot, item in new_record["evidence_slots"].items() if item["filled"]],
            "missing_slots": [slot for slot, item in new_record["evidence_slots"].items() if not item["filled"]],
        }
        new_record["completed_skill_prompt"] = build_completed_skill_prompt(new_record, skill, method, evidence, max_chars)
        new_record["prompt"] = new_record["completed_skill_prompt"]
    else:
        new_record["prompt"] = build_prompt(new_record, skill, method, evidence, max_chars)
    new_record["messages"] = [
        {"role": "user", "content": new_record["prompt"]},
        {"role": "assistant", "content": str(new_record.get("gold_answer", "")).strip()},
    ]
    return new_record


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    facts, alias_index, idf, token_index = load_kb(Path(args.kb_dir), args.max_facts)
    fact_id_to_idx = {fact["id"]: idx for idx, fact in enumerate(facts)}
    records = iter_jsonl(Path(args.input), args.limit)
    augmented = []
    skill_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    for record in records:
        evidence = retrieve(
            record,
            facts,
            fact_id_to_idx,
            alias_index,
            idf,
            token_index,
            args.method,
            args.top_k,
            args.max_candidate_facts,
            args.schema_version,
        )
        augmented.append(augment_record(record, evidence, args.method, args.max_prompt_evidence_chars, args.schema_version))
        skill_counts[infer_record_skill(record, args.schema_version)] += 1
        evidence_counts["with_evidence" if evidence else "without_evidence"] += 1
    write_jsonl(Path(args.output), augmented)
    summary = {
        "input": args.input,
        "output": args.output,
        "kb_dir": args.kb_dir,
        "method": args.method,
        "schema_version": args.schema_version,
        "limit": args.limit,
        "records": len(records),
        "facts_loaded": len(facts),
        "skill_counts": dict(skill_counts),
        "evidence_counts": dict(evidence_counts),
    }
    summary_path = Path(args.output).with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
