#!/usr/bin/env python
"""Attach skill-aware KB retrieval evidence to normalized QA JSONL records."""

from __future__ import annotations

import argparse
import hashlib
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


# Six task-aligned skills. Each is a typed record: which KB sources/relations the
# skill-based retriever may use, its evidence budget top_k (>0 for every skill, so
# each task still retrieves and the retrieval baselines have something to compare
# against), the ordered solving STEPS shown to the model, and the answer format.
SKILL_SCHEMA_V1: dict[str, dict[str, Any]] = {
    "fda_label_factual": {
        "required_slots": ("label_context",),
        "sources": {"fdarxbench_label"},
        "relations": {"label_context"},
        "top_k": 3,
        "steps": (
            "Locate the label section relevant to the question.",
            "Extract the exact fact asked for (event, dose, contraindication, population, ...).",
            "Answer concisely, grounded only in the label text.",
        ),
        "answer_format": "short_label_grounded_answer",
    },
    "fda_label_multihop": {
        "required_slots": ("label_context_primary", "label_context_secondary"),
        "sources": {"fdarxbench_label"},
        "relations": {"label_context"},
        "top_k": 4,
        "steps": (
            "Identify the two or more label sections the question connects.",
            "Extract the key fact from each section.",
            "Combine them into one coherent, grounded answer.",
        ),
        "answer_format": "synthesized_label_answer",
    },
    "molecule_property_numeric": {
        "required_slots": ("molecule_structure", "property_definition"),
        "sources": {"drugchat_pubchem", "drugchat_chembl"},
        "relations": {"instruction_fact"},
        "top_k": 3,
        "steps": (
            "Read the molecular structure from the SELFIES/SMILES string.",
            "Identify the requested property (e.g., HOMO-LUMO gap, logP).",
            "Use similar retrieved molecules as reference, then give the numeric value only.",
        ),
        "answer_format": "numeric_or_short_property_value",
    },
    "molecule_description": {
        "required_slots": ("molecule_structure",),
        "sources": {"drugchat_pubchem", "drugchat_chembl"},
        "relations": {"instruction_fact"},
        "top_k": 3,
        "steps": (
            "Parse the structure: identify its class, functional groups, or natural-product source.",
            "Use similar retrieved compound facts as reference.",
            "Write a concise natural-language description.",
        ),
        "answer_format": "natural_language_description",
    },
    "molecule_design": {
        "required_slots": ("design_requirement",),
        "sources": {"drugchat_pubchem", "drugchat_chembl"},
        "relations": {"instruction_fact"},
        "top_k": 3,
        "steps": (
            "Read the design requirement (role, class, scaffold, target property).",
            "Recall similar molecules from the retrieved compound facts.",
            "Output a molecule string (SELFIES) that satisfies the requirement.",
        ),
        "answer_format": "molecule_string",
    },
    "biomedical_open_qa": {
        "required_slots": ("question_entities", "biomedical_fact"),
        "sources": {"primekg", "drugchat_pubchem", "drugchat_chembl", "fdarxbench_label"},
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
        "top_k": 3,
        "steps": (
            "Identify the biomedical entities in the question (drug, disease, protein, pathway).",
            "Retrieve the relevant relations/facts for those entities.",
            "Answer concisely from the retrieved facts.",
        ),
        "answer_format": "concise_biomedical_answer",
    },
}


# Schema components that Exp04 (RQ5) can ablate. When a component is listed in
# `--ablate` it is REMOVED from the skill prompt/route, so the additive grid
# "uniform RAG -> full SkillRAG" is obtained by removing more/fewer components.
# The empty set (default) is the full schema and reproduces the current output
# byte-for-byte.
ABLATABLE_COMPONENTS = ("skill_id", "source_routing", "solving_steps", "answer_format", "evidence_org")


def parse_ablate(spec: Any) -> set[str]:
    """Parse an --ablate spec into a validated set of components to REMOVE.

    None / empty -> empty set -> full schema (unchanged, byte-identical output).
    Accepts a comma-separated str ("a,b") or an existing iterable of names.
    """
    if not spec:
        return set()
    if isinstance(spec, (set, frozenset, list, tuple)):
        components = {str(part).strip() for part in spec}
    else:
        components = {part.strip() for part in str(spec).split(",")}
    components = {part for part in components if part}
    unknown = components - set(ABLATABLE_COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown --ablate components {sorted(unknown)}; valid: {list(ABLATABLE_COMPONENTS)}")
    return components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kb-dir", default="outputs/knowledge_bank/unified")
    parser.add_argument("--method", default="skill_hybrid", choices=["bm25", "kg", "hybrid", "skill_hybrid", "bm25_skillrouted"])
    parser.add_argument("--schema-version", default="legacy", choices=["legacy", "v1"])
    parser.add_argument(
        "--skill-override",
        default="none",
        choices=["none", "oracle", "random", "wrong", "predicted"],
        help=(
            "Force the routed skill for Exp06 controls. 'none' (default) keeps the "
            "current gold-field routing byte-identical; 'oracle' = gold-field skill; "
            "'predicted' = question-only deployable router; 'random' = deterministic "
            "sha1(id)-hashed skill; 'wrong' = deterministic confusion-map skill."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-facts", type=int, default=0, help="Debug limit for KB facts.")
    parser.add_argument("--max-candidate-facts", type=int, default=20000)
    parser.add_argument("--max-prompt-evidence-chars", type=int, default=3500)
    parser.add_argument(
        "--ablate",
        default="",
        help=(
            "Exp04/RQ5 schema ablation. Comma-separated components to REMOVE, "
            f"subset of {{{','.join(ABLATABLE_COMPONENTS)}}}. Empty (default) keeps "
            "the full schema and is byte-identical to the un-ablated output. Only "
            "the v1 skill prompt/route honor these components."
        ),
    )
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
        return "fda_label_factual"
    if "property" in task_type or "molecular weight" in question or "logp" in question:
        return "molecule_property_numeric"
    if "description guided" in task_type or "design" in task_type or "synthesize" in question:
        return "molecule_design"
    if "molecular description" in task_type or "describe" in question:
        return "molecule_description"
    return "biomedical_open_qa"


# ---------------------------------------------------------------------------
# Exp06 skill-override controls. The routed skill is the only thing these touch;
# everything downstream (source filter, prompt, slots) follows from it. The
# default `skill_override="none"` bypasses ALL of this so the produced records
# are byte-identical to the pre-Exp06 pipeline.
#
# NOTE: the *current* routers (infer_skill / infer_schema_v1_skill) read gold
# task fields (source / task_type / input_type), so the deployed-as-is routing
# already IS the gold-field (oracle) skill. `--skill-override predicted` swaps in
# a question-ONLY router that is what a real deployment could use.
SKILL_NAMES_V1: tuple[str, ...] = tuple(SKILL_SCHEMA_V1.keys())
SKILL_NAMES_LEGACY: tuple[str, ...] = tuple(LEGACY_SKILL_CONFIG.keys())

# Fixed plausible-but-wrong confusion map: each gold skill -> one other skill
# (deterministic, and never the identity), for the wrong-skill sensitivity control.
CONFUSION_MAP_V1: dict[str, str] = {
    "fda_label_factual": "fda_label_multihop",
    "fda_label_multihop": "fda_label_factual",
    "molecule_property_numeric": "molecule_description",
    "molecule_description": "molecule_property_numeric",
    "molecule_design": "molecule_description",
    "biomedical_open_qa": "fda_label_factual",
}
CONFUSION_MAP_LEGACY: dict[str, str] = {
    "fda_label_qa": "biomedical_open_qa",
    "molecule_property": "molecule_description",
    "molecule_description": "molecule_property",
    "molecule_design": "molecule_description",
    "biomedical_open_qa": "fda_label_qa",
}


def skill_names(schema_version: str) -> tuple[str, ...]:
    return SKILL_NAMES_V1 if schema_version == "v1" else SKILL_NAMES_LEGACY


def gold_field_skill(record: dict[str, Any], schema_version: str) -> str:
    """The current gold-field router (reads source/task_type/input_type)."""
    if schema_version == "v1":
        return infer_schema_v1_skill(record)
    return infer_skill(record)


def predict_skill_question_only(record: dict[str, Any], schema_version: str = "v1") -> str:
    """Deployable router: infer the skill from ONLY ``record["question"]`` text.

    Unlike ``gold_field_skill``, this reads NO gold task fields (source,
    task_type, input_type, decoded_smiles), so it is safe to deploy. Cue-based:
    lexical patterns in the question decide the skill.
    """
    question = norm_text(record.get("question", ""))
    v1 = schema_version == "v1"
    if any(cue in question for cue in ("design", "generate a molecule", "synthesize", "create a molecule", "propose a molecule")):
        return "molecule_design"
    if any(cue in question for cue in ("logp", "molecular weight", "homo-lumo", "how many", "predict the", "solubility", "what is the value")):
        return "molecule_property_numeric" if v1 else "molecule_property"
    if any(cue in question for cue in ("describe", "description", "what type of molecule", "what kind of molecule")):
        return "molecule_description"
    if any(cue in question for cue in ("contraindicat", "indicat", "dosage", "dose", "adverse", "warning", "label", "boxed", "black box")):
        if any(cue in question for cue in (" and ", " both ", "combined", "as well as", "in addition", "along with")):
            return "fda_label_multihop" if v1 else "fda_label_qa"
        return "fda_label_factual" if v1 else "fda_label_qa"
    return "biomedical_open_qa"


def random_skill(record: dict[str, Any], schema_version: str) -> str:
    """Deterministic pseudo-random skill via sha1(id) mod n_skills (negative control)."""
    names = skill_names(schema_version)
    digest = hashlib.sha1(str(record.get("id", "")).encode("utf-8")).hexdigest()
    return names[int(digest, 16) % len(names)]


def wrong_skill(record: dict[str, Any], schema_version: str) -> str:
    """Deterministic plausible-but-wrong skill from a fixed confusion map."""
    gold = gold_field_skill(record, schema_version)
    cmap = CONFUSION_MAP_V1 if schema_version == "v1" else CONFUSION_MAP_LEGACY
    fallback = "biomedical_open_qa" if gold != "biomedical_open_qa" else skill_names(schema_version)[0]
    return cmap.get(gold, fallback)


def apply_skill_override(record: dict[str, Any], schema_version: str, skill_override: str) -> str:
    if skill_override == "oracle":
        return gold_field_skill(record, schema_version)
    if skill_override == "predicted":
        return predict_skill_question_only(record, schema_version)
    if skill_override == "random":
        return random_skill(record, schema_version)
    if skill_override == "wrong":
        return wrong_skill(record, schema_version)
    return gold_field_skill(record, schema_version)


def infer_record_skill(record: dict[str, Any], schema_version: str, skill_override: str = "none") -> str:
    if skill_override and skill_override != "none":
        return apply_skill_override(record, schema_version, skill_override)
    if schema_version == "v1":
        return infer_schema_v1_skill(record)
    return infer_skill(record)


def skill_config(skill: str, schema_version: str) -> dict[str, Any]:
    if schema_version == "v1":
        return SKILL_SCHEMA_V1.get(skill, SKILL_SCHEMA_V1["biomedical_open_qa"])
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
    ablate: set[str] | None = None,
    skill_override: str = "none",
) -> list[dict[str, Any]]:
    ablate = ablate or set()
    skill = infer_record_skill(record, schema_version, skill_override)
    # bm25_skillrouted = pure bm25 scoring (no alias boost) + skill source filter,
    # k NOT capped -> isolates *routing* from evidence-budget vs plain bm25.
    # Ablating "source_routing" drops the skill source filter so retrieval runs
    # unconstrained over the whole KB, like the uniform methods.
    use_skill_filter = method in {"skill_hybrid", "bm25_skillrouted"} and "source_routing" not in ablate
    if schema_version == "v1" and method != "bm25_skillrouted":
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
    # Minimal baseline prompt: instruction + the question's own label/context +
    # the retrieved evidence + question/answer. No skill/method/task metadata, so
    # every retrieval baseline shares an identical prompt and the ONLY thing that
    # differs is the content of "Retrieved KB evidence" (i.e. which retriever ran).
    lines = [
        "You are answering a drug and molecular QA task with external knowledge.",
        "",
        "Use the provided information to answer the question. If the answer is not available, respond: Information not found!",
        "",
        "Answer in the same style as the gold answer.",
    ]
    if record.get("input_molecule_or_context"):
        label = "Label/context evidence" if record.get("source") == "fdarxbench" else "Input"
        lines.extend(["", f"{label}:", str(record["input_molecule_or_context"])])
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
    ablate: set[str] | None = None,
) -> str:
    # Same instruction + known information + Question/Answer as the baseline prompt.
    # The skill layer adds three things: the routed skill, its ordered solving steps,
    # the answer format, and skill-based (source-constrained) retrieved knowledge in
    # place of the baseline's generic evidence dump.
    #
    # `ablate` (Exp04/RQ5) REMOVES individual schema components; the empty/None
    # default keeps every component and is byte-identical to the un-ablated prompt.
    ablate = ablate or set()
    config = skill_config(skill, "v1")
    lines = [
        "You are answering a drug and molecular QA task with external knowledge.",
        "",
        "Use the provided information to answer the question. If the answer is not available, respond: Information not found!",
        "",
        "Answer in the same style as the gold answer.",
    ]
    schema_lines: list[str] = []
    if "skill_id" not in ablate:
        schema_lines.append(f"Skill: {skill}")
    if "solving_steps" not in ablate:
        schema_lines.append("How to solve:")
        for i, step in enumerate(config.get("steps", ()), 1):
            schema_lines.append(f"  {i}. {step}")
    if "answer_format" not in ablate:
        schema_lines.append(f"Answer format: {config['answer_format']}")
    if schema_lines:
        lines.append("")
        lines.extend(schema_lines)
    if record.get("input_molecule_or_context"):
        lines.extend(["", "Known information:", str(record["input_molecule_or_context"])])
    # Ablating "evidence_org" falls back to the plain baseline evidence dump header.
    evidence_header = "Retrieved KB evidence:" if "evidence_org" in ablate else "Skill-based retrieved knowledge:"
    lines.extend(["", evidence_header, evidence_block(evidence, max_chars)])
    lines.extend(["", f"Question: {record.get('question', '')}", "Answer:"])
    return "\n".join(lines)


def augment_record(
    record: dict[str, Any],
    evidence: list[dict[str, Any]],
    method: str,
    max_chars: int,
    schema_version: str,
    ablate: set[str] | None = None,
    skill_override: str = "none",
) -> dict[str, Any]:
    ablate = ablate or set()
    new_record = dict(record)
    skill = infer_record_skill(record, schema_version, skill_override)
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
    # Only record the ablated component set when non-empty, so an un-ablated run
    # (ablate=None/empty) emits a byte-identical skill_schema.
    if ablate:
        skill_schema["ablate"] = sorted(ablate)
    new_record["skill_schema"] = skill_schema
    new_record["retrieved_kb_evidence"] = evidence
    if schema_version == "v1":
        new_record["evidence_slots"] = evidence_slot_fills(new_record, skill, evidence)
        new_record["slot_fill_status"] = {
            "all_required_filled": all(item["filled"] for item in new_record["evidence_slots"].values()),
            "filled_slots": [slot for slot, item in new_record["evidence_slots"].items() if item["filled"]],
            "missing_slots": [slot for slot, item in new_record["evidence_slots"].items() if not item["filled"]],
        }
        new_record["completed_skill_prompt"] = build_completed_skill_prompt(new_record, skill, method, evidence, max_chars, ablate)
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
    ablate = parse_ablate(args.ablate)
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
            ablate,
            args.skill_override,
        )
        augmented.append(
            augment_record(
                record,
                evidence,
                args.method,
                args.max_prompt_evidence_chars,
                args.schema_version,
                ablate,
                args.skill_override,
            )
        )
        skill_counts[infer_record_skill(record, args.schema_version, args.skill_override)] += 1
        evidence_counts["with_evidence" if evidence else "without_evidence"] += 1
    write_jsonl(Path(args.output), augmented)
    summary = {
        "input": args.input,
        "output": args.output,
        "kb_dir": args.kb_dir,
        "method": args.method,
        "schema_version": args.schema_version,
        "skill_override": args.skill_override,
        "limit": args.limit,
        "records": len(records),
        "facts_loaded": len(facts),
        "skill_counts": dict(skill_counts),
        "evidence_counts": dict(evidence_counts),
    }
    if ablate:
        summary["ablate"] = sorted(ablate)
    summary_path = Path(args.output).with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
