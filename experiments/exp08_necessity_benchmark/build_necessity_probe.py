#!/usr/bin/env python
"""Exp08 — Necessity probe construction SCAFFOLD (🔴 new-data).

Builds *candidate* items for the source-ambiguous, retrieval-sensitive necessity
benchmark defined in SPEC.md. Given the unified KB `facts.jsonl`, it:

  (a) finds "collision" aliases/entities that appear across >=2 KB sources
      (from each fact's `aliases`/`entities`, with a token fallback),
  (b) forms candidate items where the gold answer lives in source A while a
      lexically-similar decoy fact exists in a *different* source B,
  (c) emits candidate QA records in the STANDARD normalized schema so the
      existing augment/train/eval pipeline can consume them, and
  (d) writes `candidates.jsonl` + `provenance.jsonl` (A/B source pair per item).

IMPORTANT — this is MECHANICAL ASSEMBLY ONLY. The emitted `gold_answer` is a
PROVISIONAL value lifted from source A's fact; it is flagged
`metadata.validation.validated = false`. Human validation of gold answers and of
requirements R1-R4 (SPEC.md §5) is REQUIRED and NOT YET DONE. Nothing here
fabricates validated gold.

Runs offline on a tiny slice via `--max-facts` (KB read cap) and `--limit`
(candidate cap). Does not require any `outputs/` artifact to run its unit tests:
`build_probe()` operates on an in-memory facts list.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Import the shipped skill schema so "allowed sources" here never drift from the
# retriever's definition (mirrors experiments/exp01_source_heterogeneity).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from retrieval.augment_with_skill_kb import SKILL_SCHEMA_V1  # noqa: E402

KB_SOURCES = ["primekg", "drugchat_chembl", "drugchat_pubchem", "fdarxbench_label"]

# Default gold-source priority: most *restrictive* intended skill first so the
# generated pairs are discriminative (biomedical_open_qa / primekg is a catch-all
# whose allowed set covers every source, so it can never exclude a decoy — it is
# tried last). Override with --gold-sources.
DEFAULT_GOLD_SOURCES = ["fdarxbench_label", "drugchat_pubchem", "drugchat_chembl", "primekg"]

# Which skill a probe item is intended to require, keyed by its gold source A.
# Used ONLY at construction time to pick discriminative decoys and to define the
# oracle / wrong-skill scopes; never fed to the deployed router (SPEC.md §3).
SOURCE_TO_INTENDED_SKILL = {
    "fdarxbench_label": "fda_label_factual",
    "primekg": "biomedical_open_qa",
    "drugchat_chembl": "molecule_description",
    "drugchat_pubchem": "molecule_description",
}

# Relation -> question template (question surfaces the collision alias). Phrasing
# is provisional; the curator must confirm it is natural and source-ambiguous
# (SPEC.md §5, criterion 5).
QUESTION_TEMPLATES = {
    "label_context": "According to its regulatory label, what does {alias} specify or warn about?",
    "indication": "What is {alias} indicated to treat?",
    "off-label use": "What off-label use has been reported for {alias}?",
    "contraindication": "What is a contraindication associated with {alias}?",
    "drug_drug": "What drug-drug interaction involves {alias}?",
    "drug_effect": "What effect is associated with {alias}?",
    "drug_protein": "Which protein target does {alias} act on?",
    "disease_protein": "Which protein is associated with {alias}?",
    "instruction_fact": "What property or description is reported for {alias}?",
}
DEFAULT_QUESTION = "Regarding {alias}, what does the knowledge base report for the '{relation}' relation?"

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.'-]*")

VALIDATION_TODO = (
    "TODO(validation): human must verify gold_answer correctness, R1 (no answer "
    "leak), R2 (both sources retrievable), R3 (decoy answer is wrong), and "
    "question naturalness/source-ambiguity before this item enters the benchmark "
    "(see SPEC.md §5). gold_answer is PROVISIONAL and UNVALIDATED."
)


def norm_alias(text: Any) -> str:
    return " ".join(str(text).lower().split())


def tokens(text: Any) -> set[str]:
    return set(TOKEN_RE.findall(str(text).lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def entity_strings(fact: dict[str, Any]) -> list[str]:
    """Collect candidate alias strings from a fact's `aliases` and `entities`."""
    out: list[str] = []
    for a in fact.get("aliases") or []:
        out.append(str(a))
    for e in fact.get("entities") or []:
        if isinstance(e, dict):
            out.append(str(e.get("name") or e.get("id") or e.get("text") or ""))
        else:
            out.append(str(e))
    return [s for s in out if s.strip()]


def load_facts(facts_path: Path, max_facts: int = 0) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    with facts_path.open(encoding="utf-8") as f:
        for line in f:
            if max_facts and len(facts) >= max_facts:
                break
            if line.strip():
                facts.append(json.loads(line))
    return facts


def build_collision_index(
    facts: list[dict[str, Any]], min_alias_len: int = 3
) -> dict[str, dict[str, list[int]]]:
    """alias -> {source: [fact_idx, ...]} for aliases spanning >=2 distinct sources."""
    index: dict[str, dict[str, list[int]]] = {}
    for i, fact in enumerate(facts):
        source = str(fact.get("source", ""))
        seen: set[str] = set()
        for raw in entity_strings(fact):
            na = norm_alias(raw)
            if len(na) < min_alias_len or na.isdigit() or na in seen:
                continue
            seen.add(na)
            index.setdefault(na, {}).setdefault(source, []).append(i)
    return {alias: smap for alias, smap in index.items() if len(smap) >= 2}


def provisional_gold_answer(fact: dict[str, Any]) -> str:
    """Best-effort UNVALIDATED gold value lifted from source A's fact."""
    meta = fact.get("metadata") or {}
    for key in ("object", "tail", "value", "answer"):
        if meta.get(key):
            return str(meta[key]).strip()
    return str(fact.get("text", "")).strip()


def make_question(alias: str, gold_fact: dict[str, Any]) -> str:
    relation = str(gold_fact.get("relation", "") or "")
    template = QUESTION_TEMPLATES.get(relation, DEFAULT_QUESTION)
    return template.format(alias=alias, relation=relation or "fact")


def build_prompt(question: str) -> str:
    """Neutral base prompt (no evidence, empty input): the augmenter fills in
    retrieved evidence later. Input is empty so no answer leaks (R1)."""
    return "\n".join(
        [
            "You are answering a drug and molecular QA task with external knowledge.",
            "",
            "Use the provided information to answer the question. If the answer is not available, respond: Information not found!",
            "",
            f"Question: {question}",
            "Answer:",
        ]
    )


def select_pair(
    alias: str,
    smap: dict[str, list[int]],
    facts: list[dict[str, Any]],
    gold_sources: list[str],
    decoy_sources: list[str],
    min_sim: float,
    require_discriminative: bool,
) -> dict[str, Any] | None:
    """Pick (gold fact in source A, lexically-similar decoy fact in source B).

    Returns a dict describing the pair, or None if no qualifying decoy exists.
    """
    gold_source = next((s for s in gold_sources if smap.get(s)), None)
    if gold_source is None:
        return None
    gold_idx = smap[gold_source][0]
    gold_fact = facts[gold_idx]
    intended_skill = SOURCE_TO_INTENDED_SKILL.get(gold_source, "biomedical_open_qa")
    allowed = set(SKILL_SCHEMA_V1[intended_skill]["sources"])
    gold_tokens = tokens(gold_fact.get("text", ""))

    best: tuple[tuple[bool, float], str, int, float] | None = None
    for source, idxs in smap.items():
        if source == gold_source:
            continue
        if decoy_sources and source not in decoy_sources:
            continue
        if require_discriminative and source in allowed:
            continue
        for di in idxs:
            decoy_fact = facts[di]
            sim = jaccard(gold_tokens, tokens(decoy_fact.get("text", "")))
            if sim < min_sim:
                continue
            distinct_relation = decoy_fact.get("relation") != gold_fact.get("relation")
            score = (distinct_relation, sim)
            if best is None or score > best[0]:
                best = (score, source, di, sim)
    if best is None:
        return None

    _, decoy_source, decoy_idx, sim = best
    decoy_fact = facts[decoy_idx]
    return {
        "alias": alias,
        "gold_source": gold_source,
        "gold_fact": gold_fact,
        "decoy_source": decoy_source,
        "decoy_fact": decoy_fact,
        "intended_skill": intended_skill,
        "surface_ambiguity_score": round(sim, 4),
        "decoy_excluded_by_intended_skill": decoy_source not in allowed,
        "distinct_relation": decoy_fact.get("relation") != gold_fact.get("relation"),
    }


def build_record(pair: dict[str, Any], index: int, source_path: str) -> dict[str, Any]:
    """Emit one candidate in the STANDARD normalized QA schema."""
    alias = pair["alias"]
    gold_fact = pair["gold_fact"]
    decoy_fact = pair["decoy_fact"]
    question = make_question(alias, gold_fact)
    gold_answer = provisional_gold_answer(gold_fact)
    prompt = build_prompt(question)
    item_id = f"necessity_{index:05d}"
    record = {
        "id": item_id,
        "source": "necessity_probe",
        "source_path": source_path,
        "source_index": index,
        "question": question,
        "input_molecule_or_context": "",  # R1: answer must not live in the input
        "input_type": "none",
        "decoded_smiles": None,
        "gold_answer": gold_answer,  # PROVISIONAL / UNVALIDATED (see validation)
        "task_type": "necessity_probe",
        "drug_name": alias,
        "metadata": {
            "intended_skill": pair["intended_skill"],
            "gold_source": pair["gold_source"],
            "decoy_source": pair["decoy_source"],
            "gold_fact_id": gold_fact.get("id"),
            "decoy_fact_id": decoy_fact.get("id"),
            "gold_relation": gold_fact.get("relation"),
            "decoy_relation": decoy_fact.get("relation"),
            "collision_alias": alias,
            "surface_ambiguity_score": pair["surface_ambiguity_score"],
            "decoy_excluded_by_intended_skill": pair["decoy_excluded_by_intended_skill"],
            "distinct_relation": pair["distinct_relation"],
            "validation": {
                "validated": False,
                "gold_answer_is_provisional": True,
                "todo": VALIDATION_TODO,
            },
        },
        "retrieved_kg_evidence": [],
        "retrieved_molecule_evidence": [],
        "prompt": prompt,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": gold_answer},
        ],
    }
    return record


def build_provenance(pair: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """A/B source pair for one item; kept OUT of every eval arm."""
    return {
        "id": record["id"],
        "collision_alias": pair["alias"],
        "gold_source": pair["gold_source"],
        "decoy_source": pair["decoy_source"],
        "gold_fact_id": pair["gold_fact"].get("id"),
        "decoy_fact_id": pair["decoy_fact"].get("id"),
        "gold_relation": pair["gold_fact"].get("relation"),
        "decoy_relation": pair["decoy_fact"].get("relation"),
        "intended_skill": pair["intended_skill"],
        "surface_ambiguity_score": pair["surface_ambiguity_score"],
        "decoy_excluded_by_intended_skill": pair["decoy_excluded_by_intended_skill"],
        "distinct_relation": pair["distinct_relation"],
        "validated": False,
    }


def build_probe(
    facts: list[dict[str, Any]],
    gold_sources: list[str] | None = None,
    decoy_sources: list[str] | None = None,
    min_sim: float = 0.1,
    limit: int = 0,
    require_discriminative: bool = True,
    source_path: str = "experiments/exp08_necessity_benchmark/candidates.jsonl",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Pure, IO-free probe construction over an in-memory facts list.

    Returns (candidate_records, provenance_records, stats).
    """
    gold_sources = gold_sources or list(DEFAULT_GOLD_SOURCES)
    decoy_sources = decoy_sources or list(KB_SOURCES)
    collisions = build_collision_index(facts)

    candidates: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for alias in sorted(collisions):
        if limit and len(candidates) >= limit:
            break
        pair = select_pair(
            alias,
            collisions[alias],
            facts,
            gold_sources,
            decoy_sources,
            min_sim,
            require_discriminative,
        )
        if pair is None:
            continue
        record = build_record(pair, len(candidates), source_path)
        candidates.append(record)
        provenance.append(build_provenance(pair, record))

    stats = {
        "n_facts": len(facts),
        "n_collision_aliases": len(collisions),
        "n_candidates": len(candidates),
        "n_discriminative": sum(
            1 for p in provenance if p["decoy_excluded_by_intended_skill"]
        ),
        "gold_source_counts": _counts(p["gold_source"] for p in provenance),
        "decoy_source_counts": _counts(p["decoy_source"] for p in provenance),
        "validated": 0,
        "note": "All gold answers are PROVISIONAL/UNVALIDATED; run curation (SPEC.md §5).",
    }
    return candidates, provenance, stats


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kb-dir", default="outputs/knowledge_bank/unified")
    p.add_argument("--facts", default="", help="Override path to facts.jsonl (else <kb-dir>/facts.jsonl).")
    p.add_argument("--output-dir", default="experiments/exp08_necessity_benchmark/candidates")
    p.add_argument("--max-facts", type=int, default=0, help="Cap KB facts read (tiny-slice runs).")
    p.add_argument("--limit", type=int, default=0, help="Cap emitted candidate items.")
    p.add_argument("--gold-sources", default=",".join(DEFAULT_GOLD_SOURCES), help="Comma list, priority order for gold source A.")
    p.add_argument("--decoy-sources", default=",".join(KB_SOURCES), help="Comma list of eligible decoy sources.")
    p.add_argument("--min-decoy-similarity", type=float, default=0.1, help="Min token-Jaccard(gold,decoy) for a decoy.")
    p.add_argument(
        "--allow-nondiscriminative",
        action="store_true",
        help="Permit decoys whose source is in the intended skill's allowed set (weakens R4).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    facts_path = Path(args.facts) if args.facts else Path(args.kb_dir) / "facts.jsonl"
    if not facts_path.exists():
        raise SystemExit(
            f"[exp08] facts file not found: {facts_path}\n"
            "        Build the unified KB first (kb/build_unified_kb.py) or pass --facts."
        )
    facts = load_facts(facts_path, args.max_facts)
    candidates, provenance, stats = build_probe(
        facts,
        gold_sources=[s for s in args.gold_sources.split(",") if s],
        decoy_sources=[s for s in args.decoy_sources.split(",") if s],
        min_sim=args.min_decoy_similarity,
        limit=args.limit,
        require_discriminative=not args.allow_nondiscriminative,
    )
    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "candidates.jsonl", candidates)
    write_jsonl(out_dir / "provenance.jsonl", provenance)
    (out_dir / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[exp08] {stats['n_candidates']} candidates from {stats['n_collision_aliases']} "
          f"collision aliases -> {out_dir}")
    print(f"[exp08] {stats['n_discriminative']} discriminative; ALL gold PROVISIONAL/UNVALIDATED "
          "-> run curation (SPEC.md §5).")


if __name__ == "__main__":
    main()
