#!/usr/bin/env python
"""Exp02 oracle_source augmenter -- the schema's best-case source-routing UPPER BOUND.

`oracle_source` = the strong dense retriever (dense_bge, BAAI/bge-large-en-v1.5)
run over a candidate pool HARD-RESTRICTED, per record, to the KB sources allowed
for the record's GOLD-labelled skill (`infer_schema_v1_skill`, which reads the gold
`source`/`task_type` fields). The dense top-k is taken over the *entire* allowed-
source subset of the KB, so it is the best answer the schema's source routing could
ever give -- an oracle. This is deliberately NOT deployable: it consumes gold task
labels. The deployed SkillRAG must instead route on a question-only *predicted*
skill.

Relationship to the existing `dense_bge_skillrouted`
(`retrieval/augment_baseline_rag.py`):
  * BOTH route on the same gold-field rule (`infer_schema_v1_skill`) and BOTH use
    the same dense retriever + evidence budget -- so neither is deployable as-is.
  * `dense_bge_skillrouted` is an APPROXIMATION: it dense-searches the whole KB for
    a fixed 300-deep pool, then *post-filters* that pool by allowed sources and
    keeps top-k. Allowed-source facts that never made the top-300 are lost.
  * `oracle_source` restricts the pool to the allowed sources *before* ranking and
    searches that subset exhaustively, so the top-k is guaranteed to be the true
    dense-best allowed-source facts across the WHOLE KB. It is therefore an upper
    bound ON TOP OF `dense_bge_skillrouted`: `dense_bge_skillrouted <= oracle_source`
    by construction (same retriever, strictly larger effective candidate set).

Everything except which facts land in `retrieved_kb_evidence` is delegated to the
shared pipeline (`augment_with_skill_kb.augment_record`), so the emitted
`prompt`/`messages`/`skill_schema` are byte-identical in shape to every other
method. This script is additive: it imports the baseline modules and touches none
of them.

Method label written into records: `oracle_source`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path("/playpen-jfs/jesse/drug_microbiome")  # consolidated-folder anchor
RETRIEVAL_DIR = Path(__file__).resolve().parents[2] / "retrieval"
sys.path.insert(0, str(RETRIEVAL_DIR))  # so `augment_with_skill_kb`, `retrievers` resolve

import augment_with_skill_kb as base  # noqa: E402  (stdlib-only module)

DENSE_ENCODER = "bge"  # oracle rides the strong dense retriever, matching dense_bge


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kb-dir", default=str(REPO / "outputs/knowledge_bank/unified"))
    p.add_argument("--emb-dir", default=str(REPO / "outputs/knowledge_bank/embeddings"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--top-k", type=int, default=8, help="Evidence budget, same default as dense_bge.")
    p.add_argument("--schema-version", default="legacy", choices=["legacy", "v1"],
                   help="Passed to augment_record for prompt assembly (routing always uses gold v1 skill).")
    p.add_argument("--max-candidate-facts", type=int, default=20000)  # accepted for CLI parity; unused (exhaustive subset)
    p.add_argument("--max-prompt-evidence-chars", type=int, default=3500)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def allowed_sources_for(record: dict[str, Any], skill_fn: Callable[[dict[str, Any]], str]) -> tuple[str, frozenset[str]]:
    """Gold skill for a record and the frozenset of KB sources that skill allows."""
    skill = skill_fn(record)
    allowed = base.SKILL_SCHEMA_V1.get(skill, {}).get("sources") or set()
    return skill, frozenset(allowed)


def oracle_retrieve(
    records: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    q_vecs: np.ndarray,
    kb_matrix: np.ndarray,
    top_k: int,
    skill_fn: Callable[[dict[str, Any]], str] | None = None,
) -> tuple[list[list[dict[str, Any]]], list[tuple[str, frozenset[str]]]]:
    """Per-record dense top-k restricted to the gold-skill's allowed KB sources.

    Records are grouped by their allowed-source SET (only a handful of distinct sets
    exist across the 6 skills), and one exhaustive dense sub-index is built per set,
    so the returned evidence is the exact dense-best allowed-source facts. Pure numpy
    + faiss/DenseIndex -- no encoder, no GPU -- so it is unit-testable offline.
    """
    from retrievers import DenseIndex, fact_to_evidence  # local import: keeps --help torch-free

    if skill_fn is None:
        skill_fn = base.infer_schema_v1_skill
    q_vecs = np.asarray(q_vecs, dtype=np.float32)

    routing = [allowed_sources_for(r, skill_fn) for r in records]
    groups: dict[frozenset[str], list[int]] = {}
    for i, (_skill, allowed) in enumerate(routing):
        groups.setdefault(allowed, []).append(i)

    evidence_per_rec: list[list[dict[str, Any]]] = [[] for _ in records]
    for allowed, rec_idxs in groups.items():
        if allowed:
            sub_to_global = np.array(
                [i for i, f in enumerate(facts) if f.get("source") in allowed], dtype=np.int64
            )
        else:  # skill with no declared sources (should not happen for SKILL_SCHEMA_V1): whole KB
            sub_to_global = np.arange(len(facts), dtype=np.int64)
        if sub_to_global.size == 0:
            continue
        sub_index = DenseIndex(kb_matrix[sub_to_global])
        k = min(top_k, int(sub_to_global.size))
        scores, local_idxs = sub_index.search(q_vecs[rec_idxs], k)
        for row, rec_i in enumerate(rec_idxs):
            evidence: list[dict[str, Any]] = []
            for local, score in zip(local_idxs[row], scores[row]):
                local = int(local)
                if local < 0:
                    continue
                evidence.append(fact_to_evidence(facts[int(sub_to_global[local])], float(score)))
            evidence_per_rec[rec_i] = evidence
    return evidence_per_rec, routing


def main() -> None:
    args = parse_args()
    import augment_baseline_rag as dense_base  # noqa: E402  (reuse its embedding loader)
    from retrievers import build_encoder  # noqa: E402

    facts, _alias_index, _idf, _token_index = base.load_kb(Path(args.kb_dir))
    fact_id_to_idx = {fact["id"]: idx for idx, fact in enumerate(facts)}
    records = base.iter_jsonl(Path(args.input), args.limit)

    print(f"[oracle] method=oracle_source dense={DENSE_ENCODER} records={len(records)}", flush=True)
    kb_matrix = dense_base.load_embeddings(Path(args.emb_dir), DENSE_ENCODER, fact_id_to_idx, len(facts))

    encoder = build_encoder(DENSE_ENCODER, device=args.device)
    queries = [base.query_text(r) for r in records]
    q_vecs = encoder.encode(queries, is_query=True)

    evidence_per_rec, routing = oracle_retrieve(records, facts, q_vecs, kb_matrix, args.top_k)

    augmented = []
    skill_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    for i, record in enumerate(records):
        evidence = evidence_per_rec[i]
        new_record = base.augment_record(
            record, evidence, "oracle_source", args.max_prompt_evidence_chars, args.schema_version
        )
        augmented.append(new_record)
        skill_counts[routing[i][0]] += 1
        for item in evidence:
            source_counts[str(item.get("source"))] += 1
        evidence_counts["with_evidence" if evidence else "without_evidence"] += 1
        if (i + 1) % 250 == 0:
            print(f"[oracle] {i + 1}/{len(records)}", flush=True)

    base.write_jsonl(Path(args.output), augmented)
    summary = {
        "input": args.input,
        "output": args.output,
        "method": "oracle_source",
        "dense_encoder": DENSE_ENCODER,
        "routing_rule": "infer_schema_v1_skill (GOLD labels -> allowed sources) -- UPPER BOUND",
        "records": len(records),
        "facts_loaded": len(facts),
        "top_k": args.top_k,
        "gold_skill_counts": dict(skill_counts),
        "retrieved_source_counts": dict(source_counts),
        "evidence_counts": dict(evidence_counts),
    }
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
