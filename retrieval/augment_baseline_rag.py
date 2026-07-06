#!/usr/bin/env python
"""Attach RAG-baseline retrieval evidence to normalized QA JSONL records.

This mirrors `augment_with_skill_kb.py` (same folder) exactly for everything except the
retriever: it reuses that module's KB loader, BM25 scorer, prompt builder, and
record builder (imported, not copied), so the emitted `prompt` / `messages` /
`skill_schema` are byte-identical to the existing bm25/kg/hybrid runs. The ONLY
difference across methods is which facts land in `retrieved_kb_evidence`.

Methods:
  dense_bge     dense bi-encoder retrieval (BAAI/bge-large-en-v1.5)
  dense_medcpt  dense bi-encoder retrieval (ncbi/MedCPT, biomedical)
  rrf           reciprocal-rank fusion of BM25 (+) dense (BGE)
  rerank        RRF first stage -> MedCPT cross-encoder reranker
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/playpen-jfs/jesse/drug_microbiome")  # consolidated-folder anchor (was: parent.parent)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # retrievers.py + augment_with_skill_kb.py sit beside this file

# Reuse the proposed method's machinery so prompts stay identical across methods.
import augment_with_skill_kb as base  # noqa: E402

from retrievers import (  # noqa: E402
    DenseIndex,
    MedCPTCrossEncoder,
    build_encoder,
    fact_to_evidence,
    reciprocal_rank_fusion,
)

DENSE_FOR_METHOD = {
    "dense_bge": "bge",
    "dense_medcpt": "medcpt",
    "rrf": "bge",
    "rerank": "bge",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", required=True, choices=list(DENSE_FOR_METHOD))
    p.add_argument("--kb-dir", default=str(REPO / "outputs/knowledge_bank/unified"))
    p.add_argument("--emb-dir", default=str(REPO / "outputs/knowledge_bank/embeddings"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--first-stage-k", type=int, default=50, help="Candidates before RRF/rerank.")
    p.add_argument("--max-candidate-facts", type=int, default=20000)
    p.add_argument("--max-prompt-evidence-chars", type=int, default=3500)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_embeddings(emb_dir: Path, encoder: str, fact_id_to_idx: dict[str, int], n_facts: int) -> np.ndarray:
    """Load the cached KB matrix and reorder its rows to match load_kb's fact order."""
    sub = emb_dir / encoder
    matrix = np.load(sub / "facts.f16.npy").astype(np.float32)
    ids = json.loads((sub / "ids.json").read_text())
    if len(ids) != n_facts:
        raise SystemExit(
            f"Embedding count {len(ids)} != KB fact count {n_facts} for encoder={encoder}. "
            f"Rebuild with build_kb_embeddings.py against the current KB."
        )
    # Reorder rows so row i corresponds to facts[i] (load_kb order).
    reordered = np.empty_like(matrix)
    for row, fact_id in enumerate(ids):
        reordered[fact_id_to_idx[fact_id]] = matrix[row]
    return reordered


def bm25_rank(record: dict[str, Any], facts, idf, token_index, avgdl, first_stage_k: int) -> list[int]:
    """Pure-lexical BM25 ranking (no alias boost) -> ranked fact indices."""
    raw_terms = base.tokens(base.query_text(record))
    query_terms = base.informative_terms(raw_terms, idf, token_index)
    candidate_idxs: set[int] = set()
    for term in query_terms:
        candidate_idxs.update(token_index.get(term, []))
    scored = []
    for idx in candidate_idxs:
        score = base.bm25_score(query_terms, facts[idx].get("tokens", []), idf, avgdl)
        if score > 0:
            scored.append((score, idx))
    scored.sort(reverse=True)
    return [idx for _, idx in scored[: first_stage_k * 4]]


def main() -> None:
    args = parse_args()
    method = args.method
    facts, alias_index, idf, token_index = base.load_kb(Path(args.kb_dir))
    fact_id_to_idx = {fact["id"]: idx for idx, fact in enumerate(facts)}
    avgdl = sum(len(f.get("tokens", [])) for f in facts) / max(len(facts), 1)
    records = base.iter_jsonl(Path(args.input), args.limit)

    # Dense stage (all methods here use a dense encoder as at least one stage).
    dense_name = DENSE_FOR_METHOD[method]
    print(f"[augment] method={method} dense={dense_name} records={len(records)}", flush=True)
    kb_matrix = load_embeddings(Path(args.emb_dir), dense_name, fact_id_to_idx, len(facts))
    dense_index = DenseIndex(kb_matrix)
    encoder = build_encoder(dense_name, device=args.device)

    queries = [base.query_text(r) for r in records]
    q_vecs = encoder.encode(queries, is_query=True)

    needs_lexical = method in {"rrf", "rerank"}
    cross = MedCPTCrossEncoder(device=args.device) if method == "rerank" else None

    # First-stage dense candidates for everyone.
    dense_scores, dense_idxs = dense_index.search(q_vecs, max(args.first_stage_k, args.top_k))

    augmented = []
    skill_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    for i, record in enumerate(records):
        d_idx = [int(x) for x in dense_idxs[i] if x >= 0]
        d_score = {int(idx): float(s) for idx, s in zip(dense_idxs[i], dense_scores[i]) if idx >= 0}

        if method in {"dense_bge", "dense_medcpt"}:
            chosen = d_idx[: args.top_k]
            evidence = [fact_to_evidence(facts[idx], d_score.get(idx, 0.0)) for idx in chosen]
        else:
            lex_idx = bm25_rank(record, facts, idf, token_index, avgdl, args.first_stage_k)
            fused = reciprocal_rank_fusion([d_idx[: args.first_stage_k], lex_idx[: args.first_stage_k]])
            ranked = sorted(fused, key=lambda idx: fused[idx], reverse=True)
            if method == "rrf":
                chosen = ranked[: args.top_k]
                evidence = [fact_to_evidence(facts[idx], fused[idx]) for idx in chosen]
            else:  # rerank
                cand = ranked[: args.first_stage_k]
                passages = [str(facts[idx].get("text") or "") for idx in cand]
                ce_scores = cross.score(base.query_text(record), passages)
                order = np.argsort(-ce_scores)[: args.top_k]
                evidence = [fact_to_evidence(facts[cand[j]], float(ce_scores[j])) for j in order]

        new_record = base.augment_record(record, evidence, method, args.max_prompt_evidence_chars, "legacy")
        augmented.append(new_record)
        skill_counts[base.infer_record_skill(record, "legacy")] += 1
        evidence_counts["with_evidence" if evidence else "without_evidence"] += 1
        if (i + 1) % 250 == 0:
            print(f"[augment] {i + 1}/{len(records)}", flush=True)

    base.write_jsonl(Path(args.output), augmented)
    summary = {
        "input": args.input,
        "output": args.output,
        "method": method,
        "dense_encoder": dense_name,
        "reranker": "medcpt_cross" if method == "rerank" else None,
        "records": len(records),
        "facts_loaded": len(facts),
        "top_k": args.top_k,
        "first_stage_k": args.first_stage_k,
        "skill_counts": dict(skill_counts),
        "evidence_counts": dict(evidence_counts),
    }
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
