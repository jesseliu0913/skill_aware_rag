#!/usr/bin/env python
"""Encode the unified KB once per dense encoder and cache the matrix.

Passages are encoded in `facts.jsonl` order; `ids.json` records the fact-id at
each row so retrieval can align an embedding row back to the KB fact regardless
of how the KB is later reloaded. Output (float16 to halve disk / memory):

  outputs/knowledge_bank/embeddings/<encoder>/facts.f16.npy   [N, dim]
  outputs/knowledge_bank/embeddings/<encoder>/ids.json        [N] fact ids
  outputs/knowledge_bank/embeddings/<encoder>/meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from retrievers import build_encoder

REPO = Path("/playpen-jfs/jesse/drug_microbiome")  # consolidated-folder anchor (was: parent.parent)


def passage_text(fact: dict) -> str:
    title = str(fact.get("title") or "").strip()
    text = str(fact.get("text") or "").strip()
    if title and title.lower() not in text.lower():
        return f"{title}. {text}"
    return text or title


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", required=True, choices=["bge", "medcpt"])
    p.add_argument("--kb-dir", default=str(REPO / "outputs/knowledge_bank/unified"))
    p.add_argument("--out-dir", default=str(REPO / "outputs/knowledge_bank/embeddings"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-facts", type=int, default=0, help="Debug cap on facts encoded.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    facts_path = Path(args.kb_dir) / "facts.jsonl"
    ids: list[str] = []
    passages: list[str] = []
    with facts_path.open(encoding="utf-8") as f:
        for line in f:
            if args.max_facts and len(ids) >= args.max_facts:
                break
            fact = json.loads(line)
            ids.append(fact["id"])
            passages.append(passage_text(fact))
    print(f"[kb] {len(ids)} facts loaded from {facts_path}", flush=True)

    encoder = build_encoder(args.encoder)
    print(f"[encode] encoder={args.encoder} batch={args.batch_size}", flush=True)
    chunks = []
    step = max(args.batch_size * 8, 1024)
    for i in range(0, len(passages), step):
        chunk = encoder.encode(passages[i : i + step], is_query=False, batch_size=args.batch_size)
        chunks.append(chunk.astype(np.float16))
        print(f"[encode] {min(i + step, len(passages))}/{len(passages)}", flush=True)
    matrix = np.concatenate(chunks, axis=0)

    out_dir = Path(args.out_dir) / args.encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "facts.f16.npy", matrix)
    (out_dir / "ids.json").write_text(json.dumps(ids))
    (out_dir / "meta.json").write_text(
        json.dumps(
            {"encoder": args.encoder, "n_facts": len(ids), "dim": int(matrix.shape[1]), "dtype": "float16"},
            indent=2,
        )
    )
    print(f"[done] wrote {out_dir}/facts.f16.npy shape={matrix.shape}", flush=True)


if __name__ == "__main__":
    main()
