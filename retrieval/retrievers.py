#!/usr/bin/env python
"""Retrieval engines for the RAG baselines.

All retrievers operate over the SAME unified KB used by the proposed method
(`outputs/knowledge_bank/unified/facts.jsonl`) and return evidence in the exact
dict shape consumed by `augment_with_skill_kb.augment_record` (in this folder), so the
downstream prompt/train/eval pipeline is byte-for-byte identical across methods
and the retriever is the only independent variable.

Encoders (all pre-downloaded to the HF cache, loaded local_files_only):
  - BGE  : BAAI/bge-large-en-v1.5              (general dense bi-encoder)
  - MedCPT: ncbi/MedCPT-{Query,Article}-Encoder (biomedical dense bi-encoder)
  - MedCPT-Cross-Encoder                        (biomedical reranker)
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch

# Recommended query instruction for BGE v1.5 retrieval (passages get none).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

MEDCPT_QUERY_MAXLEN = 64
MEDCPT_ARTICLE_MAXLEN = 512
MEDCPT_CROSS_MAXLEN = 512


def _resolve(repo_or_path: str) -> str:
    """Resolve a HF repo id to its local snapshot dir (offline-safe)."""
    from pathlib import Path

    if Path(repo_or_path).exists():
        return repo_or_path
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id=repo_or_path, local_files_only=True)
    except Exception:
        # Fall back to the repo id; from_pretrained resolves it via HF_HOME under
        # HF_HUB_OFFLINE=1 (both set in the sbatch environment).
        return repo_or_path


# --------------------------------------------------------------------------- #
# Dense bi-encoders
# --------------------------------------------------------------------------- #
class BGEEncoder:
    """SentenceTransformer wrapper for BAAI/bge-large-en-v1.5."""

    name = "bge"

    def __init__(self, model_id: str = "BAAI/bge-large-en-v1.5", device: str = "cuda") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(_resolve(model_id), device=device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], is_query: bool, batch_size: int = 256) -> np.ndarray:
        if is_query:
            texts = [BGE_QUERY_INSTRUCTION + t for t in texts]
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)


class MedCPTEncoder:
    """Asymmetric biomedical bi-encoder (separate query / article encoders).

    Embedding = CLS ([:, 0, :]) of the last hidden state, L2-normalized so a dot
    product is cosine similarity (matches how MedCPT is used for retrieval).
    """

    name = "medcpt"

    def __init__(
        self,
        query_id: str = "ncbi/MedCPT-Query-Encoder",
        article_id: str = "ncbi/MedCPT-Article-Encoder",
        device: str = "cuda",
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.q_tok = AutoTokenizer.from_pretrained(_resolve(query_id))
        self.q_model = AutoModel.from_pretrained(_resolve(query_id)).to(device).eval()
        self.a_tok = AutoTokenizer.from_pretrained(_resolve(article_id))
        self.a_model = AutoModel.from_pretrained(_resolve(article_id)).to(device).eval()

    @torch.inference_mode()
    def _encode(self, model, tok, encoded, batch_size: int) -> np.ndarray:
        out = []
        for i in range(0, len(encoded["input_ids"]), batch_size):
            batch = {k: v[i : i + batch_size].to(self.device) for k, v in encoded.items()}
            cls = model(**batch).last_hidden_state[:, 0, :]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            out.append(cls.cpu().float().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def encode(self, texts: list[str], is_query: bool, batch_size: int = 128) -> np.ndarray:
        if is_query:
            enc = self.q_tok(
                texts, truncation=True, padding=True, return_tensors="pt", max_length=MEDCPT_QUERY_MAXLEN
            )
            return self._encode(self.q_model, self.q_tok, enc, batch_size)
        enc = self.a_tok(
            texts, truncation=True, padding=True, return_tensors="pt", max_length=MEDCPT_ARTICLE_MAXLEN
        )
        return self._encode(self.a_model, self.a_tok, enc, batch_size)


def build_encoder(name: str, device: str = "cuda"):
    if name == "bge":
        return BGEEncoder(device=device)
    if name == "medcpt":
        return MedCPTEncoder(device=device)
    raise ValueError(f"Unknown encoder: {name}")


# --------------------------------------------------------------------------- #
# Cross-encoder reranker
# --------------------------------------------------------------------------- #
class MedCPTCrossEncoder:
    name = "medcpt_cross"

    def __init__(self, model_id: str = "ncbi/MedCPT-Cross-Encoder", device: str = "cuda") -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(_resolve(model_id))
        self.model = AutoModelForSequenceClassification.from_pretrained(_resolve(model_id)).to(device).eval()

    @torch.inference_mode()
    def score(self, query: str, passages: list[str], batch_size: int = 64) -> np.ndarray:
        if not passages:
            return np.zeros((0,), dtype=np.float32)
        pairs = [[query, p] for p in passages]
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = self.tok(
                pairs[i : i + batch_size],
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=MEDCPT_CROSS_MAXLEN,
            ).to(self.device)
            logits = self.model(**batch).logits.squeeze(dim=1)
            scores.append(logits.cpu().float().numpy().reshape(-1))
        return np.concatenate(scores, axis=0)


# --------------------------------------------------------------------------- #
# Dense search over a precomputed KB matrix
# --------------------------------------------------------------------------- #
class DenseIndex:
    """Cosine (inner-product on L2-normalized vectors) search via faiss, with a
    numpy fallback if faiss is unavailable."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.ascontiguousarray(matrix.astype(np.float32))
        self._faiss = None
        try:
            import faiss

            index = faiss.IndexFlatIP(self.matrix.shape[1])
            index.add(self.matrix)
            self._faiss = index
        except Exception:
            self._faiss = None

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries.astype(np.float32))
        if self._faiss is not None:
            scores, idxs = self._faiss.search(queries, top_k)
            return scores, idxs
        sims = queries @ self.matrix.T
        idxs = np.argpartition(-sims, min(top_k, sims.shape[1] - 1), axis=1)[:, :top_k]
        row = np.arange(sims.shape[0])[:, None]
        order = np.argsort(-sims[row, idxs], axis=1)
        idxs = idxs[row, order]
        scores = sims[row, idxs]
        return scores, idxs


# --------------------------------------------------------------------------- #
# Rank fusion
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(ranked_lists: Iterable[list[int]], k: int = 60) -> dict[int, float]:
    """Standard RRF (Cormack et al. 2009): score(d) = sum 1/(k + rank_i(d))."""
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_idx in enumerate(ranked):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


# --------------------------------------------------------------------------- #
# Evidence formatting (identical dict shape to augment_with_skill_kb)
# --------------------------------------------------------------------------- #
def fact_to_evidence(fact: dict[str, Any], score: float, alias_hit: bool = False) -> dict[str, Any]:
    return {
        "id": fact["id"],
        "score": round(float(score), 4),
        "source": fact.get("source"),
        "relation": fact.get("relation"),
        "title": fact.get("title"),
        "text": fact.get("text"),
        "entities": fact.get("entities", []),
        "metadata": fact.get("metadata", {}),
        "alias_hit": alias_hit,
    }
