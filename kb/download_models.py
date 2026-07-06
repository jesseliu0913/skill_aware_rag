#!/usr/bin/env python
"""Pre-download retrieval encoders into the HF cache so Slurm compute nodes can
run fully offline (local_files_only=True), consistent with the LM snapshots.

Encoders:
  - BAAI/bge-large-en-v1.5      general-domain dense bi-encoder
  - ncbi/MedCPT-Query-Encoder   biomedical dense query encoder
  - ncbi/MedCPT-Article-Encoder biomedical dense passage encoder
  - ncbi/MedCPT-Cross-Encoder   biomedical cross-encoder reranker
"""

from __future__ import annotations

from huggingface_hub import snapshot_download

MODELS = [
    "BAAI/bge-large-en-v1.5",
    "ncbi/MedCPT-Query-Encoder",
    "ncbi/MedCPT-Article-Encoder",
    "ncbi/MedCPT-Cross-Encoder",
]


def main() -> None:
    for repo_id in MODELS:
        print(f"[download] {repo_id}", flush=True)
        path = snapshot_download(repo_id=repo_id)
        print(f"[done]     {repo_id} -> {path}", flush=True)


if __name__ == "__main__":
    main()
