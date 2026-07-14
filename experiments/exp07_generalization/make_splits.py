#!/usr/bin/env python
"""Exp07 — Generalization splits for shifted / unseen settings (RQ7).

Produces deterministic held-out train/test splits from a normalized QA JSONL file
so we can test whether skill-aware planning generalizes *beyond memorized
surface identity* rather than overfitting to specific drugs, skills, or sources.

Three modes, all seed-deterministic and all emitting a disjointness proof:

  --mode drug_holdout
      Partition so the set of drug/molecule identities in TEST is DISJOINT from
      TRAIN. For FDA records the identity key is `drug_name`; for Mol records it
      is the molecule identity (`decoded_smiles`, falling back to
      `input_molecule_or_context`). A held-out fraction of *distinct identities*
      (not records) is routed to test, so no test drug/molecule was ever seen in
      training. This isolates generalization from drug-name memorization.

  --mode skill_holdout --holdout-skill S
      Route every record to its skill via
      `retrieval.augment_with_skill_kb.infer_schema_v1_skill` (question-only
      deployed router) and move ALL records whose routed skill == S out of train
      and into test. This simulates *adding a skill declaratively*: the training
      distribution never contains skill S, yet the schema can still route it.

  --mode source_holdout --holdout-source SRC   (optional)
      Move ALL records whose `source` field == SRC out of train and into test —
      a coarser task-family holdout.

Each run writes `train.jsonl`, `test.jsonl`, and `split_summary.json` (counts +
a disjointness proof: |train_keys ∩ test_keys| == 0) to --output-dir.

Pure stdlib, no KB / model / GPU access — runnable and testable offline.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

# Reuse the shipped, question-only skill router so the skill holdout here never
# drifts from the retriever's own routing.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from retrieval.augment_with_skill_kb import infer_schema_v1_skill  # noqa: E402


# --------------------------------------------------------------------------- #
# Identity keys
# --------------------------------------------------------------------------- #
def drug_key(record: dict[str, Any]) -> str:
    """Surface identity a model could memorize.

    FDA records carry `drug_name`; Mol records have no drug name, so fall back to
    the molecule identity (decoded SMILES, then the raw input molecule/context).
    Normalized to lowercase/stripped so trivial casing/spacing does not leak an
    identity across the split. Empty string means "no identity" (see partition).
    """
    for field in ("drug_name", "decoded_smiles", "input_molecule_or_context"):
        val = record.get(field)
        if val is None:
            continue
        key = str(val).strip().lower()
        if key:
            return key
    return ""


def skill_key(record: dict[str, Any]) -> str:
    """Routed skill (question-only v1 router)."""
    return infer_schema_v1_skill(record)


def source_key(record: dict[str, Any]) -> str:
    return str(record.get("source", "")).strip()


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Partition primitives
# --------------------------------------------------------------------------- #
def partition_by_holdout_keys(
    records: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], str],
    holdout_keys: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Records whose key is in `holdout_keys` -> test; all others -> train.

    Records with an empty key are always kept in train (they carry no identity
    that could leak, so they cannot violate disjointness).
    """
    train, test = [], []
    for rec in records:
        key = key_fn(rec)
        if key and key in holdout_keys:
            test.append(rec)
        else:
            train.append(rec)
    return train, test


def choose_drug_holdout_keys(
    records: list[dict[str, Any]], test_frac: float, seed: int
) -> set[str]:
    """Deterministically pick a fraction of DISTINCT drug/molecule identities."""
    keys = sorted({drug_key(r) for r in records if drug_key(r)})
    if not keys:
        return set()
    rng = random.Random(seed)
    shuffled = keys[:]
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_frac))
    n_test = min(n_test, len(shuffled))
    return set(shuffled[:n_test])


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def split_drug_holdout(
    records: list[dict[str, Any]], test_frac: float, seed: int
) -> tuple[list, list, Callable]:
    holdout = choose_drug_holdout_keys(records, test_frac, seed)
    train, test = partition_by_holdout_keys(records, drug_key, holdout)
    return train, test, drug_key


def split_skill_holdout(
    records: list[dict[str, Any]], holdout_skill: str
) -> tuple[list, list, Callable]:
    train, test = partition_by_holdout_keys(records, skill_key, {holdout_skill})
    return train, test, skill_key


def split_source_holdout(
    records: list[dict[str, Any]], holdout_source: str
) -> tuple[list, list, Callable]:
    train, test = partition_by_holdout_keys(records, source_key, {holdout_source})
    return train, test, source_key


# --------------------------------------------------------------------------- #
# Summary / proof
# --------------------------------------------------------------------------- #
def key_set(records: list[dict[str, Any]], key_fn: Callable) -> set[str]:
    return {k for k in (key_fn(r) for r in records) if k}


def build_summary(
    mode: str,
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    key_fn: Callable,
    extra: dict[str, Any],
) -> dict[str, Any]:
    train_keys = key_set(train, key_fn)
    test_keys = key_set(test, key_fn)
    overlap = sorted(train_keys & test_keys)
    return {
        "mode": mode,
        "n_train": len(train),
        "n_test": len(test),
        "n_total": len(train) + len(test),
        "n_train_keys": len(train_keys),
        "n_test_keys": len(test_keys),
        "key_overlap_count": len(overlap),
        "disjoint": len(overlap) == 0,
        "key_overlap_examples": overlap[:10],
        **extra,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, help="Normalized QA JSONL (per INFRA_REF schema).")
    p.add_argument("--output-dir", required=True, help="Dir for train.jsonl/test.jsonl/split_summary.json.")
    p.add_argument("--mode", required=True, choices=["drug_holdout", "skill_holdout", "source_holdout"])
    p.add_argument("--seed", type=int, default=13, help="Determinism seed (drug_holdout key shuffle).")
    p.add_argument("--test-frac", type=float, default=0.2, help="drug_holdout: fraction of DISTINCT identities held out.")
    p.add_argument("--holdout-skill", default="", help="skill_holdout: routed skill to remove from train.")
    p.add_argument("--holdout-source", default="", help="source_holdout: `source` value to remove from train.")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = read_jsonl(Path(args.input))
    extra: dict[str, Any] = {"seed": args.seed}

    if args.mode == "drug_holdout":
        train, test, key_fn = split_drug_holdout(records, args.test_frac, args.seed)
        extra["test_frac"] = args.test_frac
    elif args.mode == "skill_holdout":
        if not args.holdout_skill:
            raise SystemExit("skill_holdout requires --holdout-skill S")
        train, test, key_fn = split_skill_holdout(records, args.holdout_skill)
        extra["holdout_skill"] = args.holdout_skill
        # Proof-relevant: the held-out skill must be entirely absent from train.
        train_skills = sorted(key_set(train, skill_key))
        extra["train_skills"] = train_skills
        extra["holdout_skill_in_train"] = args.holdout_skill in train_skills
    elif args.mode == "source_holdout":
        if not args.holdout_source:
            raise SystemExit("source_holdout requires --holdout-source SRC")
        train, test, key_fn = split_source_holdout(records, args.holdout_source)
        extra["holdout_source"] = args.holdout_source
    else:  # pragma: no cover - argparse restricts choices
        raise SystemExit(f"unknown mode {args.mode}")

    out_dir = Path(args.output_dir)
    n_train = write_jsonl(out_dir / "train.jsonl", train)
    n_test = write_jsonl(out_dir / "test.jsonl", test)
    summary = build_summary(args.mode, train, test, key_fn, extra)
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[exp07] mode={args.mode} train={n_train} test={n_test} "
        f"disjoint={summary['disjoint']} -> {out_dir}"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
