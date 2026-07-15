#!/usr/bin/env python
"""Exp05 — KB perturbation for skill-routing stress tests (RQ6).

Reads a unified knowledge bank (`facts.jsonl` + `alias_index.json`, see
INFRA_REF / `retrieval/augment_with_skill_kb.load_kb`) and writes a *perturbed*
KB under an output directory that the existing retrieval/augment_*.py consume
UNCHANGED. The exact fact schema is preserved:

    id, source, source_id, title, text, aliases, entities, relation,
    metadata, tokens

Two perturbation modes make retrieval harder:

  --mode noisy --noise-frac F
      Keep every original fact, then additionally inject `round(F * N)` random
      off-task synthetic distractor facts (new ids, plausible-looking text,
      randomly assigned existing `source`). Retrieval must sift through more
      candidates to find the right one.

  --mode decoy --decoy-per N   [--decoy-frac P]
      Sample a fraction P of the original facts; for each, emit N DECOY copies
      whose `text` is near-duplicated (lightly perturbed) but RELABELED to a
      DIFFERENT (wrong) `source`. A lexically near-identical wrong-source fact
      now competes with the right one, stressing source routing. Each decoy
      carries a metadata flag {"decoy": true}.

      🔴 HEURISTIC: decoy construction is a synthetic near-duplicate + source
      relabel, NOT a hand-curated paraphrase corpus. It is a lower-cost proxy
      for the paper's "new-data" decoy corpus and is marked as such in RESULTS.

Determinism: all randomness flows through a single `random.Random(seed)`; there
is NO time-based randomness, so a fixed --seed reproduces byte-identical output.

The perturbation functions operate on plain lists of dicts and are import-safe
(no file IO at import), so tests can exercise them on tiny inline fixtures.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

# Mirror retrieval/augment_with_skill_kb.py tokenizer so recomputed `tokens`
# fields match what the augmenter would have produced (schema fidelity).
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.'-]*")

# Off-task-ish vocabulary used to synthesize plausible-looking distractor text.
# Deliberately generic biomedical/chemistry filler: looks like a real fact but
# does not answer any task in the benchmark.
_NOISE_SUBJECTS = [
    "compound", "receptor", "pathway", "enzyme", "metabolite", "transporter",
    "ligand", "isoform", "assay", "biomarker", "cofactor", "substrate",
]
_NOISE_PREDICATES = [
    "is associated with", "modulates", "is co-expressed with", "inhibits",
    "shares a scaffold with", "is annotated for", "is upregulated in",
    "has been reported alongside",
]
_NOISE_OBJECTS = [
    "the mitochondrial matrix", "an orphan GPCR", "a phase-II conjugate",
    "the cytochrome P450 family", "a lysosomal disorder", "an off-target kinase",
    "a synthetic analogue", "an unrelated tissue panel",
]
_NOISE_TITLE_A = ["Uncurated", "Auxiliary", "Peripheral", "Ancillary", "Tangential"]
_NOISE_TITLE_B = ["annotation", "note", "record", "observation", "entry"]

# Light suffixes appended to decoy text so the near-duplicate stays a superset
# of the original (original text remains a prefix) while differing slightly.
_DECOY_SUFFIXES = [
    " (cf. related source record)",
    " [duplicate annotation]",
    " -- see alternative source",
    " (mirrored entry)",
]


def tokenize(text: Any) -> list[str]:
    """Sorted unique lowercase tokens, matching the augmenter's `tokens` field."""
    return sorted(set(TOKEN_RE.findall(str(text).lower())))


def kb_sources(facts: list[dict[str, Any]]) -> list[str]:
    """Deterministically ordered list of distinct `source` values present."""
    return sorted({str(f.get("source", "")) for f in facts if f.get("source")})


def make_noise_fact(idx: int, rng, sources: list[str]) -> dict[str, Any]:
    """Build one synthetic off-task distractor fact with the full schema."""
    subj = rng.choice(_NOISE_SUBJECTS)
    pred = rng.choice(_NOISE_PREDICATES)
    obj = rng.choice(_NOISE_OBJECTS)
    tag = rng.randrange(1000, 9999)
    text = f"The candidate {subj} #{tag} {pred} {obj}."
    title = f"{rng.choice(_NOISE_TITLE_A)} {rng.choice(_NOISE_TITLE_B)} {tag}"
    source = rng.choice(sources) if sources else "synthetic"
    fact = {
        "id": f"noise_{idx:08d}",
        "source": source,
        "source_id": f"noise-{tag}",
        "title": title,
        "text": text,
        "aliases": [],
        "entities": [],
        "relation": "distractor",
        "metadata": {"synthetic_noise": True},
        "tokens": tokenize(text),
    }
    return fact


def inject_noise(
    facts: list[dict[str, Any]], noise_frac: float, rng, sources: list[str] | None = None
) -> list[dict[str, Any]]:
    """Return `round(noise_frac * len(facts))` fresh synthetic distractor facts."""
    if noise_frac < 0:
        raise ValueError("noise-frac must be >= 0")
    if sources is None:
        sources = kb_sources(facts)
    n_inject = int(round(noise_frac * len(facts)))
    return [make_noise_fact(i, rng, sources) for i in range(n_inject)]


def wrong_source(own: str, all_sources: list[str], rng) -> str:
    """Pick a DIFFERENT source than `own` (deterministic)."""
    candidates = [s for s in all_sources if s != own]
    if not candidates:
        return f"{own}_wrong"
    return rng.choice(candidates)


def make_decoys(
    facts: list[dict[str, Any]],
    decoy_per: int,
    rng,
    decoy_frac: float = 1.0,
    all_sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    """For a sampled fraction of facts, emit `decoy_per` wrong-source near-dups."""
    if decoy_per < 0:
        raise ValueError("decoy-per must be >= 0")
    if not 0.0 <= decoy_frac <= 1.0:
        raise ValueError("decoy-frac must be in [0, 1]")
    if all_sources is None:
        all_sources = kb_sources(facts)
    n_seed = int(round(decoy_frac * len(facts)))
    # Sample deterministically without disturbing the original ordering.
    if n_seed >= len(facts):
        seed_idxs = list(range(len(facts)))
    else:
        seed_idxs = sorted(rng.sample(range(len(facts)), n_seed))
    decoys: list[dict[str, Any]] = []
    for i in seed_idxs:
        orig = facts[i]
        base_text = str(orig.get("text", ""))
        for k in range(decoy_per):
            suffix = rng.choice(_DECOY_SUFFIXES)
            src = wrong_source(str(orig.get("source", "")), all_sources, rng)
            decoy = dict(orig)  # shallow copy of full schema
            decoy["id"] = f"decoy_{orig.get('id', i)}_{k}"
            decoy["source"] = src
            # Near-duplicate: original text stays a prefix (lightly perturbed).
            decoy["text"] = base_text + suffix
            decoy["aliases"] = []  # avoid alias-index collisions with the original
            md = dict(orig.get("metadata") or {})
            md["decoy"] = True
            md["decoy_of"] = orig.get("id")
            md["decoy_true_source"] = orig.get("source")
            decoy["metadata"] = md
            decoy["tokens"] = tokenize(decoy["text"])
            decoys.append(decoy)
    return decoys


def perturb(
    facts: list[dict[str, Any]],
    mode: str,
    rng,
    noise_frac: float = 0.0,
    decoy_per: int = 0,
    decoy_frac: float = 1.0,
) -> list[dict[str, Any]]:
    """Return the full perturbed fact list (originals preserved, then additions)."""
    sources = kb_sources(facts)
    out = list(facts)  # originals preserved verbatim, in order
    if mode == "noisy":
        out.extend(inject_noise(facts, noise_frac, rng, sources))
    elif mode == "decoy":
        out.extend(make_decoys(facts, decoy_per, rng, decoy_frac, sources))
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    return out


def read_facts(path: Path) -> list[dict[str, Any]]:
    facts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                facts.append(json.loads(line))
    return facts


def write_facts(path: Path, facts: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for fact in facts:
            f.write(json.dumps(fact, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Perturb a unified KB (facts.jsonl + alias_index.json) for Exp05 stress tests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--kb-dir", required=True,
                   help="Input KB dir containing facts.jsonl and alias_index.json "
                        "(e.g. outputs/knowledge_bank/unified).")
    p.add_argument("--out-dir", required=True,
                   help="Output dir for the perturbed KB (facts.jsonl + copied alias_index.json).")
    p.add_argument("--mode", required=True, choices=["noisy", "decoy"],
                   help="noisy: inject off-task distractors; decoy: wrong-source near-duplicates.")
    p.add_argument("--noise-frac", type=float, default=1.0,
                   help="[noisy] inject round(F * N) synthetic distractor facts.")
    p.add_argument("--decoy-per", type=int, default=1,
                   help="[decoy] number of wrong-source decoy copies per sampled fact.")
    p.add_argument("--decoy-frac", type=float, default=1.0,
                   help="[decoy] fraction of original facts sampled as decoy seeds.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed (random.Random(seed)); fixed seed => reproducible output.")
    p.add_argument("--limit", type=int, default=0,
                   help="Read at most this many input facts (0 = all); for quick dry-runs.")
    return p


def main(argv: list[str] | None = None) -> int:
    import random

    args = build_arg_parser().parse_args(argv)
    kb_dir = Path(args.kb_dir)
    out_dir = Path(args.out_dir)
    in_facts = kb_dir / "facts.jsonl"
    in_alias = kb_dir / "alias_index.json"
    if not in_facts.exists():
        print(f"ERROR: missing {in_facts}", file=sys.stderr)
        return 2

    facts = read_facts(in_facts)
    if args.limit:
        facts = facts[: args.limit]
    rng = random.Random(args.seed)
    out_facts = perturb(
        facts, args.mode, rng,
        noise_frac=args.noise_frac, decoy_per=args.decoy_per, decoy_frac=args.decoy_frac,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_facts(out_dir / "facts.jsonl", out_facts)
    # Copy alias_index.json unchanged: injected/decoy facts carry no aliases, so
    # the existing alias index stays valid for the original facts.
    if in_alias.exists():
        shutil.copyfile(in_alias, out_dir / "alias_index.json")
    else:
        (out_dir / "alias_index.json").write_text("{}\n", encoding="utf-8")

    added = len(out_facts) - len(facts)
    print(f"mode={args.mode} seed={args.seed} originals={len(facts)} "
          f"added={added} total={len(out_facts)} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
