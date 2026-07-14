#!/usr/bin/env python
"""Exp01 — Knowledge-source heterogeneity metrics (RQ1, paper Fig. 1 / Table).

Extends the preliminary per-task source-mixture study (`pre_design/`) with the
quantitative heterogeneity metrics the paper needs to argue that drug-discovery
question types induce *distinct* knowledge-source mixtures, hence uniform
retrieval is structurally mismatched.

Inputs (already produced by `pre_design/analyze_preliminary.py`):
  - pre_design/results/evidence_by_source.csv     (per dataset,qtype,method,source share)
  - pre_design/results/source_answer_support.csv  (per source gold-token recall)

Outputs (written to --output-dir, default experiments/exp01_source_heterogeneity/results):
  - heterogeneity_metrics.csv   per (dataset, question_type, method): dominant share,
                                source entropy (bits + normalized), effective #sources,
                                oracle-source share, oracle gold-support share
  - cross_task_divergence.csv   per (dataset, method): pairwise JS divergence between
                                question types' source distributions + mean
  - REPORT.md                   short human-readable synthesis

This is pure post-hoc analysis over committed CSVs; it does not touch the KB or
run any model, so it is runnable today.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

# Import the shipped skill schema so "allowed sources" here never drift from the
# retriever's definition.
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from retrieval.augment_with_skill_kb import SKILL_SCHEMA_V1  # noqa: E402

KB_SOURCES = ["primekg", "drugchat_chembl", "drugchat_pubchem", "fdarxbench_label"]

# Question-type strata (as emitted by pre_design/analyze_preliminary.py) → schema skill.
# Refusal has no retrieval skill of its own; it inherits the FDA-label source scope.
QUESTION_TYPE_TO_SKILL = {
    "fda_factual": "fda_label_factual",
    "fda_multihop": "fda_label_multihop",
    "fda_refusal": "fda_label_factual",
    "molecule_property": "molecule_property_numeric",
    "molecule_description": "molecule_description",
    "molecule_design": "molecule_design",
    "biomedical_open_qa": "biomedical_open_qa",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence-csv", default="pre_design/results/evidence_by_source.csv")
    p.add_argument("--support-csv", default="pre_design/results/source_answer_support.csv")
    p.add_argument("--output-dir", default="experiments/exp01_source_heterogeneity/results")
    p.add_argument(
        "--method",
        default="hybrid",
        help="Which retriever's mixtures to headline in the report (all methods are written to CSV).",
    )
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def source_distribution(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, float]]:
    """(dataset, question_type, method) -> normalized source share over KB_SOURCES."""
    raw: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: {s: 0.0 for s in KB_SOURCES})
    for r in rows:
        key = (r["dataset"], r["question_type"], r["method"])
        src = r["source"]
        if src in raw[key]:
            raw[key][src] += float(r.get("mean_within_prompt_share", 0.0) or 0.0)
    dist: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, shares in raw.items():
        total = sum(shares.values())
        dist[key] = {s: (v / total if total > 0 else 0.0) for s, v in shares.items()}
    return dist


def entropy_bits(dist: dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def effective_sources(dist: dict[str, float]) -> float:
    """Perplexity of the source distribution: 2**H. 1.0 = single-source."""
    return 2.0 ** entropy_bits(dist)


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence (base 2, bounded [0,1]) between two source dists."""
    m = {s: 0.5 * (p.get(s, 0.0) + q.get(s, 0.0)) for s in KB_SOURCES}

    def kl(a: dict[str, float], b: dict[str, float]) -> float:
        return sum(a[s] * math.log2(a[s] / b[s]) for s in KB_SOURCES if a.get(s, 0.0) > 0 and b.get(s, 0.0) > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def support_by_source(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, float]]:
    """(dataset, question_type, method) -> per-source mean gold-token recall."""
    out: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for r in rows:
        key = (r["dataset"], r["question_type"], r["method"])
        out[key][r["source"]] = float(r.get("mean_gold_token_recall", 0.0) or 0.0)
    return out


def allowed_sources(question_type: str) -> set[str]:
    skill = QUESTION_TYPE_TO_SKILL.get(question_type)
    if not skill:
        return set(KB_SOURCES)
    return set(SKILL_SCHEMA_V1[skill]["sources"])


def compute_metrics(
    dist: dict[tuple[str, str, str], dict[str, float]],
    support: dict[tuple[str, str, str], dict[str, float]],
) -> list[dict[str, Any]]:
    out = []
    for (dataset, qtype, method), sd in sorted(dist.items()):
        allowed = allowed_sources(qtype)
        dominant = max(sd, key=sd.get)
        oracle_share = sum(sd[s] for s in sd if s in allowed)
        supp = support.get((dataset, qtype, method), {})
        # Fraction of total gold-support recall that the schema's allowed sources carry.
        total_supp = sum(supp.values())
        oracle_supp_share = sum(v for s, v in supp.items() if s in allowed)
        out.append(
            {
                "dataset": dataset,
                "question_type": qtype,
                "method": method,
                "skill": QUESTION_TYPE_TO_SKILL.get(qtype, ""),
                "dominant_source": dominant,
                "dominant_share": round(sd[dominant], 4),
                "source_entropy_bits": round(entropy_bits(sd), 4),
                "source_entropy_norm": round(entropy_bits(sd) / math.log2(len(KB_SOURCES)), 4),
                "effective_sources": round(effective_sources(sd), 4),
                "oracle_source_share": round(oracle_share, 4),
                "oracle_support_share": round(oracle_supp_share / total_supp, 4) if total_supp > 0 else 0.0,
                **{f"share_{s}": round(sd[s], 4) for s in KB_SOURCES},
            }
        )
    return out


def compute_divergence(dist: dict[tuple[str, str, str], dict[str, float]]) -> list[dict[str, Any]]:
    """Pairwise JS divergence between question types (same dataset+method)."""
    by_dm: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for (dataset, qtype, method), sd in dist.items():
        by_dm[(dataset, method)][qtype] = sd
    out = []
    for (dataset, method), qmap in sorted(by_dm.items()):
        qtypes = sorted(qmap)
        pair_vals = []
        for i in range(len(qtypes)):
            for j in range(i + 1, len(qtypes)):
                jsd = js_divergence(qmap[qtypes[i]], qmap[qtypes[j]])
                pair_vals.append(jsd)
                out.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "question_type_a": qtypes[i],
                        "question_type_b": qtypes[j],
                        "js_divergence": round(jsd, 4),
                    }
                )
        if pair_vals:
            out.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "question_type_a": "ALL",
                    "question_type_b": "MEAN",
                    "js_divergence": round(sum(pair_vals) / len(pair_vals), 4),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build_report(metrics: list[dict[str, Any]], divergence: list[dict[str, Any]], method: str) -> str:
    lines = [
        "# Exp01 — Knowledge-source heterogeneity (RQ1)",
        "",
        f"Headline retriever: `{method}`. Full per-method values are in the CSVs.",
        "",
        "## Per-task source concentration",
        "",
        "| Dataset | Question type | Dominant source | Dominant share | Eff. #sources | Oracle-source share | Oracle support share |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for m in metrics:
        if m["method"] != method:
            continue
        lines.append(
            f"| {m['dataset']} | {m['question_type']} | {m['dominant_source']} | "
            f"{m['dominant_share']:.3f} | {m['effective_sources']:.2f} | "
            f"{m['oracle_source_share']:.3f} | {m['oracle_support_share']:.3f} |"
        )
    lines += ["", "## Cross-task source divergence (mean pairwise JS)", ""]
    lines.append("| Dataset | Method | Mean pairwise JS divergence |")
    lines.append("|---|---|---:|")
    for d in divergence:
        if d["question_type_a"] == "ALL" and d["question_type_b"] == "MEAN":
            lines.append(f"| {d['dataset']} | {d['method']} | {d['js_divergence']:.3f} |")
    lines += [
        "",
        "Reading: a high dominant-share with low effective-sources means the task is",
        "single-source; a high mean pairwise JS divergence means different question",
        "types draw on different sources — the necessity argument for the skill schema.",
        "Oracle-source / support share near 1.0 means the schema's allowed-source set",
        "covers where the evidence (and its answer support) actually lives.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    evidence_rows = read_rows(Path(args.evidence_csv))
    support_rows = read_rows(Path(args.support_csv))
    dist = source_distribution(evidence_rows)
    support = support_by_source(support_rows)
    metrics = compute_metrics(dist, support)
    divergence = compute_divergence(dist)

    out_dir = Path(args.output_dir)
    write_csv(out_dir / "heterogeneity_metrics.csv", metrics)
    write_csv(out_dir / "cross_task_divergence.csv", divergence)
    (out_dir / "REPORT.md").write_text(build_report(metrics, divergence, args.method), encoding="utf-8")
    print(f"[exp01] wrote {len(metrics)} metric rows, {len(divergence)} divergence rows -> {out_dir}")


if __name__ == "__main__":
    main()
