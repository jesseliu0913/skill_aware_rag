#!/usr/bin/env python
"""Create paper-ready figures for the personalized-retrieval preliminary study."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


QUESTION_LABELS = {
    "fda_factual": "FDA factual",
    "fda_multihop": "FDA multihop",
    "fda_refusal": "FDA refusal",
    "biomedical_open_qa": "Open biomedical QA",
    "molecule_description": "Molecule description",
    "molecule_design": "Molecule design",
    "molecule_property": "Molecule property",
}
METHOD_LABELS = {
    "bm25": "BM25",
    "kg": "KG",
    "hybrid": "Hybrid",
    "dense_bge": "BGE",
    "dense_medcpt": "MedCPT",
    "rrf": "RRF",
    "rerank": "Rerank",
}
SOURCE_LABELS = {
    "primekg": "PrimeKG",
    "drugchat_pubchem": "PubChem",
    "drugchat_chembl": "ChEMBL",
    "fdarxbench_label": "FDA labels",
}
MODEL_LABELS = {
    "qwen2_5_7b": "Qwen2.5-7B",
    "qwen2_5_3b": "Qwen2.5-3B",
    "llama3_2_3b": "Llama-3.2-3B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--results-dir", type=Path, default=base / "results")
    parser.add_argument("--output-dir", type=Path, default=base / "figures")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=240)
    plt.close(fig)


def figure_source_support(results_dir: Path, output_dir: Path) -> None:
    rows = [
        row
        for row in read_csv(results_dir / "evidence_by_source.csv")
        if row["method"] == "hybrid"
    ]
    questions = list(QUESTION_LABELS)
    sources = list(SOURCE_LABELS)
    counts = {
        (row["question_type"], row["source"]): float(row["avg_facts"])
        * int(row["examples_with_evidence"])
        for row in rows
    }
    values = np.array([[counts.get((question, source), 0.0) for source in sources] for question in questions])
    totals = values.sum(axis=1, keepdims=True)
    shares = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
    colors = ("#3B7EA1", "#59A14F", "#E3A72F", "#C44E52")
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    y = np.arange(len(questions))
    left = np.zeros(len(questions))
    for source_idx, (source, color) in enumerate(zip(sources, colors, strict=True)):
        widths = shares[:, source_idx]
        bars = ax.barh(y, widths, left=left, height=0.68, color=color, label=SOURCE_LABELS[source])
        for row_idx, (bar, width) in enumerate(zip(bars, widths, strict=True)):
            if width >= 0.08:
                ax.text(
                    left[row_idx] + width / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if source_idx in {0, 3} else "#1f1f1f",
                )
        left += widths
    ax.set_yticks(y, [QUESTION_LABELS[question] for question in questions])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    ax.set_xlabel("Share of retrieved evidence facts")
    ax.set_ylabel("Question type")
    ax.set_title("Question types induce different knowledge-source mixtures", pad=62)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.015), frameon=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.text(
        0.01,
        -0.02,
        "Hybrid retrieval over the shared KB; proportions aggregate all retrieved facts within each question type.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "figure1_source_composition")


def figure_retriever_heterogeneity(results_dir: Path, output_dir: Path) -> None:
    rows = [
        row
        for row in read_csv(results_dir / "performance_by_question_type.csv")
        if row["partition"] == "evaluation"
    ]
    methods = list(METHOD_LABELS)
    datasets = ("fdarxbench", "mol_instructions")
    dataset_questions = {
        "fdarxbench": ["fda_factual", "fda_multihop", "fda_refusal"],
        "mol_instructions": [
            "biomedical_open_qa",
            "molecule_description",
            "molecule_design",
            "molecule_property",
        ],
    }
    aggregates: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (row["dataset"], row["question_type"], row["method"])
        aggregates.setdefault(key, []).append(float(row["utility"]))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.1), gridspec_kw={"width_ratios": [3, 4]})
    for ax, dataset in zip(axes, datasets, strict=True):
        questions = dataset_questions[dataset]
        raw = np.array(
            [
                [np.mean(aggregates.get((dataset, question, method), [0.0])) for method in methods]
                for question in questions
            ]
        )
        centered = raw - raw.mean(axis=1, keepdims=True)
        annotations = np.array([[f"{value:.3f}" for value in row] for row in raw])
        sns.heatmap(
            centered,
            annot=annotations,
            fmt="",
            cmap="RdBu_r",
            center=0,
            vmin=-0.02,
            vmax=0.02,
            linewidths=0.7,
            linecolor="white",
            xticklabels=[METHOD_LABELS[method] for method in methods],
            yticklabels=[QUESTION_LABELS[question] for question in questions],
            cbar=ax is axes[-1],
            cbar_kws={"label": "Utility relative to row mean"} if ax is axes[-1] else None,
            ax=ax,
        )
        for row_idx, row in enumerate(raw):
            best = int(np.argmax(row))
            ax.add_patch(plt.Rectangle((best, row_idx), 1, 1, fill=False, edgecolor="#111111", lw=2.0))
        ax.set_title("FDARxBench" if dataset == "fdarxbench" else "Mol-Instructions")
        ax.set_xlabel("Generic retrieval policy")
        ax.set_ylabel("Question type" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("The strongest generic retriever varies by question type", y=1.04, fontsize=13)
    fig.text(
        0.01,
        -0.08,
        "Cells show mean held-out utility across three models; color is centered within each question type. Black boxes mark row maxima.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "figure2_retriever_heterogeneity")


def figure_heldout_gain(results_dir: Path, output_dir: Path) -> None:
    rows = read_csv(results_dir / "policy_comparison.csv")
    labels = [f"{MODEL_LABELS[row['model']]}\n{'FDA' if row['dataset'] == 'fdarxbench' else 'Mol'}" for row in rows]
    gains = np.array([float(row["absolute_gain"]) * 100 for row in rows])
    colors = ["#247BA0" if gain >= 0 else "#C44E52" for gain in gains]
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    bars = ax.bar(np.arange(len(rows)), gains, color=colors, width=0.68)
    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_xticks(np.arange(len(rows)), labels)
    ax.set_ylabel("Held-out utility gain (percentage points)")
    ax.set_title("Question-type policy selection currently yields only marginal held-out gains", pad=12)
    margin = max(0.008, max(abs(gains), default=0.0) * 0.06)
    ax.set_ylim(min(-0.075, float(gains.min()) - 0.03), max(0.06, float(gains.max()) + 0.05))
    for bar, gain in zip(bars, gains, strict=True):
        vertical = gain + margin if gain >= 0 else gain - margin
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            vertical,
            f"{gain:+.2f}",
            ha="center",
            va="bottom" if gain >= 0 else "top",
            fontsize=9,
        )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure3_heldout_gain")


def main() -> None:
    args = parse_args()
    sns.set_theme(style="white", context="paper", font_scale=1.05)
    figure_source_support(args.results_dir, args.output_dir)
    figure_retriever_heterogeneity(args.results_dir, args.output_dir)
    figure_heldout_gain(args.results_dir, args.output_dir)
    print(f"Wrote 3 figures as PDF and PNG under {args.output_dir}")


if __name__ == "__main__":
    main()
