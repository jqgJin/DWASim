"""Create the submission figure for held-out benchmark multi-path fusion."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


from dwasim.paths import ROOT
RESULTS = ROOT / "results"
OUTPUT = ROOT / "figures" / "Fig6_real_multipath_fusion"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 6.5,
    }
)

COLORS = {
    "ACM": "#3F5F8A",
    "DBLP": "#C36A42",
    "path_1": "#6F7782",
    "path_2": "#AAA39B",
    "guide": "#D9D7D2",
}

CLASS_ABBREVIATIONS = {
    "ACM": {"0": "DB", "1": "WC", "2": "DM"},
    "DBLP": {"0": "DB", "1": "DM", "2": "AI", "3": "IR"},
}


def load_row(dataset: str) -> dict:
    combined = RESULTS / "real_multipath_fusion_k10.json"
    if combined.exists():
        rows = json.loads(combined.read_text(encoding="utf-8"))["rows"]
        return next(row for row in rows if row["dataset"] == dataset)
    path = RESULTS / f"real_multipath_{dataset}_k10.json"
    return json.loads(path.read_text(encoding="utf-8"))["rows"][0]


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
    )


def main() -> None:
    rows = {dataset: load_row(dataset) for dataset in ("ACM", "DBLP")}
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.20, 2.85),
        gridspec_kw={"width_ratios": [1.0, 1.16]},
        constrained_layout=True,
    )

    # a: validation evidence used for the fusion decision.
    for dataset in ("ACM", "DBLP"):
        selection = rows[dataset]["fusion_selection"]["DWASim"]
        x = np.asarray([item["weight_path_1"] for item in selection["grid"]])
        mean = np.asarray([item["macro_f1_mean"] for item in selection["grid"]])
        std = np.asarray([item["macro_f1_std"] for item in selection["grid"]])
        label = "ACM: weight on PAP" if dataset == "ACM" else "DBLP: weight on APA"
        ax_a.plot(x, mean, marker="o", markersize=3.0, linewidth=1.0,
                  markeredgewidth=0.65, color=COLORS[dataset], label=label)
        ax_a.fill_between(x, mean - std, mean + std, color=COLORS[dataset], alpha=0.10, linewidth=0)
        chosen = selection["selected"]["weight_path_1"]
        chosen_mean = selection["selected"]["macro_f1_mean"]
        ax_a.scatter([chosen], [chosen_mean], s=28, facecolor="white",
                     edgecolor=COLORS[dataset], linewidth=0.9, zorder=4)
    ax_a.set_xlabel("Weight assigned to the first path")
    ax_a.set_ylabel("Validation Macro-F1")
    ax_a.set_xticks(np.linspace(0, 1, 6))
    ax_a.set_ylim(0.10, 0.80)
    ax_a.legend(loc="lower center", ncol=1)
    ax_a.grid(axis="y", color="#D6D6D6", linewidth=0.35, alpha=0.55)
    panel_label(ax_a, "a")

    # b: Cleveland dot plot preserving every official test category.
    category_rows = []
    category_labels = []
    category_datasets = []
    for dataset in ("ACM", "DBLP"):
        classes = sorted(rows[dataset]["test_metrics"]["DWASim"]["path_1"]["per_class_f1"], key=int)
        for cls in classes:
            category_rows.append(
                [
                    rows[dataset]["test_metrics"]["DWASim"][variant]["per_class_f1"][cls]
                    for variant in ("path_1", "path_2", "validation_selected_fusion")
                ]
            )
            category_labels.append(f"{dataset}-{CLASS_ABBREVIATIONS[dataset][cls]}")
            category_datasets.append(dataset)
    category_values = np.asarray(category_rows)
    y_positions = np.arange(len(category_labels))[::-1]
    offsets = (-0.16, 0.0, 0.16)
    markers = ("o", "s", "D")
    for row, (y, dataset) in enumerate(zip(y_positions, category_datasets)):
        values = category_values[row]
        ax_b.plot(
            [values.min(), values.max()],
            [y, y],
            color=COLORS["guide"],
            linewidth=0.8,
            solid_capstyle="round",
            zorder=1,
        )
        for column, (offset, marker) in enumerate(zip(offsets, markers)):
            if column < 2:
                ax_b.scatter(
                    [values[column]],
                    [y + offset],
                    s=24,
                    marker=marker,
                    facecolor="white",
                    edgecolor=COLORS[f"path_{column + 1}"],
                    linewidth=0.8,
                    zorder=3,
                )
            else:
                ax_b.scatter(
                    [values[column]],
                    [y + offset],
                    s=30,
                    marker=marker,
                    facecolor=COLORS[dataset],
                    edgecolor="white",
                    linewidth=0.55,
                    zorder=4,
                )
    ax_b.set_yticks(y_positions, category_labels)
    ax_b.set_xlabel("Per-category F1")
    ax_b.set_xlim(-0.02, 1.02)
    ax_b.set_xticks(np.linspace(0, 1, 6))
    ax_b.axhline(3.5, color="#7D7D7D", linewidth=0.55)
    ax_b.grid(axis="x", color=COLORS["guide"], linewidth=0.35, alpha=0.65)
    ax_b.tick_params(axis="y", length=0)
    panel_label(ax_b, "b")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Wrote figure bundle to {OUTPUT.parent}")


if __name__ == "__main__":
    main()
