"""Plot path-adaptive component weights and paired held-out gains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "support_retrieval_stress.json"
DEFAULT_OUTPUT = ROOT / "figures" / "Fig7_path_adaptive_retrieval"

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
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "legend.fontsize": 6.5,
    }
)

DATASET_COLORS = {"ACM": "#3A78A1", "DBLP": "#D08145"}


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = json.loads(args.input.read_text(encoding="utf-8"))
    rows = {row["dataset"]: row for row in result["rows"]}

    weight_rows = []
    weight_labels = []
    for dataset in ("ACM", "DBLP"):
        row = rows[dataset]
        for path_name in row["paths"]:
            selected = row["parameter_selection"]["TriComponentDWASim"][
                path_name
            ]["selected"]
            weight_rows.append(selected["weights_support_magnitude_direction"])
            weight_labels.append(f"{dataset}-{path_name}")
    weights = np.asarray(weight_rows, dtype=np.float64)

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.20, 2.80),
        gridspec_kw={"width_ratios": [0.92, 1.28]},
        constrained_layout=True,
    )

    cmap = LinearSegmentedColormap.from_list(
        "component_weight", ["#F4F5F5", "#B7C9D2", "#376E8C"]
    )
    image = ax_a.imshow(weights, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax_a.set_xticks(
        np.arange(3), ["Support\n(Jaccard)", "Magnitude\n(Bray--Curtis)", "Direction\n(cosine)"]
    )
    ax_a.set_yticks(np.arange(len(weight_labels)), weight_labels)
    for row_index in range(weights.shape[0]):
        for column_index in range(weights.shape[1]):
            value = weights[row_index, column_index]
            ax_a.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.58 else "#26343D",
                fontsize=6.5,
            )
    ax_a.axhline(2.5, color="white", linewidth=2.0)
    ax_a.tick_params(length=0)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax_a, fraction=0.046, pad=0.035)
    cbar.set_label("Training-selected weight")
    cbar.outline.set_linewidth(0.5)
    ax_a.set_title("Path-specific evidence weights", loc="left", fontweight="bold")
    panel_label(ax_a, "a")

    labels = []
    points = []
    intervals = []
    colors = []
    for dataset in ("ACM", "DBLP"):
        comparison = rows[dataset]["paired_method_comparisons"][
            "TriComponentDWASim"
        ]["minus_MagnitudeOnly"]
        for metric, display in (("macro_f1", "Macro-F1"), ("ndcg_at_k", "NDCG@10")):
            record = comparison[metric]
            labels.append(f"{dataset} {display}")
            points.append(record["difference"])
            intervals.append((record["lower_95"], record["upper_95"]))
            colors.append(DATASET_COLORS[dataset])

    y = np.arange(len(labels))[::-1]
    for position, point, interval, color in zip(y, points, intervals, colors):
        ax_b.plot(interval, [position, position], color=color, linewidth=1.6)
        ax_b.scatter(
            [point], [position], s=27, facecolor=color, edgecolor="white", linewidth=0.6, zorder=3
        )
    ax_b.axvline(0.0, color="#6F6F6F", linewidth=0.8, linestyle="--")
    ax_b.set_yticks(y, labels)
    ax_b.set_xlabel("Tri-component DWASim minus magnitude-only")
    ax_b.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax_b.set_title("Paired held-out differences (95% interval)", loc="left", fontweight="bold")
    ax_b.set_xlim(-0.004, 0.048)
    panel_label(ax_b, "b")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Wrote figure bundle to {args.output.parent}")


if __name__ == "__main__":
    main()
