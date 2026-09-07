"""Plot path-adaptive component weights and paired held-out gains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


from dwasim.paths import ROOT
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
        "axes.linewidth": 0.6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "legend.fontsize": 6.5,
    }
)

DATASET_COLORS = {"ACM": "#3F5F8A", "DBLP": "#C36A42"}
PATH_MARKERS = {
    "PAP": "o",
    "PSP": "s",
    "PTP": "^",
    "APA": "o",
    "APTPA": "s",
    "APVPA": "^",
}


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


def simplex_to_xy(weights: np.ndarray) -> tuple[float, float]:
    """Map support, magnitude, and direction weights to a ternary simplex."""
    support, magnitude, direction = np.asarray(weights, dtype=np.float64)
    if not np.isclose(support + magnitude + direction, 1.0):
        raise ValueError("Evidence weights must sum to one.")
    return float(magnitude + 0.5 * direction), float(np.sqrt(3.0) * direction / 2.0)


def draw_simplex(ax) -> None:
    height = np.sqrt(3.0) / 2.0
    border = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, height], [0.0, 0.0]])
    ax.plot(border[:, 0], border[:, 1], color="#555B61", linewidth=0.85, zorder=2)
    for fraction in (0.25, 0.50, 0.75):
        remaining = 1.0 - fraction
        grid_segments = (
            ((remaining, 0.0), (0.5 * remaining, height * remaining)),
            ((fraction, 0.0), (fraction + 0.5 * remaining, height * remaining)),
            ((0.5 * fraction, height * fraction),
             (1.0 - 0.5 * fraction, height * fraction)),
        )
        for start, end in grid_segments:
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="#DEDDD9",
                linewidth=0.40,
                zorder=1,
            )
    ax.text(-0.035, -0.045, "Support", ha="right", va="top", fontsize=6.7)
    ax.text(1.035, -0.045, "Magnitude", ha="left", va="top", fontsize=6.7)
    ax.text(0.50, height + 0.045, "Direction", ha="center", va="bottom", fontsize=6.7)
    ax.set_xlim(-0.14, 1.14)
    ax.set_ylim(-0.13, height + 0.13)
    ax.set_aspect("equal")
    ax.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = json.loads(args.input.read_text(encoding="utf-8"))
    rows = {row["dataset"]: row for row in result["rows"]}

    weight_points = []
    for dataset in ("ACM", "DBLP"):
        row = rows[dataset]
        for path_name in row["paths"]:
            selected = row["parameter_selection"]["TriComponentDWASim"][
                path_name
            ]["selected"]
            weight_points.append(
                (dataset, path_name, selected["weights_support_magnitude_direction"])
            )

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.20, 2.95),
        gridspec_kw={"width_ratios": [1.02, 1.18]},
        constrained_layout=True,
    )

    draw_simplex(ax_a)
    label_offsets = {
        ("ACM", "PAP"): (5, 5, "left"),
        ("ACM", "PSP"): (-5, 6, "right"),
        ("ACM", "PTP"): (5, 5, "left"),
        ("DBLP", "APA"): (5, -8, "left"),
        ("DBLP", "APTPA"): (-5, -10, "right"),
        ("DBLP", "APVPA"): (5, -10, "left"),
    }
    for dataset, path_name, weights in weight_points:
        x, y = simplex_to_xy(np.asarray(weights))
        overlapping_direction_path = dataset == "DBLP" and path_name == "APTPA"
        ax_a.scatter(
            [x],
            [y],
            s=46 if overlapping_direction_path else 35,
            marker=PATH_MARKERS[path_name],
            facecolor="white" if overlapping_direction_path else DATASET_COLORS[dataset],
            edgecolor=DATASET_COLORS[dataset] if overlapping_direction_path else "white",
            linewidth=0.9 if overlapping_direction_path else 0.55,
            zorder=4 if overlapping_direction_path else 5,
        )
        dx, dy, alignment = label_offsets[(dataset, path_name)]
        ax_a.annotate(
            path_name,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=alignment,
            va="bottom" if dy >= 0 else "top",
            fontsize=6.4,
            color=DATASET_COLORS[dataset],
        )
    dataset_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=4.2,
               markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.5,
               label=dataset)
        for dataset, color in DATASET_COLORS.items()
    ]
    ax_a.legend(handles=dataset_handles, loc="lower center", ncol=2,
                bbox_to_anchor=(0.50, -0.02), handletextpad=0.3, columnspacing=0.9)
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
        ax_b.plot(interval, [position, position], color=color, linewidth=1.0)
        ax_b.scatter(
            [point], [position], s=24, facecolor=color, edgecolor="white", linewidth=0.55, zorder=3
        )
    ax_b.axvline(0.0, color="#6F6F6F", linewidth=0.7, linestyle="--")
    ax_b.set_yticks(y, labels)
    ax_b.set_xlabel("Tri-component DWASim minus magnitude-only")
    ax_b.grid(axis="x", color="#D6D6D6", linewidth=0.35, alpha=0.55)
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
