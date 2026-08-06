"""Create the held-out multi-path fusion figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
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
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 6.5,
    }
)

COLORS = {
    "ACM": "#3A78A1",
    "DBLP": "#D08145",
    "DWASim": "#3A78A1",
    "PathSim": "#7A7A7A",
    "HeteSim": "#B8A16A",
    "single_1": "#9DBDD3",
    "single_2": "#D7C6A5",
    "uniform": "#8FB8A8",
    "entropy": "#A9A0C6",
    "selected": "#3A78A1",
}


def load_row(dataset: str) -> dict:
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
        figsize=(7.20, 2.65),
        gridspec_kw={"width_ratios": [1.0, 1.12]},
        constrained_layout=True,
    )

    # a: validation evidence used for the fusion decision.
    for dataset in ("ACM", "DBLP"):
        selection = rows[dataset]["fusion_selection"]["DWASim"]
        x = np.asarray([item["weight_path_1"] for item in selection["grid"]])
        mean = np.asarray([item["macro_f1_mean"] for item in selection["grid"]])
        std = np.asarray([item["macro_f1_std"] for item in selection["grid"]])
        label = "ACM: weight on PAP" if dataset == "ACM" else "DBLP: weight on APA"
        ax_a.plot(x, mean, marker="o", markersize=3, linewidth=1.4, color=COLORS[dataset], label=label)
        ax_a.fill_between(x, mean - std, mean + std, color=COLORS[dataset], alpha=0.16, linewidth=0)
        chosen = selection["selected"]["weight_path_1"]
        chosen_mean = selection["selected"]["macro_f1_mean"]
        ax_a.scatter([chosen], [chosen_mean], s=34, facecolor="white", edgecolor=COLORS[dataset], linewidth=1.2, zorder=4)
    ax_a.set_xlabel("Weight assigned to the first path")
    ax_a.set_ylabel("Validation Macro-F1")
    ax_a.set_xticks(np.linspace(0, 1, 6))
    ax_a.set_ylim(0.10, 0.80)
    ax_a.legend(loc="lower center", ncol=1)
    ax_a.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax_a.set_title("Training-only fusion-weight selection", loc="left", fontweight="bold")
    panel_label(ax_a, "a")

    # b: class-level evidence, preserving every official test class.
    heat_rows = []
    heat_labels = []
    for dataset in ("ACM", "DBLP"):
        classes = sorted(rows[dataset]["test_metrics"]["DWASim"]["path_1"]["per_class_f1"], key=int)
        for cls in classes:
            heat_rows.append(
                [
                    rows[dataset]["test_metrics"]["DWASim"][variant]["per_class_f1"][cls]
                    for variant in ("path_1", "path_2", "validation_selected_fusion")
                ]
            )
            heat_labels.append(f"{dataset} class {cls}")
    heat = np.asarray(heat_rows)
    cmap = LinearSegmentedColormap.from_list("blue_muted", ["#F2F4F5", "#A8C5D7", "#356F96"])
    image = ax_b.imshow(heat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax_b.set_xticks([0, 1, 2], ["Path 1", "Path 2", "Validated"])
    ax_b.set_yticks(np.arange(len(heat_labels)), heat_labels)
    for row in range(heat.shape[0]):
        for col in range(heat.shape[1]):
            color = "white" if heat[row, col] > 0.62 else "#26343D"
            ax_b.text(
                col,
                row,
                f"{heat[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color=color,
            )
    ax_b.axhline(2.5, color="white", linewidth=2.0)
    ax_b.tick_params(length=0)
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax_b, fraction=0.036, pad=0.025)
    cbar.set_label("Per-class F1")
    cbar.outline.set_linewidth(0.5)
    ax_b.set_title("Class-level effect of DWASim fusion", loc="left", fontweight="bold")
    panel_label(ax_b, "b")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Wrote figure bundle to {OUTPUT.parent}")


if __name__ == "__main__":
    main()
