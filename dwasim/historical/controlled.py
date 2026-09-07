"""Historical global-form controlled experiments; not the final three-component evaluation.

The generator creates a typed source--object interaction matrix. Source nodes
belong to latent classes, and each class has a disjoint block of preferred
objects. Cross-class noise and within-class interaction density are controlled
independently. No manuscript value is hard-coded: CSV tables and figures are
generated from the simulations below.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import f1_score, normalized_mutual_info_score


from dwasim.paths import RESULTS_ROOT
ROOT = RESULTS_ROOT / "historical_controlled"
ROOT.mkdir(parents=True, exist_ok=True)
SEEDS = tuple(range(10))
NOISE_LEVELS = np.linspace(0.0, 1.0, 6)
LAMBDA_LEVELS = np.linspace(0.0, 1.0, 11)
N_CLASSES = 4
N_PER_CLASS = 60
N_OBJECTS = 120
CORE_SIZE = 24
TOP_K = 10


def generate_profiles(seed: int, density: float, noise: float):
    """Generate a nonnegative path-count profile matrix and class labels."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(N_CLASSES), N_PER_CLASS)
    n = labels.size
    profiles = np.zeros((n, N_OBJECTS), dtype=np.float64)

    for i, cls in enumerate(labels):
        core = np.arange(cls * CORE_SIZE, (cls + 1) * CORE_SIZE)
        outside = np.setdiff1d(np.arange(N_OBJECTS), core, assume_unique=True)

        core_mask = rng.random(core.size) < density
        outside_mask = rng.random(outside.size) < noise * density
        profiles[i, core[core_mask]] = 1.0 + rng.poisson(3.0, core_mask.sum())
        profiles[i, outside[outside_mask]] = 1.0 + rng.poisson(
            3.0, outside_mask.sum()
        )

        # Avoid empty profiles in very sparse random draws.
        if not np.any(profiles[i]):
            j = rng.choice(core)
            profiles[i, j] = 1.0

    return profiles, labels


def profile_discrepancies(x: np.ndarray):
    diff = np.abs(x[:, None, :] - x[None, :, :])
    h = np.count_nonzero(diff, axis=2).astype(np.float64)
    l1 = diff.sum(axis=2)
    b0 = max(float(h.max()), 1.0)
    value_range = float(x.max() - x.min())
    b1 = max(b0 * value_range, 1.0)
    return h, l1, b0, b1


def dwasim_from_discrepancies(h, l1, b0, b1, lam: float):
    denominator = lam * b0 + (1.0 - lam) * b1
    similarity = 1.0 - (lam * h + (1.0 - lam) * l1) / denominator
    return np.clip((similarity + similarity.T) / 2.0, 0.0, 1.0)


def pathsim(x: np.ndarray):
    counts = x @ x.T
    diagonal = np.diag(counts)
    denominator = diagonal[:, None] + diagonal[None, :]
    similarity = np.divide(
        2.0 * counts,
        denominator,
        out=np.zeros_like(counts),
        where=denominator > 0,
    )
    np.fill_diagonal(similarity, 1.0)
    return np.clip((similarity + similarity.T) / 2.0, 0.0, 1.0)


def hetesim(x: np.ndarray):
    norms = np.linalg.norm(x, axis=1)
    denominator = norms[:, None] * norms[None, :]
    similarity = np.divide(
        x @ x.T,
        denominator,
        out=np.zeros((x.shape[0], x.shape[0]), dtype=np.float64),
        where=denominator > 0,
    )
    np.fill_diagonal(similarity, 1.0)
    return np.clip((similarity + similarity.T) / 2.0, 0.0, 1.0)


def topk_macro_f1(similarity: np.ndarray, labels: np.ndarray, k: int = TOP_K):
    scores = similarity.copy()
    np.fill_diagonal(scores, -np.inf)
    neighbor_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    predictions = np.empty(labels.size, dtype=int)
    for i, neighbors in enumerate(neighbor_idx):
        votes = np.bincount(labels[neighbors], minlength=N_CLASSES)
        tied = np.flatnonzero(votes == votes.max())
        if tied.size == 1:
            predictions[i] = tied[0]
        else:
            weighted = [scores[i, neighbors[labels[neighbors] == c]].sum() for c in tied]
            predictions[i] = tied[int(np.argmax(weighted))]
    return f1_score(labels, predictions, average="macro")


def spectral_nmi(similarity: np.ndarray, labels: np.ndarray, seed: int):
    affinity = similarity.copy()
    np.fill_diagonal(affinity, 1.0)
    prediction = SpectralClustering(
        n_clusters=N_CLASSES,
        affinity="precomputed",
        assign_labels="kmeans",
        n_init=20,
        random_state=seed,
    ).fit_predict(affinity)
    return normalized_mutual_info_score(labels, prediction)


def summarize(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def run_noise_robustness():
    rows = []
    raw = {}
    for noise in NOISE_LEVELS:
        per_method = {name: {"f1": [], "nmi": []} for name in ("DWASim", "PathSim", "HeteSim")}
        for seed in SEEDS:
            x, labels = generate_profiles(seed, density=0.65, noise=float(noise))
            h, l1, b0, b1 = profile_discrepancies(x)
            matrices = {
                "DWASim": dwasim_from_discrepancies(h, l1, b0, b1, 0.5),
                "PathSim": pathsim(x),
                "HeteSim": hetesim(x),
            }
            for name, matrix in matrices.items():
                per_method[name]["f1"].append(topk_macro_f1(matrix, labels))
                per_method[name]["nmi"].append(spectral_nmi(matrix, labels, seed))

        raw[float(noise)] = per_method
        for name, metrics in per_method.items():
            f1_mean, f1_std = summarize(metrics["f1"])
            nmi_mean, nmi_std = summarize(metrics["nmi"])
            rows.append(
                {
                    "experiment": "noise_robustness",
                    "noise": float(noise),
                    "density": 0.65,
                    "lambda": 0.5 if name == "DWASim" else "",
                    "method": name,
                    "macro_f1_mean": f1_mean,
                    "macro_f1_std": f1_std,
                    "nmi_mean": nmi_mean,
                    "nmi_std": nmi_std,
                }
            )
    return rows, raw


def run_lambda_sensitivity():
    """Evaluate the support/count trade-off over increasing noise."""
    cube = np.zeros((len(NOISE_LEVELS), len(LAMBDA_LEVELS), len(SEEDS)))
    rows = []
    for di, noise in enumerate(NOISE_LEVELS):
        for si, seed in enumerate(SEEDS):
            x, labels = generate_profiles(seed + 100, density=0.65, noise=float(noise))
            h, l1, b0, b1 = profile_discrepancies(x)
            for li, lam in enumerate(LAMBDA_LEVELS):
                sim = dwasim_from_discrepancies(h, l1, b0, b1, float(lam))
                cube[di, li, si] = topk_macro_f1(sim, labels)
        for li, lam in enumerate(LAMBDA_LEVELS):
            mean, std = summarize(cube[di, li])
            rows.append(
                {
                    "experiment": "lambda_sensitivity",
                    "noise": float(noise),
                    "density": 0.65,
                    "support_share": "",
                    "lambda": float(lam),
                    "method": "DWASim",
                    "macro_f1_mean": mean,
                    "macro_f1_std": std,
                    "nmi_mean": "",
                    "nmi_std": "",
                }
            )
    return rows, cube


def configure_plotting():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
            "legend.frameon": False,
        }
    )


def plot_noise_robustness(raw):
    configure_plotting()
    colors = {"DWASim": "#3F5F8A", "PathSim": "#C36A42", "HeteSim": "#4E8C7B"}
    markers = {"DWASim": "o", "PathSim": "s", "HeteSim": "^"}
    linestyles = {"DWASim": "-", "PathSim": "--", "HeteSim": ":"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for metric, ax, ylabel, panel in (
        ("f1", axes[0], "Top-10 Macro-F1", "(a) Label prediction"),
        ("nmi", axes[1], "Spectral-clustering NMI", "(b) Clustering"),
    ):
        for method in ("DWASim", "PathSim", "HeteSim"):
            means, stds = [], []
            for noise in NOISE_LEVELS:
                mean, std = summarize(raw[float(noise)][method][metric])
                means.append(mean)
                stds.append(std)
            ax.errorbar(
                NOISE_LEVELS,
                means,
                yerr=stds,
                color=colors[method],
                marker=markers[method],
                linestyle=linestyles[method],
                linewidth=1.05,
                markersize=3.6,
                markerfacecolor="white" if method != "DWASim" else colors[method],
                markeredgewidth=0.75,
                elinewidth=0.8,
                capthick=0.8,
                capsize=2.0,
                label=method,
            )
        ax.set_xlabel("Cross-class noise probability")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.04)
        ax.grid(axis="y", color="#D6D6D6", linewidth=0.35, alpha=0.55)
        ax.text(-0.13, 1.04, panel[1], transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom")
    axes[0].legend(frameon=False, loc="upper right")
    output = ROOT / "Fig4_synthetic_robustness"
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lambda_sensitivity(cube):
    configure_plotting()
    mean_grid = cube.mean(axis=2)
    regret = mean_grid.max(axis=1, keepdims=True) - mean_grid
    noise_colors = plt.colormaps["cividis"](np.linspace(0.12, 0.90, len(NOISE_LEVELS)))
    noise_markers = ("o", "s", "^", "D", "v", "P")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
        constrained_layout=True,
    )
    for row, (noise, color, marker) in enumerate(
        zip(NOISE_LEVELS, noise_colors, noise_markers)
    ):
        axes[0].plot(
            LAMBDA_LEVELS,
            regret[row],
            color=color,
            linewidth=1.15,
            marker=marker,
            markersize=2.8,
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=rf"$\eta={noise:.1f}$",
        )
    axes[0].set_xlabel(r"Trade-off parameter $\lambda$")
    axes[0].set_ylabel("Macro-F1 regret")
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.001, max(0.033, float(regret.max()) * 1.12))
    axes[0].set_xticks(np.linspace(0, 1, 6))
    axes[0].grid(axis="y", color="#D9D7D2", linewidth=0.35, alpha=0.60)
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.50, 1.02),
        ncol=3,
        handlelength=1.5,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    axes[0].text(-0.13, 1.04, "a", transform=axes[0].transAxes, fontsize=9,
                 color="black", fontweight="bold", va="bottom")

    score_range = mean_grid.max(axis=1) - mean_grid.min(axis=1)
    y_positions = np.arange(len(NOISE_LEVELS))[::-1]
    for y, noise, value, color in zip(y_positions, NOISE_LEVELS, score_range, noise_colors):
        axes[1].plot([0.0, value], [y, y], color="#D9D7D2", linewidth=0.8, zorder=1)
        axes[1].scatter(
            [value],
            [y],
            s=27,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        axes[1].annotate(
            f"{value:.3f}",
            (value, y),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.7,
            color="#3F454A",
        )
    axes[1].set_yticks(y_positions, [rf"$\eta={value:.1f}$" for value in NOISE_LEVELS])
    axes[1].set_xlabel(r"Macro-F1 range over $\lambda$")
    axes[1].set_xlim(-0.001, max(0.036, float(score_range.max()) * 1.30))
    axes[1].grid(axis="x", color="#D9D7D2", linewidth=0.35, alpha=0.60)
    axes[1].tick_params(axis="y", length=0)
    maximum = int(np.argmax(score_range))
    axes[1].scatter(
        [score_range[maximum]],
        [y_positions[maximum]],
        s=52,
        facecolor="none",
        edgecolor="#C36A42",
        linewidth=0.9,
        zorder=4,
    )
    axes[1].text(-0.13, 1.04, "b", transform=axes[1].transAxes, fontsize=9,
                 fontweight="bold", va="bottom")
    output = ROOT / "Fig5_lambda_sensitivity"
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_results(rows):
    fields = [
        "experiment",
        "noise",
        "density",
        "support_share",
        "lambda",
        "method",
        "macro_f1_mean",
        "macro_f1_std",
        "nmi_mean",
        "nmi_std",
    ]
    with (ROOT / "synthetic_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(noise_raw, cube, elapsed):
    target_noise = min(noise_raw, key=lambda value: abs(value - 0.60))
    lines = [
        "Synthetic experiment summary (10 seeds)",
        f"Runtime: {elapsed:.2f} seconds",
        f"Controlled setting for method comparison: density=0.65, noise={target_noise:.2f}",
    ]
    for method in ("DWASim", "PathSim", "HeteSim"):
        f1_mean, f1_std = summarize(noise_raw[target_noise][method]["f1"])
        nmi_mean, nmi_std = summarize(noise_raw[target_noise][method]["nmi"])
        lines.append(
            f"{method}: Macro-F1={f1_mean:.4f} +/- {f1_std:.4f}; "
            f"NMI={nmi_mean:.4f} +/- {nmi_std:.4f}"
        )
    mean_grid = cube.mean(axis=2)
    lines.append("Best lambda by cross-class noise level:")
    for di, noise in enumerate(NOISE_LEVELS):
        li = int(mean_grid[di].argmax())
        lines.append(
            f"noise={noise:.1f}: lambda={LAMBDA_LEVELS[li]:.1f}, "
            f"Macro-F1={mean_grid[di, li]:.4f}"
        )
    (ROOT / "synthetic_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    start = time.perf_counter()
    noise_rows, noise_raw = run_noise_robustness()
    lambda_rows, cube = run_lambda_sensitivity()
    write_results(noise_rows + lambda_rows)
    plot_noise_robustness(noise_raw)
    plot_lambda_sensitivity(cube)
    write_summary(noise_raw, cube, time.perf_counter() - start)


if __name__ == "__main__":
    main()
