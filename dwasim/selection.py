"""Reference-only model selection and fixed candidate grids."""
from __future__ import annotations
import time
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from dwasim.evaluation import evaluate_affinity
from dwasim.fusion import component_views, weighted_view, adaptive_affinity, relative_affinity


def split_positions(labels: np.ndarray, seeds: list[int], fraction: float):
    splits = []
    for seed in seeds:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        reference, validation = next(splitter.split(np.zeros(labels.size), labels))
        splits.append((reference, validation))
    return splits


def simplex_weights(count: int, step: float) -> list[tuple[float, ...]]:
    """Enumerate a deterministic nonnegative simplex grid."""
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("step must divide one exactly")

    rows: list[tuple[float, ...]] = []

    def visit(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            rows.append(tuple(value / units for value in (*prefix, remaining)))
            return
        for value in range(remaining + 1):
            visit([*prefix, value], remaining - value, slots - 1)

    visit([], units, count)
    return rows


def select_path_beta(
    components: dict[str, np.ndarray],
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    splits,
    beta_grid: np.ndarray,
    k: int,
    formulation: str,
) -> dict:
    rows = []
    for beta in beta_grid:
        if formulation == "adaptive":
            affinity = adaptive_affinity(
                components["train_jaccard"], components["train_bray"], float(beta)
            )
        elif formulation == "relative":
            affinity = relative_affinity(
                components["train_jaccard"], components["train_bray"], float(beta)
            )
        else:
            raise ValueError(formulation)
        macro_scores = []
        ndcg_scores = []
        for reference, validation in splits:
            metrics, _, _ = evaluate_affinity(
                affinity[np.ix_(validation, reference)],
                train_ids[reference],
                train_labels[reference],
                train_labels[validation],
                k,
            )
            macro_scores.append(metrics["macro_f1"])
            ndcg_scores.append(metrics[f"ndcg_at_{k}"])
        rows.append(
            {
                "beta": float(beta),
                "macro_f1_mean": float(np.mean(macro_scores)),
                "macro_f1_standard_error": float(
                    np.std(macro_scores, ddof=1) / np.sqrt(len(macro_scores))
                ),
                f"ndcg_at_{k}_mean": float(np.mean(ndcg_scores)),
            }
        )
    best = max(
        rows,
        key=lambda row: (
            row["macro_f1_mean"],
            row[f"ndcg_at_{k}_mean"],
            -row["beta"],
        ),
    )
    return {"selected": best, "grid": rows}


def select_path_mixture(
    components: dict[str, np.ndarray],
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    splits,
    component_weights: list[tuple[float, ...]],
    k: int,
) -> dict:
    """Select support, magnitude, and directional component weights."""
    support = 1.0 - components["train_jaccard"]
    magnitude = 1.0 - components["train_bray"]
    direction = components["train_cosine"]
    rows = []
    for weights in component_weights:
        affinity = (
            weights[0] * support
            + weights[1] * magnitude
            + weights[2] * direction
        )
        macro_scores = []
        ndcg_scores = []
        for reference, validation in splits:
            metrics, _, _ = evaluate_affinity(
                affinity[np.ix_(validation, reference)],
                train_ids[reference],
                train_labels[reference],
                train_labels[validation],
                k,
            )
            macro_scores.append(metrics["macro_f1"])
            ndcg_scores.append(metrics[f"ndcg_at_{k}"])
        rows.append(
            {
                "weights_support_magnitude_direction": list(weights),
                "macro_f1_mean": float(np.mean(macro_scores)),
                "macro_f1_standard_error": float(
                    np.std(macro_scores, ddof=1) / np.sqrt(len(macro_scores))
                ),
                f"ndcg_at_{k}_mean": float(np.mean(ndcg_scores)),
            }
        )
    best = max(
        rows,
        key=lambda row: (
            row["macro_f1_mean"],
            row[f"ndcg_at_{k}_mean"],
            -sum(
                weight > 0
                for weight in row["weights_support_magnitude_direction"]
            ),
            row["weights_support_magnitude_direction"][1],
        ),
    )
    return {"selected": best, "grid": rows}


def select_fusion(
    train_views: list[np.ndarray],
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    splits,
    weight_grid: list[tuple[float, ...]],
    k: int,
) -> dict:
    rows = []
    for weights in weight_grid:
        macro_scores = []
        ndcg_scores = []
        for reference, validation in splits:
            fused = sum(
                weight * view[np.ix_(validation, reference)]
                for weight, view in zip(weights, train_views)
            )
            metrics, _, _ = evaluate_affinity(
                fused,
                train_ids[reference],
                train_labels[reference],
                train_labels[validation],
                k,
            )
            macro_scores.append(metrics["macro_f1"])
            ndcg_scores.append(metrics[f"ndcg_at_{k}"])
        rows.append(
            {
                "weights": list(weights),
                "macro_f1_mean": float(np.mean(macro_scores)),
                "macro_f1_standard_error": float(
                    np.std(macro_scores, ddof=1) / np.sqrt(len(macro_scores))
                ),
                f"ndcg_at_{k}_mean": float(np.mean(ndcg_scores)),
            }
        )
    best = max(
        rows,
        key=lambda row: (
            row["macro_f1_mean"],
            row[f"ndcg_at_{k}_mean"],
            -sum(weight > 0 for weight in row["weights"]),
        ),
    )
    return {"selected": best, "grid": rows}


INNER_SEEDS = list(range(20250803, 20250813))


def joint_candidates():
    # Sixty candidates, equal to 3*15 path-component candidates plus 15 fusion candidates.
    rng = np.random.default_rng(20260905)
    return [*np.eye(9).tolist(), [1 / 9] * 9, *rng.dirichlet(np.ones(9), size=50).tolist()]


def select_linear(views, candidates, ids, labels, splits, k=10):
    rows = []
    for candidate in candidates:
        affinity = weighted_view(views, candidate)
        metrics = [evaluate_affinity(affinity[np.ix_(validation, reference)], ids[reference],
                    labels[reference], labels[validation], k)[0] for reference, validation in splits]
        rows.append({"weights": list(candidate),
                     "macro_f1_mean": float(np.mean([m["macro_f1"] for m in metrics])),
                     "ndcg_at_10_mean": float(np.mean([m["ndcg_at_10"] for m in metrics]))})
    best = max(rows, key=lambda row: (row["macro_f1_mean"], row["ndcg_at_10_mean"],
                                      -np.count_nonzero(row["weights"])))
    return {"selected": best, "grid": rows}


def fit_family(family, components, ids, labels, splits, paths):
    started = time.perf_counter()
    base = [component_views(c) for c in components]
    component_grid = simplex_weights(3, 0.25)
    if family == "JointBudget60":
        selection = select_linear([view for path in base for view in path], joint_candidates(), ids, labels, splits)
        gamma = np.asarray(selection["selected"]["weights"]).reshape(3, 3)
        return {"gamma": gamma.tolist(), "component_weights": None, "path_weights": gamma.sum(axis=1).tolist(),
                "candidate_count": 60, "validation_macro_f1": selection["selected"]["macro_f1_mean"],
                "selection_seconds": time.perf_counter() - started, "joint_grid": selection["grid"]}
    if family == "SharedComponentMixture":
        shared = [sum(path[j] for path in base) / len(paths) for j in range(3)]
        selected = select_linear(shared, component_grid, ids, labels, splits)
        chosen = [selected["selected"]["weights"]] * len(paths)
        candidate_count = len(component_grid)
    else:
        chosen, candidate_count = [], 0
        for path, values in zip(paths, components):
            if family == "Cosine":
                chosen.append([0., 0., 1.])
                continue
            if family == "MagnitudeOnly":
                chosen.append([0., 1., 0.])
                continue
            if family == "Jaccard":
                chosen.append([1., 0., 0.])
                continue
            if family == "FixedEqualMixture":
                chosen.append([1 / 3] * 3)
                continue
            grid = component_grid
            if family == "OneComponentSelector":
                grid = [tuple(row) for row in np.eye(3)]
            elif family in ("NoSupport", "NoMagnitude", "NoDirection"):
                omitted = ("NoSupport", "NoMagnitude", "NoDirection").index(family)
                grid = [row for row in component_grid if row[omitted] == 0]
            elif family == "PTPNoSupport" and path == "PTP":
                grid = [row for row in component_grid if row[0] == 0]
            selection = select_path_mixture(values, ids, labels, splits, grid, 10)
            chosen.append(selection["selected"]["weights_support_magnitude_direction"])
            candidate_count += len(grid)
    train_views = [weighted_view(view, weights) for view, weights in zip(base, chosen)]
    fusion_grid = simplex_weights(len(paths), 0.25)
    fusion = select_fusion(train_views, ids, labels, splits, fusion_grid, 10)
    path_weights = fusion["selected"]["weights"]
    gamma = np.asarray(chosen) * np.asarray(path_weights)[:, None]
    return {"gamma": gamma.tolist(), "component_weights": chosen, "path_weights": path_weights,
            "candidate_count": candidate_count + len(fusion_grid),
            "validation_macro_f1": fusion["selected"]["macro_f1_mean"],
            "selection_seconds": time.perf_counter() - started}


def best_row(rows, component=False):
    return max(rows, key=lambda r: (r["macro_f1_mean"], r["ndcg_at_10_mean"],
                                    -np.count_nonzero(r["weights"]),
                                    r["weights"][1] if component else 0))
