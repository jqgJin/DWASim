"""Training-only nested optimization for the DWASim task interface.

This runner deliberately does not load the official test labels.  It compares
the manuscript's current two-path, uncalibrated majority-vote pipeline with an
enhanced pipeline that adds one schema-supported path, calibrates each path's
query-wise affinity scale, and selects neighbour count and weighted voting.

Every reported prediction is out-of-fold: a five-fold outer split estimates
performance, while four inner folds select all path and task parameters.  The
result is a development diagnostic, not a replacement for an untouched
external benchmark.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from reproduce_original import (
    CACHE_ROOT,
    PATHS,
    RESULTS_ROOT,
    full_profile,
    multiply_chain,
)
from run_corrected_protocol import (
    PAPER_PATHS,
    deterministic_topk,
    load_split,
    majority_vote,
    metric_record,
)
from similarity_baselines import BASELINE_CACHE_TAG, symmetric_path_affinities


CANDIDATE_PATHS = {
    "ACM": ("PAP", "PSP", "PTP"),
    "DBLP": ("APA", "APTPA", "APVPA"),
}


def load_training_discrepancies(dataset: str, path_name: str) -> dict:
    """Load or compute discrepancies among official training nodes only."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_ROOT / f"nested_train_discrepancies_{dataset}_{path_name}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return {key: cached[key] for key in cached.files}

    train_ids, _, _, _ = load_split(dataset)
    matrix = full_profile(dataset, path_name)
    profiles = matrix[train_ids].toarray().astype(np.float64, copy=False)
    dimensions = profiles.shape[1]
    hamming = np.rint(cdist(profiles, profiles, metric="hamming") * dimensions)
    magnitude = cdist(profiles, profiles, metric="cityblock")
    value_min = 0.0 if matrix.nnz < matrix.shape[0] * matrix.shape[1] else float(matrix.data.min())
    value_max = float(matrix.data.max()) if matrix.nnz else 0.0
    value_range = max(value_max - value_min, 1.0)
    np.savez_compressed(
        cache_path,
        hamming=hamming.astype(np.float32),
        magnitude=magnitude.astype(np.float32),
        dimensions=np.asarray(float(dimensions)),
        value_range=np.asarray(value_range),
    )
    return {
        "hamming": hamming,
        "magnitude": magnitude,
        "dimensions": float(dimensions),
        "value_range": value_range,
    }


def upper_bound_affinity(discrepancy: dict, alpha: float) -> np.ndarray:
    """DWASim affinity using the path dimension as a safe Hamming bound.

    Replacing the exact B0 by the dimension multiplies all distances for one
    path by a positive path-level constant.  It therefore leaves single-path
    rankings unchanged.  The enhanced pipeline calibrates every path before
    fusion, so the constant cannot privilege a path merely through scale.
    """
    dimensions = float(discrepancy["dimensions"])
    value_range = float(discrepancy["value_range"])
    distance = (
        float(alpha) * discrepancy["hamming"] / dimensions
        + (1.0 - float(alpha))
        * discrepancy["magnitude"]
        / (dimensions * value_range)
    )
    return np.clip(1.0 - distance, 0.0, 1.0)


def load_training_baseline_affinities(dataset: str, path_name: str) -> dict:
    """Compute PathSim and HeteSim among training nodes without test labels."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_ROOT / (
        f"nested_train_baselines_{BASELINE_CACHE_TAG}_{dataset}_{path_name}.npz"
    )
    if cache_path.exists():
        cached = np.load(cache_path)
        return {key: cached[key] for key in cached.files}

    train_ids, _, _, _ = load_split(dataset)
    config = PATHS[dataset][path_name]
    raw = multiply_chain(dataset, config["half"], transition=False)
    transition = multiply_chain(dataset, config["half"], transition=True)
    affinities = symmetric_path_affinities(
        raw,
        transition,
        train_ids,
        train_ids,
    )
    np.savez_compressed(
        cache_path,
        PathSim=affinities["PathSim"].astype(np.float32),
        HeteSim=affinities["HeteSim"].astype(np.float32),
    )
    return affinities


def load_relative_components(dataset: str, path_name: str, discrepancy: dict) -> dict:
    """Build bounded pair-relative support and magnitude discrepancies."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_ROOT / f"nested_relative_components_{dataset}_{path_name}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return {key: cached[key] for key in cached.files}

    train_ids, _, _, _ = load_split(dataset)
    profiles = full_profile(dataset, path_name)[train_ids].tocsr()
    support = profiles.copy()
    support.data = np.ones_like(support.data, dtype=np.float64)
    support = support.astype(np.float64)
    intersection = support.dot(support.T).toarray()
    support_count = np.asarray(support.sum(axis=1)).ravel()
    union = support_count[:, None] + support_count[None, :] - intersection
    jaccard = np.divide(
        union - intersection,
        union,
        out=np.zeros_like(union),
        where=union > 0,
    )

    activity = np.asarray(profiles.sum(axis=1)).ravel().astype(np.float64)
    activity_sum = activity[:, None] + activity[None, :]
    bray_curtis = np.divide(
        discrepancy["magnitude"],
        activity_sum,
        out=np.zeros_like(activity_sum),
        where=activity_sum > 0,
    )
    jaccard = np.clip(jaccard, 0.0, 1.0)
    bray_curtis = np.clip(bray_curtis, 0.0, 1.0)
    np.savez_compressed(
        cache_path,
        jaccard=jaccard.astype(np.float32),
        bray_curtis=bray_curtis.astype(np.float32),
    )
    return {"jaccard": jaccard, "bray_curtis": bray_curtis}


def relative_dwasim_affinity(components: dict, support_weight: float) -> np.ndarray:
    """Pair-relative DWASim candidate with two intrinsically bounded terms."""
    distance = (
        float(support_weight) * components["jaccard"]
        + (1.0 - float(support_weight)) * components["bray_curtis"]
    )
    return np.clip(1.0 - distance, 0.0, 1.0)


def exact_affinity_from_cache(dataset: str, path_name: str, alpha: float) -> np.ndarray:
    """Reconstruct the current manuscript affinity for a paper path."""
    cache_path = CACHE_ROOT / f"heldout_discrepancies_{dataset}_{path_name}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Run the locked real-data fusion protocol first; missing {cache_path.name}"
        )
    cached = np.load(cache_path)
    b0 = float(cached["b0"])
    b1 = float(cached["b1"])
    distance = (
        float(alpha) * cached["train_h"] / b0
        + (1.0 - float(alpha)) * cached["train_l"] / b1
    )
    return np.clip(1.0 - distance, 0.0, 1.0)


def calibrate_rows(values: np.ndarray, method: str) -> np.ndarray:
    """Place heterogeneous path affinities on a comparable query-wise scale."""
    values = np.asarray(values, dtype=np.float64)
    if method == "rank":
        if values.shape[1] <= 1:
            return np.zeros_like(values)
        return (rankdata(values, axis=1, method="average") - 1.0) / (
            values.shape[1] - 1.0
        )
    if method == "minmax":
        lower = values.min(axis=1, keepdims=True)
        span = values.max(axis=1, keepdims=True) - lower
        return np.divide(
            values - lower,
            span,
            out=np.zeros_like(values),
            where=span > 0,
        )
    raise ValueError(f"Unknown calibration method: {method}")


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


def weighted_vote_from_neighbours(
    affinity: np.ndarray,
    neighbours: np.ndarray,
    reference_labels: np.ndarray,
    gamma: float,
    prior_power: float,
) -> np.ndarray:
    """Similarity-weighted vote with optional training-prior correction."""
    classes, counts = np.unique(reference_labels, return_counts=True)
    priors = counts.astype(np.float64) / reference_labels.size
    neighbour_labels = reference_labels[neighbours]
    neighbour_values = np.take_along_axis(affinity, neighbours, axis=1)
    weights = (
        np.ones_like(neighbour_values, dtype=np.float64)
        if gamma == 0.0
        else np.clip(neighbour_values, 0.0, None) ** gamma
    )

    scores = np.empty((neighbours.shape[0], classes.size), dtype=np.float64)
    first_ranks = np.full(
        (neighbours.shape[0], classes.size),
        neighbours.shape[1] + 1,
        dtype=np.int64,
    )
    for column, (label, prior) in enumerate(zip(classes, priors)):
        mask = neighbour_labels == label
        scores[:, column] = (weights * mask).sum(axis=1) / (prior**prior_power)
        present = mask.any(axis=1)
        first_ranks[present, column] = np.argmax(mask[present], axis=1)

    maximum = scores.max(axis=1, keepdims=True)
    tied = np.isclose(scores, maximum)
    first_ranks[~tied] = neighbours.shape[1] + 1
    nearest_rank = first_ranks.min(axis=1, keepdims=True)
    eligible = tied & (first_ranks == nearest_rank)
    return classes[np.argmax(eligible, axis=1)].astype(np.int64, copy=False)


def split_score(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    return (
        float(f1_score(truth, prediction, average="macro", zero_division=0)),
        float(accuracy_score(truth, prediction)),
    )


def select_path_alphas(
    alpha_affinities: dict[str, dict[float, np.ndarray]],
    paths: tuple[str, ...],
    train_ids: np.ndarray,
    labels: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    k: int,
) -> tuple[dict[str, float], dict[str, list[dict]]]:
    selections = {}
    grids = {}
    for path_name in paths:
        rows = []
        for alpha, affinity in alpha_affinities[path_name].items():
            macro_scores = []
            accuracy_scores = []
            for reference, validation in inner_splits:
                submatrix = affinity[np.ix_(validation, reference)]
                neighbours = deterministic_topk(
                    submatrix,
                    train_ids[reference],
                    min(k, reference.size),
                    largest=True,
                )
                prediction = majority_vote(neighbours, labels[reference])
                macro, accuracy = split_score(labels[validation], prediction)
                macro_scores.append(macro)
                accuracy_scores.append(accuracy)
            rows.append(
                {
                    "alpha": float(alpha),
                    "macro_f1_mean": float(np.mean(macro_scores)),
                    "macro_f1_std": float(np.std(macro_scores, ddof=1)),
                    "accuracy_mean": float(np.mean(accuracy_scores)),
                }
            )
        selected = max(
            rows,
            key=lambda item: (
                item["macro_f1_mean"],
                item["accuracy_mean"],
                -abs(item["alpha"] - 0.5),
            ),
        )
        selections[path_name] = selected["alpha"]
        grids[path_name] = rows
    return selections, grids


def select_current_fusion(
    affinities: list[np.ndarray],
    train_ids: np.ndarray,
    labels: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    rows = []
    for weight in np.linspace(0.0, 1.0, 11):
        macro_scores = []
        accuracy_scores = []
        for reference, validation in inner_splits:
            fused = (
                float(weight) * affinities[0][np.ix_(validation, reference)]
                + (1.0 - float(weight))
                * affinities[1][np.ix_(validation, reference)]
            )
            neighbours = deterministic_topk(
                fused, train_ids[reference], min(10, reference.size), largest=True
            )
            prediction = majority_vote(neighbours, labels[reference])
            macro, accuracy = split_score(labels[validation], prediction)
            macro_scores.append(macro)
            accuracy_scores.append(accuracy)
        rows.append(
            {
                "weight_path_1": float(weight),
                "macro_f1_mean": float(np.mean(macro_scores)),
                "macro_f1_std": float(np.std(macro_scores, ddof=1)),
                "accuracy_mean": float(np.mean(accuracy_scores)),
            }
        )
    selected = max(
        rows,
        key=lambda item: (
            item["macro_f1_mean"],
            item["accuracy_mean"],
            -abs(item["weight_path_1"] - 0.5),
        ),
    )
    return {"selected": selected, "grid": rows}


def select_enhanced_config(
    affinities: list[np.ndarray],
    train_ids: np.ndarray,
    labels: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    weights: list[tuple[float, ...]],
    k_values: tuple[int, ...],
    gammas: tuple[float, ...],
    prior_powers: tuple[float, ...],
    calibrations: tuple[str, ...],
) -> dict:
    scores: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: {"macro_f1": [], "accuracy": []}
    )
    maximum_k = max(k_values)

    for reference, validation in inner_splits:
        candidate_ids = train_ids[reference]
        candidate_labels = labels[reference]
        truth = labels[validation]
        raw_views = [affinity[np.ix_(validation, reference)] for affinity in affinities]
        for calibration in calibrations:
            views = [calibrate_rows(view, calibration) for view in raw_views]
            for weight_tuple in weights:
                fused = sum(
                    weight * view for weight, view in zip(weight_tuple, views)
                )
                ordered = deterministic_topk(
                    fused,
                    candidate_ids,
                    min(maximum_k, reference.size),
                    largest=True,
                )
                for k in k_values:
                    neighbours = ordered[:, : min(k, ordered.shape[1])]
                    for gamma in gammas:
                        for prior_power in prior_powers:
                            prediction = weighted_vote_from_neighbours(
                                fused,
                                neighbours,
                                candidate_labels,
                                gamma,
                                prior_power,
                            )
                            macro, accuracy = split_score(truth, prediction)
                            key = (
                                calibration,
                                weight_tuple,
                                int(k),
                                float(gamma),
                                float(prior_power),
                            )
                            scores[key]["macro_f1"].append(macro)
                            scores[key]["accuracy"].append(accuracy)

    rows = []
    for key, values in scores.items():
        calibration, weight_tuple, k, gamma, prior_power = key
        rows.append(
            {
                "calibration": calibration,
                "weights": list(weight_tuple),
                "k": k,
                "gamma": gamma,
                "prior_power": prior_power,
                "macro_f1_mean": float(np.mean(values["macro_f1"])),
                "macro_f1_std": float(np.std(values["macro_f1"], ddof=1)),
                "accuracy_mean": float(np.mean(values["accuracy"])),
            }
        )
    selected = max(
        rows,
        key=lambda item: (
            item["macro_f1_mean"],
            item["accuracy_mean"],
            -sum(weight > 0 for weight in item["weights"]),
            -abs(item["k"] - 10),
            -abs(item["gamma"] - 1.0),
            -item["prior_power"],
            item["calibration"] == "rank",
        ),
    )
    return {"selected": selected, "configuration_count": len(rows)}


def predict_current_outer(
    affinities: list[np.ndarray],
    reference: np.ndarray,
    validation: np.ndarray,
    train_ids: np.ndarray,
    labels: np.ndarray,
    weight: float,
) -> np.ndarray:
    fused = (
        weight * affinities[0][np.ix_(validation, reference)]
        + (1.0 - weight) * affinities[1][np.ix_(validation, reference)]
    )
    neighbours = deterministic_topk(
        fused, train_ids[reference], min(10, reference.size), largest=True
    )
    return majority_vote(neighbours, labels[reference])


def predict_enhanced_outer(
    affinities: list[np.ndarray],
    reference: np.ndarray,
    validation: np.ndarray,
    train_ids: np.ndarray,
    labels: np.ndarray,
    config: dict,
) -> np.ndarray:
    views = [
        calibrate_rows(affinity[np.ix_(validation, reference)], config["calibration"])
        for affinity in affinities
    ]
    fused = sum(weight * view for weight, view in zip(config["weights"], views))
    neighbours = deterministic_topk(
        fused, train_ids[reference], min(config["k"], reference.size), largest=True
    )
    return weighted_vote_from_neighbours(
        fused,
        neighbours,
        labels[reference],
        config["gamma"],
        config["prior_power"],
    )


def paired_stratified_bootstrap(
    truth: np.ndarray,
    enhanced: np.ndarray,
    current: np.ndarray,
    iterations: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(truth == label) for label in np.unique(truth)]
    differences = {"macro_f1": [], "accuracy": []}
    for _ in range(iterations):
        sample = np.concatenate(
            [rng.choice(indices, size=indices.size, replace=True) for indices in by_class]
        )
        for metric in differences:
            if metric == "macro_f1":
                enhanced_score = f1_score(
                    truth[sample], enhanced[sample], average="macro", zero_division=0
                )
                current_score = f1_score(
                    truth[sample], current[sample], average="macro", zero_division=0
                )
            else:
                enhanced_score = accuracy_score(truth[sample], enhanced[sample])
                current_score = accuracy_score(truth[sample], current[sample])
            differences[metric].append(float(enhanced_score - current_score))

    result = {}
    for metric, values in differences.items():
        values = np.asarray(values)
        result[metric] = {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "probability_difference_positive": float(np.mean(values > 0)),
        }
    return result


def summarize_frequency(rows: list[dict], key: str) -> dict[str, int]:
    return dict(Counter(json.dumps(row[key], sort_keys=True) for row in rows))


def aggregate_oof(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    current: np.ndarray,
    bootstrap_iterations: int,
    seed: int,
) -> dict:
    classes = np.unique(truth)
    rows = {}
    for offset, (name, prediction) in enumerate(predictions.items()):
        metrics = metric_record(truth, prediction, classes)
        current_metrics = metric_record(truth, current, classes)
        rows[name] = {
            "metrics": metrics,
            "difference_from_current": {
                "macro_f1": float(
                    metrics["macro_f1"] - current_metrics["macro_f1"]
                ),
                "accuracy": float(metrics["accuracy"] - current_metrics["accuracy"]),
                "bootstrap_95_interval": paired_stratified_bootstrap(
                    truth,
                    prediction,
                    current,
                    bootstrap_iterations,
                    seed + offset,
                ),
            },
        }
    return rows


def run_dataset(
    dataset: str,
    outer_folds: int,
    inner_folds: int,
    seed: int,
    bootstrap_iterations: int,
) -> dict:
    started = time.perf_counter()
    train_ids, train_labels, _, _ = load_split(dataset)
    paths = CANDIDATE_PATHS[dataset]
    paper_paths = PAPER_PATHS[dataset]
    alphas = np.linspace(0.0, 1.0, 11)

    discrepancies = {
        path: load_training_discrepancies(dataset, path) for path in paths
    }
    alpha_affinities = {
        path: {
            float(alpha): upper_bound_affinity(discrepancies[path], float(alpha))
            for alpha in alphas
        }
        for path in paths
    }
    relative_components = {
        path: load_relative_components(dataset, path, discrepancies[path])
        for path in paths
    }
    relative_affinities = {
        path: {
            float(weight): relative_dwasim_affinity(
                relative_components[path], float(weight)
            )
            for weight in alphas
        }
        for path in paths
    }
    classical_affinities = {"PathSim": [], "HeteSim": []}
    for path in paths:
        path_affinities = load_training_baseline_affinities(dataset, path)
        for method in classical_affinities:
            classical_affinities[method].append(path_affinities[method])

    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    current_oof = np.empty(train_labels.size, dtype=np.int64)
    enhanced_oof = np.empty(train_labels.size, dtype=np.int64)
    ablation_oof = {
        "calibrated_existing_paths": np.empty(train_labels.size, dtype=np.int64),
        "new_path_only": np.empty(train_labels.size, dtype=np.int64),
        "all_paths_fixed_vote": np.empty(train_labels.size, dtype=np.int64),
    }
    method_oof = {
        method: np.empty(train_labels.size, dtype=np.int64)
        for method in classical_affinities
    }
    relative_oof = np.empty(train_labels.size, dtype=np.int64)
    fold_records = []
    weight_grid = simplex_weights(len(paths), 0.25)

    for fold, (outer_reference, outer_validation) in enumerate(
        outer.split(np.zeros(train_labels.size), train_labels)
    ):
        inner = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed + 1000 + fold,
        )
        inner_splits = []
        outer_reference_labels = train_labels[outer_reference]
        for inner_reference_local, inner_validation_local in inner.split(
            np.zeros(outer_reference.size), outer_reference_labels
        ):
            inner_splits.append(
                (
                    outer_reference[inner_reference_local],
                    outer_reference[inner_validation_local],
                )
            )

        selected_alphas, alpha_grids = select_path_alphas(
            alpha_affinities,
            paths,
            train_ids,
            train_labels,
            inner_splits,
            k=10,
        )
        selected_relative_weights, relative_weight_grids = select_path_alphas(
            relative_affinities,
            paths,
            train_ids,
            train_labels,
            inner_splits,
            k=10,
        )
        enhanced_affinities = [
            alpha_affinities[path][selected_alphas[path]] for path in paths
        ]
        current_affinities = [
            exact_affinity_from_cache(dataset, path, selected_alphas[path])
            for path in paper_paths
        ]
        selected_relative_affinities = [
            relative_affinities[path][selected_relative_weights[path]]
            for path in paths
        ]

        current_selection = select_current_fusion(
            current_affinities,
            train_ids,
            train_labels,
            inner_splits,
        )
        enhanced_selection = select_enhanced_config(
            enhanced_affinities,
            train_ids,
            train_labels,
            inner_splits,
            weight_grid,
            k_values=(5, 10, 15, 25),
            gammas=(0.0, 1.0, 2.0),
            prior_powers=(0.0, 0.5, 1.0),
            calibrations=("rank", "minmax"),
        )

        ablation_specs = {
            "calibrated_existing_paths": {
                "affinities": enhanced_affinities[:2],
                "weights": simplex_weights(2, 0.25),
                "k_values": (5, 10, 15, 25),
                "gammas": (0.0, 1.0, 2.0),
                "prior_powers": (0.0, 0.5, 1.0),
            },
            "new_path_only": {
                "affinities": enhanced_affinities[2:],
                "weights": [(1.0,)],
                "k_values": (5, 10, 15, 25),
                "gammas": (0.0, 1.0, 2.0),
                "prior_powers": (0.0, 0.5, 1.0),
            },
            "all_paths_fixed_vote": {
                "affinities": enhanced_affinities,
                "weights": weight_grid,
                "k_values": (10,),
                "gammas": (0.0,),
                "prior_powers": (0.0,),
            },
        }
        ablation_selections = {}
        for name, spec in ablation_specs.items():
            ablation_selections[name] = select_enhanced_config(
                spec["affinities"],
                train_ids,
                train_labels,
                inner_splits,
                spec["weights"],
                spec["k_values"],
                spec["gammas"],
                spec["prior_powers"],
                calibrations=("rank", "minmax"),
            )

        method_selections = {}
        for method, method_affinities in classical_affinities.items():
            method_selections[method] = select_enhanced_config(
                method_affinities,
                train_ids,
                train_labels,
                inner_splits,
                weight_grid,
                k_values=(5, 10, 15, 25),
                gammas=(0.0, 1.0, 2.0),
                prior_powers=(0.0, 0.5, 1.0),
                calibrations=("rank", "minmax"),
            )
        relative_selection = select_enhanced_config(
            selected_relative_affinities,
            train_ids,
            train_labels,
            inner_splits,
            weight_grid,
            k_values=(5, 10, 15, 25),
            gammas=(0.0, 1.0, 2.0),
            prior_powers=(0.0, 0.5, 1.0),
            calibrations=("rank", "minmax"),
        )

        current_prediction = predict_current_outer(
            current_affinities,
            outer_reference,
            outer_validation,
            train_ids,
            train_labels,
            current_selection["selected"]["weight_path_1"],
        )
        enhanced_prediction = predict_enhanced_outer(
            enhanced_affinities,
            outer_reference,
            outer_validation,
            train_ids,
            train_labels,
            enhanced_selection["selected"],
        )
        current_oof[outer_validation] = current_prediction
        enhanced_oof[outer_validation] = enhanced_prediction
        ablation_predictions = {}
        for name, spec in ablation_specs.items():
            prediction = predict_enhanced_outer(
                spec["affinities"],
                outer_reference,
                outer_validation,
                train_ids,
                train_labels,
                ablation_selections[name]["selected"],
            )
            ablation_oof[name][outer_validation] = prediction
            ablation_predictions[name] = prediction

        method_predictions = {}
        for method, method_affinities in classical_affinities.items():
            prediction = predict_enhanced_outer(
                method_affinities,
                outer_reference,
                outer_validation,
                train_ids,
                train_labels,
                method_selections[method]["selected"],
            )
            method_oof[method][outer_validation] = prediction
            method_predictions[method] = prediction
        relative_prediction = predict_enhanced_outer(
            selected_relative_affinities,
            outer_reference,
            outer_validation,
            train_ids,
            train_labels,
            relative_selection["selected"],
        )
        relative_oof[outer_validation] = relative_prediction
        current_macro, current_accuracy = split_score(
            train_labels[outer_validation], current_prediction
        )
        enhanced_macro, enhanced_accuracy = split_score(
            train_labels[outer_validation], enhanced_prediction
        )
        fold_records.append(
            {
                "fold": fold,
                "reference_nodes": int(outer_reference.size),
                "validation_nodes": int(outer_validation.size),
                "selected_alphas": selected_alphas,
                "selected_relative_support_weights": selected_relative_weights,
                "alpha_validation": alpha_grids,
                "relative_weight_validation": relative_weight_grids,
                "current_selection": current_selection["selected"],
                "enhanced_selection": enhanced_selection["selected"],
                "ablation_selections": {
                    name: selection["selected"]
                    for name, selection in ablation_selections.items()
                },
                "method_selections": {
                    name: selection["selected"]
                    for name, selection in method_selections.items()
                },
                "relative_selection": relative_selection["selected"],
                "enhanced_configuration_count": enhanced_selection[
                    "configuration_count"
                ],
                "current_metrics": {
                    "macro_f1": current_macro,
                    "accuracy": current_accuracy,
                },
                "enhanced_metrics": {
                    "macro_f1": enhanced_macro,
                    "accuracy": enhanced_accuracy,
                },
                "ablation_metrics": {
                    name: {
                        "macro_f1": split_score(
                            train_labels[outer_validation], prediction
                        )[0],
                        "accuracy": split_score(
                            train_labels[outer_validation], prediction
                        )[1],
                    }
                    for name, prediction in ablation_predictions.items()
                },
                "method_metrics": {
                    name: {
                        "macro_f1": split_score(
                            train_labels[outer_validation], prediction
                        )[0],
                        "accuracy": split_score(
                            train_labels[outer_validation], prediction
                        )[1],
                    }
                    for name, prediction in method_predictions.items()
                },
                "relative_metrics": {
                    "macro_f1": split_score(
                        train_labels[outer_validation], relative_prediction
                    )[0],
                    "accuracy": split_score(
                        train_labels[outer_validation], relative_prediction
                    )[1],
                },
            }
        )

    classes = np.unique(train_labels)
    current_metrics = metric_record(train_labels, current_oof, classes)
    enhanced_metrics = metric_record(train_labels, enhanced_oof, classes)
    paired_ci = paired_stratified_bootstrap(
        train_labels,
        enhanced_oof,
        current_oof,
        bootstrap_iterations,
        seed + 2000,
    )
    ablation_results = aggregate_oof(
        train_labels,
        ablation_oof,
        current_oof,
        bootstrap_iterations,
        seed + 3000,
    )
    method_results = aggregate_oof(
        train_labels,
        method_oof,
        current_oof,
        bootstrap_iterations,
        seed + 4000,
    )
    relative_results = aggregate_oof(
        train_labels,
        {"RelativeDWASim": relative_oof},
        current_oof,
        bootstrap_iterations,
        seed + 5000,
    )["RelativeDWASim"]
    fold_current_macro = np.asarray(
        [row["current_metrics"]["macro_f1"] for row in fold_records]
    )
    fold_enhanced_macro = np.asarray(
        [row["enhanced_metrics"]["macro_f1"] for row in fold_records]
    )
    fold_current_accuracy = np.asarray(
        [row["current_metrics"]["accuracy"] for row in fold_records]
    )
    fold_enhanced_accuracy = np.asarray(
        [row["enhanced_metrics"]["accuracy"] for row in fold_records]
    )

    return {
        "dataset": dataset,
        "candidate_paths": list(paths),
        "training_nodes": int(train_labels.size),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "current_oof_metrics": current_metrics,
        "enhanced_oof_metrics": enhanced_metrics,
        "ablation_oof_results": ablation_results,
        "same_pipeline_method_results": method_results,
        "relative_dwasim_oof_result": relative_results,
        "paired_difference": {
            "macro_f1": float(
                enhanced_metrics["macro_f1"] - current_metrics["macro_f1"]
            ),
            "accuracy": float(
                enhanced_metrics["accuracy"] - current_metrics["accuracy"]
            ),
            "bootstrap_95_interval": paired_ci,
        },
        "fold_stability": {
            "current_macro_f1_mean": float(fold_current_macro.mean()),
            "current_macro_f1_std": float(fold_current_macro.std(ddof=1)),
            "enhanced_macro_f1_mean": float(fold_enhanced_macro.mean()),
            "enhanced_macro_f1_std": float(fold_enhanced_macro.std(ddof=1)),
            "enhanced_macro_f1_fold_wins": int(
                np.sum(fold_enhanced_macro > fold_current_macro)
            ),
            "current_accuracy_mean": float(fold_current_accuracy.mean()),
            "current_accuracy_std": float(fold_current_accuracy.std(ddof=1)),
            "enhanced_accuracy_mean": float(fold_enhanced_accuracy.mean()),
            "enhanced_accuracy_std": float(fold_enhanced_accuracy.std(ddof=1)),
            "enhanced_accuracy_fold_wins": int(
                np.sum(fold_enhanced_accuracy > fold_current_accuracy)
            ),
        },
        "selection_frequency": {
            "alphas": summarize_frequency(fold_records, "selected_alphas"),
            "current": summarize_frequency(fold_records, "current_selection"),
            "enhanced": summarize_frequency(fold_records, "enhanced_selection"),
        },
        "fold_records": fold_records,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ACM", "DBLP", "all"], default="all")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "nested_multipath_optimization.json",
    )
    args = parser.parse_args()

    datasets = ("ACM", "DBLP") if args.dataset == "all" else (args.dataset,)
    started = time.perf_counter()
    rows = []
    for dataset in datasets:
        row = run_dataset(
            dataset,
            args.outer_folds,
            args.inner_folds,
            args.seed,
            args.bootstrap_iterations,
        )
        rows.append(row)
        print(
            dataset,
            json.dumps(
                {
                    "current_macro_f1": round(
                        row["current_oof_metrics"]["macro_f1"], 4
                    ),
                    "enhanced_macro_f1": round(
                        row["enhanced_oof_metrics"]["macro_f1"], 4
                    ),
                    "difference": round(
                        row["paired_difference"]["macro_f1"], 4
                    ),
                },
                sort_keys=True,
            ),
        )

    result = {
        "protocol": "five-fold-outer-four-fold-inner-training-only-optimization",
        "official_test_labels_loaded": False,
        "official_test_metrics_evaluated": False,
        "graph_setting": "transductive graph structure; training labels only",
        "selection_endpoint": "inner-validation Macro-F1",
        "bootstrap_note": (
            "Stratified paired node bootstrap quantifies conditional OOF "
            "prediction uncertainty; graph dependence is not modeled."
        ),
        "seed": args.seed,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
