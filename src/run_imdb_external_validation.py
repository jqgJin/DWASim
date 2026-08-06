"""Prespecified multi-label external validation on HGBn-IMDB.

All path and task parameters are selected from the official training split.
The official test labels are loaded only after every method has produced a
fixed test prediction.  IMDB labels are retained as five-dimensional binary
vectors throughout.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
from scipy.stats import rankdata
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
)
from sklearn.metrics.pairwise import manhattan_distances

from reproduce_original import (
    CACHE_ROOT,
    PATHS,
    PROCESSED_ROOT,
    RESULTS_ROOT,
    full_profile,
    multiply_chain,
)
from run_corrected_protocol import deterministic_topk
from similarity_baselines import (
    BASELINE_CACHE_TAG,
    hetesim_affinity,
    index_fingerprint,
    pathsim_affinity,
)


DATASET = "IMDB"
PATH_NAMES = ("MDM", "MAM", "MKM")
COMPONENT_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
K_VALUES = (10, 15)
GAMMAS = (1.0, 2.0)
PRIOR_POWERS = (0.0, 1.0)
THRESHOLDS = (0.3, 0.4, 0.5, 0.6)


class PeakRSS:
    """Poll process RSS while the validation runner is active."""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.process = psutil.Process()
        self.baseline = int(self.process.memory_info().rss)
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, int(self.process.memory_info().rss))

    def __enter__(self) -> "PeakRSS":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.peak = max(self.peak, int(self.process.memory_info().rss))
        self._stop.set()
        self._thread.join()


def load_training_labels() -> tuple[np.ndarray, np.ndarray]:
    with np.load(PROCESSED_ROOT / DATASET / "labels.npz") as labels:
        train_ids = labels["train_ids"].astype(np.int64)
        train_labels = labels["train_labels"].astype(np.int8)
    if train_labels.ndim != 2 or train_labels.shape[1] != 5:
        raise ValueError(f"Expected a five-column IMDB label matrix, found {train_labels.shape}")
    return train_ids, train_labels


def load_external_split() -> tuple[np.ndarray, np.ndarray]:
    """Load test IDs and truth only after model selection and prediction."""
    with np.load(PROCESSED_ROOT / DATASET / "labels.npz") as labels:
        test_ids = labels["test_ids"].astype(np.int64)
        test_labels = labels["test_labels"].astype(np.int8)
    return test_ids, test_labels


def load_external_ids_without_truth() -> np.ndarray:
    with np.load(PROCESSED_ROOT / DATASET / "labels.npz") as labels:
        return labels["test_ids"].astype(np.int64)


def iterative_multilabel_folds(
    labels: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic greedy iterative stratification for multi-label data."""
    labels = np.asarray(labels, dtype=np.int8)
    if labels.ndim != 2 or np.any(labels.sum(axis=1) == 0):
        raise ValueError("Iterative stratification requires nonempty binary label rows")
    if n_splits < 2 or n_splits > labels.shape[0]:
        raise ValueError("Invalid number of folds")

    rng = np.random.default_rng(seed)
    sample_count, class_count = labels.shape
    desired_sizes = np.full(n_splits, sample_count // n_splits, dtype=np.int64)
    desired_sizes[: sample_count % n_splits] += 1
    desired_labels = np.repeat(
        (labels.sum(axis=0) / n_splits)[None, :], n_splits, axis=0
    )
    remaining_sizes = desired_sizes.astype(np.float64)
    remaining_labels = desired_labels.copy()
    assignment = np.full(sample_count, -1, dtype=np.int64)
    unassigned = np.ones(sample_count, dtype=bool)
    tie_noise = rng.random((sample_count, n_splits)) * 1e-9

    while unassigned.any():
        remaining_counts = labels[unassigned].sum(axis=0)
        positive_classes = np.flatnonzero(remaining_counts > 0)
        if positive_classes.size == 0:
            leftovers = np.flatnonzero(unassigned)
            for sample in leftovers:
                fold = int(np.argmax(remaining_sizes + tie_noise[sample]))
                assignment[sample] = fold
                remaining_sizes[fold] -= 1.0
                unassigned[sample] = False
            break

        rare_count = remaining_counts[positive_classes].min()
        rare_classes = positive_classes[remaining_counts[positive_classes] == rare_count]
        selected_class = int(rare_classes[rng.integers(rare_classes.size)])
        candidates = np.flatnonzero(unassigned & (labels[:, selected_class] == 1))
        candidates = candidates[rng.permutation(candidates.size)]
        for sample in candidates:
            if not unassigned[sample]:
                continue
            available = np.flatnonzero(remaining_sizes > 0)
            if available.size == 0:
                raise RuntimeError("All fold capacities were exhausted early")
            class_need = remaining_labels[available, selected_class]
            best = available[np.isclose(class_need, class_need.max())]
            if best.size > 1:
                sample_need = remaining_labels[best][:, labels[sample] == 1].sum(axis=1)
                best = best[np.isclose(sample_need, sample_need.max())]
            if best.size > 1:
                size_need = remaining_sizes[best]
                best = best[np.isclose(size_need, size_need.max())]
            fold = int(best[np.argmax(tie_noise[sample, best])])
            assignment[sample] = fold
            unassigned[sample] = False
            remaining_sizes[fold] -= 1.0
            remaining_labels[fold] -= labels[sample]

    if np.any(assignment < 0):
        raise RuntimeError("Iterative stratification left samples unassigned")
    indices = np.arange(sample_count, dtype=np.int64)
    return [
        (indices[assignment != fold], indices[assignment == fold])
        for fold in range(n_splits)
    ]


def calibrate_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1] <= 1:
        return np.zeros_like(values)
    return (rankdata(values, axis=1, method="average") - 1.0) / (
        values.shape[1] - 1.0
    )


def simplex_weights(count: int, step: float = 0.25) -> list[tuple[float, ...]]:
    units = int(round(1.0 / step))
    rows: list[tuple[float, ...]] = []

    def visit(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            rows.append(tuple(value / units for value in (*prefix, remaining)))
            return
        for value in range(remaining + 1):
            visit([*prefix, value], remaining - value, slots - 1)

    visit([], units, count)
    return rows


def multilabel_prediction(
    affinity: np.ndarray,
    neighbours: np.ndarray,
    reference_labels: np.ndarray,
    gamma: float,
    prior_power: float,
    threshold: float,
) -> np.ndarray:
    neighbour_values = np.take_along_axis(affinity, neighbours, axis=1)
    weights = np.clip(neighbour_values, 0.0, None) ** float(gamma)
    zero_weight = np.isclose(weights.sum(axis=1), 0.0)
    weights[zero_weight] = 1.0
    neighbour_labels = reference_labels[neighbours]
    weighted = (weights[:, :, None] * neighbour_labels).sum(axis=1)
    scores = weighted / weights.sum(axis=1, keepdims=True)

    prevalence = np.clip(reference_labels.mean(axis=0), 1e-12, None)
    scores = scores / (prevalence[None, :] ** float(prior_power))
    maximum = scores.max(axis=1, keepdims=True)
    normalized = np.divide(
        scores,
        maximum,
        out=np.zeros_like(scores),
        where=maximum > 0,
    )
    prediction = (normalized >= float(threshold)).astype(np.int8)
    empty = prediction.sum(axis=1) == 0
    if empty.any():
        prediction[empty, np.argmax(scores[empty], axis=1)] = 1
    return prediction


def metric_record(truth: np.ndarray, prediction: np.ndarray) -> dict:
    per_class = f1_score(truth, prediction, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(truth, prediction, average="micro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(truth, prediction)),
        "hamming_loss": float(hamming_loss(truth, prediction)),
        "per_class_f1": {
            str(index): float(value) for index, value in enumerate(per_class)
        },
        "true_mean_cardinality": float(truth.sum(axis=1).mean()),
        "predicted_mean_cardinality": float(prediction.sum(axis=1).mean()),
    }


def component_cache(path_name: str, split_name: str) -> Path:
    return CACHE_ROOT / f"imdb_pair_components_{split_name}_{path_name}.npz"


def pair_components(
    path_name: str,
    query_ids: np.ndarray,
    reference_ids: np.ndarray,
    split_name: str,
) -> dict[str, np.ndarray | float]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = component_cache(path_name, split_name)
    if cache_path.exists():
        with np.load(cache_path) as cached:
            return {key: cached[key] for key in cached.files}

    profiles = full_profile(DATASET, path_name).tocsr().astype(np.float64)
    query = profiles[query_ids]
    reference = profiles[reference_ids]
    query_support = query.copy()
    reference_support = reference.copy()
    query_support.data = np.ones_like(query_support.data)
    reference_support.data = np.ones_like(reference_support.data)
    intersection = query_support.dot(reference_support.T).toarray()
    query_support_count = np.asarray(query_support.sum(axis=1)).ravel()
    reference_support_count = np.asarray(reference_support.sum(axis=1)).ravel()
    union = (
        query_support_count[:, None]
        + reference_support_count[None, :]
        - intersection
    )
    jaccard = np.divide(
        union - intersection,
        union,
        out=np.zeros_like(union),
        where=union > 0,
    )
    hamming = (
        query_support_count[:, None]
        + reference_support_count[None, :]
        - 2.0 * intersection
    )

    magnitude = manhattan_distances(query, reference)
    query_activity = np.asarray(query.sum(axis=1)).ravel()
    reference_activity = np.asarray(reference.sum(axis=1)).ravel()
    activity_sum = query_activity[:, None] + reference_activity[None, :]
    bray_curtis = np.divide(
        magnitude,
        activity_sum,
        out=np.zeros_like(magnitude),
        where=activity_sum > 0,
    )
    dimensions = float(profiles.shape[1])
    value_range = max(float(profiles.data.max()) if profiles.nnz else 0.0, 1.0)
    payload = {
        "jaccard": np.clip(jaccard, 0.0, 1.0).astype(np.float32),
        "bray_curtis": np.clip(bray_curtis, 0.0, 1.0).astype(np.float32),
        "hamming": hamming.astype(np.float32),
        "magnitude": magnitude.astype(np.float32),
        "dimensions": np.asarray(dimensions, dtype=np.float64),
        "value_range": np.asarray(value_range, dtype=np.float64),
    }
    np.savez_compressed(cache_path, **payload)
    return payload


def component_affinity(
    components: dict[str, np.ndarray | float],
    weight: float,
    formulation: str,
) -> np.ndarray:
    if formulation == "relative":
        distance = (
            float(weight) * components["jaccard"]
            + (1.0 - float(weight)) * components["bray_curtis"]
        )
    elif formulation == "global":
        dimensions = float(components["dimensions"])
        value_range = float(components["value_range"])
        distance = (
            float(weight) * components["hamming"] / dimensions
            + (1.0 - float(weight))
            * components["magnitude"]
            / (dimensions * value_range)
        )
    else:
        raise ValueError(f"Unknown formulation: {formulation}")
    return np.clip(1.0 - distance, 0.0, 1.0).astype(np.float32)


def baseline_affinity(
    path_name: str,
    query_ids: np.ndarray,
    reference_ids: np.ndarray,
    method: str,
) -> np.ndarray:
    query_key = index_fingerprint(query_ids)
    reference_key = index_fingerprint(reference_ids)
    cache_path = CACHE_ROOT / (
        f"imdb_{BASELINE_CACHE_TAG}_{method.lower()}_{path_name}_"
        f"q{query_key}_r{reference_key}.npz"
    )
    if cache_path.exists():
        with np.load(cache_path) as cached:
            return cached["affinity"]

    if method == "PathSim":
        half = multiply_chain(
            DATASET, PATHS[DATASET][path_name]["half"], transition=False
        )
        affinity = pathsim_affinity(
            half[query_ids],
            half[reference_ids],
        )
    elif method == "HeteSim":
        transition = multiply_chain(
            DATASET, PATHS[DATASET][path_name]["half"], transition=True
        )
        affinity = hetesim_affinity(
            transition[query_ids],
            transition[reference_ids],
        )
    else:
        raise ValueError(f"Unknown baseline: {method}")
    affinity = affinity.astype(np.float32)
    np.savez_compressed(cache_path, affinity=affinity)
    return affinity


def score_prediction(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    return (
        float(f1_score(truth, prediction, average="macro", zero_division=0)),
        float(f1_score(truth, prediction, average="micro", zero_division=0)),
    )


def select_component_weights(
    formulation: str,
    train_components: dict[str, dict[str, np.ndarray | float]],
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, float], dict[str, list[dict]]]:
    selections: dict[str, float] = {}
    grids: dict[str, list[dict]] = {}
    for path_name in PATH_NAMES:
        rows = []
        for weight in COMPONENT_WEIGHTS:
            affinity = component_affinity(
                train_components[path_name], weight, formulation
            )
            macro_scores = []
            micro_scores = []
            for reference, validation in folds:
                view = affinity[np.ix_(validation, reference)]
                neighbours = deterministic_topk(
                    view,
                    train_ids[reference],
                    min(10, reference.size),
                    largest=True,
                )
                prediction = multilabel_prediction(
                    view,
                    neighbours,
                    train_labels[reference],
                    gamma=1.0,
                    prior_power=0.0,
                    threshold=0.5,
                )
                macro, micro = score_prediction(train_labels[validation], prediction)
                macro_scores.append(macro)
                micro_scores.append(micro)
            rows.append(
                {
                    "weight": weight,
                    "macro_f1_mean": float(np.mean(macro_scores)),
                    "macro_f1_std": float(np.std(macro_scores, ddof=1)),
                    "micro_f1_mean": float(np.mean(micro_scores)),
                }
            )
        selected = max(
            rows,
            key=lambda row: (
                row["macro_f1_mean"],
                row["micro_f1_mean"],
                -abs(row["weight"] - 0.5),
            ),
        )
        selections[path_name] = float(selected["weight"])
        grids[path_name] = rows
    return selections, grids


def select_pipeline(
    affinities: list[np.ndarray],
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    k_values: tuple[int, ...] = K_VALUES,
) -> dict:
    scores: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: {"macro": [], "micro": [], "subset": []}
    )
    path_weight_grid = simplex_weights(len(affinities))
    maximum_k = max(k_values)
    for reference, validation in folds:
        views = [
            calibrate_rows(affinity[np.ix_(validation, reference)])
            for affinity in affinities
        ]
        for path_weights in path_weight_grid:
            fused = sum(
                weight * view for weight, view in zip(path_weights, views)
            )
            ordered = deterministic_topk(
                fused,
                train_ids[reference],
                min(maximum_k, reference.size),
                largest=True,
            )
            for k in k_values:
                neighbours = ordered[:, : min(k, ordered.shape[1])]
                for gamma in GAMMAS:
                    for prior_power in PRIOR_POWERS:
                        for threshold in THRESHOLDS:
                            prediction = multilabel_prediction(
                                fused,
                                neighbours,
                                train_labels[reference],
                                gamma,
                                prior_power,
                                threshold,
                            )
                            truth = train_labels[validation]
                            key = (path_weights, k, gamma, prior_power, threshold)
                            scores[key]["macro"].append(
                                f1_score(
                                    truth,
                                    prediction,
                                    average="macro",
                                    zero_division=0,
                                )
                            )
                            scores[key]["micro"].append(
                                f1_score(
                                    truth,
                                    prediction,
                                    average="micro",
                                    zero_division=0,
                                )
                            )
                            scores[key]["subset"].append(
                                accuracy_score(truth, prediction)
                            )

    rows = []
    for key, values in scores.items():
        path_weights, k, gamma, prior_power, threshold = key
        rows.append(
            {
                "path_weights": list(path_weights),
                "k": int(k),
                "gamma": float(gamma),
                "prior_power": float(prior_power),
                "threshold": float(threshold),
                "macro_f1_mean": float(np.mean(values["macro"])),
                "macro_f1_std": float(np.std(values["macro"], ddof=1)),
                "micro_f1_mean": float(np.mean(values["micro"])),
                "subset_accuracy_mean": float(np.mean(values["subset"])),
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            row["macro_f1_mean"],
            row["micro_f1_mean"],
            row["subset_accuracy_mean"],
            -sum(weight > 0 for weight in row["path_weights"]),
            -abs(row["k"] - 10),
            -abs(row["gamma"] - 1.0),
            -row["prior_power"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    return {
        "selected": selected,
        "configuration_count": len(rows),
    }


def fixed_external_prediction(
    affinities: list[np.ndarray],
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    config: dict,
) -> np.ndarray:
    views = [calibrate_rows(affinity) for affinity in affinities]
    fused = sum(
        weight * view
        for weight, view in zip(config["path_weights"], views)
    )
    neighbours = deterministic_topk(
        fused,
        train_ids,
        min(config["k"], train_ids.size),
        largest=True,
    )
    return multilabel_prediction(
        fused,
        neighbours,
        train_labels,
        config["gamma"],
        config["prior_power"],
        config["threshold"],
    )


def paired_node_bootstrap(
    truth: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    iterations: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    differences = {"macro_f1": [], "micro_f1": [], "subset_accuracy": []}
    for _ in range(iterations):
        sample = rng.integers(0, truth.shape[0], truth.shape[0])
        for name, average in (("macro_f1", "macro"), ("micro_f1", "micro")):
            first_score = f1_score(
                truth[sample], first[sample], average=average, zero_division=0
            )
            second_score = f1_score(
                truth[sample], second[sample], average=average, zero_division=0
            )
            differences[name].append(float(first_score - second_score))
        differences["subset_accuracy"].append(
            float(
                accuracy_score(truth[sample], first[sample])
                - accuracy_score(truth[sample], second[sample])
            )
        )
    return {
        name: {
            "difference": float(
                (f1_score(truth, first, average=name.split("_")[0], zero_division=0)
                 - f1_score(truth, second, average=name.split("_")[0], zero_division=0))
                if name != "subset_accuracy"
                else accuracy_score(truth, first) - accuracy_score(truth, second)
            ),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "probability_difference_positive": float(np.mean(np.asarray(values) > 0)),
        }
        for name, values in differences.items()
    }


def split_audit(
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    overall = labels.mean(axis=0)
    return [
        {
            "fold": fold,
            "reference_nodes": int(reference.size),
            "validation_nodes": int(validation.size),
            "validation_prevalence": labels[validation].mean(axis=0).tolist(),
            "maximum_absolute_prevalence_deviation": float(
                np.max(np.abs(labels[validation].mean(axis=0) - overall))
            ),
        }
        for fold, (reference, validation) in enumerate(folds)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "imdb_external_validation.json",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    with PeakRSS() as memory:
        train_ids, train_labels = load_training_labels()
        test_ids = load_external_ids_without_truth()
        folds = iterative_multilabel_folds(train_labels, args.folds, args.seed)

        train_components = {
            path: pair_components(path, train_ids, train_ids, "train_train")
            for path in PATH_NAMES
        }
        external_components = {
            path: pair_components(path, test_ids, train_ids, "test_train")
            for path in PATH_NAMES
        }

        predictions: dict[str, np.ndarray] = {}
        selections: dict[str, dict] = {}
        validation_started = time.perf_counter()

        for formulation, method_name in (
            ("relative", "RelativeDWASim"),
            ("global", "GlobalDWASim"),
        ):
            selected_components, component_grid = select_component_weights(
                formulation,
                train_components,
                train_ids,
                train_labels,
                folds,
            )
            train_affinities = [
                component_affinity(
                    train_components[path], selected_components[path], formulation
                )
                for path in PATH_NAMES
            ]
            external_affinities = [
                component_affinity(
                    external_components[path], selected_components[path], formulation
                )
                for path in PATH_NAMES
            ]
            selected_pipeline = select_pipeline(
                train_affinities, train_ids, train_labels, folds
            )
            predictions[method_name] = fixed_external_prediction(
                external_affinities,
                train_ids,
                train_labels,
                selected_pipeline["selected"],
            )
            selections[method_name] = {
                "selected_component_weights": selected_components,
                "component_validation": component_grid,
                **selected_pipeline,
            }
            if formulation == "global":
                fixed_k10 = select_pipeline(
                    train_affinities,
                    train_ids,
                    train_labels,
                    folds,
                    k_values=(10,),
                )
                predictions["GlobalDWASimFixedK10"] = fixed_external_prediction(
                    external_affinities,
                    train_ids,
                    train_labels,
                    fixed_k10["selected"],
                )
                selections["GlobalDWASimFixedK10"] = {
                    "selected_component_weights": selected_components,
                    "component_validation": component_grid,
                    **fixed_k10,
                }

        # Diagnostic component ablations use the same training-only task search
        # as the selected pair-relative formulation.  Only beta is fixed.
        for method_name, fixed_beta in (
            ("MagnitudeOnly", 0.0),
            ("EqualComponents", 0.5),
            ("SupportOnly", 1.0),
        ):
            train_affinities = [
                component_affinity(train_components[path], fixed_beta, "relative")
                for path in PATH_NAMES
            ]
            external_affinities = [
                component_affinity(external_components[path], fixed_beta, "relative")
                for path in PATH_NAMES
            ]
            selected_pipeline = select_pipeline(
                train_affinities, train_ids, train_labels, folds
            )
            predictions[method_name] = fixed_external_prediction(
                external_affinities,
                train_ids,
                train_labels,
                selected_pipeline["selected"],
            )
            selections[method_name] = {
                "fixed_component_weight": fixed_beta,
                **selected_pipeline,
            }

        for method_name in ("PathSim", "HeteSim"):
            train_affinities = [
                baseline_affinity(path, train_ids, train_ids, method_name)
                for path in PATH_NAMES
            ]
            external_affinities = [
                baseline_affinity(path, test_ids, train_ids, method_name)
                for path in PATH_NAMES
            ]
            selected_pipeline = select_pipeline(
                train_affinities, train_ids, train_labels, folds
            )
            predictions[method_name] = fixed_external_prediction(
                external_affinities,
                train_ids,
                train_labels,
                selected_pipeline["selected"],
            )
            selections[method_name] = selected_pipeline

        prediction_elapsed = time.perf_counter() - validation_started

        # Integrity boundary: this is the first point at which test truth is loaded.
        loaded_test_ids, test_labels = load_external_split()
        if not np.array_equal(test_ids, loaded_test_ids):
            raise RuntimeError("External query identifiers changed before scoring")
        metrics = {
            method: metric_record(test_labels, prediction)
            for method, prediction in predictions.items()
        }
        paired = paired_node_bootstrap(
            test_labels,
            predictions["RelativeDWASim"],
            predictions["GlobalDWASimFixedK10"],
            args.bootstrap_iterations,
            args.seed + 9000,
        )
        paired_components = {
            comparator: paired_node_bootstrap(
                test_labels,
                predictions["RelativeDWASim"],
                predictions[comparator],
                args.bootstrap_iterations,
                args.seed + 10000 + offset,
            )
            for offset, comparator in enumerate(
                ("MagnitudeOnly", "EqualComponents", "SupportOnly")
            )
        }

    result = {
        "dataset": DATASET,
        "task": "multi-label movie genre classification",
        "paths": list(PATH_NAMES),
        "path_meanings": {
            "MDM": "movie-director-movie",
            "MAM": "movie-actor-movie",
            "MKM": "movie-keyword-movie",
        },
        "training_nodes": int(train_ids.size),
        "external_test_nodes": int(test_ids.size),
        "classes": ["Romance", "Thriller", "Comedy", "Action", "Drama"],
        "protocol": {
            "test_truth_loaded_after_prediction": True,
            "selection_endpoint": "four-fold training-only Macro-F1",
            "multilabel_stratification": "deterministic greedy iterative stratification",
            "query_wise_calibration": "rank",
            "component_weight_grid": list(COMPONENT_WEIGHTS),
            "path_weight_grid": "0.25 simplex",
            "k_grid": list(K_VALUES),
            "gamma_grid": list(GAMMAS),
            "prior_power_grid": list(PRIOR_POWERS),
            "relative_to_best_threshold_grid": list(THRESHOLDS),
            "seed": args.seed,
            "fold_audit": split_audit(train_labels, folds),
        },
        "selections": selections,
        "external_metrics": metrics,
        "paired_relative_minus_fixed_global": paired,
        "paired_relative_component_ablation": paired_components,
        "uncertainty_note": (
            "Paired node bootstrap intervals are conditional on the observed graph "
            "and do not model graph-induced dependence."
        ),
        "runtime_seconds": {
            "selection_and_prediction": prediction_elapsed,
            "total_including_cache_generation": time.perf_counter() - started,
        },
        "memory": {
            "baseline_rss_mib": memory.baseline / (1024**2),
            "peak_rss_mib": memory.peak / (1024**2),
            "incremental_peak_rss_mib": (memory.peak - memory.baseline) / (1024**2),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                method: {
                    "macro_f1": round(values["macro_f1"], 4),
                    "micro_f1": round(values["micro_f1"], 4),
                    "subset_accuracy": round(values["subset_accuracy"], 4),
                }
                for method, values in metrics.items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
