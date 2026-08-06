"""Matched held-out comparison of DWASim normalizations on ACM and DBLP.

The two manuscript paths for each dataset, the top-k evaluator, validation
splits, fusion-weight grid, and tie handling are held fixed across methods.
Path-specific parameters and fusion weights are selected from official
training labels only.  Official test labels are consulted only after every
selection has been fixed.

The runner also evaluates the two pair-relative components separately and
uses a stratified paired node bootstrap to quantify conditional uncertainty on
the fixed official test predictions.  The interval conditions on the observed
graph and therefore does not model dependence induced by shared graph edges.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.metrics import accuracy_score, f1_score

from reproduce_original import CACHE_ROOT, RESULTS_ROOT, full_profile
from run_corrected_protocol import PAPER_PATHS, load_split, metric_record
from run_real_multipath_fusion import (
    baseline_affinities,
    dwasim_affinity,
    load_discrepancies,
    predict_from_affinity,
    select_fusion_weight,
    select_single_path_alpha,
    split_positions,
)


class PeakRSS:
    """Sample process RSS while the benchmark is running."""

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self.process = psutil.Process()
        self.baseline = int(self.process.memory_info().rss)
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, int(self.process.memory_info().rss))
            self._stop.wait(self.interval)

    def __enter__(self) -> "PeakRSS":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, int(self.process.memory_info().rss))


def _jaccard_discrepancy(query, reference) -> np.ndarray:
    intersection = query.dot(reference.T).toarray().astype(np.float64, copy=False)
    query_size = np.asarray(query.sum(axis=1)).ravel().astype(np.float64)
    reference_size = np.asarray(reference.sum(axis=1)).ravel().astype(np.float64)
    union = query_size[:, None] + reference_size[None, :] - intersection
    return np.divide(
        union - intersection,
        union,
        out=np.zeros_like(union),
        where=union > 0,
    )


def load_relative_components(dataset: str, path_name: str, discrepancy: dict) -> dict:
    """Return train/train and test/train Jaccard and Bray--Curtis terms."""

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_ROOT / f"heldout_relative_components_{dataset}_{path_name}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return {key: cached[key] for key in cached.files}

    train_ids, _, test_ids, _ = load_split(dataset)
    profiles = full_profile(dataset, path_name).tocsr().astype(np.float64)
    train_profiles = profiles[train_ids]
    test_profiles = profiles[test_ids]

    train_support = train_profiles.copy()
    train_support.data = np.ones_like(train_support.data)
    test_support = test_profiles.copy()
    test_support.data = np.ones_like(test_support.data)

    train_jaccard = _jaccard_discrepancy(train_support, train_support)
    test_jaccard = _jaccard_discrepancy(test_support, train_support)

    train_activity = np.asarray(train_profiles.sum(axis=1)).ravel().astype(np.float64)
    test_activity = np.asarray(test_profiles.sum(axis=1)).ravel().astype(np.float64)
    train_denominator = train_activity[:, None] + train_activity[None, :]
    test_denominator = test_activity[:, None] + train_activity[None, :]
    train_bray = np.divide(
        discrepancy["train_l"],
        train_denominator,
        out=np.zeros_like(train_denominator),
        where=train_denominator > 0,
    )
    test_bray = np.divide(
        discrepancy["test_l"],
        test_denominator,
        out=np.zeros_like(test_denominator),
        where=test_denominator > 0,
    )

    values = {
        "train_jaccard": np.clip(train_jaccard, 0.0, 1.0).astype(np.float32),
        "test_jaccard": np.clip(test_jaccard, 0.0, 1.0).astype(np.float32),
        "train_bray": np.clip(train_bray, 0.0, 1.0).astype(np.float32),
        "test_bray": np.clip(test_bray, 0.0, 1.0).astype(np.float32),
    }
    np.savez_compressed(cache_path, **values)
    return values


def relative_affinity(jaccard: np.ndarray, bray: np.ndarray, beta: float) -> np.ndarray:
    distance = float(beta) * jaccard + (1.0 - float(beta)) * bray
    return np.clip(1.0 - distance, 0.0, 1.0)


def select_path_parameter(
    affinity_grid: dict[float, np.ndarray],
    parameter_name: str,
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    splits,
    k: int,
) -> dict:
    records = []
    for parameter, affinity in affinity_grid.items():
        macro_scores = []
        accuracies = []
        for reference, validation in splits:
            prediction = predict_from_affinity(
                affinity[np.ix_(validation, reference)],
                train_ids[reference],
                train_labels[reference],
                k,
            )
            truth = train_labels[validation]
            macro_scores.append(
                float(f1_score(truth, prediction, average="macro", zero_division=0))
            )
            accuracies.append(float(accuracy_score(truth, prediction)))
        records.append(
            {
                parameter_name: float(parameter),
                "macro_f1_mean": float(np.mean(macro_scores)),
                "macro_f1_std": float(np.std(macro_scores, ddof=1)),
                "accuracy_mean": float(np.mean(accuracies)),
            }
        )
    selected = max(
        records,
        key=lambda item: (
            item["macro_f1_mean"],
            item["accuracy_mean"],
            -abs(item[parameter_name] - 0.5),
        ),
    )
    return {"selected": selected, "grid": records}


def evaluate_two_path_method(
    train_views: tuple[np.ndarray, np.ndarray],
    test_views: tuple[np.ndarray, np.ndarray],
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    splits,
    fusion_weights: np.ndarray,
    k: int,
) -> tuple[dict, dict, np.ndarray]:
    selection = select_fusion_weight(
        train_views[0],
        train_views[1],
        train_ids,
        train_labels,
        splits,
        fusion_weights,
        k,
    )
    weight = float(selection["selected"]["weight_path_1"])
    fused_test = weight * test_views[0] + (1.0 - weight) * test_views[1]
    prediction = predict_from_affinity(fused_test, train_ids, train_labels, k)
    metrics = metric_record(test_labels, prediction, np.unique(test_labels))
    return selection, metrics, prediction


def stratified_paired_bootstrap(
    truth: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    iterations: int,
    seed: int,
) -> dict:
    classes = np.unique(truth)
    class_positions = [np.flatnonzero(truth == label) for label in classes]
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = np.concatenate(
            [rng.choice(positions, size=positions.size, replace=True) for positions in class_positions]
        )
        differences[iteration] = f1_score(
            truth[sampled], first[sampled], labels=classes, average="macro", zero_division=0
        ) - f1_score(
            truth[sampled], second[sampled], labels=classes, average="macro", zero_division=0
        )
    point = f1_score(truth, first, labels=classes, average="macro", zero_division=0) - f1_score(
        truth, second, labels=classes, average="macro", zero_division=0
    )
    return {
        "difference": float(point),
        "lower_95": float(np.quantile(differences, 0.025)),
        "upper_95": float(np.quantile(differences, 0.975)),
        "probability_difference_positive": float(np.mean(differences > 0)),
    }


def run_dataset(
    dataset: str,
    k: int,
    split_seeds: list[int],
    validation_fraction: float,
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    fusion_weights: np.ndarray,
    bootstrap_iterations: int,
    chunk_size: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    train_ids, train_labels, test_ids, test_labels = load_split(dataset)
    splits = split_positions(train_labels, split_seeds, validation_fraction)
    paths = PAPER_PATHS[dataset]

    discrepancies = {
        path: load_discrepancies(dataset, path, chunk_size) for path in paths
    }
    relative_components = {
        path: load_relative_components(dataset, path, discrepancies[path]) for path in paths
    }

    global_train = []
    global_test = []
    global_path_selection = {}
    relative_train = []
    relative_test = []
    relative_path_selection = {}
    fixed_relative = {
        "MagnitudeOnly": {"train": [], "test": []},
        "EqualComponents": {"train": [], "test": []},
        "SupportOnly": {"train": [], "test": []},
    }
    fixed_betas = {"MagnitudeOnly": 0.0, "EqualComponents": 0.5, "SupportOnly": 1.0}

    for path in paths:
        global_selection, global_grid = select_single_path_alpha(
            discrepancies[path], train_ids, train_labels, splits, alpha_grid, k
        )
        alpha = float(global_selection["selected"]["alpha"])
        global_path_selection[path] = global_selection
        global_train.append(global_grid[alpha])
        global_test.append(
            dwasim_affinity(
                discrepancies[path]["test_h"],
                discrepancies[path]["test_l"],
                float(discrepancies[path]["b0"]),
                float(discrepancies[path]["b1"]),
                alpha,
            )
        )

        components = relative_components[path]
        train_grid = {
            float(beta): relative_affinity(
                components["train_jaccard"], components["train_bray"], float(beta)
            )
            for beta in beta_grid
        }
        test_grid = {
            float(beta): relative_affinity(
                components["test_jaccard"], components["test_bray"], float(beta)
            )
            for beta in beta_grid
        }
        selection = select_path_parameter(
            train_grid, "beta", train_ids, train_labels, splits, k
        )
        beta = float(selection["selected"]["beta"])
        relative_path_selection[path] = selection
        relative_train.append(train_grid[beta])
        relative_test.append(test_grid[beta])
        for name, fixed_beta in fixed_betas.items():
            fixed_relative[name]["train"].append(train_grid[fixed_beta])
            fixed_relative[name]["test"].append(test_grid[fixed_beta])

    baseline_by_path = {path: baseline_affinities(dataset, path) for path in paths}
    methods: dict[str, tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = {
        "GlobalDWASim": (tuple(global_train), tuple(global_test)),
        "PairRelativeDWASim": (tuple(relative_train), tuple(relative_test)),
    }
    for name, values in fixed_relative.items():
        methods[name] = (tuple(values["train"]), tuple(values["test"]))
    for method in ("PathSim", "HeteSim"):
        methods[method] = (
            tuple(baseline_by_path[path][method][0] for path in paths),
            tuple(baseline_by_path[path][method][1] for path in paths),
        )

    selections = {}
    metrics = {}
    predictions = {}
    method_seconds = {}
    for method, (train_views, test_views) in methods.items():
        method_started = time.perf_counter()
        selection, record, prediction = evaluate_two_path_method(
            train_views,
            test_views,
            train_ids,
            train_labels,
            test_labels,
            splits,
            fusion_weights,
            k,
        )
        selections[method] = selection
        metrics[method] = record
        predictions[method] = prediction
        method_seconds[method] = time.perf_counter() - method_started

    comparisons = {}
    for comparator in (
        "GlobalDWASim",
        "MagnitudeOnly",
        "EqualComponents",
        "SupportOnly",
        "PathSim",
        "HeteSim",
    ):
        comparisons[f"PairRelativeDWASim_minus_{comparator}"] = stratified_paired_bootstrap(
            test_labels,
            predictions["PairRelativeDWASim"],
            predictions[comparator],
            bootstrap_iterations,
            20260806 + sum(ord(character) for character in dataset + comparator),
        )

    return (
        {
            "dataset": dataset,
            "paths": list(paths),
            "reference_nodes": int(train_ids.size),
            "held_out_queries": int(test_ids.size),
            "global_path_selection": global_path_selection,
            "relative_path_selection": relative_path_selection,
            "fusion_selection": selections,
            "test_metrics": metrics,
            "paired_bootstrap": comparisons,
            "selection_and_prediction_seconds": method_seconds,
            "elapsed_seconds": time.perf_counter() - started,
        },
        predictions,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ACM", "DBLP", "all"], default="all")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=20250803)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "unified_normalization_validation.json",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=RESULTS_ROOT / "unified_normalization_predictions.npz",
    )
    args = parser.parse_args()

    alpha_grid = np.linspace(0.0, 1.0, 11)
    beta_grid = np.linspace(0.0, 1.0, 5)
    fusion_weights = np.linspace(0.0, 1.0, 11)
    split_seeds = list(range(args.seed_start, args.seed_start + args.splits))
    datasets = ("ACM", "DBLP") if args.dataset == "all" else (args.dataset,)
    started = time.perf_counter()
    rows = []
    saved_predictions = {}

    with PeakRSS() as memory:
        for dataset in datasets:
            row, predictions = run_dataset(
                dataset,
                args.k,
                split_seeds,
                args.validation_fraction,
                alpha_grid,
                beta_grid,
                fusion_weights,
                args.bootstrap_iterations,
                args.chunk_size,
            )
            rows.append(row)
            for method, prediction in predictions.items():
                saved_predictions[f"{dataset}_{method}"] = prediction
            print(
                dataset,
                json.dumps(
                    {
                        method: round(record["macro_f1"], 4)
                        for method, record in row["test_metrics"].items()
                    },
                    sort_keys=True,
                ),
            )

    result = {
        "protocol": "matched-two-path-held-out-normalization-and-component-ablation",
        "selection_endpoint": "training-only mean Macro-F1",
        "test_labels_used_for_selection": False,
        "k": args.k,
        "alpha_grid": alpha_grid.tolist(),
        "beta_grid": beta_grid.tolist(),
        "fusion_weight_grid": fusion_weights.tolist(),
        "validation_fraction": args.validation_fraction,
        "split_seeds": split_seeds,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_note": (
            "Stratified paired node bootstrap intervals are conditional on the fixed "
            "observed graph and do not model graph-induced dependence."
        ),
        "rows": rows,
        "runtime_seconds": time.perf_counter() - started,
        "memory": {
            "baseline_rss_mib": memory.baseline / (1024**2),
            "peak_rss_mib": memory.peak / (1024**2),
            "incremental_peak_rss_mib": (memory.peak - memory.baseline) / (1024**2),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(args.predictions_output, **saved_predictions)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.predictions_output}")


if __name__ == "__main__":
    main()
