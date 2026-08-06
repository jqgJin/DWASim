"""Evaluate real-data multi-path fusion under the held-out HGB protocol.

Path-specific DWASim effective weights and across-path fusion weights are
selected only on repeated stratified splits of the official training labels.
The official test labels are evaluated once after selection.  The same
validation rule is applied to PathSim and HeteSim fusion for a fair comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from reproduce_original import CACHE_ROOT, PATHS, RESULTS_ROOT, full_profile, multiply_chain
from run_corrected_protocol import (
    PAPER_PATHS,
    deterministic_topk,
    load_split,
    majority_vote,
    metric_record,
)
from similarity_baselines import symmetric_path_affinities


def split_positions(labels: np.ndarray, seeds: list[int], fraction: float):
    splits = []
    for seed in seeds:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        reference, validation = next(splitter.split(np.zeros(labels.size), labels))
        splits.append((reference, validation))
    return splits


def exact_hamming_normalizer(profiles: np.ndarray, chunk_size: int) -> float:
    """Compute max pairwise coordinate disagreement without storing all pairs."""
    maximum = 0.0
    dimensions = profiles.shape[1]
    for start in range(0, profiles.shape[0], chunk_size):
        block = profiles[start : start + chunk_size]
        hamming = np.rint(cdist(block, profiles, metric="hamming") * dimensions)
        maximum = max(maximum, float(hamming.max()))
    return max(maximum, 1.0)


def load_discrepancies(dataset: str, path_name: str, chunk_size: int) -> dict[str, np.ndarray | float]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_ROOT / f"heldout_discrepancies_{dataset}_{path_name}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return {key: cached[key] for key in cached.files}

    train_ids, _, test_ids, _ = load_split(dataset)
    matrix = full_profile(dataset, path_name)
    profiles = matrix.toarray().astype(np.float64, copy=False)
    train_profiles = profiles[train_ids]
    test_profiles = profiles[test_ids]
    dimensions = profiles.shape[1]

    b0 = exact_hamming_normalizer(profiles, chunk_size)
    value_range = float(profiles.max() - profiles.min())
    b1 = max(b0 * value_range, 1.0)
    train_h = np.rint(cdist(train_profiles, train_profiles, metric="hamming") * dimensions)
    train_l = cdist(train_profiles, train_profiles, metric="cityblock")
    test_h = np.rint(cdist(test_profiles, train_profiles, metric="hamming") * dimensions)
    test_l = cdist(test_profiles, train_profiles, metric="cityblock")
    np.savez_compressed(
        cache_path,
        train_h=train_h,
        train_l=train_l,
        test_h=test_h,
        test_l=test_l,
        b0=np.asarray(b0),
        b1=np.asarray(b1),
        value_range=np.asarray(value_range),
    )
    return {
        "train_h": train_h,
        "train_l": train_l,
        "test_h": test_h,
        "test_l": test_l,
        "b0": b0,
        "b1": b1,
        "value_range": value_range,
    }


def dwasim_affinity(h: np.ndarray, l1: np.ndarray, b0: float, b1: float, alpha: float):
    distance = alpha * h / b0 + (1.0 - alpha) * l1 / b1
    return np.clip(1.0 - distance, 0.0, 1.0)


def baseline_affinities(dataset: str, path_name: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    train_ids, _, test_ids, _ = load_split(dataset)
    config = PATHS[dataset][path_name]

    raw = multiply_chain(dataset, config["half"], transition=False)
    transition = multiply_chain(dataset, config["half"], transition=True)
    train_affinities = symmetric_path_affinities(
        raw, transition, train_ids, train_ids
    )
    test_affinities = symmetric_path_affinities(
        raw, transition, test_ids, train_ids
    )
    return {
        method: (train_affinities[method], test_affinities[method])
        for method in ("PathSim", "HeteSim")
    }


def predict_from_affinity(
    affinity: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_labels: np.ndarray,
    k: int,
) -> np.ndarray:
    neighbours = deterministic_topk(affinity, candidate_ids, k, largest=True)
    return majority_vote(neighbours, candidate_labels)


def select_single_path_alpha(
    discrepancy: dict,
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    splits,
    alphas: np.ndarray,
    k: int,
) -> tuple[dict, dict[float, np.ndarray]]:
    b0, b1 = float(discrepancy["b0"]), float(discrepancy["b1"])
    full_affinities = {
        float(alpha): dwasim_affinity(
            discrepancy["train_h"], discrepancy["train_l"], b0, b1, float(alpha)
        )
        for alpha in alphas
    }
    records = []
    for alpha in alphas:
        scores = []
        accuracies = []
        affinity = full_affinities[float(alpha)]
        for reference, validation in splits:
            prediction = predict_from_affinity(
                affinity[np.ix_(validation, reference)],
                train_ids[reference],
                train_labels[reference],
                k,
            )
            truth = train_labels[validation]
            scores.append(float(f1_score(truth, prediction, average="macro", zero_division=0)))
            accuracies.append(float(accuracy_score(truth, prediction)))
        records.append(
            {
                "alpha": float(alpha),
                "macro_f1_mean": float(np.mean(scores)),
                "macro_f1_std": float(np.std(scores, ddof=1)),
                "accuracy_mean": float(np.mean(accuracies)),
            }
        )
    selected = max(
        records,
        key=lambda item: (
            item["macro_f1_mean"],
            item["accuracy_mean"],
            -abs(item["alpha"] - 0.5),
        ),
    )
    return {"selected": selected, "grid": records}, full_affinities


def select_fusion_weight(
    affinity_1: np.ndarray,
    affinity_2: np.ndarray,
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    splits,
    weights: np.ndarray,
    k: int,
) -> dict:
    records = []
    for weight in weights:
        scores = []
        accuracies = []
        for reference, validation in splits:
            fused = (
                float(weight) * affinity_1[np.ix_(validation, reference)]
                + (1.0 - float(weight)) * affinity_2[np.ix_(validation, reference)]
            )
            prediction = predict_from_affinity(
                fused,
                train_ids[reference],
                train_labels[reference],
                k,
            )
            truth = train_labels[validation]
            scores.append(float(f1_score(truth, prediction, average="macro", zero_division=0)))
            accuracies.append(float(accuracy_score(truth, prediction)))
        records.append(
            {
                "weight_path_1": float(weight),
                "weight_path_2": float(1.0 - weight),
                "macro_f1_mean": float(np.mean(scores)),
                "macro_f1_std": float(np.std(scores, ddof=1)),
                "accuracy_mean": float(np.mean(accuracies)),
            }
        )
    selected = max(
        records,
        key=lambda item: (
            item["macro_f1_mean"],
            item["accuracy_mean"],
            -abs(item["weight_path_1"] - 0.5),
        ),
    )
    return {"selected": selected, "grid": records}


def entropy_weights(affinities: list[np.ndarray]) -> np.ndarray:
    scores = []
    n = affinities[0].shape[0]
    for affinity in affinities:
        column_sum = affinity.sum(axis=0, keepdims=True)
        probability = np.divide(
            affinity,
            column_sum,
            out=np.zeros_like(affinity),
            where=column_sum > 0,
        )
        log_probability = np.zeros_like(probability)
        np.log(probability, out=log_probability, where=probability > 0)
        entropy = -(probability * log_probability).sum(axis=0) / np.log(n)
        scores.append(1.0 - float(entropy.mean()))
    scores = np.asarray(scores)
    return scores / scores.sum()


def evaluate_variants(
    train_affinities: dict[str, tuple[np.ndarray, np.ndarray]],
    test_affinities: dict[str, tuple[np.ndarray, np.ndarray]],
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    splits,
    weights: np.ndarray,
    k: int,
) -> tuple[dict, dict]:
    selections = {}
    metrics = {}
    for method in train_affinities:
        train_1, train_2 = train_affinities[method]
        test_1, test_2 = test_affinities[method]
        selection = select_fusion_weight(
            train_1, train_2, train_ids, train_labels, splits, weights, k
        )
        selections[method] = selection
        selected_weight = selection["selected"]["weight_path_1"]
        variants = {
            "path_1": test_1,
            "path_2": test_2,
            "uniform_fusion": 0.5 * test_1 + 0.5 * test_2,
            "validation_selected_fusion": selected_weight * test_1
            + (1.0 - selected_weight) * test_2,
        }
        if method == "DWASim":
            ew = entropy_weights([train_1, train_2])
            variants["entropy_fusion"] = ew[0] * test_1 + ew[1] * test_2
            selection["entropy_weights_full_training"] = ew.tolist()

        metrics[method] = {}
        for variant, affinity in variants.items():
            prediction = predict_from_affinity(affinity, train_ids, train_labels, k)
            metrics[method][variant] = metric_record(
                test_labels, prediction, np.unique(test_labels)
            )
    return selections, metrics


def run_dataset(
    dataset: str,
    k: int,
    alphas: np.ndarray,
    weights: np.ndarray,
    seeds: list[int],
    validation_fraction: float,
    chunk_size: int,
) -> dict:
    started = time.perf_counter()
    train_ids, train_labels, test_ids, test_labels = load_split(dataset)
    splits = split_positions(train_labels, seeds, validation_fraction)
    path_1, path_2 = PAPER_PATHS[dataset]

    discrepancies = {
        path: load_discrepancies(dataset, path, chunk_size) for path in (path_1, path_2)
    }
    alpha_selection = {}
    selected_train = []
    selected_test = []
    normalizers = {}
    for path in (path_1, path_2):
        selection, train_grids = select_single_path_alpha(
            discrepancies[path], train_ids, train_labels, splits, alphas, k
        )
        alpha_selection[path] = selection
        alpha = selection["selected"]["alpha"]
        b0, b1 = float(discrepancies[path]["b0"]), float(discrepancies[path]["b1"])
        selected_train.append(train_grids[alpha])
        selected_test.append(
            dwasim_affinity(
                discrepancies[path]["test_h"],
                discrepancies[path]["test_l"],
                b0,
                b1,
                alpha,
            )
        )
        normalizers[path] = {
            "B0": b0,
            "B1": b1,
            "profile_value_range": float(discrepancies[path]["value_range"]),
        }

    train_affinities = {"DWASim": tuple(selected_train)}
    test_affinities = {"DWASim": tuple(selected_test)}
    baseline_by_path = {path: baseline_affinities(dataset, path) for path in (path_1, path_2)}
    for method in ("PathSim", "HeteSim"):
        train_affinities[method] = (
            baseline_by_path[path_1][method][0],
            baseline_by_path[path_2][method][0],
        )
        test_affinities[method] = (
            baseline_by_path[path_1][method][1],
            baseline_by_path[path_2][method][1],
        )

    fusion_selection, test_metrics = evaluate_variants(
        train_affinities,
        test_affinities,
        train_ids,
        train_labels,
        test_labels,
        splits,
        weights,
        k,
    )
    return {
        "dataset": dataset,
        "paths": [path_1, path_2],
        "reference_nodes": int(train_ids.size),
        "held_out_queries": int(test_ids.size),
        "normalizers": normalizers,
        "path_alpha_selection": alpha_selection,
        "fusion_selection": fusion_selection,
        "test_metrics": test_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ACM", "DBLP", "all"], default="all")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=20250803)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    alphas = np.linspace(0.0, 1.0, 11)
    weights = np.linspace(0.0, 1.0, 11)
    seeds = list(range(args.seed_start, args.seed_start + args.splits))
    datasets = ("ACM", "DBLP") if args.dataset == "all" else (args.dataset,)
    started = time.perf_counter()
    rows = []
    for dataset in datasets:
        row = run_dataset(
            dataset,
            args.k,
            alphas,
            weights,
            seeds,
            args.validation_fraction,
            args.chunk_size,
        )
        rows.append(row)
        print(dataset, json.dumps({
            method: {
                variant: round(record["macro_f1"], 4)
                for variant, record in variants.items()
            }
            for method, variants in row["test_metrics"].items()
        }, sort_keys=True))

    result = {
        "protocol": "real-data-held-out-multipath-fusion",
        "k": args.k,
        "alpha_grid": alphas.tolist(),
        "fusion_weight_grid": weights.tolist(),
        "validation_fraction": args.validation_fraction,
        "split_seeds": seeds,
        "test_labels_used_for_selection": False,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    output = args.output or RESULTS_ROOT / f"real_multipath_fusion_k{args.k}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
