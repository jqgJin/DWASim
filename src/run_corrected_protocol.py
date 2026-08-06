"""Run a leakage-free held-out evaluation on the recovered HGB datasets.

This runner is intentionally separate from ``reproduce_original.py``.  It uses
only ``label.dat`` nodes as references and only ``label.dat.test`` nodes as
queries.  Every query receives exactly k labelled reference neighbours, with
deterministic tie handling.

Two DWASim variants are reported from the same pairwise discrepancies:

``DWASim-current``
    The manuscript's global path-level normalization.  Its denominator is
    constant for a fixed path and therefore does not alter nearest-neighbour
    ranking.

``DWASim-historical``
    The public code's pair/order-dependent range denominator, but evaluated
    under the corrected train/test protocol.  This isolates scoring changes
    from the leakage repair.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from reproduce_original import (
    CACHE_ROOT,
    PATHS,
    PROCESSED_ROOT,
    RESULTS_ROOT,
    full_profile,
    interval_ranges,
    multiply_chain,
)
from similarity_baselines import BASELINE_CACHE_TAG, symmetric_path_affinities


PAPER_PATHS = {
    "ACM": ("PAP", "PSP"),
    "DBLP": ("APA", "APTPA"),
}


def load_split(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.load(PROCESSED_ROOT / dataset / "labels.npz")
    return (
        labels["train_ids"].astype(np.int64),
        labels["train_labels"].astype(np.int64),
        labels["test_ids"].astype(np.int64),
        labels["test_labels"].astype(np.int64),
    )


def deterministic_topk(
    values: np.ndarray,
    candidate_ids: np.ndarray,
    k: int,
    *,
    largest: bool,
) -> np.ndarray:
    """Select k candidates, resolving score ties by the candidate node ID."""
    if k <= 0 or k > candidate_ids.size:
        raise ValueError(f"k must be in [1, {candidate_ids.size}], received {k}")
    selected = np.empty((values.shape[0], k), dtype=np.int64)
    primary = -values if largest else values
    for row in range(values.shape[0]):
        order = np.lexsort((candidate_ids, primary[row]))
        selected[row] = order[:k]
    return selected


def majority_vote(neighbour_positions: np.ndarray, reference_labels: np.ndarray) -> np.ndarray:
    """Majority vote; ties go to the closest tied class, then smaller class ID."""
    predictions = np.empty(neighbour_positions.shape[0], dtype=np.int64)
    for row, positions in enumerate(neighbour_positions):
        labels = reference_labels[positions]
        classes, counts = np.unique(labels, return_counts=True)
        tied = classes[counts == counts.max()]
        if tied.size == 1:
            predictions[row] = tied[0]
            continue
        first_rank = {int(label): int(np.flatnonzero(labels == label)[0]) for label in tied}
        predictions[row] = min((int(label) for label in tied), key=lambda label: (first_rank[label], label))
    return predictions


def metric_record(truth: np.ndarray, prediction: np.ndarray, classes: np.ndarray) -> dict:
    per_class = f1_score(truth, prediction, labels=classes, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=classes, average="macro", zero_division=0)),
        "per_class_f1": {str(int(label)): float(value) for label, value in zip(classes, per_class)},
        "confusion_matrix": confusion_matrix(truth, prediction, labels=classes).tolist(),
    }


def baseline_predictions(dataset: str, path_name: str, k: int) -> dict[str, np.ndarray]:
    train_ids, train_labels, test_ids, _ = load_split(dataset)
    config = PATHS[dataset][path_name]

    raw_half = multiply_chain(dataset, config["half"], transition=False)
    transition_half = multiply_chain(dataset, config["half"], transition=True)
    affinities = symmetric_path_affinities(
        raw_half,
        transition_half,
        test_ids,
        train_ids,
    )
    pathsim_neighbours = deterministic_topk(
        affinities["PathSim"], train_ids, k, largest=True
    )
    hetesim_neighbours = deterministic_topk(
        affinities["HeteSim"], train_ids, k, largest=True
    )

    return {
        "PathSim": majority_vote(pathsim_neighbours, train_labels),
        "HeteSim": majority_vote(hetesim_neighbours, train_labels),
    }


def dwasim_predictions(
    dataset: str,
    path_name: str,
    k: int,
    lam: float,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must lie in [0, 1]")
    train_ids, train_labels, test_ids, _ = load_split(dataset)
    matrix = full_profile(dataset, path_name)
    dense = matrix.toarray().astype(np.float64, copy=False)
    reference = dense[train_ids]
    row_max = dense.max(axis=1)
    row_min = dense.min(axis=1)

    current_predictions: list[np.ndarray] = []
    historical_predictions: list[np.ndarray] = []
    for start in range(0, test_ids.size, chunk_size):
        query_ids = test_ids[start : start + chunk_size]
        query = dense[query_ids]
        hamming = np.rint(cdist(query, reference, metric="hamming") * dense.shape[1])
        l1 = cdist(query, reference, metric="cityblock")
        numerator = lam * hamming + (1.0 - lam) * l1

        current_neighbours = deterministic_topk(numerator, train_ids, k, largest=False)
        current_predictions.append(majority_vote(current_neighbours, train_labels))

        ranges = interval_ranges(row_max, row_min, query_ids)[:, train_ids]
        historical_denominator = lam + (1.0 - lam) * ranges
        historical_distance = np.divide(
            numerator,
            historical_denominator,
            out=np.full_like(numerator, np.inf),
            where=historical_denominator != 0,
        )
        historical_neighbours = deterministic_topk(
            historical_distance,
            train_ids,
            k,
            largest=False,
        )
        historical_predictions.append(majority_vote(historical_neighbours, train_labels))

    return {
        "DWASim-current": np.concatenate(current_predictions),
        "DWASim-historical": np.concatenate(historical_predictions),
    }


def run_row(dataset: str, path_name: str, k: int, lam: float, chunk_size: int) -> dict:
    started = time.perf_counter()
    train_ids, _, test_ids, test_labels = load_split(dataset)
    predictions = baseline_predictions(dataset, path_name, k)
    predictions.update(dwasim_predictions(dataset, path_name, k, lam, chunk_size))
    classes = np.unique(test_labels)

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_ROOT / (
        f"corrected_predictions_{BASELINE_CACHE_TAG}_{dataset}_{path_name}_"
        f"k{k}_lam{lam:g}.npz"
    )
    np.savez_compressed(cache_path, test_ids=test_ids, truth=test_labels, **predictions)

    return {
        "dataset": dataset,
        "path": path_name,
        "reference_nodes": int(train_ids.size),
        "held_out_queries": int(test_ids.size),
        "metrics": {
            method: metric_record(test_labels, prediction, classes)
            for method, prediction in predictions.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ACM", "DBLP", "all"], default="all")
    parser.add_argument("--path", help="one valid path; omitted means the four paper paths")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--lambda-value", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    datasets = ("ACM", "DBLP") if args.dataset == "all" else (args.dataset,)
    rows: list[dict] = []
    started = time.perf_counter()
    for dataset in datasets:
        paths = (args.path,) if args.path else PAPER_PATHS[dataset]
        for path_name in paths:
            if path_name not in PATHS[dataset]:
                raise ValueError(f"Unknown path {path_name!r} for {dataset}")
            row = run_row(dataset, path_name, args.k, args.lambda_value, args.chunk_size)
            rows.append(row)
            summary = {
                method: {
                    "accuracy": round(values["accuracy"], 4),
                    "macro_f1": round(values["macro_f1"], 4),
                }
                for method, values in row["metrics"].items()
            }
            print(dataset, path_name, json.dumps(summary, sort_keys=True))

    result = {
        "protocol": "held-out-train-reference-test-query",
        "k": args.k,
        "lambda": args.lambda_value,
        "deterministic_ties": "score, candidate node ID; vote ties by closest class then class ID",
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    output = args.output or RESULTS_ROOT / f"corrected_heldout_k{args.k}_lam{args.lambda_value:g}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
