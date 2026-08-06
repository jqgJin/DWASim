"""Reproduce the historical DWASim label-prediction protocol.

This module intentionally preserves the original protocol before any repair:

* 500 query nodes are sampled from ``label.dat``;
* the query remains eligible as one of its own top-10 neighbours;
* neighbours outside ``label.dat`` are discarded after top-10 selection;
* the denominator range used by the historical DWASim code accumulates over
  the row interval between a pair of node identifiers.

Two global positive factors in the historical DWASim implementation are not
materialized here: ``Max_non_zero`` and the final division by the largest
dissimilarity.  Both multiply every pairwise dissimilarity by a path-wide
positive constant and therefore cannot alter a top-k ranking.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import torch

from similarity_baselines import BASELINE_CACHE_TAG, symmetric_path_affinities


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = ROOT / "data" / "processed"
CACHE_ROOT = ROOT / "cache"
RESULTS_ROOT = ROOT / "results"

PATHS = {
    "ACM": {
        "PAP": {"half": [2], "full": [2, 3]},
        "PSP": {"half": [4], "full": [4, 5]},
        "PTP": {"half": [6], "full": [6, 7]},
    },
    "DBLP": {
        "APA": {"half": [0], "full": [0, 3]},
        "APTPA": {"half": [0, 1], "full": [0, 1, 4, 3]},
        "APVPA": {"half": [0, 2], "full": [0, 2, 5, 3]},
    },
    "IMDB": {
        "MDM": {"half": [0], "full": [0, 1]},
        "MAM": {"half": [2], "full": [2, 3]},
        "MKM": {"half": [4], "full": [4, 5]},
    },
}

PAPER_TARGETS = {
    ("DBLP", "APA"): {"DWASim": 0.996, "PathSim": 0.434, "HeteSim": 0.434},
    ("DBLP", "APTPA"): {"DWASim": 0.786, "PathSim": 0.928, "HeteSim": 0.928},
    ("ACM", "PAP"): {"DWASim": 0.842, "PathSim": 0.702, "HeteSim": 0.702},
    ("ACM", "PSP"): {"DWASim": 0.676, "PathSim": 0.676, "HeteSim": 0.676},
}


def load_relation(dataset: str, relation_id: int) -> sp.csr_matrix:
    return sp.load_npz(PROCESSED_ROOT / dataset / f"relation_{relation_id}.npz").tocsr()


def row_normalize(matrix: sp.csr_matrix) -> sp.csr_matrix:
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.zeros_like(sums, dtype=np.float64)
    nonzero = sums != 0
    inverse[nonzero] = 1.0 / sums[nonzero]
    return sp.diags(inverse).dot(matrix).tocsr()


def multiply_chain(dataset: str, relation_ids: list[int], transition: bool = False) -> sp.csr_matrix:
    matrices = [load_relation(dataset, relation_id) for relation_id in relation_ids]
    if transition:
        matrices = [row_normalize(matrix) for matrix in matrices]
    product = matrices[0]
    for matrix in matrices[1:]:
        product = product.dot(matrix).tocsr()
    product.eliminate_zeros()
    product.sort_indices()
    return product


def load_labels(dataset: str) -> tuple[np.ndarray, np.ndarray, int]:
    data = np.load(PROCESSED_ROOT / dataset / "labels.npz")
    train_ids = data["train_ids"].astype(np.int64)
    train_labels = data["train_labels"].astype(np.int64)
    with (PROCESSED_ROOT / dataset / "manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    target_count = int(manifest["node_counts"]["0"])
    return train_ids, train_labels, target_count


def torch_topk(scores: np.ndarray, k: int = 10) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(scores, dtype=np.float32))
    return torch.topk(tensor, k=k, dim=1, largest=True).indices.numpy()


def vote_predictions(
    top_indices: np.ndarray,
    train_ids: np.ndarray,
    train_labels: np.ndarray,
    target_count: int,
) -> np.ndarray:
    lookup = np.full(target_count, -1, dtype=np.int64)
    lookup[train_ids] = train_labels
    predictions = np.full(top_indices.shape[0], -1, dtype=np.int64)
    for row_id, neighbours in enumerate(top_indices):
        visible = [int(lookup[node]) for node in neighbours if lookup[node] >= 0]
        if visible:
            counts = Counter(visible)
            predictions[row_id] = max(counts.keys(), key=counts.get)
    return predictions


def baseline_predictions(dataset: str, path_name: str) -> dict[str, np.ndarray]:
    cache_path = CACHE_ROOT / (
        f"baseline_predictions_{BASELINE_CACHE_TAG}_{dataset}_{path_name}.npz"
    )
    if cache_path.exists():
        cached = np.load(cache_path)
        return {
            "train_ids": cached["train_ids"],
            "train_labels": cached["train_labels"],
            "PathSim": cached["PathSim"],
            "HeteSim": cached["HeteSim"],
        }

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    train_ids, train_labels, target_count = load_labels(dataset)
    config = PATHS[dataset][path_name]
    raw_half = multiply_chain(dataset, config["half"], transition=False)
    transition_half = multiply_chain(dataset, config["half"], transition=True)
    all_ids = np.arange(target_count, dtype=np.int64)
    affinities = symmetric_path_affinities(
        raw_half,
        transition_half,
        train_ids,
        all_ids,
    )
    pathsim_top = torch_topk(affinities["PathSim"])
    pathsim_predictions = vote_predictions(pathsim_top, train_ids, train_labels, target_count)
    hetesim_top = torch_topk(affinities["HeteSim"])
    hetesim_predictions = vote_predictions(hetesim_top, train_ids, train_labels, target_count)

    np.savez_compressed(
        cache_path,
        train_ids=train_ids,
        train_labels=train_labels,
        PathSim=pathsim_predictions,
        HeteSim=hetesim_predictions,
    )
    return {
        "train_ids": train_ids,
        "train_labels": train_labels,
        "PathSim": pathsim_predictions,
        "HeteSim": hetesim_predictions,
    }


def original_sample(train_ids: np.ndarray, seed: int, count: int) -> np.ndarray:
    if count > train_ids.size:
        raise ValueError(f"Cannot sample {count} unique queries from {train_ids.size} labels")
    rng = random.Random(seed)
    selected: list[int] = []
    selected_set: set[int] = set()
    while len(selected) < count:
        node = int(train_ids[rng.randint(0, train_ids.size - 1)])
        if node not in selected_set:
            selected.append(node)
            selected_set.add(node)
    return np.asarray(selected, dtype=np.int64)


def subset_accuracy(prediction_bundle: dict[str, np.ndarray], query_ids: np.ndarray, method: str) -> float:
    train_ids = prediction_bundle["train_ids"]
    train_labels = prediction_bundle["train_labels"]
    positions = {int(node): position for position, node in enumerate(train_ids)}
    indices = np.asarray([positions[int(node)] for node in query_ids], dtype=np.int64)
    return float(np.mean(prediction_bundle[method][indices] == train_labels[indices]))


def interval_ranges(row_max: np.ndarray, row_min: np.ndarray, query_ids: np.ndarray) -> np.ndarray:
    n = row_max.size
    ranges = np.empty((query_ids.size, n), dtype=np.float64)
    for row, query in enumerate(query_ids):
        right_max = np.maximum.accumulate(row_max[query:])
        right_min = np.minimum.accumulate(row_min[query:])
        ranges[row, query:] = right_max - right_min

        left_max = np.maximum.accumulate(row_max[: query + 1][::-1])[::-1]
        left_min = np.minimum.accumulate(row_min[: query + 1][::-1])[::-1]
        ranges[row, : query + 1] = left_max - left_min
    return ranges


def full_profile(dataset: str, path_name: str) -> sp.csr_matrix:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_ROOT / f"full_profile_{dataset}_{path_name}.npz"
    if cache_path.exists():
        return sp.load_npz(cache_path).tocsr()
    matrix = multiply_chain(dataset, PATHS[dataset][path_name]["full"], transition=False)
    sp.save_npz(cache_path, matrix, compressed=True)
    return matrix


def dwasim_predictions(
    dataset: str,
    path_name: str,
    query_ids: np.ndarray,
    lam: float = 0.5,
) -> np.ndarray:
    train_ids, train_labels, target_count = load_labels(dataset)
    matrix = full_profile(dataset, path_name)
    dense = matrix.toarray().astype(np.float64, copy=False)
    query_rows = dense[query_ids]

    hamming = np.rint(cdist(query_rows, dense, metric="hamming") * dense.shape[1])
    l1 = cdist(query_rows, dense, metric="cityblock")
    row_max = dense.max(axis=1)
    row_min = dense.min(axis=1)
    ranges = interval_ranges(row_max, row_min, query_ids)

    denominator = lam + (1.0 - lam) * ranges
    numerator = lam * hamming + (1.0 - lam) * l1
    dissimilarity = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.inf),
        where=denominator != 0,
    )
    top_indices = torch_topk(-dissimilarity)
    return vote_predictions(top_indices, train_ids, train_labels, target_count)


def all_labeled_dwasim_predictions(dataset: str, path_name: str) -> dict[str, np.ndarray]:
    """Cache DWASim predictions for every label visible to the historical protocol."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_ROOT / f"dwasim_predictions_{dataset}_{path_name}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return {
            "train_ids": cached["train_ids"],
            "train_labels": cached["train_labels"],
            "DWASim": cached["DWASim"],
        }
    train_ids, train_labels, _ = load_labels(dataset)
    predictions = dwasim_predictions(dataset, path_name, train_ids)
    np.savez_compressed(
        cache_path,
        train_ids=train_ids,
        train_labels=train_labels,
        DWASim=predictions,
    )
    return {"train_ids": train_ids, "train_labels": train_labels, "DWASim": predictions}


def run(dataset: str, path_name: str, seed: int, query_count: int, skip_dwasim: bool) -> dict:
    started = time.perf_counter()
    bundle = baseline_predictions(dataset, path_name)
    query_ids = original_sample(bundle["train_ids"], seed, query_count)
    positions = {int(node): position for position, node in enumerate(bundle["train_ids"])}
    query_positions = np.asarray([positions[int(node)] for node in query_ids], dtype=np.int64)
    truth = bundle["train_labels"][query_positions]

    metrics = {
        "PathSim": subset_accuracy(bundle, query_ids, "PathSim"),
        "HeteSim": subset_accuracy(bundle, query_ids, "HeteSim"),
    }
    if not skip_dwasim:
        predictions = dwasim_predictions(dataset, path_name, query_ids)
        metrics["DWASim"] = float(np.mean(predictions == truth))

    targets = PAPER_TARGETS.get((dataset, path_name), {})
    return {
        "dataset": dataset,
        "path": path_name,
        "seed": seed,
        "queries": query_count,
        "protocol": "historical-fidelity",
        "metrics": metrics,
        "paper_targets": targets,
        "absolute_gaps": {
            method: abs(value - targets[method])
            for method, value in metrics.items()
            if method in targets
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ACM", "DBLP"], required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--skip-dwasim", action="store_true")
    args = parser.parse_args()
    if args.path not in PATHS[args.dataset]:
        parser.error(f"Unknown path {args.path!r} for {args.dataset}")

    result = run(args.dataset, args.path, args.seed, args.queries, args.skip_dwasim)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    output = RESULTS_ROOT / (
        f"historical_{args.dataset}_{args.path}_seed{args.seed}_q{args.queries}.json"
    )
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
