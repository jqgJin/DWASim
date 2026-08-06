"""Select DWASim's effective normalized weight without using test labels.

The manuscript shows that the raw coefficient lambda corresponds to

    alpha = lambda * B0 / (lambda * B0 + (1-lambda) * B1).

When path counts have a wide range, a uniform lambda grid is highly skewed in
effective-weight space.  This runner therefore searches a uniform alpha grid
on repeated stratified splits of the official training labels, translates the
selected alpha back to lambda, and evaluates the official test labels once.
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

from reproduce_original import PATHS, RESULTS_ROOT, full_profile
from run_corrected_protocol import (
    PAPER_PATHS,
    deterministic_topk,
    load_split,
    majority_vote,
    metric_record,
)


def lambda_from_alpha(alpha: float, b0: float, b1: float) -> float:
    if alpha <= 0.0:
        return 0.0
    if alpha >= 1.0:
        return 1.0
    return float(alpha * b1 / (b0 * (1.0 - alpha) + alpha * b1))


def pair_discrepancies(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dimensions = query.shape[1]
    hamming = np.rint(cdist(query, reference, metric="hamming") * dimensions)
    magnitude = cdist(query, reference, metric="cityblock")
    return hamming, magnitude


def predictions_for_alpha(
    hamming: np.ndarray,
    magnitude: np.ndarray,
    b0: float,
    b1: float,
    alpha: float,
    candidate_ids: np.ndarray,
    candidate_labels: np.ndarray,
    k: int,
) -> np.ndarray:
    normalized_distance = alpha * hamming / b0 + (1.0 - alpha) * magnitude / b1
    neighbours = deterministic_topk(normalized_distance, candidate_ids, k, largest=False)
    return majority_vote(neighbours, candidate_labels)


def tune_row(
    dataset: str,
    path_name: str,
    k: int,
    alphas: np.ndarray,
    split_seeds: list[int],
    validation_fraction: float,
) -> dict:
    started = time.perf_counter()
    train_ids, train_labels, test_ids, test_labels = load_split(dataset)
    matrix = full_profile(dataset, path_name)
    dense = matrix.toarray().astype(np.float64, copy=False)
    train_profiles = dense[train_ids]
    test_profiles = dense[test_ids]

    b0 = float(dense.shape[1])
    value_range = float(dense.max() - dense.min())
    b1 = b0 * value_range
    if b1 <= 0:
        raise ValueError(f"Degenerate path representation for {dataset} {path_name}")

    train_hamming, train_magnitude = pair_discrepancies(train_profiles, train_profiles)
    validation: dict[str, dict[str, list[float]]] = {
        f"{alpha:.1f}": {"macro_f1": [], "accuracy": []} for alpha in alphas
    }

    for seed in split_seeds:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=seed,
        )
        reference_positions, validation_positions = next(
            splitter.split(np.zeros(train_labels.size), train_labels)
        )
        hamming = train_hamming[np.ix_(validation_positions, reference_positions)]
        magnitude = train_magnitude[np.ix_(validation_positions, reference_positions)]
        candidate_ids = train_ids[reference_positions]
        candidate_labels = train_labels[reference_positions]
        truth = train_labels[validation_positions]

        for alpha in alphas:
            prediction = predictions_for_alpha(
                hamming,
                magnitude,
                b0,
                b1,
                float(alpha),
                candidate_ids,
                candidate_labels,
                k,
            )
            record = validation[f"{alpha:.1f}"]
            record["macro_f1"].append(
                float(f1_score(truth, prediction, average="macro", zero_division=0))
            )
            record["accuracy"].append(float(accuracy_score(truth, prediction)))

    validation_summary: dict[str, dict] = {}
    for alpha in alphas:
        key = f"{alpha:.1f}"
        macro = np.asarray(validation[key]["macro_f1"])
        accuracy = np.asarray(validation[key]["accuracy"])
        validation_summary[key] = {
            "alpha": float(alpha),
            "lambda": lambda_from_alpha(float(alpha), b0, b1),
            "macro_f1_mean": float(macro.mean()),
            "macro_f1_std": float(macro.std(ddof=1)),
            "accuracy_mean": float(accuracy.mean()),
            "accuracy_std": float(accuracy.std(ddof=1)),
        }

    selected = max(
        validation_summary.values(),
        key=lambda item: (
            item["macro_f1_mean"],
            item["accuracy_mean"],
            -abs(item["alpha"] - 0.5),
        ),
    )

    test_hamming, test_magnitude = pair_discrepancies(test_profiles, train_profiles)
    test_prediction = predictions_for_alpha(
        test_hamming,
        test_magnitude,
        b0,
        b1,
        selected["alpha"],
        train_ids,
        train_labels,
        k,
    )
    classes = np.unique(test_labels)

    return {
        "dataset": dataset,
        "path": path_name,
        "reference_nodes": int(train_ids.size),
        "held_out_queries": int(test_ids.size),
        "normalizers": {"B0": b0, "B1": b1, "profile_value_range": value_range},
        "selected": selected,
        "validation": validation_summary,
        "test_metrics": metric_record(test_labels, test_prediction, classes),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ACM", "DBLP", "all"], default="all")
    parser.add_argument("--path")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=20250803)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    alphas = np.linspace(0.0, 1.0, 11)
    split_seeds = list(range(args.seed_start, args.seed_start + args.splits))
    datasets = ("ACM", "DBLP") if args.dataset == "all" else (args.dataset,)
    rows: list[dict] = []
    started = time.perf_counter()
    for dataset in datasets:
        paths = (args.path,) if args.path else PAPER_PATHS[dataset]
        for path_name in paths:
            if path_name not in PATHS[dataset]:
                raise ValueError(f"Unknown path {path_name!r} for {dataset}")
            row = tune_row(
                dataset,
                path_name,
                args.k,
                alphas,
                split_seeds,
                args.validation_fraction,
            )
            rows.append(row)
            print(
                dataset,
                path_name,
                "alpha",
                round(row["selected"]["alpha"], 3),
                "lambda",
                round(row["selected"]["lambda"], 6),
                "test",
                json.dumps(
                    {
                        "accuracy": round(row["test_metrics"]["accuracy"], 4),
                        "macro_f1": round(row["test_metrics"]["macro_f1"], 4),
                    }
                ),
            )

    result = {
        "protocol": "effective-weight-selected-on-repeated-stratified-training-validation",
        "k": args.k,
        "alpha_grid": alphas.tolist(),
        "validation_fraction": args.validation_fraction,
        "split_seeds": split_seeds,
        "test_labels_used_for_selection": False,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    output = args.output or RESULTS_ROOT / f"effective_weight_tuning_k{args.k}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

