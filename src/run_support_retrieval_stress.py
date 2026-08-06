"""Support-conflict stress test with retrieval and strong simple baselines.

The experiment adds a third, semantically complementary path to ACM and DBLP
and evaluates classification and label-retrieval under one held-out protocol.
All component and path-fusion parameters are selected from repeated splits of
the official training labels.  The official test labels are used only after
selection is complete.

The adaptive candidate treats Bray--Curtis magnitude discrepancy as the
anchor and activates support discrepancy only when it exposes a conflict:

    d_adapt(x, y) = d_m(x, y) + beta [d_s(x, y) - d_m(x, y)]_+.

Thus the support term cannot make a pair appear more similar than the
magnitude evidence alone.  beta=0 recovers the magnitude-only formulation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances

from reproduce_original import CACHE_ROOT, PATHS, RESULTS_ROOT, full_profile, multiply_chain
from run_corrected_protocol import deterministic_topk, load_split, majority_vote
from run_real_multipath_fusion import split_positions
from run_nested_multipath_optimization import simplex_weights
from similarity_baselines import symmetric_path_affinities


STRESS_PATHS = {
    "ACM": ("PAP", "PSP", "PTP"),
    "DBLP": ("APA", "APTPA", "APVPA"),
}

PATH_MEANINGS = {
    "PAP": "paper-author-paper",
    "PSP": "paper-subject-paper",
    "PTP": "paper-term-paper",
    "APA": "author-paper-author",
    "APTPA": "author-paper-term-paper-author",
    "APVPA": "author-paper-venue-paper-author",
}

METHOD_ORDER = (
    "TriComponentDWASim",
    "AdaptiveSupportDWASim",
    "PairRelativeDWASim",
    "MagnitudeOnly",
    "Jaccard",
    "Cosine",
    "Bhattacharyya",
    "PathSim",
    "HeteSim",
)


def _support_matrix(matrix: sp.csr_matrix) -> sp.csr_matrix:
    support = matrix.copy().astype(np.float64)
    support.data = np.ones_like(support.data)
    return support


def _row_probability_sqrt(matrix: sp.csr_matrix) -> sp.csr_matrix:
    matrix = matrix.copy().astype(np.float64)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(
        1.0,
        totals,
        out=np.zeros_like(totals, dtype=np.float64),
        where=totals > 0,
    )
    normalized = sp.diags(inverse).dot(matrix).tocsr()
    normalized.data = np.sqrt(normalized.data)
    return normalized


def _component_cache(dataset: str, path_name: str) -> Path:
    return CACHE_ROOT / f"support_retrieval_components_{dataset}_{path_name}.npz"


def load_components(dataset: str, path_name: str) -> dict[str, np.ndarray]:
    """Load train/train and test/train path-profile similarities."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = _component_cache(dataset, path_name)
    if cache_path.exists():
        with np.load(cache_path) as cached:
            return {name: cached[name] for name in cached.files}

    train_ids, _, test_ids, _ = load_split(dataset)
    profiles = full_profile(dataset, path_name).tocsr().astype(np.float64)
    train = profiles[train_ids]
    test = profiles[test_ids]
    train_support = _support_matrix(train)
    test_support = _support_matrix(test)

    train_intersection = train_support.dot(train_support.T).toarray()
    test_intersection = test_support.dot(train_support.T).toarray()
    train_support_count = np.asarray(train_support.sum(axis=1)).ravel()
    test_support_count = np.asarray(test_support.sum(axis=1)).ravel()
    train_union = (
        train_support_count[:, None]
        + train_support_count[None, :]
        - train_intersection
    )
    test_union = (
        test_support_count[:, None]
        + train_support_count[None, :]
        - test_intersection
    )
    train_jaccard = np.divide(
        train_union - train_intersection,
        train_union,
        out=np.zeros_like(train_union),
        where=train_union > 0,
    )
    test_jaccard = np.divide(
        test_union - test_intersection,
        test_union,
        out=np.zeros_like(test_union),
        where=test_union > 0,
    )

    train_l1 = manhattan_distances(train, train)
    test_l1 = manhattan_distances(test, train)
    train_activity = np.asarray(train.sum(axis=1)).ravel()
    test_activity = np.asarray(test.sum(axis=1)).ravel()
    train_activity_sum = train_activity[:, None] + train_activity[None, :]
    test_activity_sum = test_activity[:, None] + train_activity[None, :]
    train_bray = np.divide(
        train_l1,
        train_activity_sum,
        out=np.zeros_like(train_l1),
        where=train_activity_sum > 0,
    )
    test_bray = np.divide(
        test_l1,
        test_activity_sum,
        out=np.zeros_like(test_l1),
        where=test_activity_sum > 0,
    )

    train_cosine = cosine_similarity(train, train)
    test_cosine = cosine_similarity(test, train)
    train_sqrt = _row_probability_sqrt(train)
    test_sqrt = _row_probability_sqrt(test)
    train_bhattacharyya = train_sqrt.dot(train_sqrt.T).toarray()
    test_bhattacharyya = test_sqrt.dot(train_sqrt.T).toarray()

    config = PATHS[dataset][path_name]
    raw_half = multiply_chain(dataset, config["half"], transition=False)
    transition_half = multiply_chain(dataset, config["half"], transition=True)
    train_classical = symmetric_path_affinities(
        raw_half, transition_half, train_ids, train_ids
    )
    test_classical = symmetric_path_affinities(
        raw_half, transition_half, test_ids, train_ids
    )

    payload = {
        "train_jaccard": np.clip(train_jaccard, 0.0, 1.0).astype(np.float32),
        "test_jaccard": np.clip(test_jaccard, 0.0, 1.0).astype(np.float32),
        "train_bray": np.clip(train_bray, 0.0, 1.0).astype(np.float32),
        "test_bray": np.clip(test_bray, 0.0, 1.0).astype(np.float32),
        "train_cosine": np.clip(train_cosine, 0.0, 1.0).astype(np.float32),
        "test_cosine": np.clip(test_cosine, 0.0, 1.0).astype(np.float32),
        "train_bhattacharyya": np.clip(train_bhattacharyya, 0.0, 1.0).astype(np.float32),
        "test_bhattacharyya": np.clip(test_bhattacharyya, 0.0, 1.0).astype(np.float32),
        "train_pathsim": train_classical["PathSim"].astype(np.float32),
        "test_pathsim": test_classical["PathSim"].astype(np.float32),
        "train_hetesim": train_classical["HeteSim"].astype(np.float32),
        "test_hetesim": test_classical["HeteSim"].astype(np.float32),
    }
    np.savez_compressed(cache_path, **payload)
    return payload


def relative_affinity(jaccard: np.ndarray, bray: np.ndarray, beta: float) -> np.ndarray:
    distance = float(beta) * jaccard + (1.0 - float(beta)) * bray
    return np.clip(1.0 - distance, 0.0, 1.0)


def adaptive_affinity(jaccard: np.ndarray, bray: np.ndarray, beta: float) -> np.ndarray:
    conflict = np.maximum(jaccard - bray, 0.0)
    distance = bray + float(beta) * conflict
    return np.clip(1.0 - distance, 0.0, 1.0)


def ndcg_at_k_from_order(
    order: np.ndarray,
    query_labels: np.ndarray,
    reference_labels: np.ndarray,
    k: int,
) -> np.ndarray:
    selected_relevance = reference_labels[order] == query_labels[:, None]
    discounts = 1.0 / np.log2(np.arange(2, selected_relevance.shape[1] + 2))
    dcg = (selected_relevance * discounts[None, :]).sum(axis=1)
    class_counts = {
        int(label): int(np.sum(reference_labels == label))
        for label in np.unique(reference_labels)
    }
    ideal_counts = np.asarray(
        [min(k, class_counts.get(int(label), 0)) for label in query_labels],
        dtype=np.int64,
    )
    cumulative = np.concatenate(([0.0], np.cumsum(discounts)))
    ideal = cumulative[ideal_counts]
    return np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)


def evaluate_affinity(
    affinity: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_labels: np.ndarray,
    query_labels: np.ndarray,
    k: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    order = deterministic_topk(
        affinity, candidate_ids, min(k, candidate_ids.size), largest=True
    )
    prediction = majority_vote(order, candidate_labels)
    ndcg = ndcg_at_k_from_order(order, query_labels, candidate_labels, k)
    metrics = {
        "macro_f1": float(
            f1_score(query_labels, prediction, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(query_labels, prediction)),
        f"ndcg_at_{k}": float(np.mean(ndcg)),
        f"ndcg_at_{k}_standard_error": float(
            np.std(ndcg, ddof=1) / np.sqrt(ndcg.size)
        ),
    }
    return metrics, prediction, ndcg


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


def paired_bootstrap(
    truth: np.ndarray,
    first_prediction: np.ndarray,
    second_prediction: np.ndarray,
    first_ndcg: np.ndarray,
    second_ndcg: np.ndarray,
    iterations: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    classes = np.unique(truth)
    class_positions = [np.flatnonzero(truth == label) for label in classes]
    macro_differences = np.empty(iterations, dtype=np.float64)
    ndcg_differences = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sample = np.concatenate(
            [rng.choice(positions, size=positions.size, replace=True) for positions in class_positions]
        )
        macro_differences[iteration] = f1_score(
            truth[sample],
            first_prediction[sample],
            labels=classes,
            average="macro",
            zero_division=0,
        ) - f1_score(
            truth[sample],
            second_prediction[sample],
            labels=classes,
            average="macro",
            zero_division=0,
        )
        ndcg_differences[iteration] = float(
            np.mean(first_ndcg[sample] - second_ndcg[sample])
        )

    def summarize(values: np.ndarray, point: float) -> dict:
        return {
            "difference": float(point),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "probability_difference_positive": float(np.mean(values > 0)),
        }

    macro_point = f1_score(
        truth, first_prediction, average="macro", zero_division=0
    ) - f1_score(truth, second_prediction, average="macro", zero_division=0)
    ndcg_point = float(np.mean(first_ndcg - second_ndcg))
    return {
        "macro_f1": summarize(macro_differences, macro_point),
        "ndcg_at_k": summarize(ndcg_differences, ndcg_point),
    }


def conflict_strata(
    components: list[dict[str, np.ndarray]],
    magnitude_weights: list[float],
    train_ids: np.ndarray,
    magnitude_affinity: np.ndarray,
    adaptive_ndcg: np.ndarray,
    magnitude_ndcg: np.ndarray,
    k: int,
) -> dict:
    order = deterministic_topk(magnitude_affinity, train_ids, k, largest=True)
    query_rows = np.arange(order.shape[0])[:, None]
    conflict = np.zeros_like(order, dtype=np.float64)
    for weight, values in zip(magnitude_weights, components):
        gap = np.maximum(values["test_jaccard"] - values["test_bray"], 0.0)
        conflict += float(weight) * gap[query_rows, order]
    score = conflict.mean(axis=1)
    edges = np.quantile(score, [0.25, 0.5, 0.75])
    groups = np.digitize(score, edges, right=True)
    result = {}
    for group in range(4):
        mask = groups == group
        result[f"Q{group + 1}"] = {
            "queries": int(mask.sum()),
            "conflict_score_mean": float(score[mask].mean()),
            "adaptive_minus_magnitude_ndcg": float(
                (adaptive_ndcg[mask] - magnitude_ndcg[mask]).mean()
            ),
        }
    return {
        "definition": (
            "Quartiles of the mean positive support-minus-magnitude discrepancy "
            "among the magnitude-only top-k candidates; labels are not used."
        ),
        "quantile_edges": edges.tolist(),
        "groups": result,
    }


def run_dataset(
    dataset: str,
    k: int,
    split_seeds: list[int],
    validation_fraction: float,
    beta_grid: np.ndarray,
    fusion_step: float,
    bootstrap_iterations: int,
) -> dict:
    started = time.perf_counter()
    train_ids, train_labels, test_ids, test_labels = load_split(dataset)
    paths = STRESS_PATHS[dataset]
    splits = split_positions(train_labels, split_seeds, validation_fraction)
    components = {path: load_components(dataset, path) for path in paths}
    weights = simplex_weights(len(paths), fusion_step)

    train_views: dict[str, list[np.ndarray]] = {name: [] for name in METHOD_ORDER}
    test_views: dict[str, list[np.ndarray]] = {name: [] for name in METHOD_ORDER}
    parameter_selection: dict[str, dict] = {
        "TriComponentDWASim": {},
        "AdaptiveSupportDWASim": {},
        "PairRelativeDWASim": {},
    }

    for path in paths:
        values = components[path]
        tri_selection = select_path_mixture(
            values,
            train_ids,
            train_labels,
            splits,
            simplex_weights(3, 0.25),
            k,
        )
        parameter_selection["TriComponentDWASim"][path] = tri_selection
        tri_weights = tri_selection["selected"][
            "weights_support_magnitude_direction"
        ]
        train_views["TriComponentDWASim"].append(
            tri_weights[0] * (1.0 - values["train_jaccard"])
            + tri_weights[1] * (1.0 - values["train_bray"])
            + tri_weights[2] * values["train_cosine"]
        )
        test_views["TriComponentDWASim"].append(
            tri_weights[0] * (1.0 - values["test_jaccard"])
            + tri_weights[1] * (1.0 - values["test_bray"])
            + tri_weights[2] * values["test_cosine"]
        )
        for formulation, name in (
            ("adaptive", "AdaptiveSupportDWASim"),
            ("relative", "PairRelativeDWASim"),
        ):
            selection = select_path_beta(
                values,
                train_ids,
                train_labels,
                splits,
                beta_grid,
                k,
                formulation,
            )
            parameter_selection[name][path] = selection
            beta = float(selection["selected"]["beta"])
            affinity_function = (
                adaptive_affinity if formulation == "adaptive" else relative_affinity
            )
            train_views[name].append(
                affinity_function(
                    values["train_jaccard"], values["train_bray"], beta
                )
            )
            test_views[name].append(
                affinity_function(
                    values["test_jaccard"], values["test_bray"], beta
                )
            )

        fixed = {
            "MagnitudeOnly": (1.0 - values["train_bray"], 1.0 - values["test_bray"]),
            "Jaccard": (1.0 - values["train_jaccard"], 1.0 - values["test_jaccard"]),
            "Cosine": (values["train_cosine"], values["test_cosine"]),
            "Bhattacharyya": (
                values["train_bhattacharyya"],
                values["test_bhattacharyya"],
            ),
            "PathSim": (values["train_pathsim"], values["test_pathsim"]),
            "HeteSim": (values["train_hetesim"], values["test_hetesim"]),
        }
        for name, (train_affinity, test_affinity) in fixed.items():
            train_views[name].append(train_affinity)
            test_views[name].append(test_affinity)

    fusion_selection = {}
    test_metrics = {}
    predictions = {}
    per_query_ndcg = {}
    fused_test_affinities = {}
    for method in METHOD_ORDER:
        selection = select_fusion(
            train_views[method],
            train_ids,
            train_labels,
            splits,
            weights,
            k,
        )
        fusion_selection[method] = selection
        selected_weights = selection["selected"]["weights"]
        fused_test = sum(
            weight * view for weight, view in zip(selected_weights, test_views[method])
        )
        metrics, prediction, ndcg = evaluate_affinity(
            fused_test, train_ids, train_labels, test_labels, k
        )
        test_metrics[method] = metrics
        predictions[method] = prediction
        per_query_ndcg[method] = ndcg
        fused_test_affinities[method] = fused_test

    paired = {}
    for method in (
        "TriComponentDWASim",
        "AdaptiveSupportDWASim",
        "PairRelativeDWASim",
    ):
        paired[method] = {}
        for comparator in ("MagnitudeOnly", "Cosine"):
            paired[method][f"minus_{comparator}"] = paired_bootstrap(
                test_labels,
                predictions[method],
                predictions[comparator],
                per_query_ndcg[method],
                per_query_ndcg[comparator],
                bootstrap_iterations,
                20260806
                + sum(
                    ord(character)
                    for character in dataset + method + comparator
                ),
            )
    strata = conflict_strata(
        [components[path] for path in paths],
        fusion_selection["MagnitudeOnly"]["selected"]["weights"],
        train_ids,
        fused_test_affinities["MagnitudeOnly"],
        per_query_ndcg["AdaptiveSupportDWASim"],
        per_query_ndcg["MagnitudeOnly"],
        k,
    )
    return {
        "dataset": dataset,
        "paths": list(paths),
        "path_meanings": {path: PATH_MEANINGS[path] for path in paths},
        "training_reference_nodes": int(train_ids.size),
        "held_out_query_nodes": int(test_ids.size),
        "parameter_selection": parameter_selection,
        "fusion_selection": fusion_selection,
        "test_metrics": test_metrics,
        "paired_method_comparisons": paired,
        "support_conflict_strata": strata,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ACM", "DBLP", "all"], default="all")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=20250803)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--fusion-step", type=float, default=0.25)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "support_retrieval_stress.json",
    )
    args = parser.parse_args()

    beta_grid = np.asarray(
        [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        dtype=np.float64,
    )
    split_seeds = list(range(args.seed_start, args.seed_start + args.splits))
    datasets = ("ACM", "DBLP") if args.dataset == "all" else (args.dataset,)
    started = time.perf_counter()
    rows = []
    for dataset in datasets:
        row = run_dataset(
            dataset,
            args.k,
            split_seeds,
            args.validation_fraction,
            beta_grid,
            args.fusion_step,
            args.bootstrap_iterations,
        )
        rows.append(row)
        print(
            dataset,
            json.dumps(
                {
                    method: {
                        "macro_f1": round(values["macro_f1"], 4),
                        f"ndcg_at_{args.k}": round(values[f"ndcg_at_{args.k}"], 4),
                    }
                    for method, values in row["test_metrics"].items()
                },
                sort_keys=True,
            ),
        )

    result = {
        "protocol": "three-path-held-out-support-conflict-and-retrieval-stress-test",
        "selection_endpoint": "training-only mean Macro-F1 with NDCG as tie-break",
        "test_labels_used_for_selection": False,
        "k": args.k,
        "beta_grid": beta_grid.tolist(),
        "fusion_weight_grid": f"simplex step {args.fusion_step}",
        "validation_fraction": args.validation_fraction,
        "split_seeds": split_seeds,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_note": (
            "Stratified paired node-bootstrap intervals condition on the observed graph "
            "and do not model dependence induced by shared edges."
        ),
        "rows": rows,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
