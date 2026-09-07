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

from dwasim.paths import CACHE_ROOT, RESULTS_ROOT, PROCESSED_ROOT
from dwasim.data import PATHS, full_profile, multiply_chain
from dwasim.similarity import cache_identity, use_cache, pair_discrepancies
from dwasim.evaluation import deterministic_topk, majority_vote
from dwasim.data import load_split
from dwasim.selection import split_positions
from dwasim.selection import simplex_weights
from dwasim.baselines import symmetric_path_affinities


from dwasim.data import STRESS_PATHS

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


from dwasim.data import _component_cache


from dwasim.data import load_components


from dwasim.fusion import relative_affinity


from dwasim.fusion import adaptive_affinity


from dwasim.evaluation import ndcg_at_k_from_order


from dwasim.evaluation import evaluate_affinity


from dwasim.selection import select_path_beta


from dwasim.selection import select_path_mixture


from dwasim.selection import select_fusion


from dwasim.evaluation import paired_bootstrap


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
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RESULTS_ROOT / f"three_component_predictions_{dataset}.npz",
        test_ids=test_ids, truth=test_labels,
        **{f"prediction_{name}": value for name, value in predictions.items()},
        **{f"ndcg_{name}": value for name, value in per_query_ndcg.items()},
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
