"""Frozen revision controls for component necessity and selection stability.

All model selection uses official training labels. Test metrics are descriptive
revision-stage evaluations, not an untouched or preregistered holdout claim.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from dwasim.paths import RESULTS_ROOT
from dwasim.data import full_profile
from dwasim.data import load_split
from dwasim.selection import split_positions
from dwasim.selection import simplex_weights
from dwasim.diagnostics import affinity_statistics, nullable
from dwasim.data import STRESS_PATHS, load_components
from dwasim.selection import select_path_mixture, select_fusion
from dwasim.evaluation import evaluate_affinity, paired_bootstrap

FAMILIES = ("TriComponentDWASim", "Cosine", "OneComponentSelector", "FixedEqualMixture",
            "SharedComponentMixture", "JointBudget60", "NoSupport", "NoMagnitude", "NoDirection",
            "PTPNoSupport")
from dwasim.selection import INNER_SEEDS


from dwasim.fusion import component_views


from dwasim.fusion import weighted_view


from dwasim.selection import joint_candidates


from dwasim.selection import select_linear


from dwasim.selection import fit_family


from dwasim.fusion import predict_affinity


from dwasim.data import subset_components


def summarize_rows(rows, metric):
    values = np.array([row[metric] for row in rows])
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.}


def run_dataset(dataset, outer_repeats, bootstrap_iterations):
    started = time.perf_counter()
    ids, labels, test_ids, truth = load_split(dataset)
    paths = STRESS_PATHS[dataset]
    components = [load_components(dataset, path) for path in paths]
    splits = split_positions(labels, INNER_SEEDS, 0.2)
    families = [name for name in FAMILIES if name != "PTPNoSupport" or dataset == "ACM"]
    fitted, predictions, ndcg, metrics = {}, {}, {}, {}
    for family in families:
        model = fit_family(family, components, ids, labels, splits, paths)
        fitted[family] = model
        metric, pred, query_ndcg = evaluate_affinity(predict_affinity(components, model), ids, labels, truth, 10)
        metrics[family], predictions[family], ndcg[family] = metric, pred, query_ndcg
        print(dataset, family, json.dumps(metric), flush=True)

    singles, diagnostics = {}, {}
    for path, values in zip(paths, components):
        profiles = full_profile(dataset, path)
        upper = np.triu_indices(len(ids), k=1)
        scores = np.stack([v[upper] for v in component_views(values)], axis=1)
        variance, correlation = affinity_statistics(scores)
        diagnostics[path] = {"complete_profile_shape": list(profiles.shape),
            "nonzero_density": float(profiles.nnz / np.prod(profiles.shape)),
            "full_support_rows": int(np.sum(np.diff(profiles.indptr) == profiles.shape[1])),
            "component_variance": variance.tolist(), "component_correlation": nullable(correlation),
            "support_affinity_equal_one_fraction": float(np.mean(scores[:, 0] == 1)),
            "training_unordered_pairs": len(scores)}
        singles[path] = {}
        for name, affinity in (("Jaccard", 1-values["test_jaccard"]), ("MagnitudeOnly", 1-values["test_bray"]),
                              ("Cosine", values["test_cosine"]), ("Bhattacharyya", values["test_bhattacharyya"]),
                              ("PathSim", values["test_pathsim"]), ("HeteSim", values["test_hetesim"])):
            singles[path][name] = evaluate_affinity(affinity, ids, labels, truth, 10)[0]

    outer_rows = []
    outer_splits = split_positions(labels, list(range(20260910, 20260910 + outer_repeats)), 0.2)
    for index, (outer_train, outer_validation) in enumerate(outer_splits):
        inner = split_positions(labels[outer_train], INNER_SEEDS, 0.2)
        sub = subset_components(components, outer_train, outer_validation)
        record = {"repeat": index, "outer_reference_ids": ids[outer_train].tolist(),
                  "outer_validation_ids": ids[outer_validation].tolist(), "methods": {}}
        for family in ("TriComponentDWASim", "Cosine", "OneComponentSelector", "NoSupport"):
            model = fit_family(family, sub, ids[outer_train], labels[outer_train], inner, paths)
            metric = evaluate_affinity(predict_affinity(sub, model), ids[outer_train], labels[outer_train],
                                       labels[outer_validation], 10)[0]
            record["methods"][family] = {"selection": model, "metrics": metric}
        outer_rows.append(record)
        print(dataset, "outer", index + 1, {name: round(row["metrics"]["macro_f1"], 4)
               for name, row in record["methods"].items()}, flush=True)
    outer_summary = {}
    for family in ("TriComponentDWASim", "Cosine", "OneComponentSelector", "NoSupport"):
        method_rows = [row["methods"][family]["metrics"] for row in outer_rows]
        if method_rows:
            outer_summary[family] = {name: summarize_rows(method_rows, name) for name in ("macro_f1", "ndcg_at_10")}

    paired = {}
    for index, comparator in enumerate(name for name in families if name != "TriComponentDWASim"):
        paired[comparator] = paired_bootstrap(truth, predictions["TriComponentDWASim"], predictions[comparator],
            ndcg["TriComponentDWASim"], ndcg[comparator], bootstrap_iterations, 20260905 + index)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS_ROOT / f"revision_controls_{dataset}.predictions.npz", test_ids=test_ids, truth=truth,
        **{f"prediction_{key}": value for key, value in predictions.items()},
        **{f"ndcg_{key}": value for key, value in ndcg.items()})
    return {"dataset": dataset, "paths": list(paths), "selection": fitted, "test_metrics": metrics,
            "paired_final_minus_comparator": paired, "single_path_metrics": singles,
            "component_diagnostics": diagnostics, "outer_repeats": outer_rows, "outer_summary": outer_summary,
            "elapsed_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ACM", "DBLP", "all"), default="all")
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()
    datasets = ("ACM", "DBLP") if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        record = run_dataset(dataset, args.outer_repeats, args.bootstrap_iterations)
        record["protocol"] = {"inner_seeds": INNER_SEEDS, "outer_seed_start": 20260910,
            "outer_repeats": args.outer_repeats, "bootstrap_iterations": args.bootstrap_iterations,
            "joint_budget": 60, "joint_seed": 20260905, "primary_comparators": ["Cosine", "OneComponentSelector"],
            "inference": "Descriptive fixed-graph paired intervals; overlapping outer partitions are not independent graphs."}
        output = RESULTS_ROOT / f"revision_controls_{dataset}.json"
        output.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print("Wrote", output, flush=True)


if __name__ == "__main__":
    main()
