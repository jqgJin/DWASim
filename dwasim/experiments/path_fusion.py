"""Separate path availability from path-weight selection under fixed training splits."""
from __future__ import annotations
import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "2")
import argparse
import json
import time
import numpy as np
from dwasim.paths import RESULTS_ROOT
from dwasim.data import load_split
from dwasim.selection import split_positions
from dwasim.selection import simplex_weights
from dwasim.selection import INNER_SEEDS
from dwasim.data import subset_components
from dwasim.fusion import weighted_view, component_views
from dwasim.data import STRESS_PATHS, load_components
from dwasim.selection import select_path_mixture, select_fusion
from dwasim.evaluation import evaluate_affinity, paired_bootstrap

MODES = ("BestSinglePath", "UniformPathFusion", "LearnedPathFusion")


def fit_path_controls(components, ids, labels, splits, paths):
    """One shared component fit; only the subsequent path rule differs."""
    selections = [select_path_mixture(c, ids, labels, splits, simplex_weights(3, .25), 10)
                  for c in components]
    theta = [row["selected"]["weights_support_magnitude_direction"] for row in selections]
    views = [weighted_view(component_views(c), w) for c, w in zip(components, theta)]
    grids = {"BestSinglePath": [tuple(row) for row in np.eye(len(paths))],
             "UniformPathFusion": [tuple([1 / len(paths)] * len(paths))],
             "LearnedPathFusion": simplex_weights(len(paths), .25)}
    fitted = {}
    for mode in MODES:
        selection = select_fusion(views, ids, labels, splits, grids[mode], 10)
        weights = selection["selected"]["weights"]
        fitted[mode] = {"component_weights": theta, "path_weights": weights,
                        "path_selection": selection,
                        "candidate_evaluations": 15 * len(paths) + len(grids[mode]),
                        "selected_path": paths[int(np.argmax(weights))] if mode == "BestSinglePath" else None}
    return fitted


def affinity_for(components, fitted, prefix="test"):
    views = [weighted_view(component_views(c, prefix), theta)
             for c, theta in zip(components, fitted["component_weights"])]
    return sum(w * view for w, view in zip(fitted["path_weights"], views))


def run(dataset, outer_repeats=5, bootstrap_iterations=2000):
    started = time.perf_counter()
    ids, labels, test_ids, truth = load_split(dataset)
    paths = STRESS_PATHS[dataset]
    components = [load_components(dataset, path) for path in paths]
    splits = split_positions(labels, INNER_SEEDS, .2)
    fitted = fit_path_controls(components, ids, labels, splits, paths)
    metrics, predictions, ndcg = {}, {}, {}
    for mode, model in fitted.items():
        metrics[mode], predictions[mode], ndcg[mode] = evaluate_affinity(
            affinity_for(components, model), ids, labels, truth, 10)
        print(dataset, mode, metrics[mode], flush=True)
    outer = []
    for i, (reference, validation) in enumerate(split_positions(labels, range(20260910, 20260910 + outer_repeats), .2)):
        sub = subset_components(components, reference, validation)
        inner = split_positions(labels[reference], INNER_SEEDS, .2)
        models = fit_path_controls(sub, ids[reference], labels[reference], inner, paths)
        record = {"seed": 20260910 + i, "reference_ids": ids[reference].tolist(),
                  "validation_ids": ids[validation].tolist(), "methods": {}}
        for mode, model in models.items():
            score = evaluate_affinity(affinity_for(sub, model), ids[reference], labels[reference], labels[validation], 10)[0]
            record["methods"][mode] = {"selection": model, "metrics": score}
        outer.append(record)
        print(dataset, "outer", i + 1, {m: r["metrics"]["macro_f1"] for m, r in record["methods"].items()}, flush=True)
    paired = {}
    for i, comparator in enumerate(MODES[:2]):
        paired[comparator] = paired_bootstrap(truth, predictions["LearnedPathFusion"], predictions[comparator],
            ndcg["LearnedPathFusion"], ndcg[comparator], bootstrap_iterations, 20260906 + i)
    summary = {}
    for mode in MODES:
        summary[mode] = {}
        for metric in ("macro_f1", "ndcg_at_10"):
            values = np.array([row["methods"][mode]["metrics"][metric] for row in outer])
            summary[mode][metric] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))} if len(values) else None
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS_ROOT / f"path_fusion_audit_{dataset}.predictions.npz", test_ids=test_ids, truth=truth,
        **{f"prediction_{m}": x for m, x in predictions.items()}, **{f"ndcg_{m}": x for m, x in ndcg.items()})
    result = {"dataset": dataset, "paths": list(paths), "selection": fitted, "test_metrics": metrics,
              "paired_learned_minus": paired, "outer_partitions": outer, "outer_summary": summary,
              "protocol": {"k": 10, "inner_seeds": INNER_SEEDS, "component_step": .25, "fusion_step": .25,
                  "test_labels_used_for_selection": False, "bootstrap_iterations": bootstrap_iterations,
                  "note": "Revision-stage diagnostic. Uniform fusion's one candidate is evaluated for documentation, not chosen from alternatives. Outer partitions overlap."},
              "elapsed_seconds": time.perf_counter() - started}
    target = RESULTS_ROOT / f"path_fusion_audit_{dataset}.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved", target, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ACM", "DBLP", "all"), default="all")
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()
    for ds in ("ACM", "DBLP") if args.dataset == "all" else (args.dataset,):
        run(ds, args.outer_repeats, args.bootstrap_iterations)
