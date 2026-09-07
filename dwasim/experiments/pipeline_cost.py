"""One complete fresh-derived-computation run per dataset/method, plus exact pruning audit.

Run each worker serially. The OS file cache is not flushed; imports, evaluation
metrics and output serialization are excluded. No performance claims are inferred
from a single timing observation. No production caches are deleted or overwritten.
"""
from __future__ import annotations
import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "2"
import argparse
import gc
import json
import platform
import subprocess
import sys
import time
import numpy as np
import psutil
from threadpoolctl import threadpool_limits, threadpool_info
from dwasim.data import PATHS, multiply_chain
from dwasim.paths import PROCESSED_ROOT, RESULTS_ROOT
from dwasim.similarity import pair_discrepancies, mixture_affinity, active_mixture_affinity
from dwasim.evaluation import deterministic_topk, majority_vote
from dwasim.selection import split_positions
from dwasim.selection import INNER_SEEDS, fit_family
from dwasim.selection import select_fusion
from dwasim.evaluation import evaluate_affinity
import dwasim.experiments.multilabel as imdb


def load_ids(dataset):
    with np.load(PROCESSED_ROOT / dataset / "labels.npz") as record:
        return record["train_ids"].astype(np.int64), record["train_labels"].copy(), record["test_ids"].astype(np.int64)


def components_for(profiles, query_ids, reference_ids, required):
    answer = {}
    for path, profile in profiles.items():
        values = pair_discrepancies(profile[query_ids], profile[reference_ids], include_bhattacharyya=False,
                                   required_components=required)
        answer[path] = {name: value.astype(np.float32) for name, value in values.items()
                        if name in ("jaccard", "bray", "cosine")}
    return answer


def select_model(dataset, method, components, ids, labels, paths):
    if dataset == "IMDB":
        folds = imdb.iterative_multilabel_folds(labels, 4, 20260805)
        if method == "DWASim":
            theta, _ = imdb.select_three_components(components, ids, labels, folds)
        else:
            theta = {path: [0., 0., 1.] for path in paths}
        views = [mixture_affinity(components[path], theta[path]) if method == "DWASim" else components[path]["cosine"] for path in paths]
        selection = imdb.select_pipeline(views, ids, labels, folds)
        return {"component_weights": [theta[path] for path in paths],
                "path_weights": selection["selected"]["path_weights"],
                "task": selection["selected"], "candidate_evaluations": (45 if method == "DWASim" else 0) + selection["configuration_count"]}
    splits = split_positions(labels, INNER_SEEDS, .2)
    if method == "DWASim":
        packed = [{"train_" + k: v for k, v in components[path].items()} for path in paths]
        fit = fit_family("TriComponentDWASim", packed, ids, labels, splits, paths)
        return {"component_weights": fit["component_weights"], "path_weights": fit["path_weights"],
                "candidate_evaluations": fit["candidate_count"], "task": {"k": 10}}
    fit = select_fusion([components[p]["cosine"] for p in paths], ids, labels, splits, imdb.simplex_weights(3), 10)
    return {"component_weights": [[0., 0., 1.]] * len(paths), "path_weights": fit["selected"]["weights"],
            "candidate_evaluations": 15, "task": {"k": 10}}


def construct_views(components, paths, fit, method):
    return [mixture_affinity(components[path], theta) if method == "DWASim" else components[path]["cosine"]
            for path, theta in zip(paths, fit["component_weights"])]


def combine_predict(views, ids, labels, fit, dataset):
    if dataset == "IMDB":
        views = [imdb.calibrate_rows(view) for view in views]
    fused = sum(weight * view for weight, view in zip(fit["path_weights"], views))
    order = deterministic_topk(fused, ids, fit["task"]["k"], largest=True)
    if dataset == "IMDB":
        c = fit["task"]
        prediction = imdb.multilabel_prediction(fused, order, labels, c["gamma"], c["prior_power"], c["threshold"])
    else:
        prediction = majority_vote(order, labels)
    return fused, order, prediction


def worker(dataset, method):
    process = psutil.Process()
    with threadpool_limits(limits=2):
        started = time.perf_counter()
        ids, labels, test_ids = load_ids(dataset)
        paths = list(PATHS[dataset])
        profiles = {path: multiply_chain(dataset, PATHS[dataset][path]["full"]).astype(np.float64)
                    for path in paths}
        profile_seconds = time.perf_counter() - started
        required = ("jaccard", "bray", "cosine") if method == "DWASim" else ("cosine",)
        mark = time.perf_counter()
        training = components_for(profiles, ids, ids, required)
        training_seconds = time.perf_counter() - mark
        print(dataset, method, "profiles/train components", round(profile_seconds, 3), round(training_seconds, 3), flush=True)
        mark = time.perf_counter()
        fit = select_model(dataset, method, training, ids, labels, paths)
        selection_seconds = time.perf_counter() - mark
        del training
        gc.collect()
        print(dataset, method, "selection", round(selection_seconds, 3), fit, flush=True)
        mark = time.perf_counter()
        testing = components_for(profiles, test_ids, ids, required)
        query_seconds = time.perf_counter() - mark
        mark = time.perf_counter()
        views = construct_views(testing, paths, fit, method)
        fused, order, prediction = combine_predict(views, ids, labels, fit, dataset)
        predict_seconds = time.perf_counter() - mark
        memory = process.memory_info()
        peak = getattr(memory, "peak_wset", memory.rss) / 2**20
        full_elapsed = time.perf_counter() - started
        # Save outputs in memory for a correctness check; this extra audit is not
        # included in the full pipeline cost or its recorded peak memory.
        del testing, views
        gc.collect()
        mark = time.perf_counter()
        # The cosine baseline deliberately stores and fuses float32 arrays.
        # Do not route it through the float64-weight mixture arithmetic: that
        # would change rounding, even though the mathematical weight is one.
        active_views = [(active_mixture_affinity(profiles[path][test_ids], profiles[path][ids], theta)
                         if method == "DWASim" else pair_discrepancies(
                             profiles[path][test_ids], profiles[path][ids],
                             include_bhattacharyya=False, required_components=("cosine",))["cosine"].astype(np.float32))
                        if weight != 0 else np.zeros((len(test_ids), len(ids)), dtype=np.float32)
                        for path, theta, weight in zip(paths, fit["component_weights"], fit["path_weights"])]
        active_query_seconds = time.perf_counter() - mark
        mark = time.perf_counter()
        active_fused, active_order, active_prediction = combine_predict(active_views, ids, labels, fit, dataset)
        active_predict_seconds = time.perf_counter() - mark
        exact_scores = bool(np.array_equal(fused, active_fused))
        exact_order = bool(np.array_equal(order, active_order))
        exact_predictions = bool(np.array_equal(prediction, active_prediction))
        if not (exact_scores and exact_order and exact_predictions):
            raise AssertionError("Zero-coefficient pruning changed scores, rankings or predictions")
        with np.load(PROCESSED_ROOT / dataset / "labels.npz") as source:
            truth = source["test_labels"]
        if dataset == "IMDB":
            metrics = imdb.metric_record(truth, prediction)
            historical = json.loads((RESULTS_ROOT / "imdb_external_validation.json").read_text(encoding="utf-8"))
            expected = historical["external_metrics"]["TriComponentDWASim" if method == "DWASim" else "Cosine"]
        else:
            metrics = evaluate_affinity(fused, ids, labels, truth, 10)[0]
            historical = json.loads((RESULTS_ROOT / "support_retrieval_stress.json").read_text(encoding="utf-8"))
            expected = next(row for row in historical["rows"] if row["dataset"] == dataset)["test_metrics"]["TriComponentDWASim" if method == "DWASim" else "Cosine"]
        if not np.isclose(metrics["macro_f1"], expected["macro_f1"], atol=1e-12, rtol=0):
            raise AssertionError((dataset, method, "fresh computation changed published Macro-F1", metrics, expected))
        active_terms = sum(sum(value != 0 for value in theta) for theta, w in zip(fit["component_weights"], fit["path_weights"]) if w != 0)
        result = {"dataset": dataset, "method": method, "paths": paths, "reference_count": len(ids), "query_count": len(test_ids),
                  "selection": fit, "active_path_count": sum(w != 0 for w in fit["path_weights"]), "active_term_count": active_terms,
                  "profile_seconds": profile_seconds, "training_components_seconds": training_seconds,
                  "selection_seconds": selection_seconds, "query_components_seconds": query_seconds,
                  "ranking_prediction_seconds": predict_seconds, "pipeline_wall_seconds": full_elapsed,
                  "stage_sum_seconds": profile_seconds + training_seconds + selection_seconds + query_seconds + predict_seconds,
                  "pipeline_peak_MiB": peak, "active_query_seconds": active_query_seconds,
                  "active_ranking_prediction_seconds": active_predict_seconds,
                  "pruning_checks": {"scores_identical": exact_scores, "rankings_identical": exact_order, "predictions_identical": exact_predictions},
                  "test_metrics": metrics, "environment": {"python": sys.version, "platform": platform.platform(),
                      "processor": platform.processor(), "threads": 2, "threadpools": threadpool_info()},
                  "timing_scope": "One serial full run per dataset/method. Derived profile/component caches not read. Prepared relation files and OS file caching permitted. Imports, metrics, bootstrap, output serialization and post-run pruning audit excluded. Peak is whole-process high-water mark before pruning audit."}
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        target = RESULTS_ROOT / f"pipeline_cost_{dataset}_{method}.json"
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("Saved", target, "seconds", round(full_elapsed, 3), "exact pruning", exact_predictions, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ACM", "DBLP", "IMDB", "all"), default="all")
    parser.add_argument("--method", choices=("DWASim", "Cosine", "all"), default="all")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.dataset, args.method)
    else:
        for ds in ("ACM", "DBLP", "IMDB") if args.dataset == "all" else (args.dataset,):
            for method in ("DWASim", "Cosine") if args.method == "all" else (args.method,):
                subprocess.run([sys.executable, "-m", "dwasim.experiments.pipeline_cost", "--worker", "--dataset", ds, "--method", method], check=True)
