"""Frozen minimal-change ACM/DBLP extension; retains every specified outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from dwasim.paths import RESULTS_ROOT
from dwasim.data import load_split
from dwasim.selection import split_positions
from dwasim.selection import INNER_SEEDS
from dwasim.fusion import component_views
from dwasim.data import STRESS_PATHS, load_components
from dwasim.evaluation import evaluate_affinity, paired_bootstrap
from dwasim.kernels import ETAS, FAMILIES, target_statistics, solve_target, score, hierarchical_weights

OUTER_SEEDS = tuple(range(20260910, 20260915))


def fit(kernels, ids, labels):
    """Neither query matrices nor query labels are accepted by this interface."""
    started = time.perf_counter()
    candidates = {family: {eta: [] for eta in ETAS} for family in FAMILIES}
    for reference, validation in split_positions(labels, INNER_SEEDS, 0.2):
        stats = target_statistics(
            [kernel[np.ix_(reference, reference)] for kernel in kernels],
            labels[reference],
        )
        queries = [k[np.ix_(validation, reference)] for k in kernels]
        for family, allowed in FAMILIES.items():
            for eta in ETAS:
                model = solve_target(stats, eta, allowed)
                metrics = evaluate_affinity(
                    score(queries, model["gamma"]), ids[reference],
                    labels[reference], labels[validation], 10,
                )[0]
                candidates[family][eta].append(metrics)
    stats = target_statistics(kernels, labels)
    models, grids = {}, {}
    for family, allowed in FAMILIES.items():
        rows = []
        for eta, records in candidates[family].items():
            rows.append({
                "eta": eta,
                "macro_f1": float(np.mean([fold["macro_f1"] for fold in records])),
                "ndcg_at_10": float(np.mean([fold["ndcg_at_10"] for fold in records])),
                "folds": records,
            })
        selected = max(
            rows, key=lambda row: (row["macro_f1"], row["ndcg_at_10"], row["eta"])
        )
        model = solve_target(stats, selected["eta"], allowed)
        model.update(hierarchical_weights(model["gamma"]))
        model["validation_macro_f1"] = selected["macro_f1"]
        models[family], grids[family] = model, rows
    return {
        "models": models,
        "grids": grids,
        "seconds": time.perf_counter() - started,
        "centered_norms": stats["scales"].tolist(),
        "centered_kernel_gram": stats["gram"].tolist(),
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def run(dataset, output):
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"kernel_target_{dataset}.json"
    checkpoint = output / f"kernel_target_{dataset}.checkpoint.json"
    if target.exists() or checkpoint.exists():
        raise FileExistsError("Use a new output directory; completed and partial outcomes are preserved")
    started = time.perf_counter()
    ids, labels, test_ids, truth = load_split(dataset)
    components = [load_components(dataset, p) for p in STRESS_PATHS[dataset]]
    kernels = [k for c in components for k in component_views(c)]
    test_kernels = [k for c in components for k in component_views(c, "test")]
    del components
    outer = []
    for seed, (reference, validation) in zip(OUTER_SEEDS, split_positions(labels, OUTER_SEEDS, .2)):
        fitted = fit([k[np.ix_(reference, reference)] for k in kernels], ids[reference], labels[reference])
        queries = [k[np.ix_(validation, reference)] for k in kernels]
        metrics = {}
        for family, model in fitted["models"].items():
            metrics[family] = evaluate_affinity(
                score(queries, model["gamma"]), ids[reference],
                labels[reference], labels[validation], 10,
            )[0]
        outer.append({"seed": seed, "reference_ids": ids[reference].tolist(),
                      "validation_ids": ids[validation].tolist(), "fit": fitted, "metrics": metrics})
        write_json(checkpoint, {"dataset": dataset, "outer": outer})
        print(dataset, "outer", seed, {m: round(v["macro_f1"], 4) for m, v in metrics.items()}, flush=True)
    fitted = fit(kernels, ids, labels)
    write_json(output / f"kernel_target_{dataset}.frozen_fit.json", fitted)
    metrics, predictions, ndcgs = {}, {}, {}
    for family, model in fitted["models"].items():
        metrics[family], predictions[family], ndcgs[family] = evaluate_affinity(
            score(test_kernels, model["gamma"]), ids, labels, truth, 10)
        print(dataset, "official", family, metrics[family], "eta", model["eta"], flush=True)
    paired = {family: paired_bootstrap(truth, predictions["KernelTarget"], predictions[family],
             ndcgs["KernelTarget"], ndcgs[family], 2000, 20261001+i)
             for i, family in enumerate(FAMILIES) if family != "KernelTarget"}
    legacy = RESULTS_ROOT / f"regularized_selection_{dataset}.predictions.npz"
    if legacy.exists():
        with np.load(legacy) as old:
            np.testing.assert_array_equal(old["test_ids"], test_ids)
            np.testing.assert_array_equal(old["truth"], truth)
            for i, mode in enumerate(("Original", "ScreenedShrinkage", "CosineShrinkage")):
                paired[mode] = paired_bootstrap(truth, predictions["KernelTarget"], old[f"prediction_{mode}"],
                    ndcgs["KernelTarget"], old[f"ndcg_{mode}"], 2000, 20261020+i)
    pred_path = output / f"kernel_target_{dataset}.predictions.npz"
    np.savez_compressed(pred_path, truth=truth, test_ids=test_ids,
                       **{f"prediction_{k}": v for k, v in predictions.items()},
                       **{f"ndcg_{k}": v for k, v in ndcgs.items()})
    summary = {}
    for family in FAMILIES:
        summary[family] = {}
        for metric in ("macro_f1", "ndcg_at_10"):
            values = [fold["metrics"][family][metric] for fold in outer]
            summary[family][metric] = {
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)),
            }
    result = {"dataset": dataset, "fit": fitted, "outer": outer, "outer_summary": summary,
              "test_metrics": metrics, "paired_full_minus": paired,
              "elapsed_seconds": time.perf_counter()-started,
              "prediction_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest(),
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               (Path(__file__), Path(__file__).resolve().parents[1] / "kernels.py")},
              "protocol": {"etas": ETAS, "families": FAMILIES, "inner_seeds": INNER_SEEDS,
                           "outer_seeds": OUTER_SEEDS, "k": 10, "bootstrap_iterations": 2000,
                           "validation_labels_in_inner_target": False, "test_labels_in_fit": False,
                           "outer_partitions_overlap": True, "previously_inspected_test_sets": True,
                           "inference_precision": "float64 accumulation of unchanged cached affinities",
                           "interval_scope": "unadjusted paired-query descriptive intervals, fixed graph and fitted model"}}
    write_json(target, result)
    print("Completed", dataset, "seconds", result["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ACM", "DBLP", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "kernel_target")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        for dataset in (("ACM", "DBLP") if args.dataset == "all" else (args.dataset,)):
            run(dataset, args.output_dir)
