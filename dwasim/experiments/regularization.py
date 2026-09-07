"""Frozen component-screening and path-shrinkage experiment on ACM/DBLP.

No test labels enter fit_models. Outer results are descriptive overlapping
training-label partitions. Historical test exposure is explicitly acknowledged.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "2")
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
from dwasim.selection import simplex_weights
from dwasim.selection import INNER_SEEDS
from dwasim.fusion import component_views, weighted_view
from dwasim.data import subset_components
from dwasim.data import STRESS_PATHS, load_components
from dwasim.evaluation import evaluate_affinity, paired_bootstrap

TOLERANCE = .005
ALPHAS = (0., .25, .5, .75, 1.)
MODES = ("Original", "Uniform", "Shrinkage", "Screened", "ScreenedUniform",
         "ScreenedShrinkage", "Cosine", "SingleSelector", "CosineShrinkage", "SingleSelectorShrinkage")


def score_candidates(views, candidates, ids, labels, splits):
    rows = []
    for weights in candidates:
        affinity = weighted_view(views, weights)
        scores = [evaluate_affinity(affinity[np.ix_(q, r)], ids[r], labels[r], labels[q], 10)[0]
                  for r, q in splits]
        rows.append({"weights": list(weights),
                     "macro_f1_mean": float(np.mean([s["macro_f1"] for s in scores])),
                     "ndcg_at_10_mean": float(np.mean([s["ndcg_at_10"] for s in scores])),
                     "split_scores": scores})
    return rows


from dwasim.selection import best_row


def screen_row(rows, tolerance=TOLERANCE):
    if tolerance < 0 or not rows:
        raise ValueError("Nonempty candidates and nonnegative tolerance required")
    peak = max(r["macro_f1_mean"] for r in rows)
    eligible = [r for r in rows if peak - r["macro_f1_mean"] <= tolerance + 1e-12]
    return max(eligible, key=lambda r: (-np.count_nonzero(r["weights"]), r["macro_f1_mean"],
                                       r["ndcg_at_10_mean"], r["weights"][1]))


def shrink_weights(weights, alpha):
    weights = np.asarray(weights, dtype=float)
    if not 0 <= alpha <= 1 or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Weights must form a simplex and alpha must be in [0, 1]")
    return ((1-alpha)*weights + alpha / len(weights)).tolist()


def shrink_row(rows, tolerance=TOLERANCE):
    if tolerance < 0 or not rows:
        raise ValueError("Nonempty candidates and nonnegative tolerance required")
    peak = max(r["macro_f1_mean"] for r in rows)
    return max((r for r in rows if peak-r["macro_f1_mean"] <= tolerance+1e-12),
               key=lambda r: r["alpha"])


def fit_models(components, ids, labels, splits):
    """Fit using train component matrices and training labels ONLY.

    Test matrices, IDs, labels and outcomes are intentionally absent from the
    fitting interface; train-only dictionaries suffice, as asserted in tests.
    """
    start = time.perf_counter()
    grid = simplex_weights(3, .25)
    grids = [score_candidates(component_views(c), grid, ids, labels, splits) for c in components]
    singles = [best_row([r for r in rows if np.count_nonzero(r["weights"]) == 1], True) for rows in grids]
    original = [best_row(rows, True) for rows in grids]
    screened = [screen_row(rows) for rows in grids]
    choices = {"Original": [r["weights"] for r in original],
               "Screened": [r["weights"] for r in screened],
               "Cosine": [[0., 0., 1.] for _ in components],
               "SingleSelector": [r["weights"] for r in singles]}
    models, fusion_audit = {}, {}
    for family, theta in choices.items():
        started = time.perf_counter()
        views = [weighted_view(component_views(c), w) for c, w in zip(components, theta)]
        fusion = score_candidates(views, simplex_weights(len(views), .25), ids, labels, splits)
        chosen = best_row(fusion)
        # alpha=0 reuses the selected fusion row; the other four are new evaluations.
        shrink = [dict(chosen, alpha=0.)]
        extra = score_candidates(views, [shrink_weights(chosen["weights"], a) for a in ALPHAS[1:]], ids, labels, splits)
        shrink.extend(dict(row, alpha=a) for row, a in zip(extra, ALPHAS[1:]))
        regularized = shrink_row(shrink)
        component_budget = 45 if family in ("Original", "Screened") else 9 if family == "SingleSelector" else 0

        def model(row, alpha, budget):
            gamma = np.asarray(theta)*np.asarray(row["weights"])[:, None]
            return {"component_weights": theta, "path_weights": row["weights"], "alpha": alpha,
                    "active_terms": int(np.count_nonzero(gamma)),
                    "active_paths": int(np.count_nonzero(row["weights"])),
                    "validation_macro_f1": row["macro_f1_mean"],
                    "candidate_evaluations": budget}

        models[family] = model(chosen, 0., component_budget+15)
        target = "Shrinkage" if family == "Original" else family+"Shrinkage"
        models[target] = model(regularized, regularized["alpha"], component_budget+19)
        if family in ("Original", "Screened"):
            models["Uniform" if family == "Original" else "ScreenedUniform"] = model(shrink[-1], 1., component_budget+1)
        fusion_audit[family] = {"learned_grid": fusion, "shrinkage_grid": shrink,
                                "fusion_and_shrinkage_seconds": time.perf_counter()-started}
    diagnostics = []
    for full, screened_row, single in zip(original, screened, singles):
        diagnostics.append({"best_mixture_minus_single": [a["macro_f1"]-b["macro_f1"]
                            for a, b in zip(full["split_scores"], single["split_scores"])],
                            "screened_minus_single": [a["macro_f1"]-b["macro_f1"]
                            for a, b in zip(screened_row["split_scores"], single["split_scores"])],
                            "best_weights": full["weights"], "screened_weights": screened_row["weights"],
                            "single_weights": single["weights"]})
    return {"models": models, "component_grids": grids, "component_diagnostics": diagnostics,
            "fusion_audit": fusion_audit, "selection_seconds": time.perf_counter()-start}


def affinity_for(components, model, prefix="test"):
    # Preserve the original two-stage stored-precision arithmetic.
    views = [weighted_view(component_views(c, prefix), w) for c, w in zip(components, model["component_weights"])]
    return sum(w*v for w, v in zip(model["path_weights"], views))


def save_json(path, value):
    stage = path.with_suffix(".partial.json")
    stage.write_text(json.dumps(value, indent=2), encoding="utf-8")
    stage.replace(path)


def run(dataset, output_root):
    start = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / f"regularized_selection_{dataset}.json"
    if target.exists():
        raise FileExistsError(f"Preserving completed results: {target}; use a new output directory")
    ids, labels, test_ids, truth = load_split(dataset)
    components = [load_components(dataset, p) for p in STRESS_PATHS[dataset]]
    rows = []
    for seed, (r, q) in zip(range(20260910, 20260915), split_positions(labels, range(20260910, 20260915), .2)):
        subset = subset_components(components, r, q)
        train_only = [{k: v for k, v in c.items() if k.startswith("train_")} for c in subset]
        fitted = fit_models(train_only, ids[r], labels[r], split_positions(labels[r], INNER_SEEDS, .2))
        metrics = {name: evaluate_affinity(affinity_for(subset, fitted["models"][name]), ids[r], labels[r], labels[q], 10)[0]
                   for name in MODES}
        rows.append({"seed": seed, "reference_ids": ids[r].tolist(), "validation_ids": ids[q].tolist(),
                     "fit": fitted, "metrics": metrics})
        save_json(output_root / f"regularized_selection_{dataset}.checkpoint.json", {"outer": rows})
        print(dataset, "outer", seed, {m: round(s["macro_f1"], 4) for m, s in metrics.items()}, flush=True)
    train_only = [{k: v for k, v in c.items() if k.startswith("train_")} for c in components]
    fitted = fit_models(train_only, ids, labels, split_positions(labels, INNER_SEEDS, .2))
    # Only now evaluate official test outcomes; there is no data-dependent rerun.
    metrics, predictions, ndcgs = {}, {}, {}
    for mode in MODES:
        metrics[mode], predictions[mode], ndcgs[mode] = evaluate_affinity(
            affinity_for(components, fitted["models"][mode]), ids, labels, truth, 10)
        print(dataset, "test", mode, metrics[mode], "alpha", fitted["models"][mode]["alpha"], flush=True)
    previous = RESULTS_ROOT / f"path_fusion_audit_{dataset}.json"
    if previous.exists():
        reference = json.loads(previous.read_text(encoding="utf-8"))
        for name, old in (("Original", "LearnedPathFusion"), ("Uniform", "UniformPathFusion")):
            for metric in ("macro_f1", "ndcg_at_10"):
                if not np.isclose(metrics[name][metric], reference["test_metrics"][old][metric], rtol=0, atol=1e-12):
                    raise AssertionError(f"Original reproduction failed: {dataset}/{name}/{metric}")
    paired = {m: paired_bootstrap(truth, predictions["ScreenedShrinkage"], predictions[m],
              ndcgs["ScreenedShrinkage"], ndcgs[m], 2000, 20260930+i)
              for i, m in enumerate(("Original", "Uniform", "Cosine", "SingleSelector"))}
    outer_summary = {m: {metric: {"mean": float(np.mean([r["metrics"][m][metric] for r in rows])),
                      "sample_sd": float(np.std([r["metrics"][m][metric] for r in rows], ddof=1))}
                      for metric in ("macro_f1", "ndcg_at_10")} for m in MODES}
    prediction_path = output_root / f"regularized_selection_{dataset}.predictions.npz"
    np.savez_compressed(prediction_path, test_ids=test_ids, truth=truth,
                       **{f"prediction_{m}": p for m, p in predictions.items()},
                       **{f"ndcg_{m}": p for m, p in ndcgs.items()})
    result = {"dataset": dataset, "paths": list(STRESS_PATHS[dataset]), "fit": fitted,
              "outer": rows, "outer_summary": outer_summary, "test_metrics": metrics,
              "paired_combined_minus": paired, "elapsed_seconds": time.perf_counter()-start,
              "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "protocol": {"tolerance": TOLERANCE, "alphas": ALPHAS, "inner_seeds": INNER_SEEDS,
                           "outer_seeds": list(range(20260910, 20260915)), "k": 10,
                           "component_step": .25, "path_step": .25, "bootstrap_iterations": 2000,
                           "test_labels_used_in_fitting": False, "outer_partitions_overlap": True,
                           "previously_inspected_test_sets": True,
                           "interval_scope": "Unadjusted paired query intervals conditional on fitted models; graph dependence excluded"}}
    save_json(target, result)
    print("Completed", dataset, "seconds", result["elapsed_seconds"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ACM", "DBLP", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        for dataset in (("ACM", "DBLP") if args.dataset == "all" else (args.dataset,)):
            run(dataset, args.output_dir)
