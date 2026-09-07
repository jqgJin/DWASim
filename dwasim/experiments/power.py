"""Fixed path-wise integer-power calibration with matched component controls."""
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
from dwasim.fusion import component_views, weighted_view
from dwasim.selection import best_row
from dwasim.selection import simplex_weights
from dwasim.data import STRESS_PATHS, load_components
from dwasim.evaluation import evaluate_affinity, paired_bootstrap

from dwasim.fusion import POWERS
FAMILIES = (
    "PowerDWASim", "PowerMagnitude", "PowerCosine",
    "PowerNoSupport", "PowerNoMagnitude", "PowerNoDirection",
)
OUTER_SEEDS = tuple(range(20260910, 20260915))


from dwasim.fusion import power_affinity


def within_weights(grids, family):
    if family in ("PowerMagnitude", "PowerCosine"):
        weights = [0.0, 1.0, 0.0] if family == "PowerMagnitude" else [0.0, 0.0, 1.0]
        return [weights.copy() for _ in grids]
    omitted = {
        "PowerNoSupport": 0,
        "PowerNoMagnitude": 1,
        "PowerNoDirection": 2,
    }.get(family)
    selected = []
    for rows in grids:
        eligible = [
            row for row in rows
            if omitted is None or row["weights"][omitted] == 0
        ]
        selected.append(best_row(eligible, component=True)["weights"])
    return selected


def candidate_rank(row):
    """Keep the original metric, power, and sparsity tie-breaking order."""
    return (
        row["macro_f1"],
        row["ndcg_at_10"],
        -row["power"],
        -np.count_nonzero(row["weights"]),
    )


def fit(components, grids, ids, labels):
    started = time.perf_counter()
    splits = split_positions(labels, INNER_SEEDS, 0.2)
    models = {}
    candidates = {}
    for family in FAMILIES:
        theta = within_weights(grids, family)
        views = [
            weighted_view(component_views(path), weights)
            for path, weights in zip(components, theta)
        ]
        rows = []
        for power in POWERS:
            transformed = [v if power == 1 else v**power for v in views]
            for weights in simplex_weights(3, 0.25):
                affinity = weighted_view(transformed, weights)
                folds = []
                for reference, validation in splits:
                    metrics = evaluate_affinity(
                        affinity[np.ix_(validation, reference)],
                        ids[reference], labels[reference], labels[validation], 10,
                    )[0]
                    folds.append(metrics)
                rows.append({
                    "power": power,
                    "weights": list(weights),
                    "folds": folds,
                    "macro_f1": float(np.mean([fold["macro_f1"] for fold in folds])),
                    "ndcg_at_10": float(np.mean([fold["ndcg_at_10"] for fold in folds])),
                })
        selected = max(rows, key=candidate_rank)
        uncalibrated = max(
            [row for row in rows if row["power"] == 1], key=candidate_rank
        )
        for name, candidate in ((family, selected), (family + "P1", uncalibrated)):
            models[name] = {
                "component_weights": theta,
                "path_weights": candidate["weights"],
                "power": candidate["power"],
                "validation_macro_f1": candidate["macro_f1"],
            }
        candidates[family] = rows
    return {
        "models": models,
        "grids": candidates,
        "cached_component_selection": True,
        "fusion_seconds": time.perf_counter() - started,
    }


def affinity(components, model, prefix="test"):
    views = [
        weighted_view(component_views(path, prefix), weights)
        for path, weights in zip(components, model["component_weights"])
    ]
    return power_affinity(views, model["path_weights"], model["power"])


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")


def run(dataset, output):
    output.mkdir(parents=True,exist_ok=True)
    target = output/f"kernel_power_{dataset}.json"
    checkpoint = output/f"kernel_power_{dataset}.checkpoint.json"
    if target.exists() or checkpoint.exists():
        raise FileExistsError("Existing outcomes are protected; choose a new output folder")
    start = time.perf_counter()
    oldpath = RESULTS_ROOT/f"regularized_selection_{dataset}.json"
    old = json.loads(oldpath.read_text())
    if old["protocol"]["inner_seeds"] != INNER_SEEDS or old["protocol"]["outer_seeds"] != list(OUTER_SEEDS):
        raise AssertionError("Cached component-selection partitions differ")
    ids, labels, test_ids, truth = load_split(dataset)
    components = [load_components(dataset,p) for p in STRESS_PATHS[dataset]]
    outer = []
    for index, (reference, validation) in enumerate(
        split_positions(labels, OUTER_SEEDS, 0.2)
    ):
        cached = old["outer"][index]
        np.testing.assert_array_equal(ids[reference], cached["reference_ids"])
        np.testing.assert_array_equal(ids[validation], cached["validation_ids"])
        subset = []
        for path in components:
            split_path = {}
            for prefix, positions in (("train", reference), ("test", validation)):
                for name in ("jaccard", "bray", "cosine"):
                    split_path[f"{prefix}_{name}"] = path[f"train_{name}"][
                        np.ix_(positions, reference)
                    ]
            subset.append(split_path)
        fitted = fit(
            subset, cached["fit"]["component_grids"], ids[reference], labels[reference]
        )
        metrics = {}
        for name, model in fitted["models"].items():
            metrics[name] = evaluate_affinity(
                affinity(subset, model), ids[reference],
                labels[reference], labels[validation], 10,
            )[0]
        for metric in ("macro_f1","ndcg_at_10"):
            if abs(metrics["PowerDWASimP1"][metric]-cached["metrics"]["Original"][metric]) > 1e-12:
                raise AssertionError("p=1 does not reproduce frozen outer control")
        outer.append({
            "seed": OUTER_SEEDS[index],
            "reference_ids": ids[reference].tolist(),
            "validation_ids": ids[validation].tolist(),
            "fit": fitted,
            "metrics": metrics,
        })
        write_json(checkpoint,{"outer":outer})
        print(dataset,"power outer",OUTER_SEEDS[index],{m:round(metrics[m]["macro_f1"],4) for m in FAMILIES},flush=True)
    fitted = fit(components,old["fit"]["component_grids"],ids,labels)
    write_json(output/f"kernel_power_{dataset}.frozen_fit.json",fitted)
    metrics,predictions,ndcgs = {},{},{}
    for name,model in fitted["models"].items():
        metrics[name],predictions[name],ndcgs[name] = evaluate_affinity(affinity(components,model),ids,labels,truth,10)
        print(dataset,"power official",name,"p",model["power"],metrics[name],flush=True)
    for metric in ("macro_f1","ndcg_at_10"):
        if abs(metrics["PowerDWASimP1"][metric]-old["test_metrics"]["Original"][metric]) > 1e-12:
            raise AssertionError("p=1 does not reproduce frozen official control")
    paired = {name:paired_bootstrap(truth,predictions["PowerDWASim"],predictions[name],ndcgs["PowerDWASim"],ndcgs[name],2000,20261101+i)
              for i,name in enumerate(metrics) if name != "PowerDWASim"}
    predpath = output/f"kernel_power_{dataset}.predictions.npz"
    np.savez_compressed(predpath,truth=truth,test_ids=test_ids,**{f"prediction_{k}":v for k,v in predictions.items()},
                       **{f"ndcg_{k}":v for k,v in ndcgs.items()})
    summary = {}
    for name in metrics:
        summary[name] = {}
        for metric in ("macro_f1", "ndcg_at_10"):
            values = [fold["metrics"][name][metric] for fold in outer]
            summary[name][metric] = {
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)),
            }
    result = {"dataset":dataset,"fit":fitted,"outer":outer,"outer_summary":summary,"test_metrics":metrics,
              "paired_full_minus":paired,"elapsed_seconds":time.perf_counter()-start,
              "prediction_sha256":hashlib.sha256(predpath.read_bytes()).hexdigest(),
              "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "component_selection_source_sha256":hashlib.sha256(oldpath.read_bytes()).hexdigest(),
              "protocol":{"powers":POWERS,"inner_seeds":INNER_SEEDS,"outer_seeds":OUTER_SEEDS,"k":10,
                           "component_candidates_full":45,"fusion_candidates_per_family":45,
                           "test_labels_in_fit":False,"previously_inspected_test_sets":True,
                           "timing_excludes_cached_component_selection":True,"bootstrap_iterations":2000}}
    write_json(target,result)
    print("Completed power",dataset,"seconds",result["elapsed_seconds"],flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=("ACM","DBLP","all"),default="all")
    parser.add_argument("--output-dir",type=Path,default=RESULTS_ROOT/"kernel_power")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        for dataset in (("ACM","DBLP") if args.dataset == "all" else (args.dataset,)):
            run(dataset,args.output_dir)
