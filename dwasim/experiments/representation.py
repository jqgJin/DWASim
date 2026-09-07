"""Revision-stage representation and discrepancy-factorial diagnostics."""
import json
import numpy as np
from dwasim.paths import RESULTS_ROOT
from dwasim.data import load_split
from dwasim.experiments.normalization import load_discrepancies
from dwasim.selection import split_positions
from dwasim.data import STRESS_PATHS, load_components
from dwasim.evaluation import evaluate_affinity
from dwasim.selection import select_fusion
from dwasim.selection import fit_family, select_linear, INNER_SEEDS
from dwasim.fusion import predict_affinity


def main():
    for dataset in ("ACM", "DBLP"):
        ids, labels, test_ids, truth = load_split(dataset)
        paths = STRESS_PATHS[dataset]
        splits = split_positions(labels, INNER_SEEDS, .2)
        output = {"dataset": dataset, "paths": paths, "representation": {}, "factorial": {},
                  "protocol": "Ten training-only splits; no representation chosen from test scores."}
        for representation in ("complete", "half"):
            components = [load_components(dataset, p, representation) for p in paths]
            output["representation"][representation] = {}
            for family in ("TriComponentDWASim", "Cosine", "MagnitudeOnly", "OneComponentSelector"):
                model = fit_family(family, components, ids, labels, splits, paths)
                metrics = evaluate_affinity(predict_affinity(components, model), ids, labels, truth, 10)[0]
                output["representation"][representation][family] = {"selection": model, "metrics": metrics}
                print(dataset, representation, family, metrics, flush=True)
        # Fix the original two paths and independently change H -> J and L/B1 -> Bray.
        pair_paths = paths[:2]
        local = [load_components(dataset, p) for p in pair_paths]
        global_values = [load_discrepancies(dataset, p, 128) for p in pair_paths]
        for support in ("H", "J"):
            for magnitude in ("GlobalL", "Bray"):
                selected, train_views, test_views = [], [], []
                for comp, glob in zip(local, global_values):
                    views = {}
                    for prefix in ("train", "test"):
                        first = glob[f"{prefix}_h"] / float(glob["b0"]) if support == "H" else comp[f"{prefix}_jaccard"]
                        second = glob[f"{prefix}_l"] / float(glob["b1"]) if magnitude == "GlobalL" else comp[f"{prefix}_bray"]
                        views[prefix] = [1-first, 1-second]
                    choice = select_linear(views["train"], [(a, 1-a) for a in np.linspace(0, 1, 11)], ids, labels, splits)
                    weights = choice["selected"]["weights"]
                    selected.append(weights)
                    train_views.append(sum(w*v for w, v in zip(weights, views["train"])))
                    test_views.append(sum(w*v for w, v in zip(weights, views["test"])))
                fusion = select_fusion(train_views, ids, labels, splits, [(a, 1-a) for a in np.linspace(0, 1, 11)], 10)
                weights = fusion["selected"]["weights"]
                affinity = sum(w*v for w, v in zip(weights, test_views))
                metrics = evaluate_affinity(affinity, ids, labels, truth, 10)[0]
                output["factorial"][f"{support}_{magnitude}"] = {"component_weights": selected, "path_weights": weights, "metrics": metrics}
                print(dataset, support, magnitude, metrics, flush=True)
        (RESULTS_ROOT / f"representation_controls_{dataset}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
