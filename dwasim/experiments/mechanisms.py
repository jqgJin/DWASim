"""Frozen mechanism diagnostics with disjoint train/validation/test observations.

These generated nonnegative profiles isolate stipulated mechanisms; they are
not empirical HGB data and do not establish performance on unseen networks.
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split
from dwasim.similarity import pair_discrepancies
from dwasim.paths import RESULTS_ROOT
from dwasim.selection import fit_family
from dwasim.fusion import predict_affinity
from dwasim.selection import split_positions
from dwasim.evaluation import evaluate_affinity

METHODS = ("TriComponentDWASim", "Cosine", "MagnitudeOnly", "Jaccard", "OneComponentSelector", "FixedEqualMixture")
SCENARIOS = ("Support", "Magnitude", "Direction", "Complementary", "Conflicting")


def generate(kind, labels, rng):
    n, q = len(labels), 60
    core = np.arange(q)[None, :] // 15 == labels[:, None]
    if kind == "Support":
        support = rng.random((n, q)) < np.where(core, .65, .12)
        counts = 1 + rng.poisson(3, (n, q))
        activity = rng.lognormal(0, 1.2, (n, 1))
        return np.rint(support * counts * activity)
    if kind == "Magnitude":
        return np.array([1, 2, 4, 8])[labels, None] * np.ones((n, q)) + rng.poisson(.25, (n, q))
    if kind == "Direction":
        activity = rng.lognormal(0, 1.2, (n, 1))
        return (1 + 3 * core + rng.gamma(1, .75, (n, q))) * activity
    raise ValueError(kind)


def main():
    records = []
    for scenario in SCENARIOS:
        for seed in range(20260920, 20260930):
            rng = np.random.default_rng(seed)
            labels = np.repeat(np.arange(4), 60)
            ids = np.arange(len(labels))
            train, test = train_test_split(ids, test_size=.4, stratify=labels, random_state=seed)
            kinds = [scenario] * 3 if scenario in SCENARIOS[:3] else ["Support", "Magnitude", "Direction"]
            components = []
            for index, kind in enumerate(kinds):
                view_labels = rng.permutation(labels) if scenario == "Conflicting" and index == 2 else labels
                profiles = generate(kind, view_labels, rng)
                values = {}
                for prefix, queries in (("train", train), ("test", test)):
                    comp = pair_discrepancies(profiles[queries], profiles[train])
                    values.update({f"{prefix}_{key}": value for key, value in comp.items()})
                components.append(values)
            splits = split_positions(labels[train], list(range(20260901, 20260906)), .2)
            row = {"scenario": scenario, "seed": seed, "train_ids": train.tolist(), "test_ids": test.tolist(), "methods": {}}
            for method in METHODS:
                model = fit_family(method, components, ids[train], labels[train], splits, ("View1", "View2", "View3"))
                metric = evaluate_affinity(predict_affinity(components, model), ids[train], labels[train], labels[test], 10)[0]
                row["methods"][method] = {"selection": model, "metrics": metric}
            records.append(row)
        print(scenario, {m: round(float(np.mean([r["methods"][m]["metrics"]["macro_f1"] for r in records if r["scenario"] == scenario])), 4) for m in METHODS}, flush=True)
    output = {"protocol": {"n": 240, "categories": 4, "dimensions": 60, "train_fraction": .6, "k": 10,
              "inner_splits": 5, "seeds": list(range(20260920, 20260930)), "scenarios": SCENARIOS,
              "interpretation": "Ten independent generated profile collections; not independent empirical graphs."}, "runs": records}
    (RESULTS_ROOT / "controlled_mechanisms.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
