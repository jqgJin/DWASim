"""Run the four historical label-prediction rows in the original dataset order."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from reproduce_original import (
    PAPER_TARGETS,
    baseline_predictions,
    dwasim_predictions,
    subset_accuracy,
)
from search_historical_seed import sample_from_rng


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"

TABLE_PATHS = {
    "ACM": ["PAP", "PSP"],
    "DBLP": ["APA", "APTPA"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3434)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--skip-dwasim", action="store_true")
    args = parser.parse_args()

    bundles = {
        (dataset, path): baseline_predictions(dataset, path)
        for dataset, paths in TABLE_PATHS.items()
        for path in paths
    }
    rng = random.Random(args.seed)
    query_ids = {
        "ACM": sample_from_rng(bundles[("ACM", "PAP")]["train_ids"], rng, args.queries),
        "DBLP": sample_from_rng(bundles[("DBLP", "APA")]["train_ids"], rng, args.queries),
    }

    rows: list[dict] = []
    started = time.perf_counter()
    for dataset, paths in TABLE_PATHS.items():
        for path in paths:
            row_started = time.perf_counter()
            bundle = bundles[(dataset, path)]
            metrics = {
                "PathSim": subset_accuracy(bundle, query_ids[dataset], "PathSim"),
                "HeteSim": subset_accuracy(bundle, query_ids[dataset], "HeteSim"),
            }
            if not args.skip_dwasim:
                positions = {
                    int(node): position for position, node in enumerate(bundle["train_ids"])
                }
                query_positions = np.asarray(
                    [positions[int(node)] for node in query_ids[dataset]], dtype=np.int64
                )
                truth = bundle["train_labels"][query_positions]
                predictions = dwasim_predictions(dataset, path, query_ids[dataset])
                metrics["DWASim"] = float(np.mean(predictions == truth))

            targets = PAPER_TARGETS[(dataset, path)]
            row = {
                "dataset": dataset,
                "path": path,
                "metrics": metrics,
                "paper_targets": targets,
                "absolute_gaps": {
                    method: abs(value - targets[method])
                    for method, value in metrics.items()
                },
                "elapsed_seconds": time.perf_counter() - row_started,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    output = {
        "protocol": "historical-fidelity",
        "dataset_order": list(TABLE_PATHS),
        "seed": args.seed,
        "queries_per_dataset": args.queries,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_ROOT / f"historical_table_seed{args.seed}_q{args.queries}.json"
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False)
    print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)
    print(f"saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
