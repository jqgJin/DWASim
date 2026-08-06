"""Search deterministic seeds that best match the paper's historical baseline table.

The published code did not set a random seed.  It sampled ACM first and DBLP
second from one process-global pseudo-random stream.  Once per-node baseline
predictions have been cached by ``reproduce_original.py``, testing candidate
seeds is inexpensive and helps separate random-sampling variation from a true
implementation or data-version mismatch.
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
from pathlib import Path

import numpy as np

from reproduce_original import CACHE_ROOT, all_labeled_dwasim_predictions, baseline_predictions


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"

TARGET_COUNTS = {
    ("ACM", "PAP", "DWASim"): 421,
    ("ACM", "PAP", "PathSim"): 351,
    ("ACM", "PAP", "HeteSim"): 351,
    ("ACM", "PSP", "DWASim"): 338,
    ("ACM", "PSP", "PathSim"): 338,
    ("ACM", "PSP", "HeteSim"): 338,
    ("DBLP", "APA", "DWASim"): 498,
    ("DBLP", "APA", "PathSim"): 217,
    ("DBLP", "APA", "HeteSim"): 217,
    ("DBLP", "APTPA", "DWASim"): 393,
    ("DBLP", "APTPA", "PathSim"): 464,
    ("DBLP", "APTPA", "HeteSim"): 464,
}


def sample_from_rng(train_ids: np.ndarray, rng: random.Random, count: int) -> np.ndarray:
    selected: list[int] = []
    selected_set: set[int] = set()
    while len(selected) < count:
        node = int(train_ids[rng.randint(0, train_ids.size - 1)])
        if node not in selected_set:
            selected.append(node)
            selected_set.add(node)
    return np.asarray(selected, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seed", type=int, default=100_000)
    parser.add_argument("--keep", type=int, default=20)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--include-dwasim", action="store_true")
    args = parser.parse_args()

    bundles = {
        (dataset, path): baseline_predictions(dataset, path)
        for dataset, paths in {"ACM": ["PAP", "PSP"], "DBLP": ["APA", "APTPA"]}.items()
        for path in paths
    }
    if args.include_dwasim:
        for key, bundle in bundles.items():
            dwasim = all_labeled_dwasim_predictions(*key)
            bundle["DWASim"] = dwasim["DWASim"]
    position_maps = {
        key: {int(node): position for position, node in enumerate(bundle["train_ids"])}
        for key, bundle in bundles.items()
    }

    best: list[tuple[int, int, dict]] = []
    exact: list[dict] = []
    for seed in range(args.max_seed + 1):
        rng = random.Random(seed)
        acm_ids = sample_from_rng(bundles[("ACM", "PAP")]["train_ids"], rng, args.queries)
        dblp_ids = sample_from_rng(bundles[("DBLP", "APA")]["train_ids"], rng, args.queries)
        query_ids = {"ACM": acm_ids, "DBLP": dblp_ids}

        observed: dict[str, int] = {}
        total_gap = 0
        active_targets = {
            key: value
            for key, value in TARGET_COUNTS.items()
            if args.include_dwasim or key[2] != "DWASim"
        }
        for (dataset, path, method), target in active_targets.items():
            bundle = bundles[(dataset, path)]
            positions = np.asarray(
                [position_maps[(dataset, path)][int(node)] for node in query_ids[dataset]],
                dtype=np.int64,
            )
            count = int(np.sum(bundle[method][positions] == bundle["train_labels"][positions]))
            observed[f"{dataset}:{path}:{method}"] = count
            total_gap += abs(count - target)

        record = {"seed": seed, "total_absolute_count_gap": total_gap, "counts": observed}
        if total_gap == 0:
            exact.append(record)
            if len(exact) >= args.keep:
                break
        item = (-total_gap, -seed, record)
        if len(best) < args.keep:
            heapq.heappush(best, item)
        elif item > best[0]:
            heapq.heapreplace(best, item)

    best_records = [item[2] for item in sorted(best, reverse=True)]
    output = {
        "searched_seed_range": [0, args.max_seed],
        "queries": args.queries,
        "targets": {
            f"{d}:{p}:{m}": value
            for (d, p, m), value in TARGET_COUNTS.items()
            if args.include_dwasim or m != "DWASim"
        },
        "includes_dwasim": args.include_dwasim,
        "exact_matches": exact,
        "best_matches": best_records,
        "cache_root": str(CACHE_ROOT),
    }
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_ROOT / "historical_seed_search.json"
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
