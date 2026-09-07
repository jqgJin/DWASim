"""Isolated query-batch timings; not an end-to-end scalability claim."""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import numpy as np
import psutil
from sklearn.metrics.pairwise import cosine_similarity
from dwasim.paths import RESULTS_ROOT
from dwasim.data import PATHS, full_profile, multiply_chain
from dwasim.similarity import pair_discrepancies, mixture_affinity
from dwasim.baselines import pathsim_affinity


def worker(method, candidates):
    matrix = full_profile("DBLP", "APTPA") if method != "NativePathSim" else multiply_chain("DBLP", PATHS["DBLP"]["APTPA"]["half"])
    # Disjoint fixed query/candidate rows; profile construction is outside timing.
    query, reference = matrix[-64:], matrix[:candidates]
    process = psutil.Process()
    before = process.memory_info().rss
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        if method == "ThreeComponents":
            score = mixture_affinity(pair_discrepancies(query, reference, include_bhattacharyya=False), [1/3]*3)
        elif method == "Cosine":
            score = cosine_similarity(query, reference)
        else:
            score = pathsim_affinity(query, reference)
        timings.append(time.perf_counter() - started)
        assert score.shape == (64, candidates) and np.all(np.isfinite(score))
    memory = process.memory_info()
    return {"method": method, "queries": 64, "candidates": candidates, "dimensions": matrix.shape[1],
            "seconds": timings, "median_seconds": float(np.median(timings)), "rss_before_MiB": before/2**20,
            "process_peak_MiB": getattr(memory, "peak_wset", memory.rss)/2**20}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("ThreeComponents", "Cosine", "NativePathSim"))
    parser.add_argument("--candidates", type=int, default=2000)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.candidates)))
        return
    rows = []
    for count in (500, 1000, 2000):
        for method in ("ThreeComponents", "Cosine", "NativePathSim"):
            run = subprocess.run([sys.executable, "-m", "dwasim.experiments.kernel_cost", "--worker", method, "--candidates", str(count)], capture_output=True, text=True, check=True)
            row = json.loads(run.stdout.strip().splitlines()[-1])
            rows.append(row)
            print(row, flush=True)
    result = {"rows": rows, "environment": {"python": sys.version, "platform": platform.platform(),
              "processor": platform.processor(), "threads": os.environ.get("OMP_NUM_THREADS")},
              "scope": "Five measured repetitions in a separate process per method/size, profiles already constructed. Peak RSS is whole-process peak including loading, not incremental kernel memory. No model selection or end-to-end preprocessing timing."}
    (RESULTS_ROOT / "revision_cost.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
