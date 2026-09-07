"""Benchmark relations, profiles, partitions, and component caches."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from dwasim.paths import ROOT, PROCESSED_ROOT, CACHE_ROOT, RESULTS_ROOT
from dwasim.similarity import cache_identity, use_cache, pair_discrepancies
from dwasim.baselines import symmetric_path_affinities


PATHS = {
    "ACM": {
        "PAP": {"half": [2], "full": [2, 3]},
        "PSP": {"half": [4], "full": [4, 5]},
        "PTP": {"half": [6], "full": [6, 7]},
    },
    "DBLP": {
        "APA": {"half": [0], "full": [0, 3]},
        "APTPA": {"half": [0, 1], "full": [0, 1, 4, 3]},
        "APVPA": {"half": [0, 2], "full": [0, 2, 5, 3]},
    },
    "IMDB": {
        "MDM": {"half": [0], "full": [0, 1]},
        "MAM": {"half": [2], "full": [2, 3]},
        "MKM": {"half": [4], "full": [4, 5]},
    },
}


def load_relation(dataset: str, relation_id: int) -> sp.csr_matrix:
    return sp.load_npz(PROCESSED_ROOT / dataset / f"relation_{relation_id}.npz").tocsr()


def row_normalize(matrix: sp.csr_matrix) -> sp.csr_matrix:
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.zeros_like(sums, dtype=np.float64)
    nonzero = sums != 0
    inverse[nonzero] = 1.0 / sums[nonzero]
    return sp.diags(inverse).dot(matrix).tocsr()


def multiply_chain(dataset: str, relation_ids: list[int], transition: bool = False) -> sp.csr_matrix:
    matrices = [load_relation(dataset, relation_id) for relation_id in relation_ids]
    if transition:
        matrices = [row_normalize(matrix) for matrix in matrices]
    product = matrices[0]
    for matrix in matrices[1:]:
        product = product.dot(matrix).tocsr()
    product.eliminate_zeros()
    product.sort_indices()
    return product


def full_profile(dataset: str, path_name: str) -> sp.csr_matrix:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    identity = cache_identity(PROCESSED_ROOT, dataset, path_name, PATHS[dataset][path_name]["full"])
    cache_path = CACHE_ROOT / f"full_profile_{dataset}_{path_name}_{identity}.npz"
    if use_cache(cache_path):
        return sp.load_npz(cache_path).tocsr()
    matrix = multiply_chain(dataset, PATHS[dataset][path_name]["full"], transition=False)
    sp.save_npz(cache_path, matrix, compressed=True)
    return matrix


def load_split(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.load(PROCESSED_ROOT / dataset / "labels.npz")
    return (
        labels["train_ids"].astype(np.int64),
        labels["train_labels"].astype(np.int64),
        labels["test_ids"].astype(np.int64),
        labels["test_labels"].astype(np.int64),
    )


STRESS_PATHS = {
    "ACM": ("PAP", "PSP", "PTP"),
    "DBLP": ("APA", "APTPA", "APVPA"),
}


def _component_cache(dataset: str, path_name: str, representation: str = "complete") -> Path:
    train_ids, _, test_ids, _ = load_split(dataset)
    path_key = path_name if representation == "complete" else f"{path_name}_{representation}"
    identity = cache_identity(PROCESSED_ROOT, dataset, path_key, train_ids, test_ids)
    return CACHE_ROOT / f"support_retrieval_components_{dataset}_{path_key}_{identity}.npz"


def load_components(dataset: str, path_name: str, representation: str = "complete") -> dict[str, np.ndarray]:
    """Load content-addressed complete-profile and native classical affinities."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if representation not in ("complete", "half"):
        raise ValueError("Representation must be complete or half")
    cache_path = _component_cache(dataset, path_name, representation)
    if use_cache(cache_path):
        with np.load(cache_path) as cached:
            return {name: cached[name] for name in cached.files}
    train_ids, _, test_ids, _ = load_split(dataset)
    profiles = (full_profile(dataset, path_name) if representation == "complete" else
                multiply_chain(dataset, PATHS[dataset][path_name]["half"])).tocsr().astype(np.float64)
    train, test = profiles[train_ids], profiles[test_ids]
    payload = {}
    for prefix, query in (("train", train), ("test", test)):
        values = pair_discrepancies(query, train)
        for name in ("jaccard", "bray", "cosine", "bhattacharyya"):
            payload[f"{prefix}_{name}"] = values[name].astype(np.float32)
    config = PATHS[dataset][path_name]
    raw_half = multiply_chain(dataset, config["half"], transition=False)
    transition_half = multiply_chain(dataset, config["half"], transition=True)
    for prefix, queries in (("train", train_ids), ("test", test_ids)):
        classical = symmetric_path_affinities(raw_half, transition_half, queries, train_ids)
        for method in ("PathSim", "HeteSim"):
            payload[f"{prefix}_{method.lower()}"] = classical[method].astype(np.float32)
    np.savez_compressed(cache_path, **payload)
    return payload


def subset_components(components, references, queries):
    answer = []
    for values in components:
        row = {}
        for name in ("jaccard", "bray", "cosine", "bhattacharyya", "pathsim", "hetesim"):
            if f"train_{name}" in values:
                row[f"train_{name}"] = values[f"train_{name}"][np.ix_(references, references)]
                row[f"test_{name}"] = values[f"train_{name}"][np.ix_(queries, references)]
        answer.append(row)
    return answer
