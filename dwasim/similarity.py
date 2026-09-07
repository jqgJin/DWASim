"""Shared, tested discrepancy definitions and versioned cache identities.

Zero profiles have zero distance to each other by explicit mathematical convention.
This convention is not evidence of a shared positive interaction.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import manhattan_distances

CACHE_VERSION = "dwasim_components_v3_20260905"


def array_fingerprint(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()[:20]


@lru_cache(maxsize=256)
def _file_hash(path: str, size: int, modified_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_fingerprint(processed_root: Path, dataset: str) -> str:
    folder = Path(processed_root) / dataset
    files = sorted(folder.glob("relation_*.npz")) + [folder / "labels.npz"]
    if len(files) < 2 or not all(path.exists() for path in files):
        raise FileNotFoundError(f"Prepared relations and labels are required in {folder}")
    records = []
    for path in files:
        stat = path.stat()
        records.append((path.name, _file_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns)))
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()[:20]


def cache_identity(processed_root: Path, dataset: str, *identifiers) -> str:
    values = [CACHE_VERSION, dataset_fingerprint(processed_root, dataset)]
    for value in identifiers:
        values.append(array_fingerprint(value) if isinstance(value, np.ndarray) else value)
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:24]


def use_cache(path: Path) -> bool:
    return os.environ.get("DWASIM_REBUILD_CACHE", "0") != "1" and path.exists()


def nonnegative_csr(values) -> sp.csr_matrix:
    matrix = sp.csr_matrix(values, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if np.any(~np.isfinite(matrix.data)) or np.any(matrix.data < 0):
        raise ValueError("Profiles must contain finite nonnegative values")
    return matrix


def count_disagreement(query, reference, chunk_size: int = 128) -> np.ndarray:
    query = nonnegative_csr(query)
    reference = nonnegative_csr(reference)
    if query.shape[1] != reference.shape[1]:
        raise ValueError("Query and reference coordinate systems differ")
    answer = np.empty((query.shape[0], reference.shape[0]), dtype=np.float64)
    dense_reference = reference.toarray()
    for start in range(0, query.shape[0], chunk_size):
        view = query[start:start + chunk_size].toarray()
        answer[start:start + len(view)] = np.rint(
            cdist(view, dense_reference, metric="hamming") * query.shape[1]
        )
    return answer


def global_normalizers(profiles, chunk_size: int = 128) -> tuple[float, float]:
    profiles = nonnegative_csr(profiles)
    dense = profiles.toarray()
    maximum = 0.0
    for start in range(0, len(dense), chunk_size):
        if dense.shape[1]:
            maximum = max(maximum, float(np.rint(
                cdist(dense[start:start + chunk_size], dense, "hamming") * dense.shape[1]
            ).max(initial=0.0)))
    value_range = float(dense.max(initial=0.0) - dense.min(initial=0.0))
    # If all entries are positive, the minimum must not be replaced by zero.
    if dense.size:
        value_range = float(dense.max() - dense.min())
    return maximum, maximum * value_range


def pair_discrepancies(query, reference, *, include_hamming: bool = False,
                       include_bhattacharyya: bool = True,
                       required_components: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    query = nonnegative_csr(query)
    reference = nonnegative_csr(reference)
    if query.shape[1] != reference.shape[1]:
        raise ValueError("Query and reference coordinate systems differ")
    required = {"jaccard", "bray", "cosine"} if required_components is None else set(required_components)
    if not required <= {"jaccard", "bray", "cosine"}:
        raise ValueError("Unknown requested affinity component")
    result = {}
    if "jaccard" in required:
        qs, rs = query.copy(), reference.copy()
        qs.data[:] = 1.0
        rs.data[:] = 1.0
        intersection = qs.dot(rs.T).toarray()
        union = np.diff(qs.indptr)[:, None] + np.diff(rs.indptr)[None, :] - intersection
        jaccard = np.divide(union - intersection, union, out=np.zeros_like(union), where=union > 0)
        result["jaccard"] = np.clip(jaccard, 0, 1)
    qa = np.asarray(query.sum(axis=1)).ravel()
    ra = np.asarray(reference.sum(axis=1)).ravel()
    both_zero = (qa == 0)[:, None] & (ra == 0)[None, :]
    if "bray" in required:
        magnitude = manhattan_distances(query, reference)
        total = qa[:, None] + ra[None, :]
        bray = np.divide(magnitude, total, out=np.zeros_like(magnitude), where=total > 0)
        result.update(bray=np.clip(bray, 0, 1), magnitude=magnitude)
    if "cosine" in required:
        qn = np.sqrt(np.asarray(query.multiply(query).sum(axis=1)).ravel())
        rn = np.sqrt(np.asarray(reference.multiply(reference).sum(axis=1)).ravel())
        norm_product = qn[:, None] * rn[None, :]
        cosine = np.divide(query.dot(reference.T).toarray(), norm_product,
                           out=np.zeros_like(norm_product), where=norm_product > 0)
        cosine[both_zero] = 1.0
        result["cosine"] = np.clip(cosine, 0, 1)
    if include_bhattacharyya:
        qprob = sp.diags(np.divide(1., qa, out=np.zeros_like(qa), where=qa > 0)).dot(query).tocsr()
        rprob = sp.diags(np.divide(1., ra, out=np.zeros_like(ra), where=ra > 0)).dot(reference).tocsr()
        qprob.data = np.sqrt(qprob.data)
        rprob.data = np.sqrt(rprob.data)
        bhattacharyya = qprob.dot(rprob.T).toarray()
        bhattacharyya[both_zero] = 1.0
        result["bhattacharyya"] = np.clip(bhattacharyya, 0, 1)
    if include_hamming:
        result["hamming"] = count_disagreement(query, reference)
    return result


def mixture_affinity(components: dict, weights) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (3,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Three nonnegative component weights must sum to one")
    return np.clip(weights[0] * (1 - components["jaccard"])
                   + weights[1] * (1 - components["bray"])
                   + weights[2] * components["cosine"], 0, 1)


def active_mixture_affinity(query, reference, weights, dtype=np.float32) -> np.ndarray:
    """Evaluate exactly the nonzero terms, preserving stored-component precision."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (3,) or np.any(~np.isfinite(weights)) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Three finite nonnegative component weights must sum to one")
    required = tuple(name for name, weight in zip(("jaccard", "bray", "cosine"), weights) if weight != 0)
    values = pair_discrepancies(query, reference, include_bhattacharyya=False, required_components=required)
    shape = (query.shape[0], reference.shape[0])
    components = {name: values[name].astype(dtype) if name in values else np.zeros(shape, dtype=dtype)
                  for name in ("jaccard", "bray", "cosine")}
    return mixture_affinity(components, weights)


def safe_entropy_weights(affinities: list[np.ndarray]) -> np.ndarray:
    if not affinities:
        raise ValueError("At least one affinity is required")
    shape = np.asarray(affinities[0]).shape
    if len(shape) != 2 or not all(np.asarray(a).shape == shape for a in affinities):
        raise ValueError("Affinity shapes must agree")
    if shape[0] <= 1:
        return np.full(len(affinities), 1 / len(affinities))
    scores = []
    for affinity in affinities:
        affinity = np.asarray(affinity, dtype=float)
        if np.any(~np.isfinite(affinity)) or np.any(affinity < 0):
            raise ValueError("Affinities must be finite and nonnegative")
        total = affinity.sum(axis=0, keepdims=True)
        probability = np.divide(affinity, total, out=np.zeros_like(affinity), where=total > 0)
        logarithm = np.zeros_like(probability)
        np.log(probability, out=logarithm, where=probability > 0)
        entropy = -(probability * logarithm).sum(axis=0) / np.log(shape[0])
        concentration = np.where(total.ravel() > 0, 1 - entropy, 0)
        scores.append(float(np.maximum(concentration, 0).mean()) if shape[1] else 0.)
    scores = np.asarray(scores)
    if scores.sum() <= 1e-12:
        return np.full(len(scores), 1 / len(scores))
    return scores / scores.sum()
