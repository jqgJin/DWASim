"""Reference implementations of the classical path-based similarities.

The functions in this module deliberately operate on *half-path profiles*.
For a symmetric meta-path, let ``X`` be the raw half-path count matrix and
``U`` the product of the row-stochastic relation matrices along that half.
Then PathSim is the Dice-normalized product of rows of ``X``, whereas HeteSim
is the cosine-normalized product of rows of ``U``.

Keeping the two inputs explicit is important: normalizing only the final
half-path product is not equivalent to multiplying transition matrices for a
multi-relation half-path.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib

import numpy as np
import scipy.sparse as sp


BASELINE_CACHE_TAG = "classical_v2"


def index_fingerprint(values: Sequence[int] | np.ndarray) -> str:
    """Return an order-sensitive cache fingerprint for a node-index vector."""
    array = np.asarray(values, dtype=np.dtype("<i8"))
    if array.ndim != 1:
        raise ValueError("node indices must be one-dimensional")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(np.asarray(contiguous.shape, dtype=np.dtype("<i8")).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()[:16]


def _as_nonnegative_csr(matrix: sp.spmatrix | np.ndarray, name: str) -> sp.csr_matrix:
    """Return a float64 CSR matrix after validating a path-profile input."""
    if sp.issparse(matrix):
        value = matrix.tocsr().astype(np.float64, copy=False)
    else:
        array = np.asarray(matrix, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional matrix")
        value = sp.csr_matrix(array)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if value.data.size and (not np.all(np.isfinite(value.data)) or np.any(value.data < 0)):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return value


def _validated_pair(
    query: sp.spmatrix | np.ndarray,
    reference: sp.spmatrix | np.ndarray | None,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    query_csr = _as_nonnegative_csr(query, "query profiles")
    reference_csr = (
        query_csr
        if reference is None
        else _as_nonnegative_csr(reference, "reference profiles")
    )
    if query_csr.shape[1] != reference_csr.shape[1]:
        raise ValueError(
            "query and reference profiles must share the same terminal dimension"
        )
    return query_csr, reference_csr


def pathsim_affinity(
    query_half_counts: sp.spmatrix | np.ndarray,
    reference_half_counts: sp.spmatrix | np.ndarray | None = None,
) -> np.ndarray:
    """Compute PathSim for a symmetric path from raw half-path counts.

    A zero profile has similarity zero to every profile, including another
    zero profile. Nonzero self-similarity follows from the formula and is one.
    """
    query, reference = _validated_pair(query_half_counts, reference_half_counts)
    product = query.dot(reference.T).toarray()
    query_squared_norm = np.asarray(query.multiply(query).sum(axis=1)).ravel()
    reference_squared_norm = np.asarray(reference.multiply(reference).sum(axis=1)).ravel()
    denominator = query_squared_norm[:, None] + reference_squared_norm[None, :]
    affinity = np.divide(
        2.0 * product,
        denominator,
        out=np.zeros_like(product, dtype=np.float64),
        where=denominator > 0,
    )
    return np.clip(affinity, 0.0, 1.0)


def hetesim_affinity(
    query_transition_profiles: sp.spmatrix | np.ndarray,
    reference_transition_profiles: sp.spmatrix | np.ndarray | None = None,
) -> np.ndarray:
    """Compute HeteSim from the two transition-based half-path profiles.

    For the symmetric paths evaluated in this project, the left and right
    half-path schemas coincide. For a general path, callers may provide
    different query and reference transition profiles as long as their shared
    middle representation has the same width. No symmetry is imposed.
    """
    query, reference = _validated_pair(
        query_transition_profiles, reference_transition_profiles
    )
    product = query.dot(reference.T).toarray()
    query_norm = np.sqrt(np.asarray(query.multiply(query).sum(axis=1)).ravel())
    reference_norm = np.sqrt(
        np.asarray(reference.multiply(reference).sum(axis=1)).ravel()
    )
    denominator = query_norm[:, None] * reference_norm[None, :]
    affinity = np.divide(
        product,
        denominator,
        out=np.zeros_like(product, dtype=np.float64),
        where=denominator > 0,
    )
    return np.clip(affinity, 0.0, 1.0)


def symmetric_path_affinities(
    raw_half_profiles: sp.spmatrix | np.ndarray,
    transition_half_profiles: sp.spmatrix | np.ndarray,
    query_ids: Sequence[int] | np.ndarray,
    reference_ids: Sequence[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute both baselines from separate raw and transition profiles."""
    raw = _as_nonnegative_csr(raw_half_profiles, "raw half-path profiles")
    transition = _as_nonnegative_csr(
        transition_half_profiles, "transition half-path profiles"
    )
    if raw.shape[0] != transition.shape[0]:
        raise ValueError("raw and transition profiles must describe the same nodes")

    query_index = np.asarray(query_ids, dtype=np.int64)
    reference_index = np.asarray(reference_ids, dtype=np.int64)
    return {
        "PathSim": pathsim_affinity(raw[query_index], raw[reference_index]),
        "HeteSim": hetesim_affinity(
            transition[query_index], transition[reference_index]
        ),
    }
