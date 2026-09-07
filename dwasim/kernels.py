"""Regularized centered-kernel target fitting for unchanged DWASim affinities.

Only reference Gram matrices and reference labels enter fitting. Centering is
used to learn coefficients; inference uses the original, uncentered affinities.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

ETAS = (0.001, 0.01, 0.1, 1.0, 10.0)
FAMILIES = {
    "KernelTarget": tuple(range(9)),
    "MagnitudeTarget": (1, 4, 7),
    "CosineTarget": (2, 5, 8),
    "NoSupportTarget": (1, 2, 4, 5, 7, 8),
    "NoMagnitudeTarget": (0, 2, 3, 5, 6, 8),
    "NoDirectionTarget": (0, 1, 3, 4, 6, 7),
}


def center(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A square Gram matrix is required")
    if not np.isfinite(matrix).all():
        raise ValueError("Gram matrix must be finite")
    return (
        matrix
        - matrix.mean(axis=0)[None, :]
        - matrix.mean(axis=1)[:, None]
        + matrix.mean()
    )


def target_statistics(kernels, labels):
    """Build the centered kernel Gram matrix and reference-label alignment."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size < 2 or not np.isfinite(labels).all():
        raise ValueError("At least two finite single-label reference nodes required")
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if len(counts) < 2:
        raise ValueError("At least two reference categories required")
    indicator = np.eye(len(counts))[inverse] / np.sqrt(counts)[None, :]
    target = center(indicator @ indicator.T)
    target /= np.linalg.norm(target)
    normalized = []
    scales = []
    active = []
    for index, kernel in enumerate(kernels):
        kernel = np.asarray(kernel, dtype=np.float64)
        if kernel.shape != (len(labels), len(labels)) or not np.allclose(
            kernel, kernel.T, atol=2e-6
        ):
            raise ValueError("Reference Gram matrices must be square and symmetric")
        centered = center(kernel)
        norm = float(np.linalg.norm(centered))
        scales.append(norm)
        if norm > 1e-12 * max(1.0, float(np.linalg.norm(kernel))):
            active.append(index)
            normalized.append((centered / norm).ravel())
    if not active:
        return {
            "gram": np.empty((0, 0)),
            "alignment": np.empty(0),
            "scales": np.asarray(scales),
            "active": np.asarray(active, dtype=int),
        }
    design = np.stack(normalized)
    gram = design @ design.T
    return {
        "gram": (gram + gram.T) / 2,
        "alignment": design @ target.ravel(),
        "scales": np.asarray(scales),
        "active": np.asarray(active, dtype=int),
    }


def solve_target(stats, eta, allowed=None):
    """Fit nonnegative coefficients and map them back to uncentered kernels."""
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and strictly positive")
    size = len(stats["scales"])
    allowed = np.arange(size) if allowed is None else np.asarray(allowed, dtype=int)
    if (
        allowed.size == 0
        or len(np.unique(allowed)) != len(allowed)
        or np.any((allowed < 0) | (allowed >= size))
    ):
        raise ValueError("Distinct valid eligible kernel indices required")
    keep = np.flatnonzero(np.isin(stats["active"], allowed))
    gamma = np.zeros(size)
    vector = np.zeros(size)
    if not len(keep):
        gamma[allowed] = 1 / len(allowed)
        return {
            "gamma": gamma.tolist(),
            "eta": float(eta),
            "v": vector.tolist(),
            "kkt_residual": 0.0,
            "fallback": "all eligible reference kernels centered-constant",
        }
    active = stats["active"][keep]
    gram = stats["gram"][np.ix_(keep, keep)]
    alignment = stats["alignment"][keep]
    quadratic = gram + eta * np.eye(len(keep))
    # Q = L L.T converts the quadratic fit to nonnegative least squares.
    cholesky = np.linalg.cholesky(quadratic)
    solution = nnls(
        cholesky.T, np.linalg.solve(cholesky, alignment), maxiter=1000
    )[0]
    gradient = quadratic @ solution - alignment
    residual = max(
        float(np.max(np.maximum(-gradient, 0))),
        float(np.max(np.abs(solution * gradient))),
    )
    if residual > 1e-7:
        raise ArithmeticError(f"KKT check failed: {residual}")
    raw = solution / stats["scales"][active]
    fallback = None
    if raw.sum() <= 1e-15:
        gamma[active] = 1 / len(active)
        fallback = "zero nonnegative label alignment"
    else:
        gamma[active] = raw / raw.sum()
    vector[active] = solution
    return {
        "gamma": gamma.tolist(),
        "eta": float(eta),
        "v": vector.tolist(),
        "kkt_residual": residual,
        "fallback": fallback,
        "objective": float(0.5 * solution @ quadratic @ solution - alignment @ solution),
        "active_training_kernels": active.tolist(),
    }


def score(kernels, gamma):
    gamma = np.asarray(gamma, dtype=np.float64)
    if len(kernels) != len(gamma) or np.any(gamma < 0) or not np.isclose(gamma.sum(), 1):
        raise ValueError("A simplex coefficient for each kernel is required")
    result = np.zeros_like(kernels[0], dtype=np.float64)
    for kernel, weight in zip(kernels, gamma):
        if weight:
            result += weight * np.asarray(kernel, dtype=np.float64)
    return result


def hierarchical_weights(gamma):
    gamma = np.asarray(gamma, dtype=np.float64).reshape(-1, 3)
    paths = gamma.sum(axis=1)
    within = np.divide(
        gamma, paths[:, None],
        out=np.full_like(gamma, 1 / 3),
        where=paths[:, None] > 0,
    )
    return {"path_weights": paths.tolist(), "component_weights": within.tolist()}
