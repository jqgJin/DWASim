"""Stable descriptive diagnostics; never used for fitting or prediction."""
import numpy as np


def affinity_statistics(scores):
    """Population variance and Pearson correlation across all supplied rows.

    float64 accumulation avoids float32 loss on dense, nearly constant affinities.
    A constant column has undefined correlation, represented by NaN, not zero.
    """
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected at least two finite observations by components")
    centered = x - x.mean(axis=0)
    covariance = centered.T @ centered / x.shape[0]
    variance = np.diag(covariance).copy()
    denominator = np.sqrt(variance[:, None] * variance[None, :])
    correlation = np.divide(covariance, denominator,
                            out=np.full_like(covariance, np.nan), where=denominator > 0)
    if np.any(np.abs(correlation[np.isfinite(correlation)]) > 1 + 1e-12):
        raise ArithmeticError("Pearson correlation outside numerical tolerance")
    return variance, np.clip(correlation, -1, 1)


def nullable(matrix):
    """Serialize undefined correlations as strict JSON null, not NaN."""
    return [[float(v) if np.isfinite(v) else None for v in row] for row in matrix]


def query_effect_summary(delta):
    """Retain every query and distinguish exact ties from improvements/decreases."""
    x = np.asarray(delta, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite query-effect vector")
    return {"n": int(x.size), "mean": float(x.mean()),
            "median": float(np.median(x)), "improved": int(np.sum(x > 0)),
            "unchanged": int(np.sum(x == 0)), "decreased": int(np.sum(x < 0)),
            "quantiles": np.quantile(x, [0, .25, .5, .75, 1]).tolist()}
