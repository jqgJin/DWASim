"""Component affinities and fixed-weight path fusion."""
from __future__ import annotations
import numpy as np


def component_views(components, prefix="train"):
    return [1 - components[f"{prefix}_jaccard"], 1 - components[f"{prefix}_bray"],
            components[f"{prefix}_cosine"]]


def weighted_view(views, weights):
    return sum(float(weight) * view for weight, view in zip(weights, views) if weight != 0)


def predict_affinity(components, fitted, prefix="test"):
    return sum(weighted_view(component_views(values, prefix), gamma)
               for values, gamma in zip(components, fitted["gamma"]) if np.sum(gamma) > 0)


POWERS = (1, 2, 4)


def power_affinity(views, weights, power):
    if power not in POWERS:
        raise ValueError("Only the fixed positive integer power grid is supported")
    return weighted_view([v if power == 1 else v**power for v in views], weights)


def relative_affinity(jaccard: np.ndarray, bray: np.ndarray, beta: float) -> np.ndarray:
    distance = float(beta) * jaccard + (1.0 - float(beta)) * bray
    return np.clip(1.0 - distance, 0.0, 1.0)


def adaptive_affinity(jaccard: np.ndarray, bray: np.ndarray, beta: float) -> np.ndarray:
    conflict = np.maximum(jaccard - bray, 0.0)
    distance = bray + float(beta) * conflict
    return np.clip(1.0 - distance, 0.0, 1.0)
