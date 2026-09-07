"""Deterministic single-label prediction, retrieval, and paired effects."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def deterministic_topk(
    values: np.ndarray,
    candidate_ids: np.ndarray,
    k: int,
    *,
    largest: bool,
) -> np.ndarray:
    """Select k candidates, resolving score ties by the candidate node ID."""
    if k <= 0 or k > candidate_ids.size:
        raise ValueError(f"k must be in [1, {candidate_ids.size}], received {k}")
    candidate_order = np.argsort(candidate_ids, kind="stable")
    primary = -values if largest else values
    ordered = np.argsort(primary[:, candidate_order], axis=1, kind="stable")[:, :k]
    return candidate_order[ordered]


def majority_vote(neighbour_positions: np.ndarray, reference_labels: np.ndarray) -> np.ndarray:
    """Majority vote; ties go to the closest tied class, then smaller class ID."""
    classes = np.unique(reference_labels)
    neighbour_labels = reference_labels[neighbour_positions]
    matches = neighbour_labels[:, :, None] == classes[None, None, :]
    counts = matches.sum(axis=1)
    tied = counts == counts.max(axis=1, keepdims=True)
    first_rank = np.where(tied, matches.argmax(axis=1), neighbour_positions.shape[1] + 1)
    return classes[first_rank.argmin(axis=1)].astype(np.int64)


def metric_record(truth: np.ndarray, prediction: np.ndarray, classes: np.ndarray) -> dict:
    per_class = f1_score(truth, prediction, labels=classes, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=classes, average="macro", zero_division=0)),
        "per_class_f1": {str(int(label)): float(value) for label, value in zip(classes, per_class)},
        "confusion_matrix": confusion_matrix(truth, prediction, labels=classes).tolist(),
    }


def ndcg_at_k_from_order(
    order: np.ndarray,
    query_labels: np.ndarray,
    reference_labels: np.ndarray,
    k: int,
) -> np.ndarray:
    selected_relevance = reference_labels[order] == query_labels[:, None]
    discounts = 1.0 / np.log2(np.arange(2, selected_relevance.shape[1] + 2))
    dcg = (selected_relevance * discounts[None, :]).sum(axis=1)
    class_counts = {
        int(label): int(np.sum(reference_labels == label))
        for label in np.unique(reference_labels)
    }
    ideal_counts = np.asarray(
        [min(k, class_counts.get(int(label), 0)) for label in query_labels],
        dtype=np.int64,
    )
    cumulative = np.concatenate(([0.0], np.cumsum(discounts)))
    ideal = cumulative[ideal_counts]
    return np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)


def evaluate_affinity(
    affinity: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_labels: np.ndarray,
    query_labels: np.ndarray,
    k: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    order = deterministic_topk(
        affinity, candidate_ids, min(k, candidate_ids.size), largest=True
    )
    prediction = majority_vote(order, candidate_labels)
    ndcg = ndcg_at_k_from_order(order, query_labels, candidate_labels, k)
    metrics = {
        "macro_f1": float(
            f1_score(query_labels, prediction, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(query_labels, prediction)),
        f"ndcg_at_{k}": float(np.mean(ndcg)),
        f"ndcg_at_{k}_standard_error": float(
            np.std(ndcg, ddof=1) / np.sqrt(ndcg.size)
        ),
    }
    return metrics, prediction, ndcg


def paired_bootstrap(
    truth: np.ndarray,
    first_prediction: np.ndarray,
    second_prediction: np.ndarray,
    first_ndcg: np.ndarray,
    second_ndcg: np.ndarray,
    iterations: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    classes = np.unique(truth)
    class_positions = [np.flatnonzero(truth == label) for label in classes]
    macro_differences = np.empty(iterations, dtype=np.float64)
    ndcg_differences = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sample = np.concatenate(
            [rng.choice(positions, size=positions.size, replace=True) for positions in class_positions]
        )
        macro_differences[iteration] = f1_score(
            truth[sample],
            first_prediction[sample],
            labels=classes,
            average="macro",
            zero_division=0,
        ) - f1_score(
            truth[sample],
            second_prediction[sample],
            labels=classes,
            average="macro",
            zero_division=0,
        )
        ndcg_differences[iteration] = float(
            np.mean(first_ndcg[sample] - second_ndcg[sample])
        )

    def summarize(values: np.ndarray, point: float) -> dict:
        return {
            "difference": float(point),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "probability_difference_positive": float(np.mean(values > 0)),
        }

    macro_point = f1_score(
        truth, first_prediction, average="macro", zero_division=0
    ) - f1_score(truth, second_prediction, average="macro", zero_division=0)
    ndcg_point = float(np.mean(first_ndcg - second_ndcg))
    return {
        "macro_f1": summarize(macro_differences, macro_point),
        "ndcg_at_k": summarize(ndcg_differences, ndcg_point),
    }
