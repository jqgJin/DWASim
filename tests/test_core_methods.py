from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reproduce_original import original_sample  # noqa: E402
from run_corrected_protocol import deterministic_topk, majority_vote  # noqa: E402
from run_imdb_external_validation import (  # noqa: E402
    iterative_multilabel_folds,
    multilabel_prediction,
)
from run_nested_multipath_optimization import (  # noqa: E402
    calibrate_rows,
    relative_dwasim_affinity,
    simplex_weights,
    weighted_vote_from_neighbours,
)
from run_real_multipath_fusion import entropy_weights  # noqa: E402
from run_support_retrieval_stress import (  # noqa: E402
    adaptive_affinity,
    ndcg_at_k_from_order,
)
from similarity_baselines import (  # noqa: E402
    hetesim_affinity,
    index_fingerprint,
    pathsim_affinity,
    symmetric_path_affinities,
)
from tune_effective_weight import lambda_from_alpha  # noqa: E402


class ClassicalSimilarityFormulaTests(unittest.TestCase):
    def test_pathsim_and_hetesim_are_independent(self) -> None:
        profiles = sp.csr_matrix(
            np.array(
                [
                    [1.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0],
                ]
            )
        )
        pathsim = pathsim_affinity(profiles)
        hetesim = hetesim_affinity(profiles)

        self.assertAlmostEqual(pathsim[0, 1], 2.0 / 3.0)
        self.assertAlmostEqual(hetesim[0, 1], 1.0 / np.sqrt(2.0))
        self.assertFalse(np.allclose(pathsim, hetesim))
        np.testing.assert_allclose(pathsim, pathsim.T)
        np.testing.assert_allclose(hetesim, hetesim.T)

    def test_equal_norm_profiles_can_match(self) -> None:
        profiles = sp.csr_matrix(
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        )
        np.testing.assert_allclose(
            pathsim_affinity(profiles),
            hetesim_affinity(profiles),
        )

    def test_hetesim_accepts_rectangular_halves(self) -> None:
        left = sp.csr_matrix(np.array([[1.0, 0.0], [1.0, 1.0]]))
        right = sp.csr_matrix(
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )
        affinity = hetesim_affinity(left, right)
        self.assertEqual(affinity.shape, (2, 3))
        self.assertAlmostEqual(affinity[0, 0], 1.0)
        self.assertAlmostEqual(affinity[0, 1], 0.0)
        self.assertAlmostEqual(affinity[1, 2], 1.0)

    def test_wrapper_keeps_count_and_transition_inputs_separate(self) -> None:
        raw = sp.csr_matrix(np.array([[2.0, 0.0], [1.0, 1.0]]))
        transition = sp.csr_matrix(np.array([[1.0, 0.0], [0.25, 0.75]]))
        affinities = symmetric_path_affinities(raw, transition, [0, 1], [0, 1])
        np.testing.assert_allclose(affinities["PathSim"], pathsim_affinity(raw))
        np.testing.assert_allclose(
            affinities["HeteSim"], hetesim_affinity(transition)
        )

    def test_cache_fingerprint_uses_identity_and_order(self) -> None:
        self.assertEqual(index_fingerprint([1, 2, 3]), index_fingerprint([1, 2, 3]))
        self.assertNotEqual(index_fingerprint([1, 2, 3]), index_fingerprint([3, 2, 1]))
        self.assertNotEqual(index_fingerprint([1, 2, 3]), index_fingerprint([4, 5, 6]))


class ComponentAndRetrievalTests(unittest.TestCase):
    def test_adaptive_support_is_bounded_and_nests_magnitude_only(self) -> None:
        jaccard = np.asarray([[0.0, 0.8], [0.8, 0.0]])
        bray = np.asarray([[0.0, 0.3], [0.3, 0.0]])
        magnitude = adaptive_affinity(jaccard, bray, beta=0.0)
        adaptive = adaptive_affinity(jaccard, bray, beta=0.5)
        np.testing.assert_allclose(magnitude, 1.0 - bray)
        self.assertTrue(np.all((adaptive >= 0.0) & (adaptive <= 1.0)))
        self.assertLess(adaptive[0, 1], magnitude[0, 1])

    def test_ndcg_matches_a_hand_ranked_binary_example(self) -> None:
        order = np.asarray([[0, 1]], dtype=np.int64)
        query_labels = np.asarray([1], dtype=np.int64)
        reference_labels = np.asarray([1, 0, 1], dtype=np.int64)
        observed = ndcg_at_k_from_order(
            order, query_labels, reference_labels, k=2
        )[0]
        ideal = 1.0 + 1.0 / np.log2(3.0)
        self.assertAlmostEqual(observed, 1.0 / ideal)


class ProtocolUtilityTests(unittest.TestCase):
    def test_historical_sampling_is_unique_and_seeded(self) -> None:
        train_ids = np.arange(100, dtype=np.int64)
        first = original_sample(train_ids, seed=3434, count=50)
        second = original_sample(train_ids, seed=3434, count=50)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(np.unique(first).size, 50)

    def test_topk_and_vote_ties_are_deterministic(self) -> None:
        candidate_ids = np.asarray([30, 10, 20, 40])
        scores = np.asarray([[0.5, 0.5, 0.7, 0.5]])
        selected = deterministic_topk(scores, candidate_ids, 3, largest=True)
        self.assertEqual(selected.tolist(), [[2, 1, 0]])

        labels = np.asarray([0, 1, 1, 0])
        prediction = majority_vote(np.asarray([[2, 0, 1, 3]]), labels)
        self.assertEqual(prediction.tolist(), [1])

    def test_effective_weight_conversion(self) -> None:
        b0, b1 = 100.0, 900.0
        lam = lambda_from_alpha(0.5, b0, b1)
        recovered = lam * b0 / (lam * b0 + (1.0 - lam) * b1)
        self.assertAlmostEqual(lam, 0.9)
        self.assertAlmostEqual(recovered, 0.5)

    def test_entropy_fusion_weights_are_normalized(self) -> None:
        first = np.asarray([[0.8, 0.1], [0.2, 0.7], [0.4, 0.3]])
        second = np.asarray([[0.3, 0.5], [0.6, 0.2], [0.1, 0.9]])
        weights = entropy_weights([first, second])
        self.assertTrue(np.all(weights >= 0.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_simplex_grid_is_complete_and_normalized(self) -> None:
        weights = simplex_weights(3, 0.25)
        self.assertEqual(len(weights), 15)
        self.assertEqual(len(set(weights)), 15)
        for row in weights:
            self.assertAlmostEqual(sum(row), 1.0)

    def test_row_calibration_preserves_ties_and_order(self) -> None:
        values = np.asarray([[0.2, 0.8, 0.2, 0.5]])
        ranked = calibrate_rows(values, "rank")
        scaled = calibrate_rows(values, "minmax")
        self.assertAlmostEqual(ranked[0, 0], ranked[0, 2])
        self.assertGreater(ranked[0, 1], ranked[0, 3])
        self.assertAlmostEqual(float(scaled.min()), 0.0)
        self.assertAlmostEqual(float(scaled.max()), 1.0)

    def test_prior_correction_uses_reference_labels_only(self) -> None:
        affinity = np.asarray(
            [[0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6]]
        )
        neighbours = np.asarray([[0, 1, 2, 9]])
        labels = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        plain = weighted_vote_from_neighbours(
            affinity, neighbours, labels, gamma=0.0, prior_power=0.0
        )
        balanced = weighted_vote_from_neighbours(
            affinity, neighbours, labels, gamma=0.0, prior_power=1.0
        )
        self.assertEqual(plain.tolist(), [0])
        self.assertEqual(balanced.tolist(), [1])

    def test_relative_dwasim_is_bounded_and_has_identity(self) -> None:
        components = {
            "jaccard": np.asarray([[0.0, 0.5], [0.5, 0.0]]),
            "bray_curtis": np.asarray([[0.0, 0.25], [0.25, 0.0]]),
        }
        affinity = relative_dwasim_affinity(components, support_weight=0.4)
        self.assertTrue(np.all((affinity >= 0.0) & (affinity <= 1.0)))
        np.testing.assert_allclose(np.diag(affinity), 1.0)
        np.testing.assert_allclose(affinity, affinity.T)

    def test_multilabel_prediction_never_returns_an_empty_set(self) -> None:
        affinity = np.asarray([[0.9, 0.8, 0.2], [0.0, 0.0, 0.0]])
        neighbours = np.asarray([[0, 1], [0, 1]])
        reference_labels = np.asarray(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int8
        )
        prediction = multilabel_prediction(
            affinity,
            neighbours,
            reference_labels,
            gamma=1.0,
            prior_power=0.0,
            threshold=0.6,
        )
        self.assertEqual(prediction.shape, (2, 3))
        self.assertTrue(np.all(prediction.sum(axis=1) >= 1))

    def test_multilabel_folds_are_reproducible_and_complete(self) -> None:
        labels = np.asarray(
            [
                [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1],
                [0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0],
                [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1],
            ],
            dtype=np.int8,
        )
        first = iterative_multilabel_folds(labels, 3, 20260805)
        second = iterative_multilabel_folds(labels, 3, 20260805)
        seen: list[int] = []
        for (_, validation), (_, repeated_validation) in zip(first, second):
            np.testing.assert_array_equal(validation, repeated_validation)
            seen.extend(validation.tolist())
        self.assertEqual(sorted(seen), list(range(labels.shape[0])))


if __name__ == "__main__":
    unittest.main()
