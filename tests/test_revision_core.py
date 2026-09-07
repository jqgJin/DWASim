from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dwasim.similarity import pair_discrepancies, count_disagreement, mixture_affinity, global_normalizers, safe_entropy_weights, array_fingerprint


class RevisionDefinitionTests(unittest.TestCase):
    def test_count_disagreement_is_not_support_difference(self):
        x, y = [[1, 2]], [[3, 2]]
        result = pair_discrepancies(x, y, include_hamming=True)
        self.assertEqual(result["hamming"][0, 0], 1)
        self.assertEqual(result["jaccard"][0, 0], 0)
        self.assertAlmostEqual(result["bray"][0, 0], 0.25)

    def test_zero_profile_conventions_and_identity(self):
        x = np.array([[0., 0.], [0., 0.], [1., 2.]])
        result = pair_discrepancies(x, x)
        self.assertEqual(result["cosine"][0, 1], 1)
        self.assertEqual(result["cosine"][0, 2], 0)
        self.assertEqual(result["jaccard"][0, 1], 0)
        self.assertEqual(result["jaccard"][0, 2], 1)
        for weights in ((0.25, 0.25, 0.5), (0, 0, 1), (0, 1, 0), (1, 0, 0)):
            affinity = mixture_affinity(result, weights)
            np.testing.assert_allclose(np.diag(affinity), 1)
            np.testing.assert_allclose(affinity, affinity.T)
            self.assertEqual(affinity[0, 1], 1)
            self.assertEqual(affinity[0, 2], 0)

    def test_proportional_profiles_preserve_direction_but_not_magnitude(self):
        result = pair_discrepancies([[1, 2]], [[2, 4]])
        self.assertAlmostEqual(result["cosine"][0, 0], 1)
        self.assertGreater(result["bray"][0, 0], 0)

    def test_sparse_explicit_zero_is_not_support(self):
        x = sp.csr_matrix(([1., 0.], ([0, 0], [0, 1])), shape=(1, 2))
        self.assertEqual(pair_discrepancies(x, [[1, 0]])["jaccard"][0, 0], 0)

    def test_invalid_profiles_and_weights_fail(self):
        for values in ([[-1, 0]], [[np.nan, 0]], [[np.inf, 0]]):
            with self.assertRaises(ValueError):
                pair_discrepancies(values, [[0, 0]])
        with self.assertRaises(ValueError):
            mixture_affinity(pair_discrepancies([[1]], [[1]]), (0.5, 0.5, 0.5))

    def test_global_bounds_match_hand_calculation(self):
        x = np.array([[2, 2, 1], [2, 2, 1], [2, 2, 0], [0, 0, 1]])
        self.assertEqual(global_normalizers(x), (3., 6.))
        self.assertEqual(global_normalizers(np.ones((3, 2))), (0., 0.))
        self.assertEqual(global_normalizers(np.zeros((3, 2))), (0., 0.))

    def test_entropy_degenerate_inputs_are_finite(self):
        for shape in ((2, 2), (1, 1), (2, 0)):
            for value in (0., 1., 3.):
                weights = safe_entropy_weights([np.full(shape, value)] * 2)
                np.testing.assert_allclose(weights, [0.5, 0.5])

    def test_empty_path_gets_no_information_weight(self):
        weights = safe_entropy_weights([np.zeros((2, 2)), np.eye(2)])
        np.testing.assert_allclose(weights, [0., 1.])

    def test_fingerprint_uses_shape_dtype_and_values(self):
        a = np.array([1, 2, 3], dtype=np.int64)
        self.assertNotEqual(array_fingerprint(a), array_fingerprint(a[::-1]))
        self.assertNotEqual(array_fingerprint(a), array_fingerprint(a.reshape(1, 3)))
        self.assertNotEqual(array_fingerprint(a), array_fingerprint(a.astype(float)))

    def test_global_scaling_does_not_change_single_path_ranking(self):
        distance = count_disagreement([[1, 2]], [[1, 2], [2, 2], [2, 3]])
        np.testing.assert_array_equal(np.argsort(distance), np.argsort(-(1 - distance / 8)))

    def test_vectorized_ranking_and_vote_match_reference(self):
        from dwasim.evaluation import deterministic_topk, majority_vote
        rng = np.random.default_rng(20260905)
        ids = rng.permutation(71) + 100
        values = rng.integers(0, 5, size=(31, 71)).astype(float)
        for largest in (False, True):
            primary = -values if largest else values
            expected = np.array([np.lexsort((ids, row))[:10] for row in primary])
            actual = deterministic_topk(values, ids, 10, largest=largest)
            np.testing.assert_array_equal(actual, expected)
            labels = rng.choice([2, 5, 9], 71)
            expected_votes = []
            for positions in expected:
                selected = labels[positions]
                classes, counts = np.unique(selected, return_counts=True)
                tied = classes[counts == counts.max()]
                expected_votes.append(min(tied, key=lambda c: (np.flatnonzero(selected == c)[0], c)))
            np.testing.assert_array_equal(majority_vote(actual, labels), expected_votes)


if __name__ == "__main__":
    unittest.main()
