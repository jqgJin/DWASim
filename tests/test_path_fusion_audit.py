import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dwasim.similarity import active_mixture_affinity, pair_discrepancies, mixture_affinity
from dwasim.experiments.path_fusion import fit_path_controls
from dwasim.selection import split_positions


class ActiveInferenceTests(unittest.TestCase):
    def test_exact_pruning_including_zero_profiles_and_float32_storage(self):
        rng = np.random.default_rng(82)
        q = rng.integers(0, 5, size=(9, 7)).astype(float)
        r = rng.integers(0, 5, size=(12, 7)).astype(float)
        q[0] = 0
        r[0] = 0
        q, r = sp.csr_matrix(q), sp.csr_matrix(r)
        all_values = {key: value.astype(np.float32) for key, value in pair_discrepancies(q, r).items()}
        for weights in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (.5, 0, .5), (.25, .25, .5)):
            np.testing.assert_array_equal(active_mixture_affinity(q, r, weights), mixture_affinity(all_values, weights))

    def test_direction_only_does_not_execute_magnitude_kernel(self):
        x = sp.csr_matrix([[0., 0.], [1., 2.], [2., 4.]])
        with patch("dwasim.similarity.manhattan_distances", side_effect=AssertionError("unused magnitude executed")):
            scores = active_mixture_affinity(x, x, (0, 0, 1))
        np.testing.assert_allclose(scores, [[1, 0, 0], [0, 1, 1], [0, 1, 1]], atol=1e-7)

    def test_best_path_is_selected_on_training_labels(self):
        labels = np.repeat([0, 1], 30)
        ids = np.arange(len(labels))
        signal = (labels[:, None] == labels[None, :]).astype(np.float32)
        constant = np.ones_like(signal)
        components = [{"train_jaccard": 1-v, "train_bray": 1-v, "train_cosine": v}
                      for v in (constant, signal, constant)]
        fitted = fit_path_controls(components, ids, labels, split_positions(labels, [11, 12], .2), ("a", "b", "c"))
        self.assertEqual(fitted["BestSinglePath"]["selected_path"], "b")
        self.assertEqual(fitted["UniformPathFusion"]["path_weights"], [1/3]*3)
        theta = fitted["BestSinglePath"]["component_weights"]
        self.assertEqual(theta, fitted["UniformPathFusion"]["component_weights"])
        self.assertEqual(theta, fitted["LearnedPathFusion"]["component_weights"])


if __name__ == "__main__":
    unittest.main()
