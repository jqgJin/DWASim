import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dwasim.experiments.regularization import screen_row, shrink_row, shrink_weights, fit_models, affinity_for, MODES
from dwasim.selection import split_positions
from dwasim.experiments.path_fusion import fit_path_controls, affinity_for as original_affinity


def row(weights, score):
    return {"weights": weights, "macro_f1_mean": score, "ndcg_at_10_mean": score}


class RegularizationTests(unittest.TestCase):
    def test_screen_prefers_simple_only_within_tolerance(self):
        candidates = [row([0, 0, 1], .8), row([.5, 0, .5], .804), row([1, 0, 0], .79)]
        self.assertEqual(screen_row(candidates)["weights"], [0, 0, 1])
        candidates[1]["macro_f1_mean"] = .806
        self.assertEqual(screen_row(candidates)["weights"], [.5, 0, .5])

    def test_shrinkage_endpoints_and_simplex(self):
        weights = [.25, 0, .75]
        np.testing.assert_array_equal(shrink_weights(weights, 0), weights)
        np.testing.assert_allclose(shrink_weights(weights, 1), [1/3]*3)
        for alpha in (0, .25, .5, .75, 1):
            result = shrink_weights(weights, alpha)
            self.assertAlmostEqual(sum(result), 1)
            self.assertTrue(all(v >= 0 for v in result))
        with self.assertRaises(ValueError):
            shrink_weights(weights, 2)

    def test_shrinkage_tolerance_does_not_use_secondary_metric(self):
        rows = [dict(row([0, 0, 1], .81), alpha=0), dict(row([1/3]*3, .806), alpha=1)]
        self.assertEqual(shrink_row(rows)["alpha"], 1)
        rows[1]["macro_f1_mean"] = .804
        self.assertEqual(shrink_row(rows)["alpha"], 0)

    def test_train_only_fit_original_identity_and_matched_baselines(self):
        rng = np.random.default_rng(93)
        labels = np.repeat([0, 1], 20)
        ids = np.arange(len(labels))
        comps = []
        for _ in range(3):
            views = [rng.random((40, 40), dtype=np.float32) for _ in range(3)]
            comps.append({"train_jaccard": 1-views[0], "train_bray": 1-views[1], "train_cosine": views[2]})
        splits = split_positions(labels, [11, 12], .2)
        result = fit_models(comps, ids, labels, splits)
        self.assertEqual(set(result["models"]), set(MODES))
        reference = fit_path_controls(comps, ids, labels, splits, ("a", "b", "c"))
        np.testing.assert_array_equal(affinity_for(comps, result["models"]["Original"], "train"),
                                      original_affinity(comps, reference["LearnedPathFusion"], "train"))
        for model in result["models"].values():
            self.assertAlmostEqual(sum(model["path_weights"]), 1)
            self.assertTrue(all(abs(sum(w)-1) < 1e-12 for w in model["component_weights"]))
        self.assertEqual(result["models"]["ScreenedShrinkage"]["candidate_evaluations"], 64)
        self.assertEqual(result["models"]["CosineShrinkage"]["candidate_evaluations"], 19)
        self.assertEqual(result["models"]["SingleSelectorShrinkage"]["candidate_evaluations"], 28)


if __name__ == "__main__":
    unittest.main()
