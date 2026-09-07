import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dwasim.similarity import pair_discrepancies
from dwasim.kernels import center, target_statistics, solve_target, score, hierarchical_weights
from dwasim.fusion import power_affinity


class KernelTargetTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(724)
        self.labels = np.repeat(np.arange(3), 8)
        self.kernels = []
        for _ in range(3):
            profiles = rng.poisson(1.2, (24, 9)).astype(float)
            profiles[:2] = 0
            values = pair_discrepancies(profiles, profiles)
            self.kernels.extend([1-values["jaccard"], 1-values["bray"], values["cosine"]])

    def test_component_and_fusion_psd_including_empty_profiles(self):
        for matrix in self.kernels:
            self.assertGreater(np.linalg.eigvalsh(matrix).min(), -1e-10)
            np.testing.assert_allclose(np.diag(matrix), 1, atol=1e-14)
            self.assertEqual(matrix[0, 1], 1.)
            self.assertEqual(matrix[0, 2], 0.)
        mixed = score(self.kernels, np.arange(1, 10)/45)
        self.assertGreater(np.linalg.eigvalsh(mixed).min(), -1e-10)

    def test_hilbert_distance_triangle_and_rank(self):
        kernel = score(self.kernels, np.full(9, 1/9))
        dist = np.sqrt(np.maximum(0, 2*(1-kernel)))
        for pivot in range(len(dist)):
            self.assertTrue(np.all(dist <= dist[:, pivot, None]+dist[None, pivot, :]+1e-7))
        np.testing.assert_array_equal(np.argsort(-kernel[3], kind="stable"), np.argsort(dist[3], kind="stable"))

    def test_bray_curtis_itself_is_not_a_metric(self):
        x = np.array([[1., 0], [1., 1], [0., 1]])
        bray = pair_discrepancies(x, x)["bray"]
        self.assertGreater(bray[0, 2], bray[0, 1]+bray[1, 2])

    def test_centering_and_psd_gram(self):
        matrix = self.kernels[0]
        h = np.eye(24)-np.ones((24,24))/24
        np.testing.assert_allclose(center(matrix), h@matrix@h, atol=1e-14)
        stats = target_statistics(self.kernels, self.labels)
        self.assertGreater(np.linalg.eigvalsh(stats["gram"]).min(), -1e-10)

    def test_nnls_kkt_and_simplex(self):
        stats = target_statistics(self.kernels, self.labels)
        model = solve_target(stats, .1)
        self.assertLess(model["kkt_residual"], 1e-8)
        self.assertAlmostEqual(sum(model["gamma"]), 1.)
        self.assertTrue(all(g >= 0 for g in model["gamma"]))

    def test_identical_components_receive_equal_regularized_coefficients(self):
        k = (self.labels[:,None] == self.labels).astype(float)
        stats = target_statistics([k,k,k], self.labels)
        model = solve_target(stats, .1)
        np.testing.assert_allclose(model["gamma"], np.ones(3)/3, atol=1e-12)

    def test_restriction_and_hierarchical_identity(self):
        stats = target_statistics(self.kernels, self.labels)
        model = solve_target(stats, .1, (1,4,7))
        gamma = np.asarray(model["gamma"])
        self.assertTrue(np.all(gamma[[0,2,3,5,6,8]] == 0))
        hierarchy = hierarchical_weights(gamma)
        reconstructed = np.asarray(hierarchy["component_weights"])*np.asarray(hierarchy["path_weights"])[:,None]
        np.testing.assert_allclose(reconstructed.ravel(), gamma)

    def test_reference_permutation_invariance(self):
        perm = np.random.default_rng(99).permutation(24)
        original = solve_target(target_statistics(self.kernels, self.labels), .1)
        permuted = solve_target(target_statistics([k[np.ix_(perm,perm)] for k in self.kernels],self.labels[perm]), .1)
        np.testing.assert_allclose(original["gamma"], permuted["gamma"], atol=1e-10)

    def test_all_constant_fallback_and_validation(self):
        stats = target_statistics([np.ones((24,24))]*3, self.labels)
        model = solve_target(stats, .1)
        self.assertIsNotNone(model["fallback"])
        np.testing.assert_allclose(model["gamma"], [1/3]*3)
        for eta in (0,-1,np.nan):
            with self.assertRaises(ValueError):
                solve_target(stats, eta)
        with self.assertRaises(ValueError):
            target_statistics([np.eye(3)], self.labels)
        with self.assertRaises(ValueError):
            target_statistics(self.kernels, np.ones(24))

    def test_deterministic_target_perturbation_bound(self):
        stats = target_statistics(self.kernels, self.labels)
        eta = .1
        old = solve_target(stats, eta)
        newstats = dict(stats, alignment=stats["alignment"]+np.linspace(-1e-4,1e-4,9))
        new = solve_target(newstats, eta)
        self.assertLessEqual(np.linalg.norm(np.array(new["v"])-old["v"]),
                             np.linalg.norm(newstats["alignment"]-stats["alignment"])/eta+1e-9)

    def test_integer_powers_psd_and_path_order(self):
        views = [score(self.kernels[i:i+3],[.25,.25,.5]) for i in (0,3,6)]
        for power in (1,2,4):
            matrix = power_affinity(views,[.25,.25,.5],power)
            self.assertGreater(np.linalg.eigvalsh(matrix).min(), -1e-10)
            np.testing.assert_allclose(np.diag(matrix),1,atol=1e-14)
            for view in views:
                np.testing.assert_array_equal(np.argsort(view,axis=1,kind="stable"),
                                              np.argsort(view**power,axis=1,kind="stable"))

    def test_pre_fusion_power_can_change_cross_path_order(self):
        views = [np.array([[.9,.6]]),np.array([[.1,.6]])]
        self.assertEqual(np.argmax(power_affinity(views,[.5,.5],1)),1)
        self.assertEqual(np.argmax(power_affinity(views,[.5,.5],2)),0)
        base = power_affinity(views,[.5,.5],1)
        self.assertEqual(np.argmax(base),np.argmax(base**2))

    def test_power_perturbation_bound(self):
        rng = np.random.default_rng(97)
        views = [rng.random((20,30)) for _ in range(3)]
        perturbed = [np.clip(v+rng.uniform(-.001,.001,v.shape),0,1) for v in views]
        for power in (1,2,4):
            a = power_affinity(views,[.25,.25,.5],power)
            b = power_affinity(perturbed,[.25,.25,.5],power)
            self.assertLessEqual(np.max(np.abs(a-b)),power*.001+1e-12)

    def test_inner_fit_has_no_query_argument(self):
        import inspect
        from dwasim.experiments.kernel_target import fit
        self.assertEqual(tuple(inspect.signature(fit).parameters), ("kernels","ids","labels"))
        labels = np.repeat(np.arange(3),10)
        base = (labels[:,None] == labels).astype(float)
        # Deterministic block kernels allow fast complete inner reselection.
        answer = fit([base.copy() for _ in range(9)],np.arange(30),labels)
        for model in answer["models"].values():
            self.assertIsNone(model["fallback"])
            self.assertAlmostEqual(model["validation_macro_f1"],1.)


if __name__ == "__main__":
    unittest.main()
