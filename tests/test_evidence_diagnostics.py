"""Data-independent checks for manuscript diagnostic statistics."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dwasim.diagnostics import affinity_statistics, nullable, query_effect_summary


class EvidenceDiagnosticsTests(unittest.TestCase):
    def test_float32_input_float64_accumulation(self):
        x = np.arange(3000, dtype=np.float32) / 3000
        scores = np.column_stack((x, x, 1-x))
        variance, corr = affinity_statistics(scores)
        self.assertEqual(variance.dtype, np.float64)
        np.testing.assert_allclose(corr, [[1,1,-1],[1,1,-1],[-1,-1,1]], atol=1e-12)
        self.assertLessEqual(np.max(np.abs(corr)), 1)

    def test_constant_is_undefined(self):
        variance, corr = affinity_statistics([[1,0],[1,1],[1,2]])
        self.assertEqual(variance[0], 0)
        self.assertTrue(np.isnan(corr[0]).all())
        self.assertIsNone(nullable(corr)[0][1])
        self.assertEqual(corr[1,1], 1)

    def test_all_queries_retained(self):
        d = np.array([-.2,0,0,.1,.5])
        r = query_effect_summary(d)
        self.assertEqual(r["n"],r["improved"]+r["unchanged"]+r["decreased"])
        self.assertEqual((r["improved"],r["unchanged"],r["decreased"]),(2,2,1))
        self.assertAlmostEqual(r["mean"],.08)

    def test_invalid_input_rejected(self):
        for x in ([[1,2]], [[1,2],[2,np.nan]]):
            with self.assertRaises(ValueError):
                affinity_statistics(x)
        with self.assertRaises(ValueError):
            query_effect_summary([])


if __name__ == "__main__":
    unittest.main()
