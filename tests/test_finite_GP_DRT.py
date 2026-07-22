import importlib.util
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tutorials" / "finite_GP_DRT.py"
)
SPEC = importlib.util.spec_from_file_location("finite_GP_DRT", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load {MODULE_PATH}")
FINITE_GP_DRT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINITE_GP_DRT)


class NearestPDTests(unittest.TestCase):
    def test_repairs_indefinite_matrix(self):
        result = FINITE_GP_DRT.nearest_PD([[1, 2], [2, 1]])

        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(result, result.T)
        np.linalg.cholesky(result)

    def test_repairs_rank_deficient_matrix(self):
        result = FINITE_GP_DRT.nearest_PD(np.ones((2, 2)))

        np.linalg.cholesky(result)

    def test_preserves_spd_matrix(self):
        matrix = np.array([[2.0, 0.5], [0.5, 1.0]])

        result = FINITE_GP_DRT.nearest_PD(matrix)

        np.testing.assert_allclose(result, matrix)


class IsPDTests(unittest.TestCase):
    def test_accepts_spd_matrix(self):
        self.assertTrue(
            FINITE_GP_DRT.is_PD(np.array([[2.0, 0.5], [0.5, 1.0]]))
        )

    def test_rejects_invalid_inputs(self):
        invalid_inputs = {
            "empty": np.array([]),
            "nonsymmetric": np.array([[1.0, 1.0], [0.0, 1.0]]),
            "nonsquare": np.ones((2, 3)),
            "nan": np.array([[1.0, np.nan], [np.nan, 1.0]]),
            "infinity": np.array([[1.0, 0.0], [0.0, np.inf]]),
        }

        for name, matrix in invalid_inputs.items():
            with self.subTest(name=name):
                self.assertFalse(FINITE_GP_DRT.is_PD(matrix))


class GenerateTMGTests(unittest.TestCase):
    def test_rejects_infeasible_initial_condition_before_sampling(self):
        output = StringIO()
        with mock.patch.object(FINITE_GP_DRT, "randn") as mocked_randn:
            with redirect_stdout(output):
                result = FINITE_GP_DRT.generate_tmg(
                    F=np.array([[1.0]]),
                    g=np.array([0.0]),
                    M=np.eye(1),
                    mu_r=np.zeros(1),
                    initial_X=np.array([0.0]),
                    L=2,
                )

        self.assertIsNone(result)
        mocked_randn.assert_not_called()
        self.assertIn("Error: inconsistent initial condition", output.getvalue())


if __name__ == "__main__":
    unittest.main()
