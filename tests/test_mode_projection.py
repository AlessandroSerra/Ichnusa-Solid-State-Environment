"""Internal projection engine contracts, independent of vector input kind."""

import unittest

import numpy as np
from numba.core.registry import CPUDispatcher

from isse import (
    PhononModes,
    calculate_temperature,
    project_displacements,
    project_velocities,
)
from isse.helpers.mode_projection import _project_batch_numba, _project_vectors
from isse.structures import Atoms


class ModeProjectionTests(unittest.TestCase):
    def setUp(self):
        self.reference = Atoms(
            symbols=["H"],
            cell=np.eye(3),
            positions=np.zeros((1, 3)),
            masses=np.array([4.0]),
        )
        self.modes = PhononModes(
            self.reference,
            [[0.0, 0.0, 0.0]],
            np.zeros((1, 3)),
            (1j * np.eye(3)).reshape(1, 3, 1, 3),
        )

    def test_required_vector_iterator_and_generic_values(self):
        with self.assertRaisesRegex(TypeError, "vector_batches"):
            _project_vectors([self.reference], self.reference, modes=self.modes)
        # The frame has neither velocities nor any displacement: only the
        # caller-supplied vector batch defines what is projected.
        vectors = np.array([[[1.0, 2.0, 3.0]]])
        result = _project_vectors(
            [self.reference],
            self.reference,
            modes=self.modes,
            vector_batches=iter([vectors]),
            return_complex=True,
        )
        np.testing.assert_allclose(result[4][0, 0], [-2j, -4j, -6j])
        np.testing.assert_allclose(result[1][0, 0], [4, 16, 36])
        np.testing.assert_allclose(result[2], [56])
        np.testing.assert_allclose(result[3], [0])
        self.assertIsInstance(_project_batch_numba, CPUDispatcher)
        self.assertTrue(_project_batch_numba.nopython_signatures)

    def test_both_workflows_import_shared_engine(self):
        self.assertIs(
            project_velocities.__globals__["_project_vectors"], _project_vectors
        )
        self.assertIs(
            project_displacements.__globals__["_project_vectors"], _project_vectors
        )
        self.assertEqual(_project_vectors.__module__, "isse.helpers.mode_projection")

    def test_no_backend_selection_or_fallback(self):
        for workflow in (
            project_velocities,
            project_displacements,
            calculate_temperature,
        ):
            for backend in ("numpy", "numba"):
                with self.assertRaisesRegex(TypeError, "backend"):
                    workflow(
                        [self.reference], self.reference, self.modes, backend=backend
                    )


if __name__ == "__main__":
    unittest.main()
