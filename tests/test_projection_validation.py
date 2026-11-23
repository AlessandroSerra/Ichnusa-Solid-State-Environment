"""Shared frame validation, numerical integrity and partial-mode temperatures."""

import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from isse import (
    PhononModes,
    calculate_temperature,
    project_displacements,
    project_velocities,
)
from isse.constants import AMU_A2_FS2_TO_EV, KB_EV_K
from isse.helpers.mode_projection import _compute_parseval_errors
from isse.structures import Atoms


class ProjectionValidationTests(unittest.TestCase):
    def setUp(self):
        self.reference = Atoms(
            symbols=["H", "He"],
            cell=np.diag([2.0, 2.0, 2.0]),
            positions=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            masses=np.array([1.0, 4.0]),
        )
        self.modes = PhononModes(
            self.reference,
            [[0.0, 0.0, 0.0]],
            np.arange(6.0)[None, :],
            np.eye(6, dtype=complex).reshape(1, 6, 2, 3),
        )
        self.frames = [deepcopy(self.reference) for _ in range(2)]
        for i, frame in enumerate(self.frames):
            frame.velocities = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * (i + 1)
            frame.positions += 0.01 * (i + 1)

    def test_every_frame_species_checked_by_both_workflows(self):
        for workflow in (project_velocities, project_displacements):
            for batch_size in (1, 2):
                with self.subTest(workflow=workflow.__name__, batch_size=batch_size):
                    frames = deepcopy(self.frames)
                    frames[1].symbols.reverse()
                    with self.assertRaisesRegex(
                        ValueError, "frame 1 symbols.*atom order"
                    ):
                        workflow(
                            frames, self.reference, self.modes, batch_size=batch_size
                        )

    def test_every_frame_cell_checked_by_both_workflows(self):
        for workflow in (project_velocities, project_displacements):
            for value in (2.01, np.nan, np.inf):
                with self.subTest(workflow=workflow.__name__, value=value):
                    frames = deepcopy(self.frames)
                    frames[1].cell[0, 0] = value
                    with self.assertRaisesRegex(
                        ValueError, "frame 1 cell.*fixed reference"
                    ):
                        workflow(frames, self.reference, self.modes)
            frames = deepcopy(self.frames)
            frames[1].cell = np.eye(2)
            with self.assertRaisesRegex(ValueError, "frame 1 cell"):
                workflow(frames, self.reference, self.modes)

    def test_optional_masses_checked_when_present(self):
        bad_masses = (
            [1.0, 5.0],
            [np.nan, 4.0],
            [1.0, np.inf],
            [0.0, 4.0],
            [-1.0, 4.0],
            [1.0],
        )
        for workflow in (project_velocities, project_displacements):
            for masses in bad_masses:
                with self.subTest(workflow=workflow.__name__, masses=masses):
                    frames = deepcopy(self.frames)
                    frames[0].masses = None  # Reference supplies projection masses.
                    frames[1].masses = np.array(masses)
                    with self.assertRaisesRegex(
                        ValueError, "frame 1 masses.*projection masses"
                    ):
                        workflow(frames, self.reference, self.modes)

    def test_no_redundant_frame_masses_required(self):
        for workflow in (project_velocities, project_displacements):
            with self.subTest(workflow=workflow.__name__):
                expected = workflow(self.frames, self.reference, self.modes)
                frames = deepcopy(self.frames)
                for frame in frames:
                    frame.masses = None
                actual = workflow(frames, self.reference, self.modes)
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a, b)
                # Small metadata roundoff is accepted consistently with the
                # primitive/projection mass comparison.
                frames[1].masses = self.reference.masses + 1e-8
                workflow(frames, self.reference, self.modes)

    def test_first_frame_mass_fallback_and_later_validation(self):
        reference = deepcopy(self.reference)
        reference.masses = None
        for workflow in (project_velocities, project_displacements):
            with self.subTest(workflow=workflow.__name__):
                frames = deepcopy(self.frames)
                frames[1].masses = None
                actual = workflow(frames, reference, self.modes)
                expected = workflow(self.frames, self.reference, self.modes)
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a, b)
                frames[1].masses = np.array([1.0, 5.0])
                with self.assertRaisesRegex(ValueError, "frame 1 masses"):
                    workflow(frames, reference, self.modes)

    def test_partial_gamma_temperatures_retain_all_selected_modes(self):
        # Deliberately reordered, purely selected modes with no acoustic IDs.
        # Both a list shorter than three and one longer than three must retain
        # all supplied entries, not drop the first three by position.
        for indices in ([3, 4], [5, 3, 0, 4]):
            partial = PhononModes(
                self.reference,
                self.modes.qpoints,
                self.modes.frequencies[:, indices],
                self.modes.eigenvectors[:, indices],
            )
            for keyword in (False, True):
                with self.subTest(indices=indices, keyword=keyword):
                    args = () if keyword else (partial,)
                    kwargs = {"modes": partial} if keyword else {}
                    with self.assertLogs(
                        "isse.phonon_temperatures", level="WARNING"
                    ) as logs:
                        result = calculate_temperature(
                            self.frames,
                            self.reference,
                            *args,
                            parseval_tolerance=None,
                            **kwargs,
                        )
                    self.assertTrue(
                        any(
                            "Partial Gamma mode set" in message
                            for message in logs.output
                        )
                    )
                    factor = AMU_A2_FS2_TO_EV / KB_EV_K
                    full_squared = (
                        self.frames[0].velocities
                        * np.sqrt(self.reference.masses)[:, None]
                    ).ravel() ** 2
                    expected = full_squared[indices] * 2.5 * factor  # mean(1², 2²)
                    np.testing.assert_allclose(result["mode_temperatures"][0], expected)
                    self.assertAlmostEqual(
                        result["mean_mode_temperature"], expected.mean()
                    )

    def test_complete_gamma_mode_list_preserves_legacy_average(self):
        result = calculate_temperature(self.frames, self.reference, self.modes)
        self.assertAlmostEqual(
            result["mean_mode_temperature"],
            result["mode_temperatures"][0, 3:].mean(),
        )

    def test_zero_norm_nan_convention_and_mixed_frames(self):
        for workflow in (project_velocities, project_displacements):
            for tolerance in (None, 1e-6):
                with self.subTest(workflow=workflow.__name__, tolerance=tolerance):
                    frames = deepcopy(self.frames)
                    frames[0].positions = self.reference.positions.copy()
                    frames[0].velocities[:] = 0
                    result = workflow(
                        frames,
                        self.reference,
                        self.modes,
                        parseval_tolerance=tolerance,
                        return_complex=True,
                    )
                    self.assertTrue(np.isnan(result[3][0]))
                    self.assertTrue(np.isfinite(result[3][1]))
                    np.testing.assert_array_equal(result[1][0], 0)
                    np.testing.assert_array_equal(result[4][0], 0)
                    only_zero = workflow(
                        frames[:1],
                        self.reference,
                        self.modes,
                        parseval_tolerance=tolerance,
                    )
                    self.assertTrue(np.isnan(only_zero[3][0]))

    def test_nonfinite_outputs_never_hidden_by_parseval_filtering(self):
        for tolerance in (None, 1e-6):
            for field in (0, 1, 2):
                for value in (np.nan, np.inf):
                    with self.subTest(tolerance=tolerance, field=field, value=value):
                        outputs = [
                            np.zeros((2, 1, 6)),
                            np.zeros(2),
                            np.zeros((2, 1, 6), dtype=complex),
                        ]
                        outputs[field].flat[0] = value
                        with (
                            patch(
                                "isse.helpers.mode_projection._project_batch_numba",
                                return_value=outputs,
                            ),
                            self.assertRaisesRegex(RuntimeError, "Non-finite"),
                        ):
                            project_velocities(
                                self.frames,
                                self.reference,
                                self.modes,
                                parseval_tolerance=tolerance,
                                return_complex=True,
                            )

    def test_finite_input_overflow_rejected_for_both_workflows(self):
        for workflow in (project_velocities, project_displacements):
            for tolerance in (None, 1e-6):
                with self.subTest(workflow=workflow.__name__, tolerance=tolerance):
                    frames = deepcopy(self.frames)
                    frames[0].velocities[:] = 1e200
                    frames[0].positions[:] = 1e200
                    kwargs = (
                        {"position_mode": "cartesian"}
                        if workflow is project_displacements
                        else {}
                    )
                    with self.assertRaisesRegex(RuntimeError, "Non-finite"):
                        workflow(
                            frames,
                            self.reference,
                            self.modes,
                            parseval_tolerance=tolerance,
                            **kwargs,
                        )

    def test_parseval_sum_and_relative_error_overflow(self):
        with self.assertRaisesRegex(RuntimeError, "Non-finite Parseval"):
            _compute_parseval_errors(np.array([[[1e308, 1e308]]]), np.ones(1))
        with self.assertRaisesRegex(RuntimeError, "Non-finite Parseval"):
            _compute_parseval_errors(np.array([[[1e308]]]), np.array([1e-308]))
        with self.assertRaisesRegex(RuntimeError, "zero atomic norm"):
            _compute_parseval_errors(np.ones((1, 1, 1)), np.zeros(1))


if __name__ == "__main__":
    unittest.main()
