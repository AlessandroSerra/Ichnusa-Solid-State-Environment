"""Analytic normal-coordinate and periodic-image regression tests."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from isse import PhononModes, project_displacements, project_velocities
from isse.constants import BOHR_TO_ANGSTROM
from isse.structures import Atoms, Trajectory


class DisplacementProjectionTests(unittest.TestCase):
    def setUp(self):
        primitive = Atoms(
            symbols=["H"],
            cell=np.eye(3),
            positions=np.zeros((1, 3)),
            masses=np.array([4.0]),
        )
        self.q = np.array([[i / 4, 0, 0] for i in range(4)])
        self.e = np.broadcast_to(
            1j * np.eye(3).reshape(1, 3, 1, 3), (4, 3, 1, 3)
        ).copy()
        self.modes = PhononModes(primitive, self.q, np.ones((4, 3)), self.e)
        self.reference = Atoms(
            symbols=["H"] * 4,
            cell=np.diag([4.0, 1.0, 1.0]),
            positions=np.array([[i, 0, 0] for i in range(4)], dtype=float),
            masses=np.full(4, 4.0),
        )
        self.displacements = np.arange(-30, 30, dtype=float).reshape(5, 4, 3) / 1000
        self.frames = self.make_frames(self.displacements)
        self.expected = -1j * np.einsum(
            "ql,tla->tqa",
            np.exp(-2j * np.pi * self.q[:, :1] * np.arange(4)),
            self.displacements,
        )

    def make_frames(self, displacements, wrapped=False):
        frames = []
        for u in displacements:
            positions = self.reference.positions + u
            if wrapped:
                positions = (
                    positions @ np.linalg.inv(self.reference.cell) % 1
                ) @ self.reference.cell
            frames.append(
                Atoms(
                    symbols=self.reference.symbols.copy(),
                    cell=self.reference.cell.copy(),
                    positions=positions,
                    masses=self.reference.masses.copy(),
                )
            )
        return frames

    def project(self, **kwargs):
        return project_displacements(
            self.frames, self.reference, modes=self.modes, **kwargs
        )

    def test_analytic_complex_and_parseval(self):
        result = self.project(return_complex=True, batch_size=2)
        np.testing.assert_allclose(result[4], self.expected, atol=1e-15)
        np.testing.assert_allclose(result[1], np.abs(result[4]) ** 2, atol=1e-16)
        np.testing.assert_allclose(
            result[2], 4 * (self.displacements**2).sum(axis=(1, 2)), atol=1e-16
        )
        np.testing.assert_allclose(result[3], 0, atol=1e-14)
        self.assertEqual(len(self.project()), 4)

    def test_wrapped_positions_crossing_boundaries(self):
        frames = self.make_frames(self.displacements, wrapped=True)
        original_positions = [f.positions.copy() for f in frames]
        result = project_displacements(
            frames, self.reference, self.modes, return_complex=True
        )
        np.testing.assert_allclose(result[4], self.expected, atol=1e-14)
        for f, original in zip(frames, original_positions):
            np.testing.assert_array_equal(f.positions, original)
        # Direct Cartesian differences deliberately retain the image jumps.
        direct = project_displacements(
            frames,
            self.reference,
            self.modes,
            return_complex=True,
            position_mode="cartesian",
        )
        self.assertFalse(np.allclose(direct[4], self.expected))

    def test_unwrapped_and_direct_modes(self):
        for frame in self.frames:
            frame.unwrapped_positions = frame.positions.copy()
            frame.positions[:] = 0  # Explicit unwrapped mode must ignore positions.
        result = self.project(position_mode="unwrapped", return_complex=True)
        np.testing.assert_allclose(result[4], self.expected, atol=1e-15)
        # A common image origin in reference and trajectory is subtracted,
        # not silently reimaged per frame.
        shift = np.array([4.0, -1.0, 2.0])
        self.reference.unwrapped_positions = self.reference.positions + shift
        for frame in self.frames:
            frame.unwrapped_positions += shift
        shifted = self.project(position_mode="unwrapped", return_complex=True)
        np.testing.assert_allclose(shifted[4], self.expected, atol=1e-14)
        direct = project_displacements(
            self.make_frames(self.displacements),
            self.reference,
            self.modes,
            position_mode="cartesian",
            return_complex=True,
        )
        np.testing.assert_allclose(direct[4], self.expected, atol=1e-15)

    def test_unwrapped_large_excursions_are_not_folded(self):
        displacements = self.displacements + np.array([5.0, 2.0, -3.0])
        frames = self.make_frames(displacements, wrapped=True)
        for frame, u in zip(frames, displacements):
            frame.unwrapped_positions = self.reference.positions + u
        unwrapped = project_displacements(
            frames,
            self.reference,
            self.modes,
            position_mode="unwrapped",
            return_complex=True,
        )
        mic = project_displacements(
            frames, self.reference, self.modes, return_complex=True
        )
        self.assertFalse(np.allclose(mic[4], unwrapped[4]))
        np.testing.assert_allclose(
            unwrapped[2], 4 * (displacements**2).sum(axis=(1, 2))
        )

    def test_triclinic_minimum_image(self):
        cell = np.array([[4.0, 0.0, 0.0], [0.2, 1.0, 0.0], [0.1, 0.1, 1.0]])
        primitive = deepcopy(self.modes.primitive)
        primitive.cell = cell / np.array([4.0, 1.0, 1.0])[:, None]
        modes = PhononModes(primitive, self.q, np.ones((4, 3)), self.e)
        ref = deepcopy(self.reference)
        ref.cell = cell
        ref.positions = np.array([[i / 4, 0, 0] for i in range(4)]) @ cell
        frames = []
        for u in self.displacements:
            pos = ref.positions + u
            wrapped = (pos @ np.linalg.inv(cell) % 1) @ cell
            frames.append(Atoms(ref.symbols, cell, wrapped, masses=ref.masses))
        result = project_displacements(frames, ref, modes, return_complex=True)
        np.testing.assert_allclose(result[4], self.expected, atol=1e-14)

    def test_partial_gamma(self):
        modes = PhononModes(
            self.modes.primitive, self.q[:1], np.ones((1, 3)), self.e[:1]
        )
        with self.assertRaisesRegex(ValueError, "Incomplete basis"):
            project_displacements(self.frames, self.reference, modes)
        result = project_displacements(
            self.frames,
            self.reference,
            modes,
            return_complex=True,
            parseval_tolerance=None,
        )
        np.testing.assert_allclose(result[4], self.expected[:, :1], atol=1e-15)
        self.assertTrue(np.all(result[1].sum(axis=(1, 2)) < result[2]))

    def test_zero_displacement_and_no_drift_removal(self):
        result = project_displacements(
            [deepcopy(self.reference)], self.reference, self.modes, return_complex=True
        )
        np.testing.assert_array_equal(result[4], 0)
        self.assertTrue(np.isnan(result[3][0]))
        translation = np.full((1, 4, 3), 0.1)
        result = project_displacements(
            self.make_frames(translation),
            self.reference,
            self.modes,
            return_complex=True,
        )
        # A rigid translation is a physical Gamma coordinate unless removed by caller.
        np.testing.assert_allclose(result[4][0, 0], -0.4j)
        np.testing.assert_allclose(result[4][0, 1:], 0, atol=1e-15)

    def test_fixed_basis_derivative_matches_velocity_projection(self):
        omega = 0.02  # fs^-1, only for this analytic synthetic test
        times = np.arange(5) * 3.0
        pattern = np.zeros((4, 3))
        pattern[:, 0] = 0.1 * np.cos(2 * np.pi * np.arange(4) / 4)
        displacements = np.cos(omega * times)[:, None, None] * pattern
        velocities = -omega * np.sin(omega * times)[:, None, None] * pattern
        frames = self.make_frames(displacements)
        for frame, v in zip(frames, velocities):
            frame.velocities = v
        coordinates = project_displacements(
            frames, self.reference, self.modes, return_complex=True
        )[4]
        qdot = project_velocities(
            frames, self.reference, self.modes, return_complex=True
        )[4]
        modal_pattern = -1j * np.einsum(
            "ql,la->qa",
            np.exp(-2j * np.pi * self.q[:, :1] * np.arange(4)),
            pattern,
        )
        np.testing.assert_allclose(
            coordinates,
            np.cos(omega * times)[:, None, None] * modal_pattern,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            qdot,
            -omega * np.sin(omega * times)[:, None, None] * modal_pattern,
            atol=1e-15,
        )

    def test_lazy_batch_invariance_and_mass_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "backing"
            path.touch()
            reads = []

            def reader(path, offset):
                reads.append(offset)
                return self.frames[offset]

            trajectory = Trajectory(path, range(5), reader)
            for batch_size in (1, 2, 10):
                reads.clear()
                result = project_displacements(
                    trajectory,
                    self.reference,
                    self.modes,
                    batch_size=batch_size,
                    return_complex=True,
                )
                self.assertEqual(reads, list(range(5)))
                np.testing.assert_allclose(result[4], self.expected, atol=1e-15)
        self.reference.masses = None
        np.testing.assert_allclose(
            self.project(return_complex=True)[4], self.expected, atol=1e-15
        )

    def test_legacy_file_input(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "modes.evec"
            lines = [
                "# Lattice vectors",
                *[" ".join(map(str, row)) for row in np.eye(3) / BOHR_TO_ANGSTROM],
                "# Number of phonon modes: 3",
                "# Number of k points : 4",
            ]
            for iq, q in enumerate(self.q):
                lines.append(f"## kpoint {iq + 1} : " + " ".join(map(str, q)))
                for im, e in enumerate(self.e[iq]):
                    lines.append(f"### mode {im + 1} : 0")
                    lines.extend(f"{z.real} {z.imag}" for z in e.ravel())
            path.write_text("\n".join(lines))
            result = project_displacements(
                self.frames, self.reference, path, return_complex=True
            )
            np.testing.assert_allclose(result[4], self.expected, atol=1e-15)

    def test_input_validation(self):
        with self.assertRaisesRegex(ValueError, "position_mode"):
            self.project(position_mode="auto")
        with self.assertRaisesRegex(ValueError, "unwrapped_positions"):
            self.project(position_mode="unwrapped")
        with self.assertRaises(ValueError):
            self.project(batch_size=0)
        with self.assertRaisesRegex(ValueError, "No trajectory"):
            project_displacements([], self.reference, self.modes)
        self.frames[0].cell[0, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "fixed reference cell"):
            self.project()
        self.frames[0].cell = self.reference.cell.copy()
        self.frames[0].symbols[0] = "He"
        with self.assertRaisesRegex(ValueError, "atom order"):
            self.project()
        self.frames[0].symbols = self.reference.symbols.copy()
        self.frames[0].positions[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "positions must be finite"):
            self.project()
        self.frames[0].positions = np.zeros((3, 3))
        with self.assertRaisesRegex(ValueError, "shape"):
            self.project()


if __name__ == "__main__":
    unittest.main()
