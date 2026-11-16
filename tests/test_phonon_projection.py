"""Analytic projection regressions; no ALAMODE calculation is run."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from isse import PhononModes, calculate_temperature, project_velocities
from isse.constants import BOHR_TO_ANGSTROM
from isse.io.parse_alamode import read_alamode_evec
from isse.structures import Atoms, Trajectory


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.primitive = Atoms(
            symbols=["H"],
            positions=np.zeros((1, 3)),
            cell=np.eye(3),
            masses=np.array([4.0]),
        )
        self.q = np.array([[i / 4, 0, 0] for i in range(4)])
        # Non-real eigenvectors test conjugation separately from spatial phases.
        self.e = np.broadcast_to(
            1j * np.eye(3).reshape(1, 3, 1, 3), (4, 3, 1, 3)
        ).copy()
        self.modes = PhononModes(self.primitive, self.q, np.ones((4, 3)), self.e)
        self.reference = Atoms(
            symbols=["H"] * 4,
            cell=np.diag([4.0, 1.0, 1.0]),
            positions=np.array([[i, 0, 0] for i in range(4)], dtype=float),
            masses=np.full(4, 4.0),
        )
        self.velocities = np.arange(60, dtype=float).reshape(5, 4, 3) / 17
        self.frames = [
            Atoms(
                symbols=self.reference.symbols,
                cell=self.reference.cell,
                positions=self.reference.positions,
                masses=self.reference.masses,
                velocities=v,
            )
            for v in self.velocities
        ]

    def project(self, **kwargs):
        return project_velocities(
            self.frames, self.reference, modes=self.modes, **kwargs
        )

    def test_analytic_complex_and_parseval(self):
        expected = -1j * np.einsum(
            "ql,tla->tqa",
            np.exp(-2j * np.pi * self.q[:, :1] * np.arange(4)),
            self.velocities,
        )
        r = self.project(batch_size=2, return_complex=True)
        np.testing.assert_allclose(r[4], expected, atol=1e-13)
        np.testing.assert_allclose(r[1], np.abs(r[4]) ** 2, atol=1e-12)
        np.testing.assert_allclose(r[1].sum(axis=(1, 2)), r[2], atol=1e-11)
        np.testing.assert_allclose(r[3], 0, atol=1e-14)
        self.assertEqual(len(self.project()), 4)

    def test_lazy_trajectory_batch_invariance(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "backing"
            path.touch()
            reads = []

            def reader(path, offset):
                reads.append(offset)
                return self.frames[offset]

            trajectory = Trajectory(path, list(range(len(self.frames))), reader)
            self.assertEqual(reads, [])
            for batch_size in (1, 2, 10):
                reads.clear()
                actual = project_velocities(
                    trajectory,
                    self.reference,
                    self.modes,
                    batch_size=batch_size,
                    return_complex=True,
                )
                self.assertEqual(reads, list(range(len(self.frames))))
                expected = self.project(return_complex=True)
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a, b, atol=1e-12)

    def test_gamma_partial_no_renormalization(self):
        gamma = PhononModes(self.primitive, self.q[:1], np.ones((1, 3)), self.e[:1])
        r = project_velocities(
            self.frames,
            self.reference,
            gamma,
            parseval_tolerance=None,
            return_complex=True,
        )
        np.testing.assert_allclose(r[4][:, 0], -1j * self.velocities.sum(axis=1))
        self.assertTrue(np.all(r[1].sum(axis=(1, 2)) < r[2]))
        with self.assertRaisesRegex(ValueError, "Incomplete basis"):
            project_velocities(self.frames, self.reference, gamma)
        partial_mode = PhononModes(
            self.primitive, self.q[:1], np.ones((1, 1)), self.e[:1, :1]
        )
        r = project_velocities(
            self.frames, self.reference, partial_mode, parseval_tolerance=None
        )
        self.assertEqual(r[1].shape, (5, 1, 1))

    def write_evec(self, path):
        lines = [
            "# Lattice vectors",
            *[" ".join(map(str, row)) for row in np.eye(3) / BOHR_TO_ANGSTROM],
            "# Number of phonon modes: 3",
            "# Number of k points : 4",
        ]
        for iq, q in enumerate(self.q):
            lines.append(f"## kpoint {iq + 1} : " + " ".join(map(str, q)))
            for im, e in enumerate(self.e[iq]):
                lines.append(f"### mode {im + 1} : {(-1 if im == 0 else 1) * 1e-6}")
                lines.extend(f"{z.real} {z.imag}" for z in e.ravel())
        path.write_text("\n".join(lines) + "\n")

    def test_legacy_and_explicit_evec_equivalence(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "modes.evec"
            self.write_evec(path)
            modes = PhononModes.from_evec(path, primitive=self.primitive)
            self.assertAlmostEqual(modes.frequencies[0, 0], -109.7373156816)
            expected = self.project(return_complex=True)
            explicit = project_velocities(
                self.frames, self.reference, modes, return_complex=True
            )
            # Exercise the historical spglib mapping on this unambiguous basis.
            legacy = project_velocities(
                self.frames, self.reference, str(path), 2, 1e-6, return_complex=True
            )
            default_legacy = project_velocities(
                self.frames, self.reference, evec_filepath=path
            )
            self.assertEqual(len(default_legacy), 4)
            for a, b in zip(default_legacy, expected[:4]):
                np.testing.assert_allclose(a, b, atol=1e-12)
            for result in (explicit, legacy):
                for a, b in zip(result, expected):
                    np.testing.assert_allclose(a, b, atol=1e-12)
            path.write_text(path.read_text().rsplit("### mode", 1)[0])
            with self.assertRaisesRegex(ValueError, "Incomplete mode"):
                read_alamode_evec(path)

    def test_anphon_adapter_contract(self):
        ph = Mock()
        ph.frequencies.return_value = (np.ones((4, 3)), self.e.reshape(4, 3, 3))
        with self.assertRaisesRegex(ValueError, "Confirm"):
            PhononModes.from_anphon(ph, self.q, primitive=self.primitive)
        ph.frequencies.assert_not_called()
        modes = PhononModes.from_anphon(
            ph, self.q, primitive=self.primitive, source_order_confirmed=True
        )
        np.testing.assert_array_equal(ph.frequencies.call_args.args[0], self.q)
        self.assertEqual(ph.frequencies.call_args.kwargs, {"eigenvectors": True})
        np.testing.assert_array_equal(modes.eigenvectors, self.e)
        ph.frequencies.return_value = (np.ones((4, 3)), np.zeros((4, 1, 3, 3)))
        with self.assertRaisesRegex(ValueError, "anphon eigenvectors"):
            PhononModes.from_anphon(
                ph, self.q, primitive=self.primitive, source_order_confirmed=True
            )

    def test_shape_normalization_masses_validation(self):
        for e in (self.e[..., :2], self.e * 2, np.zeros_like(self.e), self.e * np.nan):
            with self.assertRaises(ValueError):
                PhononModes(self.primitive, self.q, np.ones((4, 3)), e)
        e = self.e.copy()
        e[:, 1] = e[:, 0]
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            PhononModes(self.primitive, self.q, np.ones((4, 3)), e)
        bad = deepcopy(self.primitive)
        bad.masses = np.array([-1.0])
        with self.assertRaisesRegex(ValueError, "masses"):
            PhononModes(bad, self.q, np.ones((4, 3)), self.e)
        self.reference.masses[0] = 5
        with self.assertRaisesRegex(ValueError, "Mapped masses"):
            self.project()

    def test_primitive_requires_explicit_masses(self):
        primitive = deepcopy(self.primitive)
        primitive.masses = None
        with self.assertRaisesRegex(ValueError, "explicit atomic masses"):
            PhononModes(primitive, self.q, np.ones((4, 3)), self.e)
        with self.assertRaisesRegex(TypeError, "isse.structures.Atoms"):
            PhononModes(object(), self.q, np.ones((4, 3)), self.e)

    def test_constructor_copies_inputs(self):
        self.primitive.positions[:] = 7
        self.primitive.cell[:] = 8
        self.primitive.masses[:] = 9
        self.primitive.symbols[0] = "He"
        self.q[:] = 2
        self.e[:] = 0
        np.testing.assert_array_equal(self.modes.primitive.positions, np.zeros((1, 3)))
        np.testing.assert_array_equal(self.modes.primitive.cell, np.eye(3))
        np.testing.assert_array_equal(self.modes.primitive.masses, [4.0])
        self.assertEqual(self.modes.primitive.symbols, ["H"])
        np.testing.assert_array_equal(self.modes.qpoints[:, 0], np.arange(4) / 4)
        self.assertTrue(np.any(self.modes.eigenvectors != 0))

    def test_primitive_images_are_not_wrapped(self):
        primitive = deepcopy(self.primitive)
        primitive.positions[0, 0] += 1.0
        modes = PhononModes(primitive, self.q, np.ones((4, 3)), self.e)
        original = self.project(return_complex=True)
        shifted = project_velocities(
            self.frames,
            self.reference,
            modes,
            return_complex=True,
        )
        expected_phase = np.exp(2j * np.pi * self.q[:, 0])[None, :, None]
        np.testing.assert_allclose(shifted[4], original[4] * expected_phase, atol=1e-12)
        np.testing.assert_allclose(shifted[1], original[1], atol=1e-12)

    def test_explicit_multiatom_order(self):
        # Intentionally opposite to the trajectory order, with distinct species.
        primitive = Atoms(
            symbols=["He", "H"],
            positions=np.array([[0.4, 0.2, 0], [0, 0, 0]]),
            cell=np.eye(3),
            masses=np.array([4.0, 1.0]),
        )
        e = np.eye(6).reshape(1, 6, 2, 3).astype(complex)
        modes = PhononModes(primitive, [[0, 0, 0]], np.ones((1, 6)), e)
        ref = Atoms(
            symbols=["H", "He"],
            cell=np.eye(3),
            positions=np.array([[0.0, 0, 0], [0.4, 0.2, 0]]),
            masses=np.array([1.0, 4.0]),
            velocities=np.array([[1.0, 2, 3], [4.0, 5, 6]]),
        )
        r = project_velocities([ref], ref, modes, return_complex=True)
        np.testing.assert_allclose(r[4][0, 0], [8, 10, 12, 1, 2, 3])
        ref.symbols = ["He", "H"]
        with self.assertRaisesRegex(ValueError, "Mapped species"):
            project_velocities([ref], ref, modes)

    def test_mapping_validation(self):
        self.reference.positions[0, 0] = 0.1
        with self.assertRaisesRegex(ValueError, "mapping"):
            self.project()
        self.reference.positions[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.project()

    def test_noncommensurate_and_duplicate_q(self):
        for q in (self.q + [0.01, 0, 0], np.zeros_like(self.q)):
            modes = PhononModes(self.primitive, q, np.ones((4, 3)), self.e)
            with self.assertRaises(ValueError):
                project_velocities(self.frames, self.reference, modes)

    def test_mass_fallback_zero_and_empty(self):
        self.reference.masses = None
        self.project()
        self.frames[0].velocities[:] = 0
        r = self.project()
        self.assertTrue(np.isnan(r[3][0]))
        with self.assertRaisesRegex(ValueError, "No trajectory"):
            project_velocities([], self.reference, self.modes)

    def test_parseval_rejects_incomplete_legacy(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "modes.evec"
            self.write_evec(path)
            # A normalized but incomplete legacy Gamma-only basis retains
            # the historical RuntimeError unless validation is disabled.
            text = path.read_text().split("## kpoint 2")[0]
            path.write_text(text.replace("k points : 4", "k points : 1"))
            with self.assertRaisesRegex(RuntimeError, "Parseval error"):
                project_velocities(self.frames, self.reference, path)
            r = project_velocities(
                self.frames, self.reference, path, parseval_tolerance=None
            )
            self.assertEqual(r[1].shape, (5, 1, 3))

    def test_input_errors(self):
        with self.assertRaises(ValueError):
            self.project(batch_size=0)
        with self.assertRaises(TypeError):
            self.project(backend="numpy")
        with self.assertRaises(ValueError):
            project_velocities(
                self.frames, self.reference, self.modes, modes=self.modes
            )
        with self.assertRaises(ValueError):
            project_velocities(self.frames, self.reference)
        self.frames[0].velocities = None
        with self.assertRaisesRegex(ValueError, "does not contain velocities"):
            self.project()

    def test_temperature_interface(self):
        a = calculate_temperature(self.frames, self.reference, modes=self.modes)
        b = calculate_temperature(self.frames, self.reference, self.modes)
        np.testing.assert_allclose(a["mode_temperatures"], b["mode_temperatures"])


if __name__ == "__main__":
    unittest.main()
