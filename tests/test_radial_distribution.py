"""Regression tests for the fused total/partial RDF histogram."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from isse.radial_distribution import _histogram_rdf_numba, calculate_rdf
from isse.structures import Atoms, Trajectory


def reference_counts(positions, cells, cutoff, nbins, species_ids, pair_index, npairs):
    """Small independent oracle using a linear solve for fractional vectors."""
    total = np.zeros((len(positions), nbins))
    partial = np.zeros((len(positions), npairs, nbins))
    for f, (frame, cell) in enumerate(zip(positions, cells)):
        for i in range(len(frame)):
            for j in range(i + 1, len(frame)):
                fractional = np.linalg.solve(cell.T, frame[j] - frame[i])
                fractional -= np.rint(fractional)
                distance = np.linalg.norm(fractional @ cell)
                if distance < cutoff:
                    b = int(distance / (cutoff / nbins))
                    total[f, b] += 2
                    if species_ids is not None:
                        p = pair_index[species_ids[i], species_ids[j]]
                        if p >= 0:
                            partial[f, p, b] += 2
    return total, partial


class RDFTests(unittest.TestCase):
    def test_counts_against_independent_reference(self):
        rng = np.random.default_rng(42)
        cells = np.array([np.diag([8., 9., 10.]),
                          [[8., 0., 0.], [1.5, 9., 0.], [.5, 1., 10.]]])
        positions = np.einsum('fai,fij->faj', rng.uniform(-1, 2, (2, 19, 3)), cells)
        ids = np.arange(19, dtype=np.int64) % 2
        for pairs, npairs in [(None, 0), (np.array([[0, 1], [1, 2]]), 3),
                              (np.array([[-1, 0], [0, -1]]), 1)]:
            species_ids = ids if pairs is not None else None
            with self.subTest(npairs=npairs):
                actual = _histogram_rdf_numba(positions, cells, np.linalg.inv(cells),
                                              3.7, 3.7 / 13, 13,
                                              species_ids, pairs, npairs)
                expected = reference_counts(positions, cells, 3.7, 13,
                                            species_ids, pairs, npairs)
                for a, e in zip(actual, expected):
                    assert_array_equal(a, e)
                if npairs == 3:
                    assert_array_equal(actual[0], actual[1].sum(axis=1))

    def test_cutoff_bins_and_periodic_boundary(self):
        cells = np.array([np.eye(3) * 16])
        # Distances 0, 1 (an exact bin edge), 2 (excluded cutoff), and
        # 15.5 (wrapped to 0.5).
        positions = np.array([[[0., 0., 0.], [0., 0., 0.], [1., 0., 0.],
                               [2., 0., 0.], [15.5, 0., 0.]]])
        total, partial = _histogram_rdf_numba(positions, cells, np.linalg.inv(cells),
                                             2., 1., 2)
        assert_array_equal(total, [[6., 8.]])
        self.assertEqual(partial.shape, (1, 0, 2))

    def test_single_frame_normalization_and_selected_pairs(self):
        atoms = Atoms(symbols=['Si', 'O', 'Si'], cell=np.eye(3) * 8,
                      positions=np.array([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]]))
        total_only = calculate_rdf(atoms, 4., 4)
        all_pairs = calculate_rdf(atoms, 4., 4, partial=True)
        selected = calculate_rdf(atoms, 4., 4, species_pairs=[('Si', 'O'), ('O', 'Si')])
        for result in (all_pairs, selected):
            assert_array_equal(result['g_r'], total_only['g_r'])
            self.assertEqual(result['g_r'].shape, (4,))
        self.assertEqual(set(selected['partial_g_r']), {('O', 'Si')})
        shells = 4 * np.pi / 3 * np.diff(np.arange(5.) ** 3)
        assert_allclose(total_only['g_r'], np.array([0, 2, 2, 2]) * 512 / (6 * shells))
        assert_allclose(selected['partial_g_r'][('O', 'Si')],
                        np.array([0, 2, 2, 0]) * 512 / (4 * shells))

    def test_batch_invariance(self):
        frames = [Atoms(symbols=['Si', 'O', 'Si'], cell=np.eye(3) * (8 + i),
                        positions=np.array([[0., 0., 0.], [1. + i * .1, 0., 0.],
                                            [3., 0., 0.]])) for i in range(3)]
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'trajectory'
            path.touch()
            trajectory = Trajectory(path, range(len(frames)), lambda path, offset: frames[offset])
            first = calculate_rdf(trajectory, 4., 8, batch_size=1, partial=True)
            second = calculate_rdf(trajectory, 4., 8, batch_size=2, partial=True)
        assert_array_equal(first['g_r'], second['g_r'])
        for pair in first['partial_g_r']:
            assert_array_equal(first['partial_g_r'][pair], second['partial_g_r'][pair])
        self.assertEqual(first['g_r'].shape, (3, 8))

    def test_single_atom(self):
        atoms = Atoms(symbols=['Si'], cell=np.eye(3) * 8, positions=np.zeros((1, 3)))
        result = calculate_rdf(atoms, 3., 6, partial=True)
        assert_array_equal(result['g_r'], np.zeros(6))
        assert_array_equal(result['partial_g_r'][('Si', 'Si')], np.zeros(6))


if __name__ == '__main__':
    unittest.main()
