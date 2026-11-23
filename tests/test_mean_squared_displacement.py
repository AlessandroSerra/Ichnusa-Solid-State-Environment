"""Analytical and direct-reference tests for all-origins MSD."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from isse import calculate_msd, masses_from_symbols
from isse.structures import Atoms, Trajectory


def make_frames(positions, *, symbols=None, cell=None, unwrapped=None, masses=None, times=None):
    if symbols is None:
        symbols = ["Li"] * positions.shape[1]
    if cell is None:
        cell = np.eye(3) * 10
    return [
        Atoms(symbols, cell, p,
              unwrapped_positions=None if unwrapped is None else unwrapped[i],
              masses=masses, info={} if times is None else {"timestep": str(times[i])})
        for i, p in enumerate(positions)
    ]


def direct_msd(positions):
    return np.array([
        np.mean(np.sum((positions[lag:] - positions[:len(positions) - lag]) ** 2, axis=-1))
        for lag in range(len(positions))
    ])


class MSDTests(unittest.TestCase):
    def test_fft_matches_direct_all_origins_without_timing_metadata(self):
        rng = np.random.default_rng(12)
        positions = 1e6 + rng.normal(size=(37, 5, 3)).cumsum(axis=0)
        frames = make_frames(positions.copy())
        actual = calculate_msd(frames, position_mode="cartesian")
        assert_allclose(actual, direct_msd(positions), atol=1e-12)
        assert_array_equal(np.stack([f.positions for f in frames]), positions)

    def test_ballistic_motion_time_axis_and_lag_limit(self):
        steps = np.arange(9)
        positions = steps[:, None, None] * np.array([[[1., 2., 0.], [0., 0., 0.]]])
        frames = make_frames(positions)
        timed_frames = make_frames(positions, times=100 + steps * 2.5)
        times, msd = calculate_msd(timed_frames, position_mode="cartesian")
        assert_array_equal(times, steps * 2.5)
        assert_allclose(msd, 2.5 * steps**2, atol=1e-12)
        assert_allclose(calculate_msd(frames, position_mode="cartesian", max_lag=2), msd[:3])
        assert_array_equal(calculate_msd(frames, max_lag=0), [0.])
        assert_array_equal(calculate_msd(frames[:1]), [0.])
        assert_array_equal(calculate_msd(make_frames(np.ones_like(positions))), np.zeros(9))

    def test_every_selects_lags_without_skipping_time_origins(self):
        positions = np.zeros((8, 1, 3))
        positions[:, 0, 0] = np.arange(8) ** 2
        frames = make_frames(positions)
        expected = direct_msd(positions)
        actual = calculate_msd(frames, every=2, position_mode="cartesian")
        assert_allclose(actual, expected[::2])
        timed_frames = make_frames(positions, times=100 + np.arange(8) * 2.5)
        times, msd = calculate_msd(
            timed_frames, every=2, max_lag=5, position_mode="cartesian",
        )
        assert_array_equal(times, [0., 5., 10.])
        assert_allclose(msd, expected[:6:2])
        assert_array_equal(calculate_msd(frames, every=10), [0.])

    def test_wrapped_triclinic_multiple_crossings_lazy_single_pass(self):
        cell = np.array([[8., 0., 0.], [2., 7., 0.], [.5, 1., 6.]])
        fractional = np.array([[[.9, .1, .5]]]) + np.arange(15)[:, None, None] * [.3, -.25, .05]
        frames = make_frames((fractional % 1) @ cell, cell=cell)
        reads = []

        def reader(path, offset):
            reads.append(offset)
            return frames[offset]

        with TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory"
            path.touch()
            trajectory = Trajectory(path, range(len(frames)), reader)
            # Skipping every other input frame would make the 0.6 fractional
            # displacement ambiguous. Output stride must not affect unwrapping.
            actual = calculate_msd(trajectory, every=2)

        self.assertEqual(reads, list(range(len(frames))))
        assert_allclose(actual, direct_msd(fractional @ cell)[::2], atol=1e-12)

    def test_explicit_unwrapped_is_preferred_even_for_large_steps(self):
        continuous = np.arange(7)[:, None, None] * np.array([[[8., 0., 0.]]])
        frames = make_frames(continuous % 10, unwrapped=continuous)
        expected = direct_msd(continuous)
        for mode in ("auto", "unwrapped"):
            assert_allclose(calculate_msd(frames, position_mode=mode), expected)
        assert_allclose(calculate_msd(frames, position_mode="wrapped"), expected / 16)

    def test_com_uses_all_atoms_before_species_selection(self):
        positions = np.zeros((8, 2, 3))
        positions[:, :, 0] = np.arange(8)[:, None] * [5., 1.]
        symbols = ["Li", "O"]
        for masses in (None, np.array([1., 3.])):
            frames = make_frames(positions % 10, symbols=symbols, unwrapped=positions, masses=masses)
            weights = masses_from_symbols(symbols) if masses is None else masses
            expected = ((5. - np.average([5., 1.], weights=weights)) * np.arange(8)) ** 2
            assert_allclose(calculate_msd(frames, species="Li", remove_com=True), expected)
            assert_allclose(calculate_msd(frames, species="Li"), (5 * np.arange(8)) ** 2)

    def test_com_removes_rigid_translation_across_periodic_boundaries(self):
        positions = np.arange(12)[:, None, None] * [2., 1., 0.]
        positions = positions + [[[0., 0., 0.], [3., 1., 0.]]]
        frames = make_frames(positions % 10, symbols=["Li", "O"])
        assert_allclose(calculate_msd(frames, remove_com=True), 0., atol=1e-25)

    def test_timestep_metadata_validation(self):
        positions = np.zeros((3, 1, 3))
        for times, error in [
            ([0, 2, 5], "uniformly spaced"),
            ([0, 2, 2], "strictly increasing"),
            ([0, np.inf, 4], "numeric and finite"),
            ([0, "invalid", 4], "numeric and finite"),
        ]:
            with self.subTest(times=times), self.assertRaisesRegex(ValueError, error):
                calculate_msd(make_frames(positions, times=times))

        frames = make_frames(positions, times=[10**18 + 2 * i for i in range(3)])
        lag_time, msd = calculate_msd(frames, every=2)
        assert_array_equal(lag_time, [0., 4.])
        assert_array_equal(msd, [0., 0.])
        lag_time, msd = calculate_msd(frames[:1])
        assert_array_equal(lag_time, [0.])
        assert_array_equal(msd, [0.])
        frames[1].info.clear()
        with self.assertRaisesRegex(ValueError, "mixed availability of timestep"):
            calculate_msd(frames)
        with self.assertRaisesRegex(ValueError, "mixed availability of timestep"):
            calculate_msd(frames[1:])

    def test_invalid_options(self):
        frames = make_frames(np.zeros((3, 1, 3)))
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            calculate_msd([])
        for kwargs, error in [
            ({"max_lag": 3}, "max_lag"), ({"max_lag": -1}, "max_lag"),
            ({"max_lag": 1.5}, "max_lag"), ({"species": "O"}, "no atoms"),
            ({"every": 0}, "every"), ({"every": -1}, "every"), ({"every": 1.5}, "every"),
            ({"position_mode": "unknown"}, "position_mode"),
            ({"position_mode": "unwrapped"}, "unwrapped_positions are required"),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, error):
                calculate_msd(frames, **kwargs)

    def test_inconsistent_frames(self):
        frames = make_frames(np.zeros((2, 1, 3)))
        frames[1].unwrapped_positions = np.zeros((1, 3))
        with self.assertRaisesRegex(ValueError, "mixed availability"):
            calculate_msd(frames)
        frames[1].cell = np.eye(3) * 11
        with self.assertRaisesRegex(ValueError, "fixed cell"):
            calculate_msd(frames, position_mode="wrapped")
        assert_array_equal(calculate_msd(frames, position_mode="cartesian"), [0., 0.])
        frames[1].masses = np.array([0.])
        with self.assertRaisesRegex(ValueError, "masses"):
            calculate_msd(frames, position_mode="cartesian", remove_com=True)
        frames[1].symbols = ["O"]
        with self.assertRaisesRegex(ValueError, "symbol order"):
            calculate_msd(frames, position_mode="cartesian")


if __name__ == "__main__":
    unittest.main()
