"""Regression and analytical tests for spectral frequency grids."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from isse.constants import HZ_TO_CM, PS_TO_FS
from isse.spectral import calculate_vdos, vibrational_density_of_states


def legacy_filon(corr, time_step, width=None):
    """Scalar reference for the original Filon grid and quadrature."""
    n = (len(corr) - 1) // 2 * 2
    values = np.array(corr[:n + 1], dtype=float)
    dt = time_step / PS_TO_FS
    if width is not None:
        values *= np.exp(-0.5 * (0.5 * width * np.arange(n + 1) / n) ** 2)
    dw = 2 * np.pi / (n * dt)
    result = []
    for nu in range(n + 1):
        theta = nu * dw * dt
        s = np.sin(theta)
        c = np.cos(theta)
        if abs(theta) < 1e-9:
            a, b, g = 0., 2 / 3, 4 / 3
        else:
            a = (theta**2 + theta * s * c - 2 * s**2) / theta**3
            b = 2 * (theta * (1 + c**2) - 2 * s * c) / theta**3
            g = 4 * (s - theta * c) / theta**3
        even = np.sum(values[::2] * np.cos(theta * np.arange(0, n + 1, 2)))
        even -= 0.5 * (values[0] + values[n] * np.cos(theta * n))
        odd = np.sum(values[1::2] * np.cos(theta * np.arange(1, n, 2)))
        result.append(2 * dt * (a * values[n] * np.sin(nu * dw * n * dt)
                                + b * even + g * odd))
    return np.arange(n + 1) * dw / (2 * np.pi) * HZ_TO_CM, np.array(result)


class SpectralTests(unittest.TestCase):
    def test_legacy_filon(self):
        for size in (31, 32):
            corr = np.random.default_rng(size).normal(size=size)
            original = corr.copy()
            for width in (None, 0., 3.):
                with self.subTest(size=size, width=width):
                    actual = calculate_vdos(corr, 0.1, method="filon",
                                            gaussian_filter_width=width)
                    expected = legacy_filon(corr, 0.1, width)
                    assert_array_equal(actual[0], expected[0])
                    assert_allclose(actual[1], expected[1], atol=1e-15)
            assert_array_equal(corr, original)

    def test_legacy_fft(self):
        for size in (31, 32):
            corr = np.random.default_rng(size).normal(size=size)
            for width in (None, 3.):
                values = corr.copy()
                if width is not None:
                    values *= np.exp(-0.5 * (0.5 * width * np.arange(size)
                                            / (size - 1)) ** 2)
                values[[0, -1]] *= 0.5
                frequency, spectrum = calculate_vdos(
                    corr, 0.1, gaussian_filter_width=width)
                assert_array_equal(frequency, np.fft.rfftfreq(size, 0.1 / PS_TO_FS) * HZ_TO_CM)
                assert_array_equal(spectrum, 2 * (0.1 / PS_TO_FS) * np.fft.rfft(values).real)

    def test_custom_grid_and_wrapper(self):
        for size in (101, 1000):
            corr = np.exp(-np.arange(size) / 100)
            for maximum, count in ((5000., 101), (5023., 101), (0., 1), (49., 1)):
                options = dict(method="filon", frequency_step=50., frequency_max=maximum)
                frequency, spectrum = vibrational_density_of_states(corr, 0.1, **options)
                assert_array_equal(frequency, np.arange(count) * 50.)
                self.assertEqual(spectrum.shape, frequency.shape)
                self.assertTrue(np.isfinite(spectrum).all())
                wrapped = calculate_vdos(corr, 0.1, **options)
                assert_array_equal(wrapped[0], frequency)
                assert_array_equal(wrapped[1], spectrum)

    def test_single_grid_options(self):
        corr = np.ones(101)
        nyquist = HZ_TO_CM / (2 * 0.1 / PS_TO_FS)
        frequency, spectrum = calculate_vdos(corr, 0.1, method="filon", frequency_step=1000.)
        assert_array_equal(frequency, np.arange(int(nyquist // 1000) + 1) * 1000.)
        self.assertEqual(frequency.shape, spectrum.shape)
        legacy_frequency, legacy_spectrum = calculate_vdos(corr, 0.1, method="filon")
        frequency, spectrum = calculate_vdos(corr, 0.1, method="filon", frequency_max=10000.)
        mask = legacy_frequency <= 10000.
        assert_allclose(frequency, legacy_frequency[mask])
        assert_allclose(spectrum, legacy_spectrum[mask], atol=1e-15)

    def test_matching_frequencies_and_damping(self):
        corr = np.random.default_rng(3).normal(size=101)
        step = HZ_TO_CM / (100 * 0.1 / PS_TO_FS)
        for width in (None, 4.):
            old_frequency, old_spectrum = calculate_vdos(
                corr, 0.1, method="filon", gaussian_filter_width=width)
            frequency, spectrum = calculate_vdos(
                corr, 0.1, method="filon", gaussian_filter_width=width,
                frequency_step=step, frequency_max=step * 40)
            assert_allclose(frequency, old_frequency[:41])
            assert_allclose(spectrum, old_spectrum[:41], atol=1e-14)

    def test_invalid_parameters(self):
        for name, values in (("frequency_step", (0., -1., np.nan, np.inf, -np.inf)),
                             ("frequency_max", (-1., np.nan, np.inf, -np.inf))):
            for value in values:
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        calculate_vdos(np.ones(11), 0.1, method="filon", **{name: value})
        for options in ({"frequency_step": 50.}, {"frequency_max": 0.},
                        {"frequency_step": 50., "frequency_max": 5000.}):
            with self.assertRaisesRegex(ValueError, "filon"):
                calculate_vdos(np.ones(11), 0.1, **options)
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            calculate_vdos(np.ones(11), 1., method="filon", frequency_max=HZ_TO_CM * 1000)

    def test_known_peak_and_time_units(self):
        dt_fs = 0.1
        target = 1000.
        times_ps = np.arange(10001) * dt_fs / PS_TO_FS
        corr = np.cos(2 * np.pi * target / HZ_TO_CM * times_ps)
        for width in (None, 4.):
            frequency, spectrum = calculate_vdos(
                corr, dt_fs, method="filon", frequency_step=10.,
                frequency_max=2000., gaussian_filter_width=width)
            self.assertEqual(frequency[np.argmax(spectrum)], target)
            if width is None:
                self.assertAlmostEqual(spectrum[100], times_ps[-1], delta=0.01)

    def test_zero_and_small_frequencies(self):
        # Constant VACF: transform is 2 sin(omega T) / omega, with limit 2T.
        dt_fs = 1.
        duration_ps = 100 * dt_fs / PS_TO_FS
        for step in (1e-8, 0.1):
            frequency, spectrum = calculate_vdos(
                np.ones(101), dt_fs, method="filon", frequency_step=step,
                frequency_max=5 * step)
            expected = 2 * duration_ps * np.sinc(2 * frequency / HZ_TO_CM * duration_ps)
            self.assertTrue(np.isfinite(spectrum).all())
            # The unchanged trigonometric coefficients have cancellation
            # error at small theta outside their existing zero-limit branch.
            assert_allclose(spectrum, expected, rtol=1e-6, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
