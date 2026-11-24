# Vibrational spectra and Filon frequency grids

`isse.spectral.vibrational_density_of_states` accepts a trajectory, a VACF
array, or a `(time, vacf)` tuple. Pass the sampling interval explicitly as
`time_step` in **fs** (a tuple's time axis is ignored). Internally the transform
uses ps; returned frequencies are in **cm⁻¹**.

```python
from isse.spectral import vibrational_density_of_states

frequency, vdos = vibrational_density_of_states(
    trajectory,
    time_step=0.1,
    atom_groups=atom_groups,
    group=1,
    mass_weighted=False,
    remove_com=True,
    batch_size=1000,
    gaussian_filter_width=None,
    method="filon",
    frequency_step=50.0,
    frequency_max=5000.0,
)
```

This evaluates Filon quadrature directly at 101 frequencies from 0 to 5000
cm⁻¹, regardless of the VACF sample count. There is no spectrum interpolation,
zero-padding, or added smoothing. Output frequency and spectrum arrays have
identical shapes. `calculate_vdos` exposes the same keyword-only grid options
for precomputed VACFs.

## Grid defaults and limits

- Both options omitted (`None`): preserve the original Filon grid and results,
  including its historical above-Nyquist frequency range. FFT is unchanged.
- Only `frequency_step`: use this positive, finite spacing and stop at the
  last multiple at or below Nyquist.
- Only `frequency_max`: retain the original Filon spacing, `HZ_TO_CM / T_ps`,
  but stop at the last multiple at or below the requested maximum.
- Both supplied: start at zero and include each spacing multiple not exceeding
  the maximum. A maximum of zero returns just the zero-frequency value.

`frequency_max` must be finite and non-negative. Custom grids are supported
only with `method="filon"`; supplying either option to FFT raises `ValueError`.
Explicit custom maxima above Nyquist also raise `ValueError`. In these unit
conventions the Nyquist wavenumber is
`HZ_TO_CM / (2 * time_step / PS_TO_FS)`. Frequencies above it do not represent
independent physical information from the sampled VACF.

## Grid spacing is not spectral resolution

A finer grid samples the finite-duration cosine transform more densely, but
**does not improve intrinsic spectral resolution**, whose scale is set by
inverse VACF duration (`HZ_TO_CM / T_ps` in cm⁻¹). A short VACF still produces
broad or unresolved peaks even on a finely sampled grid.

Filon integration retains an even number of time intervals: for an even
sample count it drops the final VACF sample, as before. `T_ps` is the retained
duration. Gaussian damping, endpoint treatment, spectral normalization, and
the existing Filon coefficients are unchanged. The existing small-angle
coefficient evaluation retains its floating-point cancellation limitations;
custom grids do not change the quadrature formula. Runtime scales as the
number of retained samples times the number of requested frequencies.
