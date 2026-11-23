# Normal-mode velocity projection

`PhononModes` uses `isse.structures.Atoms` and requires no additional ISSE
dependencies. Install ALAMODE 2.0 and its `anphon` Python package separately
if using the ALAMODE adapter. Neither ASE nor ALAMODE is required for legacy
files or directly supplied arrays. Numba is a required ISSE runtime dependency
and the only projection contraction implementation; there is no NumPy fallback
or backend-selection argument.

## Data contract

`isse.PhononModes(primitive, qpoints, frequencies, eigenvectors)` is independent
of the phonon solver. Its fields are:

| Field | Shape | Convention |
|---|---|---|
| `primitive` | ISSE `Atoms`, `nprim` atoms | Å, masses in amu; cell vectors by row |
| `qpoints` | `(nq, 3)` | Reduced reciprocal coordinates; phase includes 2π |
| `frequencies` | `(nq, nmodes)` | cm⁻¹; imaginary frequencies are negative |
| `eigenvectors` | `(nq, nmodes, nprim, 3)` | Complex, dimensionless, mode-first |

Eigenvectors are orthonormal eigenvectors of the **mass-weighted dynamical
matrix**, not Cartesian displacement patterns already divided by sqrt(mass).
At each q their flattened rows satisfy `e.conj() @ e.T = identity`. Validation
uses absolute tolerance 1e-6; it never silently normalizes input. Fewer than
`3*nprim` modes are allowed. Shapes, finiteness, positive masses, and a
nonsingular primitive cell are checked. Constructor inputs are copied.
The primitive must contain explicit `masses`; they are never silently inferred
from element symbols. ISSE's POSCAR parser populates masses from its element
table; replace them explicitly if ALAMODE uses different masses or isotopes.

**Ordering and gauge matter.** `primitive` must use the phonon source's exact
atom order, cell orientation, origin, and basis images. Do not sort atoms or
wrap/reimage primitive sites independently of the eigenvectors. The convention
is `u_li ~ e_i exp(+2πi q.R_l) / sqrt(m_i)`, where `R_l` is an integer primitive
cell translation. Full-position (`R_l + tau_i`) gauge eigenvectors must be
converted by their source adapter before construction. The projection engine
uses

```text
qdot[q, mode] = sum_li sqrt(m_i) e[q, mode, i]* . v_li
               * exp(-2πi q.R_l) / sqrt(Ncells)
```

The star is complex conjugation. No extra basis-position phase is inserted.
The explicit-data path maps reference atoms to this supplied primitive,
checks species, masses, integer supercell geometry, site populations and
absence of duplicate sites modulo supercell translations. Trajectory atoms
must remain in reference order, with velocities in Å/fs. The reference must
be the equilibrium structure, not a thermally distorted MD frame.

Both projection workflows validate **every frame** before batching: its species
sequence must match the reference, and its finite simulation cell must agree
within 1e-8 Å absolute tolerance (no relative tolerance). Variable-cell
trajectories must be transformed explicitly before either projection.
Projection masses come from the reference or, if absent there, from the first
frame. Later frame mass metadata is optional, but every supplied mass array
must be finite, positive, correctly shaped and agree with projection masses
within absolute and relative tolerances of 1e-6. You do not need masses in both
the reference and every frame.

**Species checks cannot detect permutations between atoms of the same chemical
element.** Matching symbols, masses, or Parseval norms do not establish atomic
identity. The caller must preserve reference atom identities/order throughout
the trajectory (for example by maintaining a stable atom-ID ordering). No
reordering is inferred or performed.

## Legacy `.evec` files

Existing positional and keyword calls and the default four outputs are unchanged:

```python
from isse import project_velocities

q, qdot2, atomic_norms, errors = project_velocities(
    trajectory, reference, "modes.evec", batch_size=100,
)
```

The file-only path retains the historical spglib-derived primitive ordering.
**Legacy text files contain no primitive basis positions or per-site species**;
ISSE cannot prove that spglib's ordering matches the file. Parseval alone also
cannot detect all ordering errors. Prefer explicit source-ordered data when
that information is available:

```python
from isse import PhononModes
from isse.io import parse_poscar

primitive = parse_poscar("source_ordered_primitive.POSCAR")  # verified against ALAMODE
modes = PhononModes.from_evec("modes.evec", primitive=primitive)
q, qdot2, norms, errors = project_velocities(trajectory, reference, modes=modes)
```

`from_evec` checks the lattice against the supplied primitive. Text eigenvalues
are signed squared frequencies in Rydberg units: conversion is
`sign(omega2) * sqrt(abs(omega2)) * 109737.3156816` cm⁻¹. Frequencies are metadata;
they are not used to scale velocity projections. Truncated mode blocks now
raise an error rather than leaving uninitialized eigenvectors.

## ALAMODE 2.0 Python API / Gamma-only example

```python
import anphon
from isse import PhononModes, project_velocities
from isse.io import parse_gpumd_dump, parse_poscar

ph = anphon.Phonon("harmonic.h5")
primitive = parse_poscar("source_ordered_primitive.POSCAR")
# IMPORTANT: verify primitive against harmonic.h5 /PrimitiveCell metadata:
# lattice (check units), fractional coordinates including basis images,
# atomic kinds/elements and their order. Set primitive masses to the masses
# used by anphon, including any general={"MASS": ...} overrides/isotopes.
# An arbitrary symmetry-equivalent POSCAR is not sufficient.
modes = PhononModes.from_anphon(
    ph, [[0.0, 0.0, 0.0]], primitive=primitive,
    source_order_confirmed=True,
)
reference = parse_poscar("supercell.POSCAR")
trajectory = parse_gpumd_dump("production.xyz")
q, qdot2, norms, deficits, qdot = project_velocities(
    trajectory, reference, modes=modes,
    parseval_tolerance=None,  # Gamma is incomplete for a multi-cell supercell
    return_complex=True,
    batch_size=50,
)
```

The adapter makes one call to `ph.frequencies(qpoints, eigenvectors=True)`.
The public API does **not** provide primitive ordering/mass metadata; hence
`primitive` and the explicit `source_order_confirmed=True` acknowledgement
are required. ISSE does not infer order from a matching atom count. The caller
must check the source metadata; this adapter does not automatically read the
force-constant HDF5 file. The source primitive's masses and mapped MD masses
must agree (relative tolerance 1e-6).

There are no ISSE-generated intermediate legacy files. Note that the verified
ALAMODE implementation itself runs its solver in a temporary directory and
reads an `.evec.h5` internally. It is currently serial-only and does not apply
LO–TO splitting exactly at Gamma, even when nonanalytic corrections are enabled.

## Direct NumPy arrays

```python
import numpy as np
from isse import PhononModes, project_velocities
from isse.structures import Atoms

primitive = Atoms(
    symbols=["H"], cell=np.eye(3), positions=np.zeros((1, 3)), masses=np.ones(1),
)
modes = PhononModes(
    primitive=primitive,
    qpoints=np.array([[0., 0., 0.]]),
    frequencies=np.zeros((1, 3)),
    eigenvectors=np.eye(3, dtype=complex).reshape(1, 3, 1, 3),
)
reference = Atoms(
    symbols=["H"], cell=np.eye(3), positions=np.zeros((1, 3)),
    masses=np.ones(1), velocities=np.array([[1., 2., 3.]]),
)
q, squared, norms, errors, complex_velocity = project_velocities(
    [reference], reference, modes=modes, return_complex=True,
)
# complex_velocity[0, 0] == [1, 2, 3]; squared[0, 0] == [1, 4, 9]
# norms[0] == 14; errors[0] == 0
```

## Outputs, completeness and interpretation

Default outputs remain `(qpoints, qdot2, atomic_norms, parseval_errors)`.
`return_complex=True` appends `qdot`, preserving phase across all batches. Both modal arrays have shape `(nframes, nq, nmodes)`.

- `qdot`: sqrt(amu) Å/fs, complex128.
- `qdot2 = abs(qdot)**2`: amu Å²/fs², float64.
- `atomic_norms = sum_atoms mass * abs(velocity)**2`: amu Å²/fs².
- `parseval_errors`: dimensionless relative norm difference; NaN for a frame
  with zero velocity norm. For a partial basis this measures the missing norm,
  not an error in the contraction.

A full basis requires all `3*nprim` modes at all `Ncells` orthogonal,
commensurate q-points. The explicit-data path checks these conditions when
Parseval validation is enabled (default tolerance 1e-6), then validates each
frame. For partial q/mode selections, including Gamma-only in a supercell,
**pass `parseval_tolerance=None`**. No rescaling recovers the missing norm.
Noncommensurate projections can also be explicitly requested with validation
disabled, but do not form a periodic-supercell normal-mode decomposition.

Disabling Parseval tolerance disables only the completeness/equality test,
**not numerical-integrity checks**. Non-finite squared moduli, atomic norms,
requested complex amplitudes, summed norms or nonzero-frame relative errors
raise `RuntimeError`, including overflow from finite inputs. A NaN Parseval
error is intentional only for a finite zero atomic norm with zero modal norm;
such frames are excluded from the maximum relative error. All-zero batches
are valid and keep their NaN error convention.

Half the squared modal velocity, multiplied by `AMU_A2_FS2_TO_EV`, is a modal
**kinetic** energy in eV. It is not the total harmonic energy: velocity
projection alone does not compute normal coordinates or potential energies.
Displacement projection, described below, returns normal coordinates but does
not compute energies automatically. `calculate_temperature` also
accepts `modes=` with the same Parseval controls. It converts
mean squared velocities to kinetic-equivalent temperatures; a partial basis
must not be interpreted as a reconstructed system temperature. Its thermal
average retains the legacy acoustic-first assumption **only for complete
per-q mode lists** (`3*nprim` modes), excluding the first three entries at
Gamma. For arbitrary partial mode lists, original acoustic-mode identities
are unavailable: all supplied modes are retained in `mean_mode_temperature`,
with a warning if Gamma is present. This avoids silently removing optical
modes. That partial-set mean is a kinetic-equivalent average of the selected
modes, not an acoustic-filtered or reconstructed system temperature.

The eigenvector's arbitrary global phase changes individual complex modal
velocities. Fix the eigenbasis across frames/runs when analyzing coherence.
Within a degenerate subspace even the individual eigenvectors are arbitrary
up to unitary rotations. The summed `abs(qdot)**2` over that subspace is
invariant; individual amplitudes/phases are not. Compare subspace-invariant
quantities or explicitly align bases before comparing independent solvers.
Trajectory sampling, not the MD integration step, sets the projection time axis.

## Displacement projection / complex normal coordinates

`project_displacements` and `project_velocities` share mapping, coefficients,
the batched Numba kernel and Parseval controls in `helpers/mode_projection.py`.
Each public workflow supplies its own vector-batch iterator; neither calls the
other, and the shared engine never selects a physical input field. It accepts the same legacy
`.evec` file or `PhononModes` inputs (including `from_anphon` and direct arrays),
uses only ISSE structures, and does **not** require frame velocities.

With an equilibrium supercell `reference` and the source-ordered `modes` above:

```python
from isse import project_displacements

q, q2, displacement_norms, deficits, Q = project_displacements(
    trajectory, reference, modes=modes,
    return_complex=True,
    parseval_tolerance=None,  # for Gamma-only or other incomplete bases
    position_mode="minimum_image",
    batch_size=50,
)
# Q.shape == (nframes, nq, nmodes); q2 == abs(Q)**2
```

The normal coordinate is

```text
u_li(t) = r_li(t) - r_li(reference)  # with image handling chosen below
Q[q, mode] = sum_li sqrt(m_i) e[q, mode, i]* . u_li
            * exp(-2πi q.R_l) / sqrt(Ncells)
```

The default four outputs are `(qpoints, q2, atomic_norms, parseval_errors)`;
`return_complex=True` appends `Q`, retaining the sign and complex phase rather
than only a squared amplitude. Units:

- `Q`: sqrt(amu) Å, complex128, shape `(nframes, nq, nmodes)`.
- `q2 = abs(Q)**2`: amu Å², same shape, float64; **not an energy**.
- `atomic_norms = sum_atoms mass * abs(displacement)**2`: amu Å²,
  shape `(nframes,)`.
- `parseval_errors`: dimensionless norm difference; NaN for zero displacement.

Full-basis Parseval is `sum(abs(Q)**2) == sum(mass * abs(u)**2)`. Its validation
is enabled by default; disable it explicitly for partial bases, without any
renormalization. Source ordering, masses, phases, and degenerate-subspace
limitations are identical to those for velocity projection.

### Reference and periodic-image handling

Use a **fixed equilibrium supercell**, not a thermally distorted initial frame
unless that is deliberately the displacement origin. Frame atom order and
symbols must match the reference. Every frame cell must equal the reference
cell within an absolute tolerance of 1e-8 Å. Variable-cell trajectories raise
an error: affine strain removal and a co-deforming basis are not silently
assumed or implemented. No COM translation, rotation, or trajectory-average
position is subtracted.

Choose one explicit `position_mode`, consistently across all frames:

- **`"minimum_image"` (default):** difference `frame.positions - reference.positions`,
  minimum-imaged with the **supercell**, not the primitive cell. Handles wrapped
  boundary crossings for small vibrational excursions. Assumes each atom stays
  within half the supercell in fractional coordinates of its reference image.
  There is no temporal unwrapping. The existing fractional-rounding MIC helper
  is approximate for strongly skewed/non-reduced cells.
- **`"unwrapped"`:** require `frame.unwrapped_positions` for every frame. Subtract
  `reference.unwrapped_positions` if supplied, otherwise `reference.positions`.
  No MIC is applied. The caller must align the trajectory's unwrapped origin
  with the reference images; diffusion/large translations are retained.
- **`"cartesian"`:** subtract `positions` directly, without MIC or unwrapping.
  Use this only for already image-aligned Cartesian positions.

For example, if the trajectory includes consistent unwrapped positions:

```python
q, q2, norms, errors, Q = project_displacements(
    trajectory, reference, "modes.evec",  # or modes=modes
    position_mode="unwrapped", return_complex=True,
    parseval_tolerance=None,  # only if the supplied basis is incomplete
)
```

With a fixed reference/basis and continuous, consistent images, `dQ/dt` is the
complex modal velocity returned by `project_velocities` in the **same** basis.
Minimum-image branch changes can make `Q` discontinuous and break that derivative
relation. Consistent eigenvector phases and aligned degenerate bases are essential
when comparing Q with qdot or comparing independent calculations. The sums of
`abs(Q)**2` over degenerate subspaces are invariant under unitary basis rotations.

## Verification and tests

Conventions were checked against official ALAMODE documentation and source at
commit [`57346cd6e00bd6277ef5a4a28f31e7c7fb2a932f`](https://github.com/ttadano/alamode/tree/57346cd6e00bd6277ef5a4a28f31e7c7fb2a932f):

- `docs/source/anphon_python.rst`, `python/anphon/_api.py`: API, units,
  mode-first layout, no public primitive accessor.
- `anphon/system.cpp`, `docs/source/hdf5_format.rst`: PrimitiveCell coordinates,
  kinds and ordering loaded from force-constant HDF5 metadata.
- `anphon/dynamical.cpp` (`calc_analytic_k`): positive translation Bloch phase,
  mass-divided force constants; normalized Hermitian eigenproblem.
- `anphon/write_phonons.cpp`: legacy signed omega² and cell-translation phase
  in generated mode patterns.

Synthetic tests cover analytic complex Fourier projections using the required
Numba kernel, full Parseval, Gamma/partial modes, explicit atomic order, malformed data and
legacy equivalence. Displacement regressions also cover wrapped boundary
crossings, explicit unwrapped origins, large excursions, triclinic cells,
lazy batch invariance, zero displacement, fixed-cell/atom-order validation and
an analytic coherent oscillation with matching modal velocities.
The adapter contract is tested with a stub; no live
ALAMODE installation, expensive phonon calculation or MD simulation is required.
Run with ISSE's standard dependencies installed:

```bash
NUMBA_NUM_THREADS=2 python -m unittest discover -s tests -v
```
