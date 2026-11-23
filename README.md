# ISSE

**Ichnusa Solid State Environment** is a Python toolkit for solid-state physics and atomistic simulation workflows.

The project provides lightweight atomistic data structures, lazy trajectory I/O, format conversion utilities, periodic-boundary helpers, radial distribution functions, and phonon-mode velocity projection workflows.

> Status: early development. APIs are usable but not yet guaranteed stable.

## Features

- `Atoms`: a compact container for one atomistic configuration.
- `Trajectory`: lazy, file-backed access to single-file or multi-file trajectories.
- Native readers and writers for selected VASP, LAMMPS, and GPUMD/extended-XYZ files.
- Explicit unit handling for LAMMPS `metal` and `real` unit styles.
- Dependency-light conversion utilities without ASE.
- Phonon velocity/displacement projection and modal temperatures from legacy ALAMODE files, NumPy arrays, or the ALAMODE 2.0 Python API; optional complex modal velocities and normal coordinates.
- Radial distribution function calculation using Numba.
- Velocity autocorrelation and vibrational density of states utilities.
- Periodic-boundary utilities for wrapping, unwrapping, and minimum-image distances.

## Installation

ISSE requires Python 3.11 or newer.

From a local checkout:

```bash
pip install .
```

or, with `uv`:

```bash
uv sync
```

Runtime dependencies are:

- `numpy`
- `spglib`
- `numba`

Numba is required at runtime for the projection, RDF, and Filon implementations.
Normal-mode projection uses Numba exclusively; there is no NumPy fallback or
backend-selection argument.

Source-independent phonon data uses ISSE's own `Atoms` structure and needs no
additional dependencies. ALAMODE is installed separately and is not required
for legacy files or NumPy-array phonon inputs; ASE is not required for any
phonon projection interface.

## Internal unit convention

ISSE converts parsed data to a consistent internal unit system:

| Quantity | Internal unit |
| --- | --- |
| Cell vectors | Å |
| Positions / unwrapped positions | Å |
| Velocities | Å/fs |
| Masses | amu |
| Forces | eV/Å |
| Energies used internally | eV |
| Time metadata, when interpreted | fs |
| Temperature | K |

This convention is important when constructing `Atoms` objects manually or when passing data to writers.

## Supported file formats

| Format name | Read | Write | Object | Notes |
| --- | --- | --- | --- | --- |
| `poscar`, `vasp` | yes | yes | `Atoms` | POSCAR-like single configuration. Positive scaling factors only. Direct and Cartesian positions supported. |
| `lammps-data` | yes | yes | `Atoms` | `atom_style atomic`; optional `Masses` and `Velocities`; orthogonal or restricted-triclinic cells. |
| `lammps-dump`, `lammpstrj` | yes | yes | `Trajectory` | Lazy multi-frame dump reader; supports `x/y/z`, `xs/ys/zs`, `xu/yu/zu`, velocities, forces, masses, ids, types. |
| `gpumd-dump`, `extxyz` | yes | yes | `Trajectory` | GPUMD/extended-XYZ style with `Lattice` and `Properties`. |

Aliases accepted by the converter include `data`, `dump`, `lammps_data`, `lammps_dump`, `xyz`, and `gpumd`.

## LAMMPS units

LAMMPS files must be read or written with an explicit unit style: `"metal"` or `"real"`. ISSE intentionally avoids silent assumptions.

### `units="metal"`

| Quantity | LAMMPS unit | ISSE internal conversion |
| --- | --- | --- |
| Positions/cell | Å | unchanged |
| Velocities | Å/ps | divided by 1000 on read; multiplied by 1000 on write |
| Forces | eV/Å | unchanged |
| Masses | amu | unchanged |

### `units="real"`

| Quantity | LAMMPS unit | ISSE internal conversion |
| --- | --- | --- |
| Positions/cell | Å | unchanged |
| Velocities | Å/fs | unchanged |
| Forces | kcal/mol/Å | multiplied by `0.0433641` on read; divided by it on write |
| Masses | amu | unchanged |

For LAMMPS data files, the parser first tries to infer the unit style from the first line, for example an ISSE-written header such as `ISSE LAMMPS data file (units metal)`. If it cannot infer the style, pass `units` explicitly.

LAMMPS data files do not store element symbols directly. Pass `symbols=[...]` in atom-type order when symbols cannot be inferred from masses or when precise labels are required:

```python
from isse.io.parse_lammps import parse_lammps

atoms = parse_lammps(
    "structure.data",
    format="data",
    units="metal",
    symbols=["Si", "O"],  # type 1 -> Si, type 2 -> O
)
```

## Basic usage

### Read a POSCAR

```python
from isse.io.parse_vasp import parse_poscar

atoms = parse_poscar("POSCAR")
print(len(atoms))
print(atoms.cell)
print(atoms.positions)
```

### Read a lazy LAMMPS trajectory

```python
from isse.io.parse_lammps import parse_lammps

trajectory = parse_lammps(
    "dump.lammpstrj",
    format="dump",
    units="metal",
    symbols=["Si"],
)

print(len(trajectory))      # number of frames
frame0 = trajectory[0]      # only this frame is read from disk
for frame in trajectory[:10]:
    print(frame.positions.shape)
```

### Concatenate trajectory segments lazily

Trajectory segments read with the same parser can be concatenated without
loading frames into memory:

```python
from isse.io.parse_gpumddump import parse_gpumd_dump

trajectory = sum([
    parse_gpumd_dump("traj2ps.xyz"),
    parse_gpumd_dump("traj10ps.xyz"),
    parse_gpumd_dump("traj30ps.xyz"),
])

print(len(trajectory))
print(trajectory.paths)     # all backing files
frame = trajectory[-1]      # only this frame is read
```

### Convert files

The conversion API uses ISSE's native readers and writers:

```python
from isse.convert import convert

convert(
    "dump.lammpstrj",
    "trajectory.xyz",
    infile_type="lammps-dump",
    outfile_type="gpumd-dump",
    input_units="metal",
    symbols=["Si"],
)
```

Select one frame when writing a single-frame output from a trajectory:

```python
convert(
    "dump.lammpstrj",
    "frame10.POSCAR",
    infile_type="lammps-dump",
    outfile_type="poscar",
    input_units="metal",
    symbols=["Si"],
    frame=10,
    fractional=True,
)
```

Replicate before writing:

```python
convert(
    "POSCAR",
    "supercell.data",
    infile_type="poscar",
    outfile_type="lammps-data",
    output_units="metal",
    replicate=(2, 2, 2),
)
```

### Write files directly

```python
from isse.io.write_vasp import write_poscar
from isse.io.write_lammps import write_lammps_dump

write_poscar("POSCAR.out", atoms, direct=True)
write_lammps_dump("out.lammpstrj", trajectory, units="metal", fractional=False)
```

## Data model

### `Atoms`

`Atoms` represents one atomistic configuration:

```python
from isse.structures import Atoms
```

Main fields:

- `symbols`: list of chemical symbols.
- `cell`: `(3, 3)` array, lattice vectors by row.
- `positions`: `(n_atoms, 3)` Cartesian positions in Å.
- `unwrapped_positions`: optional `(n_atoms, 3)` array in Å.
- `velocities`: optional `(n_atoms, 3)` array in Å/fs.
- `masses`: optional `(n_atoms,)` array in amu.
- `forces`: optional `(n_atoms, 3)` array in eV/Å.
- `arrays`: extra per-atom arrays such as `id`, `type`, or `groups`.
- `info`: frame/configuration metadata such as `timestep` or POSCAR header.

Cell convention:

```text
cartesian_position = fractional_position @ cell
```

### `Trajectory`

`Trajectory` provides lazy sequence semantics over frames stored in a file:

- `trajectory[i]` reads one frame.
- `trajectory[start:stop]` returns another lazy trajectory view.
- Iteration reads frames on demand.
- Memory usage scales with the number of frame offsets, not with the full trajectory size.

## Radial distribution function

```python
from isse.radial_distribution import calculate_rdf

rdf = calculate_rdf(trajectory, r_max=8.0, n_bins=400)
print(rdf["r"])       # shape (n_bins,)
print(rdf["g_r"])     # shape (n_frames, n_bins)

partial_rdf = calculate_rdf(trajectory, r_max=8.0, n_bins=400, partial=True)
print(partial_rdf["partial_g_r"][("O", "Si")])  # shape (n_frames, n_bins)
```

The RDF implementation assumes periodic boundary conditions and returns one RDF per frame for trajectory inputs. Passing a single `Atoms` object returns one-dimensional arrays for that single configuration. RDF histograms are computed with Numba, parallelized over frames.

## Phonon modal temperatures

```python
from isse.io.parse_gpumddump import parse_gpumd_dump
from isse.io.parse_vasp import parse_poscar
from isse.phonon_temperatures import calculate_temperature

trajectory = parse_gpumd_dump("production.xyz")
reference = parse_poscar("POSCAR")

results = calculate_temperature(
    trajectory=trajectory,
    reference_atoms=reference,
    evec_filepath="modes.evec",
    batch_size=100,
)

print(results["qpoints"])
print(results["mode_temperatures"])
print(results["mean_mode_temperature"])
```

Requirements for this workflow:

- The trajectory frames must include velocities in Å/fs.
- Masses must be available in the reference or first frame. Frame masses are otherwise optional; any supplied masses must agree with the projection masses.
- Every frame must match the reference species sequence and fixed cell (absolute tolerance 1e-8 Å). Species checks cannot detect permutations of same-element atoms; stable atomic identity/order is the caller's responsibility.
- The ALAMODE `.evec` file is read by `isse.io.parse_alamode.read_alamode_evec`.
- A Parseval consistency check is performed by default; pass `parseval_tolerance=None` to disable completeness checking, not numerical-integrity checks. Non-finite results always raise an error; valid zero-norm frames retain NaN relative errors.
- For partial Gamma mode lists, `calculate_temperature` retains all supplied modes and logs a warning instead of guessing acoustic indices. Complete mode lists keep the legacy first-three-acoustic assumption.

### Source-independent phonons and complex modal velocities

Use `PhononModes` to supply arrays directly, read an `.evec` with an explicitly
ordered primitive, or adapt `anphon.Phonon` without ISSE-generated intermediate
files. Both `project_velocities` and `calculate_temperature` accept `modes=`.

```python
from isse import PhononModes, project_velocities

# ph is anphon.Phonon("harmonic.h5"). primitive is an isse.structures.Atoms object
# whose order, basis images and masses have been checked against ALAMODE.
modes = PhononModes.from_anphon(
    ph, [[0, 0, 0]], primitive=primitive, source_order_confirmed=True,
)
q, squared, norms, deficits, complex_velocity = project_velocities(
    trajectory, reference, modes=modes,
    return_complex=True, parseval_tolerance=None,
)
```

Gamma-only projections in a multi-cell supercell are incomplete: disable
Parseval validation explicitly; no partial-basis renormalization is applied.
Complex velocities have units sqrt(amu) Å/fs; their squared moduli are not total
harmonic energies. Eigenvector phases and degenerate-subspace rotations affect
individual modal amplitudes.

See [the projection guide](docs/phonon_projection.md) for complete examples,
array shapes, normalization, phase/ordering conventions, source verification,
partial-basis interpretation, and test instructions.

### Displacement projection

Project deviations from the same fixed equilibrium reference onto normal
coordinates, using the same phonon data and shared Numba projection engine:

```python
from isse import project_displacements

q, q2, norms, deficits, Q = project_displacements(
    trajectory, reference, modes=modes,
    return_complex=True, parseval_tolerance=None,  # Gamma-only / partial basis
    position_mode="minimum_image",
)
```

`Q` has units sqrt(amu) Å; `q2 = abs(Q)**2` is not an energy. Both have shape
`(nframes, nq, nmodes)`. Velocities are not required. Default displacements
are minimum-image differences from the reference, using the supercell.
Use `position_mode="unwrapped"` for explicit, image-aligned unwrapped positions,
or `"cartesian"` for direct differences without image correction. Variable
cells are rejected; no COM translation, rotation or strain is removed.
See the [projection guide](docs/phonon_projection.md#displacement-projection--complex-normal-coordinates)
for image conventions, completeness checks and phase limitations.

## Development notes

Public workflows, including `PhononModes`, `project_velocities`,
`project_displacements`, and `calculate_temperature`, are also available from the top-level `isse` package.

Recommended checks while developing:

```bash
python -m compileall src/isse
```

If tests are available in your checkout:

```bash
python -m pytest
```

## License

See `LICENSE`.
