"""Prepare trajectory velocities for shared Numba normal-mode projection."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .helpers.mode_projection import _iter_validated_frames, _project_vectors
from .phonon_modes import PhononModes
from .structures import Atoms, Trajectory


def _iter_velocity_batches(
    trajectory: Trajectory,
    reference_atoms: Atoms,
    batch_size: int,
) -> Iterator[NDArray[np.float64]]:
    """Yield finite float64 velocity batches, shape (batch_frames, natoms, 3)."""
    natoms = len(reference_atoms)
    batch: list[NDArray[np.float64]] = []
    for iframe, atoms in enumerate(_iter_validated_frames(trajectory, reference_atoms)):
        velocities = atoms.velocities
        if velocities is None:
            raise ValueError(f"Trajectory frame {iframe} does not contain velocities.")
        if velocities.shape != (natoms, 3):
            raise ValueError(
                f"Expected velocities with shape ({natoms}, 3) in frame "
                f"{iframe}, found {velocities.shape}."
            )
        if not np.isfinite(velocities).all():
            raise ValueError(f"Non-finite velocities in trajectory frame {iframe}.")
        batch.append(velocities)
        if len(batch) == batch_size:
            yield np.ascontiguousarray(batch, dtype=np.float64)
            batch.clear()
    if batch:
        yield np.ascontiguousarray(batch, dtype=np.float64)


def project_velocities(
    trajectory: Trajectory,
    reference_atoms: Atoms,
    evec_filepath: str | Path | PhononModes | None = None,
    batch_size: int = 100,
    parseval_tolerance: float | None = 1e-6,
    *,
    modes: PhononModes | None = None,
    return_complex: bool = False,
) -> tuple:
    """Project trajectory velocities onto phonon eigenmodes using Numba.

    The reference is used for masses, primitive mapping and frame-independent
    coefficients. Velocity validation/batching is done here; the shared engine
    in helpers.mode_projection performs contraction and Parseval validation.
    Numba is required and there is no alternative projection backend.

    Parameters
    ----------
    trajectory : Trajectory
        Lazy trajectory yielding Atoms in reference order, with velocities
        in Å/fs and shape (natoms, 3). Every frame must match reference
        species/order and cell (absolute cell tolerance 1e-8 Å). Species
        checks cannot detect permutations between atoms of the same element.
        Frame masses are optional; if present they must match projection
        masses (absolute/relative tolerances 1e-6).
    reference_atoms : Atoms
        Equilibrium supercell corresponding to the simulation. Masses in amu
        must be present here or in the first trajectory frame.
    evec_filepath : str, pathlib.Path or PhononModes, optional
        Legacy ALAMODE eigenvector file or source-independent modes. File-only
        calls retain historical spglib mapping and assume primitive ordering
        matches the file. Use PhononModes.from_evec for explicit source order.
    batch_size : int, optional
        Maximum trajectory frames processed in each batch (default 100).
    parseval_tolerance : float or None, optional
        Maximum relative Parseval error (default 1e-6). Pass None for partial
        q/mode sets, including Gamma-only projections in multi-cell supercells.
    modes : PhononModes, optional
        Source-independent phonons with an explicitly ordered ISSE primitive.
        Supply either this or evec_filepath, not both.
    return_complex : bool, optional
        Append complex modal velocities as a fifth output (default False).

    Returns
    -------
    qpoints : numpy.ndarray
        Reduced q-point coordinates, shape (nqpoints, 3).
    qdot2 : numpy.ndarray
        Squared modal velocities, shape (nframes, nqpoints, nmodes), in
        amu Å²/fs². These are not total harmonic energies: half their value
        times AMU_A2_FS2_TO_EV is the modal kinetic energy in eV.
    atomic_norms : numpy.ndarray
        Mass-weighted Cartesian velocity norms, shape (nframes,), in amu Å²/fs².
    parseval_errors : numpy.ndarray
        Relative norm differences, shape (nframes,); NaN for zero atomic norm.
        With partial bases and validation disabled this is a completeness deficit.
    qdot : numpy.ndarray, optional
        With return_complex=True: complex128, shape (nframes, nqpoints, nmodes),
        in sqrt(amu) Å/fs. Uses conjugated eigenvectors, exp(-2πi q.R_l), and
        normalization 1/sqrt(number of primitive cells). No partial-basis
        renormalization is performed. Individual phases depend on eigenvector
        phase and arbitrary rotations within degenerate subspaces; compare
        subspace-invariant squared norms when necessary.

    Raises
    ------
    FileNotFoundError
        If the ALAMODE eigenvector file does not exist.
    ValueError
        For missing/invalid masses, velocities, phonon input, nonpositive batch
        size, or an empty trajectory.
    RuntimeError
        If the per-frame Parseval error exceeds the requested tolerance, or
        projection/norm results are non-finite even with validation disabled.
    """
    return _project_vectors(
        trajectory,
        reference_atoms,
        evec_filepath,
        batch_size,
        parseval_tolerance,
        modes=modes,
        return_complex=return_complex,
        vector_batches=_iter_velocity_batches(
            trajectory,
            reference_atoms=reference_atoms,
            batch_size=batch_size,
        ),
    )
