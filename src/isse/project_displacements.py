"""Projection of reference-relative displacements onto phonon normal modes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .helpers.mode_projection import _iter_validated_frames, _project_vectors
from .helpers.periodic import minimum_image_displacements
from .phonon_modes import PhononModes
from .structures import Atoms, Trajectory

PositionMode = Literal["minimum_image", "unwrapped", "cartesian"]


def _iter_displacement_batches(
    trajectory: Trajectory,
    reference_atoms: Atoms,
    batch_size: int,
    position_mode: PositionMode,
) -> Iterator[NDArray[np.float64]]:
    """Yield finite displacements in reference order without loading all frames."""
    natoms = len(reference_atoms)
    reference_cell = np.asarray(reference_atoms.cell, dtype=np.float64)
    reference_positions = reference_atoms.positions
    if position_mode == "unwrapped" and reference_atoms.unwrapped_positions is not None:
        reference_positions = reference_atoms.unwrapped_positions
    reference_positions = np.asarray(reference_positions, dtype=np.float64)
    if (
        reference_positions.shape != (natoms, 3)
        or not np.isfinite(reference_positions).all()
    ):
        raise ValueError("Reference positions must be finite with shape (natoms, 3)")

    batch: list[NDArray[np.float64]] = []
    for iframe, atoms in enumerate(_iter_validated_frames(trajectory, reference_atoms)):
        positions = (
            atoms.unwrapped_positions
            if position_mode == "unwrapped"
            else atoms.positions
        )
        if positions is None:
            raise ValueError(
                f"Trajectory frame {iframe} does not contain unwrapped_positions"
            )
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (natoms, 3) or not np.isfinite(positions).all():
            raise ValueError(
                f"Trajectory frame {iframe} positions must be finite with shape ({natoms}, 3)"
            )
        displacements = positions - reference_positions
        if position_mode == "minimum_image":
            displacements = minimum_image_displacements(displacements, reference_cell)
        batch.append(displacements)
        if len(batch) == batch_size:
            yield np.ascontiguousarray(batch, dtype=np.float64)
            batch.clear()
    if batch:
        yield np.ascontiguousarray(batch, dtype=np.float64)


def project_displacements(
    trajectory: Trajectory,
    reference_atoms: Atoms,
    evec_filepath: str | Path | PhononModes | None = None,
    batch_size: int = 100,
    parseval_tolerance: float | None = 1e-6,
    *,
    modes: PhononModes | None = None,
    return_complex: bool = False,
    position_mode: PositionMode = "minimum_image",
) -> tuple:
    r"""Project displacements from a fixed equilibrium reference onto phonons.

    Uses the same source ordering, mass weighting and cell-translation Bloch
    gauge as :func:`isse.project_velocities`, with the same required Numba
    contraction kernel (no alternative backend):

    ``Q[q, nu] = sum_li sqrt(m_i) e[q, nu, i]* . u_li
                 exp(-2πi q.R_l) / sqrt(Ncells)``.

    Parameters
    ----------
    trajectory : Trajectory or sequence of Atoms
        Frames in reference atom order. Velocities are not required. Positions
        are in Å; each frame's cell must equal the reference cell to 1e-8 Å.
        Species checks cannot detect permutations of same-element atoms.
        Frame masses are optional; if present they must match projection
        masses (absolute/relative tolerances 1e-6).
    reference_atoms : Atoms
        Fixed equilibrium supercell used for both atomic mapping and the
        displacement origin. Masses (amu) are taken from this structure, or
        from the first trajectory frame if absent, as for velocity projection.
    evec_filepath : str, pathlib.Path or PhononModes, optional
        Legacy ALAMODE file or source-independent modes. The legacy file-only
        interface retains the historical spglib ordering assumption.
    batch_size : int, optional
        Maximum frames in each batch (default 100).
    parseval_tolerance : float or None, optional
        Relative tolerance for sum(|Q|²) == sum(m_i |u_i|²), default 1e-6.
        Pass None for partial q/mode sets, including Gamma-only supercell
        projections. Partial projections are never renormalized.
    modes : PhononModes, optional
        Explicit source-ordered phonon data, instead of evec_filepath.
    return_complex : bool, optional
        Append complex normal coordinates Q as a fifth output (default False).
        Enable this to retain the sign/phase of coherent displacements.
    position_mode : {"minimum_image", "unwrapped", "cartesian"}, optional
        How to form displacements, consistently for every frame:

        - ``minimum_image`` (default): minimum-image difference of positions
          from reference positions, using the reference supercell, not the
          primitive cell. Assumes excursions smaller than half the supercell
          in fractional coordinates. No temporal unwrapping is performed.
          The existing fractional-rounding helper is approximate for strongly
          skewed/non-reduced cells.
        - ``unwrapped``: require frame.unwrapped_positions and subtract
          reference.unwrapped_positions if available, otherwise reference.positions.
          The caller must ensure consistent image origins; no wrapping or
          independent initial-image correction is applied.
        - ``cartesian``: subtract positions directly, without periodic image
          correction; useful for already image-aligned Cartesian data.

    Returns
    -------
    qpoints : numpy.ndarray
        Reduced q-points, shape (nq, 3).
    q2 : numpy.ndarray
        |Q|², shape (nframes, nq, nmodes), in amu Å². This is not an energy.
    atomic_norms : numpy.ndarray
        sum(m_i |u_i|²), shape (nframes,), in amu Å².
    parseval_errors : numpy.ndarray
        Relative norm difference, shape (nframes,). NaN for zero displacement
        norms; a completeness deficit rather than a validation error for
        partial bases with validation disabled. Non-finite numerical results
        raise RuntimeError even when completeness validation is disabled.
    Q : numpy.ndarray, optional
        Only with return_complex=True: complex128 normal coordinates, shape
        (nframes, nq, nmodes), in sqrt(amu) Å.

    Notes
    -----
    No center-of-mass translation, rotation, or strain is subtracted. A fixed
    harmonic basis does not define a unique treatment of variable cells;
    changing-cell trajectories must be transformed explicitly before use.
    Individual Q phases depend on eigenvector gauge, and on arbitrary basis
    rotations within degenerate subspaces, exactly as for modal velocities.
    ``dQ/dt = qdot`` only for consistent continuous displacement images and
    a fixed reference/basis; minimum-image branch jumps break that relation.
    """
    if position_mode not in ("minimum_image", "unwrapped", "cartesian"):
        raise ValueError(
            "position_mode must be 'minimum_image', 'unwrapped' or 'cartesian'"
        )
    return _project_vectors(
        trajectory,
        reference_atoms,
        evec_filepath,
        batch_size,
        parseval_tolerance,
        modes=modes,
        return_complex=return_complex,
        vector_batches=_iter_displacement_batches(
            trajectory,
            reference_atoms,
            batch_size,
            position_mode,
        ),
    )
