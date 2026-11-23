"""Internal mass-weighted normal-mode projection of Cartesian vector batches.

Numba is required: there is no optional import or alternative contraction
backend. Public velocity/displacement workflows supply their own vector batches.
This module is intentionally not re-exported from ``isse.helpers``.
"""

from __future__ import annotations

from collections.abc import Iterator
from logging import getLogger
from pathlib import Path

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from ..io.parse_alamode import read_alamode_evec
from ..phonon_modes import PhononModes
from ..structures import Atoms, Trajectory
from .cell_mapping import map_atoms_to_primitive
from .symmetry import get_scaled_positions

logger = getLogger(__name__)


def _iter_validated_frames(
    trajectory: Trajectory,
    reference_atoms: Atoms,
) -> Iterator[Atoms]:
    """Validate common frame metadata before either workflow reads vectors.

    Species cannot identify permutations between atoms of the same element;
    callers must maintain reference atom identity/order. Frame masses are
    optional, but when supplied must agree with the projection masses (from
    the reference, or the first frame when reference masses are absent).
    """
    projection_masses = reference_atoms.masses
    reference_cell = np.asarray(reference_atoms.cell, dtype=np.float64)
    for iframe, atoms in enumerate(trajectory):
        if not np.array_equal(atoms.symbols, reference_atoms.symbols):
            raise ValueError(
                f"Trajectory frame {iframe} symbols must match reference atom order"
            )
        cell = np.asarray(atoms.cell, dtype=np.float64)
        if (
            cell.shape != (3, 3)
            or not np.isfinite(cell).all()
            or not np.allclose(cell, reference_cell, atol=1e-8, rtol=0)
        ):
            raise ValueError(
                f"Trajectory frame {iframe} cell differs from the fixed reference cell"
            )
        if iframe == 0 and projection_masses is None:
            projection_masses = atoms.masses
        if atoms.masses is not None:
            frame_masses = np.asarray(atoms.masses, dtype=np.float64)
            if (
                frame_masses.shape != (len(reference_atoms),)
                or not np.isfinite(frame_masses).all()
                or np.any(frame_masses <= 0)
                or not np.allclose(
                    frame_masses, projection_masses, atol=1e-6, rtol=1e-6
                )
            ):
                raise ValueError(
                    f"Trajectory frame {iframe} masses must match the projection masses"
                )
        yield atoms


def _compute_parseval_errors(
    squared_moduli: NDArray[np.float64],
    atomic_norms: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Relative modal/Cartesian norm difference per frame; NaN for zero norm.

    Inputs have shape (nframes, nq, nmodes) and (nframes,) respectively.
    Their physical units agree and depend on the supplied Cartesian vectors.
    Non-finite inputs, norm sums or nonzero-frame relative errors raise
    RuntimeError, even if the caller disables the completeness tolerance.
    NaN is reserved for a finite zero atomic norm with zero modal norm.
    """
    if not np.isfinite(squared_moduli).all() or not np.isfinite(atomic_norms).all():
        raise RuntimeError(
            "Non-finite projection results: squared moduli or atomic norms"
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        modal_norms = squared_moduli.sum(axis=(1, 2))
        errors = np.full_like(atomic_norms, np.nan)
        nonzero = atomic_norms != 0.0
        errors[nonzero] = (
            np.abs(modal_norms[nonzero] - atomic_norms[nonzero]) / atomic_norms[nonzero]
        )
    if not np.isfinite(modal_norms).all() or not np.isfinite(errors[nonzero]).all():
        raise RuntimeError("Non-finite Parseval norms or relative errors")
    if np.any(modal_norms[~nonzero] != 0.0):
        raise RuntimeError("Nonzero modal norm for a zero atomic norm")
    return errors


def _precompute_coefficients(
    qpoints: NDArray[np.float64],
    eigenvectors: NDArray[np.complex128],
    cell_indices: NDArray[np.int32],
    basis_indices: NDArray[np.int32],
    masses: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Combine conjugated eigenvectors, translation phases and mass weights.

    Reduced qpoints: (nq, 3); eigenvectors: (nq, nmodes, nbasis, 3);
    cell_indices: (natoms, 3); basis_indices and masses: (natoms,).
    Coefficients have shape (nq, nmodes, natoms, 3), with phase
    exp(-2πi q.R_l) and normalization sqrt(mass / number_of_cells).
    Contracting with velocities gives sqrt(amu) Å/fs; with displacements,
    sqrt(amu) Å. No basis-position phase or partial-basis rescaling is added.
    """
    _, _, nbasis, ndim = eigenvectors.shape
    natoms = len(masses)
    if ndim != 3:
        raise ValueError(f"Expected three Cartesian dimensions, found {ndim}.")
    if natoms % nbasis != 0:
        raise ValueError(f"{natoms} atoms are not divisible by {nbasis} basis atoms.")

    ncells = natoms // nbasis
    phases = np.exp(-2j * np.pi * (qpoints @ cell_indices.T))
    eig_atoms = np.take(eigenvectors, basis_indices, axis=2)
    coefficients = (
        eig_atoms.conj()
        * phases[:, None, :, None]
        * np.sqrt(masses)[None, None, :, None]
        / np.sqrt(ncells)
    )
    return np.ascontiguousarray(coefficients, dtype=np.complex128)


@njit(cache=True, parallel=True)
def _project_batch_numba(
    vectors: NDArray[np.float64],
    masses: NDArray[np.float64],
    coefficients: NDArray[np.complex128],
    return_complex: bool = False,
):
    """Contract (nframes, natoms, 3) Cartesian vectors, parallel over frames.

    Returns squared moduli (nframes, nq, nmodes), mass-weighted Cartesian
    norms (nframes,), and optionally complex modal values of the same modal
    shape. Without complex output, an empty leading axis avoids allocating
    that storage. Modal units are sqrt(amu) times the input vector units;
    squared moduli and Cartesian norms have amu times squared input units.
    """
    nframes, natoms, _ = vectors.shape
    nq, nmodes, _, _ = coefficients.shape
    squared_moduli = np.empty((nframes, nq, nmodes), dtype=np.float64)
    atomic_norms = np.empty(nframes, dtype=np.float64)
    modal_values = np.empty(
        (nframes if return_complex else 0, nq, nmodes), dtype=np.complex128
    )

    for iframe in prange(nframes):
        atomic_norm = 0.0
        for iatom in range(natoms):
            x = vectors[iframe, iatom, 0]
            y = vectors[iframe, iatom, 1]
            z = vectors[iframe, iatom, 2]
            atomic_norm += masses[iatom] * (x * x + y * y + z * z)
        atomic_norms[iframe] = atomic_norm

        for iq in range(nq):
            for imode in range(nmodes):
                modal_real = 0.0
                modal_imag = 0.0
                for iatom in range(natoms):
                    x = vectors[iframe, iatom, 0]
                    y = vectors[iframe, iatom, 1]
                    z = vectors[iframe, iatom, 2]
                    cx = coefficients[iq, imode, iatom, 0]
                    cy = coefficients[iq, imode, iatom, 1]
                    cz = coefficients[iq, imode, iatom, 2]
                    modal_real += cx.real * x + cy.real * y + cz.real * z
                    modal_imag += cx.imag * x + cy.imag * y + cz.imag * z
                squared_moduli[iframe, iq, imode] = (
                    modal_real * modal_real + modal_imag * modal_imag
                )
                if return_complex:
                    modal_values[iframe, iq, imode] = modal_real + 1j * modal_imag

    return squared_moduli, atomic_norms, modal_values


def _project_vectors(
    trajectory: Trajectory,
    reference_atoms: Atoms,
    evec_filepath: str | Path | PhononModes | None = None,
    batch_size: int = 100,
    parseval_tolerance: float | None = 1e-6,
    *,
    vector_batches: Iterator[NDArray[np.float64]],
    modes: PhononModes | None = None,
    return_complex: bool = False,
) -> tuple:
    """Map, prepare coefficients, and project caller-supplied vector batches.

    The required iterator yields finite contiguous float64 arrays of shape
    (batch_frames, natoms, 3) in reference order. Reading and validation of
    physical vector data belong to the public workflow. The trajectory is
    used only for length and, if needed, fallback masses from the first frame.
    This engine never reads trajectory velocities, positions, or image fields;
    reference positions are used only for the common atomic mapping.
    """
    if isinstance(evec_filepath, PhononModes):
        if modes is not None:
            raise ValueError("Supply phonon modes only once")
        modes, evec_filepath = evec_filepath, None
    if (modes is None) == (evec_filepath is None):
        raise ValueError("Supply either evec_filepath or modes, not both")
    if modes is not None and not isinstance(modes, PhononModes):
        raise TypeError("modes must be PhononModes")
    if parseval_tolerance is not None and (
        not np.isfinite(parseval_tolerance) or parseval_tolerance < 0
    ):
        raise ValueError("parseval_tolerance must be finite and nonnegative or None")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, found {batch_size}.")
    if evec_filepath is not None:
        evec_filepath = Path(evec_filepath)
        if not evec_filepath.is_file():
            raise FileNotFoundError(
                f"ALAMODE eigenvector file not found: {evec_filepath}"
            )
    if len(trajectory) == 0:
        raise ValueError("No trajectory frames were read")
    reference_cell = np.asarray(reference_atoms.cell, dtype=np.float64)
    if (
        reference_cell.shape != (3, 3)
        or not np.isfinite(reference_cell).all()
        or abs(np.linalg.det(reference_cell)) < 1e-12
    ):
        raise ValueError(
            "Reference cell must be finite and nonsingular with shape (3, 3)"
        )

    if reference_atoms.masses is not None:
        masses = reference_atoms.masses
    elif trajectory[0].masses is not None:
        masses = trajectory[0].masses
        logger.warning(
            "No masses found in reference atoms. Using masses from trajectory."
        )
    else:
        raise ValueError(
            "The Trajectory or the reference Atoms object must contain atomic masses."
        )
    masses = np.asarray(masses, dtype=np.float64)
    if (
        masses.shape != (len(reference_atoms),)
        or not np.isfinite(masses).all()
        or np.any(masses <= 0)
    ):
        raise ValueError(
            "Atomic masses must be finite, positive and match the reference"
        )

    if modes is None:
        # Retain historical spglib mapping for callers supplying only a file.
        _, qpoints, _, eigenvectors = read_alamode_evec(evec_filepath)
        mapping_results, mapping_residuals = map_atoms_to_primitive(reference_atoms)
    else:
        primitive = modes.primitive
        qpoints = modes.qpoints
        eigenvectors = modes.eigenvectors
        mapping_results, mapping_residuals = map_atoms_to_primitive(
            reference_atoms,
            primitive_cell=primitive.cell,
            basis=get_scaled_positions(primitive),
        )
        transform = reference_atoms.cell @ np.linalg.inv(primitive.cell)
        if not np.allclose(transform, np.rint(transform), atol=1e-6, rtol=0):
            raise ValueError(
                "Reference cell must be an integer supercell of the phonon primitive"
            )
        ncells = len(reference_atoms) // len(primitive)
        if not np.isclose(abs(np.linalg.det(transform)), ncells, atol=1e-6, rtol=0):
            raise ValueError(
                "Reference cell volume is inconsistent with primitive populations"
            )
        # Positions separated by a supercell translation must not be duplicated.
        translations, sites = mapping_results
        super_scaled = translations @ np.linalg.inv(transform)
        for site in range(len(primitive)):
            points = super_scaled[sites == site]
            for i in range(len(points)):
                delta = points[i + 1 :] - points[i]
                if np.any(
                    np.all(np.isclose(delta, np.rint(delta), atol=1e-6, rtol=0), axis=1)
                ):
                    raise ValueError(
                        "Duplicate primitive site modulo the reference supercell"
                    )
        basis_indices = mapping_results[1]
        expected_symbols = np.asarray(primitive.symbols)[basis_indices]
        if not np.array_equal(expected_symbols, reference_atoms.symbols):
            raise ValueError(
                "Mapped species do not match the phonon primitive ordering"
            )
        if not np.allclose(
            masses, primitive.masses[basis_indices], atol=1e-6, rtol=1e-6
        ):
            raise ValueError("Mapped masses differ from the phonon primitive masses")
        if parseval_tolerance is not None:
            if eigenvectors.shape[1] != 3 * len(primitive) or len(qpoints) != ncells:
                raise ValueError(
                    "Incomplete basis: pass parseval_tolerance=None for partial projections"
                )
            phases = np.exp(
                -2j * np.pi * (qpoints @ translations[basis_indices == 0].T)
            )
            if not np.allclose(
                phases @ phases.conj().T / ncells, np.eye(ncells), atol=1e-6, rtol=0
            ):
                raise ValueError(
                    "q-points are not an orthogonal complete supercell basis"
                )
            if not np.allclose(
                qpoints @ transform.T, np.rint(qpoints @ transform.T), atol=1e-6, rtol=0
            ):
                raise ValueError(
                    "q-points are not commensurate with the reference supercell"
                )
    cell_indices, basis_indices = mapping_results
    logger.info(
        f"Successfully mapped atoms to primitive cell with {np.max(mapping_residuals):.2e} A max residual."
    )
    coefficients = _precompute_coefficients(
        qpoints, eigenvectors, cell_indices, basis_indices, masses
    )

    complex_batches: list[NDArray[np.complex128]] = []
    squared_modulus_batches: list[NDArray[np.float64]] = []
    atomic_norm_batches: list[NDArray[np.float64]] = []
    parseval_error_batches: list[NDArray[np.float64]] = []
    for ibatch, vectors in enumerate(vector_batches):
        squared_moduli, atomic_norms, modal_values = _project_batch_numba(
            vectors,
            masses,
            coefficients,
            return_complex,
        )
        if return_complex:
            if not np.isfinite(modal_values).all():
                raise RuntimeError("Non-finite complex projection results")
            complex_batches.append(modal_values)
        errors = _compute_parseval_errors(squared_moduli, atomic_norms)
        nonzero_errors = errors[atomic_norms != 0.0]
        max_error = float(nonzero_errors.max()) if nonzero_errors.size else float("nan")
        logger.debug(f"Batch {ibatch}: max Parseval error = {max_error}")
        if parseval_tolerance is not None and max_error > parseval_tolerance:
            raise RuntimeError(
                "Parseval error exceeds tolerance: "
                f"max={max_error:.6e} > {parseval_tolerance:.6e}."
            )
        squared_modulus_batches.append(squared_moduli)
        atomic_norm_batches.append(atomic_norms)
        parseval_error_batches.append(errors)
    if not squared_modulus_batches:
        raise ValueError(
            "No trajectory frames were read. "
            "The trajectory is empty or the parser did not yield any frames."
        )
    result = (
        qpoints,
        np.concatenate(squared_modulus_batches, axis=0),
        np.concatenate(atomic_norm_batches, axis=0),
        np.concatenate(parseval_error_batches, axis=0),
    )
    if return_complex:
        return (*result, np.concatenate(complex_batches, axis=0))
    return result
