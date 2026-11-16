"""
Source-independent, mass-weighted phonon eigenmodes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .io.parse_alamode import read_alamode_evec
from .structures import Atoms
from .constants import RY_TO_CM


@dataclass(slots=True)
class PhononModes:
    """Phonons in an explicitly ordered primitive basis.

    ``primitive`` is ISSE Atoms (Å, amu), in exactly the source's atom order
    and cell-origin/image convention. qpoints: (nq, 3), reduced reciprocal;
    frequencies: (nq, nmodes), cm^-1 (negative for imaginary modes);
    eigenvectors: (nq, nmodes, natoms, 3), dimensionless orthonormal rows of
    the mass-weighted dynamical matrix. Partial mode sets are allowed.

    The Bloch convention is ``u_li ~ e_i exp(+2πi q.R_l) / sqrt(m_i)``:
    eigenvectors must use the cell-translation gauge, not the full-position
    gauge. No normalization, mass weighting or phase correction is applied.
    """

    primitive: Atoms
    qpoints: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128]

    def __post_init__(self):
        if not isinstance(self.primitive, Atoms):
            raise TypeError("primitive must be an isse.structures.Atoms object")
        self.primitive = deepcopy(self.primitive)
        self.qpoints = np.array(self.qpoints, dtype=np.float64, copy=True)
        self.frequencies = np.array(self.frequencies, dtype=np.float64, copy=True)
        self.eigenvectors = np.array(self.eigenvectors, dtype=np.complex128, copy=True)
        n = len(self.primitive)
        if (
            n == 0
            or not np.isfinite(self.primitive.cell).all()
            or abs(np.linalg.det(self.primitive.cell)) < 1e-12
        ):
            raise ValueError("primitive must have a finite, nonsingular 3D cell")
        if not np.isfinite(self.primitive.positions).all():
            raise ValueError("primitive positions must be finite")
        masses = self.primitive.masses
        if masses is None:
            raise ValueError("primitive must contain explicit atomic masses in amu")
        if masses.shape != (n,) or not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError(
                "primitive masses must have shape (natoms,) and be finite and positive"
            )
        if (
            self.qpoints.ndim != 2
            or self.qpoints.shape[1] != 3
            or len(self.qpoints) == 0
        ):
            raise ValueError("qpoints must have shape (nq, 3), nq > 0")
        if self.frequencies.ndim != 2 or self.frequencies.shape[0] != len(self.qpoints):
            raise ValueError("frequencies must have shape (nq, nmodes)")
        nmodes = self.frequencies.shape[1]
        if not 0 < nmodes <= 3 * n:
            raise ValueError("nmodes must lie between 1 and 3 * natoms")
        if self.eigenvectors.shape != (len(self.qpoints), nmodes, n, 3):
            raise ValueError("eigenvectors must have shape (nq, nmodes, natoms, 3)")
        if not all(
            np.isfinite(a).all()
            for a in (self.qpoints, self.frequencies, self.eigenvectors)
        ):
            raise ValueError("phonon arrays must be finite")
        flat = self.eigenvectors.reshape(len(self.qpoints), nmodes, 3 * n)
        if not np.allclose(
            flat.conj() @ flat.swapaxes(1, 2), np.eye(nmodes), atol=1e-6, rtol=0
        ):
            raise ValueError(
                "eigenvectors must be orthonormal mass-weighted dynamical-matrix eigenvectors"
            )

    @classmethod
    def from_evec(cls, filename, *, primitive):
        """Read legacy text data with an explicit source-ordered ISSE primitive.

        Legacy files do not store basis positions or per-site species; these
        must be supplied, not inferred from eigenvector dimensions.
        """
        cell, qpoints, omega2, eigenvectors = read_alamode_evec(filename)
        if not np.allclose(cell, primitive.cell, atol=1e-6, rtol=1e-6):
            raise ValueError("primitive cell differs from the .evec lattice")
        # ALAMODE text writes signed squared frequencies in Rydberg units.
        frequencies = np.sign(omega2) * np.sqrt(np.abs(omega2)) * RY_TO_CM
        return cls(primitive, qpoints, frequencies, eigenvectors)

    @classmethod
    def from_anphon(cls, phonon, qpoints, *, primitive, source_order_confirmed=False):
        """Call ``anphon.Phonon.frequencies`` without importing anphon.

        The public API returns no primitive structure/ordering metadata.
        Supply the exact primitive used by ALAMODE, including masses (and
        any MASS overrides), unwrapped basis images and order. Set
        ``source_order_confirmed=True`` only after checking this against the
        force-constant file's PrimitiveCell metadata. A POSCAR with merely
        the same composition is NOT sufficient. No ordering is inferred.
        """
        if not source_order_confirmed:
            raise ValueError(
                "Confirm the primitive order, images, cell and masses against the ALAMODE source metadata"
            )
        qpoints = np.asarray(qpoints, dtype=np.float64)
        if qpoints.shape == (3,):
            qpoints = qpoints[None, :]
        if qpoints.ndim != 2 or qpoints.shape[1] != 3 or len(qpoints) == 0:
            raise ValueError("qpoints must have shape (nq, 3)")
        frequencies, vectors = phonon.frequencies(qpoints, eigenvectors=True)
        vectors = np.asarray(vectors)
        n = len(primitive)
        if vectors.shape != (len(qpoints), 3 * n, 3 * n):
            raise ValueError(
                "anphon eigenvectors must have shape (nq, 3*natoms, 3*natoms), mode axis first"
            )
        return cls(
            primitive, qpoints, frequencies, vectors.reshape(len(qpoints), 3 * n, n, 3)
        )
