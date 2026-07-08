from __future__ import annotations

from collections.abc import Iterator
from logging import getLogger

import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from .helpers.periodic import minimum_image_distances
from .structures import Trajectory

logger = getLogger(__name__)


def calculate_rdf(
    trajectory: Trajectory,
    r_max: float,
    dr: float,
    batch_size: int = 100,
    use_numba: bool = True,
) -> dict[str, NDArray[np.float64]]:
    """
    Calculate the total radial distribution function of a trajectory.

    Parameters
    ----------
    trajectory : Trajectory
        Lazy trajectory yielding one ``Atoms`` object per frame.
    r_max : float
        Maximum distance included in the RDF histogram.
    dr : float
        Histogram bin width.
    batch_size : int, optional
        Maximum number of trajectory frames processed in each batch.
    use_numba : bool, optional
        If True, use the Numba backend when available. Otherwise use the
        NumPy fallback.

    Returns
    -------
    dict
        Dictionary containing:

        - ``"r"``: bin centers, shape ``(nbins,)``;
        - ``"g_r"``: radial distribution function, shape ``(nbins,)``;
        - ``"counts"``: raw pair counts, shape ``(nbins,)``.
    """
    if r_max <= 0.0:
        raise ValueError(f"r_max must be positive, found {r_max}.")

    if dr <= 0.0:
        raise ValueError(f"dr must be positive, found {dr}.")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, found {batch_size}.")

    nbins = int(np.floor(r_max / dr))

    if nbins < 1:
        raise ValueError("r_max must be at least one bin width.")

    counts = np.zeros(nbins, dtype=np.float64)
    volumes: list[float] = []
    nframes = 0
    natoms: int | None = None

    backend_name = "numba" if use_numba and NUMBA_AVAILABLE else "numpy"
    if use_numba and not NUMBA_AVAILABLE:
        logger.warning("Numba is not available; falling back to NumPy RDF backend.")

    for positions, cells in _iter_position_batches(trajectory, batch_size=batch_size):
        if natoms is None:
            natoms = positions.shape[1]
        elif positions.shape[1] != natoms:
            raise ValueError(
                "All trajectory frames must contain the same number of atoms: "
                f"expected {natoms}, found {positions.shape[1]}."
            )

        if backend_name == "numba":
            inverse_cells = np.linalg.inv(cells)
            batch_counts = _histogram_rdf_numba(
                positions,
                cells,
                inverse_cells,
                r_max,
                dr,
                nbins,
            )
        else:
            batch_counts = _histogram_rdf_numpy(
                positions,
                cells,
                r_max,
                dr,
                nbins,
            )

        counts += batch_counts
        volumes.extend(np.abs(np.linalg.det(cells)).tolist())
        nframes += positions.shape[0]

    if natoms is None or nframes == 0:
        raise ValueError("No trajectory frames were read.")

    r, g_r = _normalize_rdf(
        counts=counts,
        natoms=natoms,
        nframes=nframes,
        volumes=np.asarray(volumes, dtype=np.float64),
        dr=dr,
    )

    return {
        "r": r,
        "g_r": g_r,
        "counts": counts,
    }


def _iter_position_batches(
    trajectory: Trajectory,
    batch_size: int,
) -> Iterator[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Yield batches of positions and cells from a lazy trajectory.

    Yields
    ------
    positions : numpy.ndarray
        Cartesian positions with shape ``(nframes_batch, natoms, 3)``.
    cells : numpy.ndarray
        Cell matrices with shape ``(nframes_batch, 3, 3)``.
    """
    position_batch: list[NDArray[np.float64]] = []
    cell_batch: list[NDArray[np.float64]] = []
    natoms: int | None = None

    for iframe, atoms in enumerate(trajectory):
        positions = atoms.positions
        cell = atoms.cell

        if positions.shape[-1] != 3 or positions.ndim != 2:
            raise ValueError(
                f"Frame {iframe} positions must have shape (n_atoms, 3), "
                f"found {positions.shape}."
            )

        if cell.shape != (3, 3):
            raise ValueError(
                f"Frame {iframe} cell must have shape (3, 3), found {cell.shape}."
            )

        if natoms is None:
            natoms = positions.shape[0]
        elif positions.shape[0] != natoms:
            raise ValueError(
                "All trajectory frames must contain the same number of atoms: "
                f"expected {natoms}, found {positions.shape[0]} in frame {iframe}."
            )

        position_batch.append(positions)
        cell_batch.append(cell)

        if len(position_batch) == batch_size:
            yield (
                np.ascontiguousarray(position_batch, dtype=np.float64),
                np.ascontiguousarray(cell_batch, dtype=np.float64),
            )
            position_batch.clear()
            cell_batch.clear()

    if position_batch:
        yield (
            np.ascontiguousarray(position_batch, dtype=np.float64),
            np.ascontiguousarray(cell_batch, dtype=np.float64),
        )


def _histogram_rdf_numpy(
    positions: NDArray[np.float64],
    cells: NDArray[np.float64],
    r_max: float,
    dr: float,
    nbins: int,
) -> NDArray[np.float64]:
    """
    Accumulate RDF pair counts using a readable NumPy fallback.

    Pair distances are computed with the minimum image convention. Each pair
    ``i < j`` contributes ``2`` counts because atom ``i`` sees atom ``j`` and
    atom ``j`` sees atom ``i``.
    """
    nframes, natoms, _ = positions.shape
    counts = np.zeros(nbins, dtype=np.float64)

    for iframe in range(nframes):
        frame_positions = positions[iframe]
        cell = cells[iframe]

        for iatom in range(natoms - 1):
            displacements = frame_positions[iatom + 1 :] - frame_positions[iatom]
            distances = minimum_image_distances(displacements, cell)
            selected = distances < r_max
            bin_indices = np.floor(distances[selected] / dr).astype(np.int64)
            counts += 2.0 * np.bincount(bin_indices, minlength=nbins)[:nbins]

    return counts


if NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True)
    def _histogram_rdf_numba(
        positions: NDArray[np.float64],
        cells: NDArray[np.float64],
        inverse_cells: NDArray[np.float64],
        r_max: float,
        dr: float,
        nbins: int,
    ) -> NDArray[np.float64]:
        """
        Accumulate RDF pair counts using a Numba backend parallel over frames.
        """
        nframes, natoms, _ = positions.shape
        counts_by_frame = np.zeros((nframes, nbins), dtype=np.float64)

        for iframe in prange(nframes):
            cell = cells[iframe]
            inverse_cell = inverse_cells[iframe]
            frame_counts = counts_by_frame[iframe]

            for iatom in range(natoms - 1):
                xi = positions[iframe, iatom, 0]
                yi = positions[iframe, iatom, 1]
                zi = positions[iframe, iatom, 2]

                for jatom in range(iatom + 1, natoms):
                    dx = positions[iframe, jatom, 0] - xi
                    dy = positions[iframe, jatom, 1] - yi
                    dz = positions[iframe, jatom, 2] - zi

                    sx = (
                        dx * inverse_cell[0, 0]
                        + dy * inverse_cell[1, 0]
                        + dz * inverse_cell[2, 0]
                    )
                    sy = (
                        dx * inverse_cell[0, 1]
                        + dy * inverse_cell[1, 1]
                        + dz * inverse_cell[2, 1]
                    )
                    sz = (
                        dx * inverse_cell[0, 2]
                        + dy * inverse_cell[1, 2]
                        + dz * inverse_cell[2, 2]
                    )

                    sx -= np.rint(sx)
                    sy -= np.rint(sy)
                    sz -= np.rint(sz)

                    dx = sx * cell[0, 0] + sy * cell[1, 0] + sz * cell[2, 0]
                    dy = sx * cell[0, 1] + sy * cell[1, 1] + sz * cell[2, 1]
                    dz = sx * cell[0, 2] + sy * cell[1, 2] + sz * cell[2, 2]

                    distance = np.sqrt(dx * dx + dy * dy + dz * dz)

                    if distance < r_max:
                        ibin = int(distance / dr)
                        frame_counts[ibin] += 2.0

        counts = np.zeros(nbins, dtype=np.float64)
        for iframe in range(nframes):
            for ibin in range(nbins):
                counts[ibin] += counts_by_frame[iframe, ibin]

        return counts


def _normalize_rdf(
    counts: NDArray[np.float64],
    natoms: int,
    nframes: int,
    volumes: NDArray[np.float64],
    dr: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Normalize RDF pair counts into ``g(r)``.
    """
    nbins = len(counts)
    edges = np.arange(nbins + 1, dtype=np.float64) * dr
    r = 0.5 * (edges[:-1] + edges[1:])

    shell_volumes = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    mean_volume = float(np.mean(volumes))
    density = natoms / mean_volume

    normalization = nframes * natoms * density * shell_volumes
    g_r = counts / normalization

    return r, np.asarray(g_r, dtype=np.float64)
