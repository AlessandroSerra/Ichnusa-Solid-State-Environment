"""Time-origin-averaged mean squared displacement of atomistic trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import chain
from operator import index
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .constants import masses_from_symbols
from .structures import Atoms


def calculate_msd(
    trajectory: Sequence[Atoms],
    *,
    every: int = 1,
    max_lag: int | None = None,
    species: str | None = None,
    position_mode: Literal["auto", "wrapped", "unwrapped", "cartesian"] = "auto",
    remove_com: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate the mean squared displacement over all available time origins.

    Parameters
    ----------
    trajectory
        Nonempty sequence of frames, including a lazy ``Trajectory``. Frames
        must be uniformly spaced and preserve atom identities and order.
    every
        Output frame-lag interval. For example, ``every=2`` returns lags
        0, 2, 4, ... while still using every available time origin for each
        lag. It does not skip input frames, including during unwrapping.
    max_lag
        Largest frame lag to return, inclusive. Defaults to ``len(trajectory) - 1``.
    species
        Restrict the average to this chemical symbol. By default use all atoms.
    position_mode
        ``"auto"`` uses explicit ``unwrapped_positions`` if present in every
        frame, or reconstructs them from wrapped ``positions`` if absent from
        every frame. Mixed availability is an error. ``"unwrapped"`` requires
        explicit unwrapped positions, ``"wrapped"`` always reconstructs from
        ``positions``, and ``"cartesian"`` uses ``positions`` as-is.
    remove_com
        Subtract the mass-weighted center-of-mass motion of the whole system
        before species selection. Use masses from the first frame if present,
        otherwise use standard masses from its symbols. Any masses supplied
        in later frames must agree with these masses.

    Returns
    -------
    ndarray or tuple of ndarray
        MSD in Å² at frame lags ``0, every, 2*every, ...`` up to ``max_lag``.
        If frames have ``info["timestep"]`` metadata, return ``(lag_time, msd)``
        with lag times in the metadata's units (e.g. simulation steps for
        LAMMPS dumps). If no frames have timing metadata, return only ``msd``.
        Partial, non-finite, or nonuniform timing metadata raises ``ValueError``.

    Notes
    -----
    Wrapped reconstruction assumes a fixed, fully periodic cell and motion
    strictly less than half a box in each fractional direction between stored
    frames. It accumulates consecutive displacements, not minimum images at
    each lag. Explicit unwrapped and Cartesian modes measure laboratory-frame
    displacements, including any motion caused by cell deformation.

    Frames are read once. Only the selected coordinate histories are stored
    for the FFT, which takes O(n_frames log n_frames) work per selected atom.
    Stable identity within a species is the caller's responsibility.
    """
    n_frames = len(trajectory)
    if n_frames == 0:
        raise ValueError("trajectory must contain at least one frame")
    try:
        every = index(every)
    except TypeError as exc:
        raise ValueError("every must be a positive integer") from exc
    if every < 1:
        raise ValueError("every must be a positive integer")
    if max_lag is None:
        max_lag = n_frames - 1
    else:
        try:
            max_lag = index(max_lag)
        except TypeError as exc:
            raise ValueError("max_lag must be an integer") from exc
        if not 0 <= max_lag < n_frames:
            raise ValueError("max_lag must be between 0 and len(trajectory) - 1")
    if position_mode not in ("auto", "wrapped", "unwrapped", "cartesian"):
        raise ValueError("position_mode must be 'auto', 'wrapped', 'unwrapped', or 'cartesian'")

    lags = np.arange(0, max_lag + 1, every, dtype=np.intp)
    positions, frame_times = _collect_displacements(trajectory, species, position_mode, remove_com)
    if frame_times is not None and n_frames > 1:
        intervals = np.diff(frame_times)
        if (
            not np.isfinite(intervals).all()
            or np.any(intervals <= 0)
            or not np.allclose(intervals, intervals[0], rtol=1e-6, atol=0)
        ):
            raise ValueError("timestep metadata must be strictly increasing and uniformly spaced")
    msd = _msd_fft(positions, lags)
    if frame_times is None:
        return msd
    return frame_times[lags], msd


def _collect_displacements(
    trajectory: Sequence[Atoms],
    species: str | None,
    position_mode: str,
    remove_com: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Read each frame once and collect displacements and optional relative times."""
    frames = iter(trajectory)
    first = next(frames)
    symbols = list(first.symbols)
    selection = np.array([species is None or s == species for s in symbols], dtype=bool)
    n_selected = np.count_nonzero(selection)
    if n_selected == 0:
        raise ValueError("no atoms match the selection")

    mode = position_mode
    if mode == "auto":
        mode = "unwrapped" if first.has_unwrapped_positions else "wrapped"

    wrapped_state = None
    previous_fractional = None
    if mode == "wrapped":
        cell = np.array(first.cell, dtype=np.float64, copy=True)
        if not np.isfinite(cell).all():
            raise ValueError("wrapped positions require a finite cell")
        try:
            inverse_cell = np.linalg.inv(cell)
        except np.linalg.LinAlgError as exc:
            raise ValueError("wrapped positions require an invertible cell") from exc
        if not np.isfinite(inverse_cell).all():
            raise ValueError("wrapped positions require a finite inverse cell")
        cumulative = np.zeros((len(symbols) if remove_com else n_selected, 3))
        wrapped_state = (cell, inverse_cell, cumulative)

    masses = weights = None
    if remove_com:
        masses = (
            np.array(first.masses, dtype=np.float64, copy=True)
            if first.masses is not None
            else masses_from_symbols(symbols)
        )
        if not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError("center-of-mass removal requires finite positive masses")
        weights = masses / masses.max()
        weights /= weights.sum()

    positions = np.empty((len(trajectory), n_selected, 3), dtype=np.float64)
    frame_times = np.empty(len(trajectory)) if "timestep" in first.info else None
    first_timestep = None
    origin = None
    for i, frame in enumerate(chain((first,), frames)):
        if ("timestep" in frame.info) != (frame_times is not None):
            raise ValueError(f"frame {i}: mixed availability of timestep metadata")
        if frame_times is not None:
            timestep = _parse_timestep(str(frame.info["timestep"]), i)
            if first_timestep is None:
                first_timestep = timestep
            try:
                frame_times[i] = timestep - first_timestep
            except OverflowError as exc:
                raise ValueError(f"frame {i}: relative timestep must be finite") from exc
            if not np.isfinite(frame_times[i]):
                raise ValueError(f"frame {i}: relative timestep must be finite")
        if frame.symbols != symbols:
            raise ValueError(f"frame {i}: atom count and symbol order must match the first frame")
        if masses is not None and frame.masses is not None and (
            not np.isfinite(frame.masses).all()
            or np.any(frame.masses <= 0)
            or not np.allclose(frame.masses, masses, atol=0, rtol=1e-6)
        ):
            raise ValueError(f"frame {i}: masses must be positive and consistent across frames")
        if position_mode == "auto" and frame.has_unwrapped_positions != (mode == "unwrapped"):
            raise ValueError(f"frame {i}: mixed availability of unwrapped_positions")

        coordinates = frame.unwrapped_positions if mode == "unwrapped" else frame.positions
        if coordinates is None:
            raise ValueError(f"frame {i}: unwrapped_positions are required")
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if weights is None:
            coordinates = coordinates[selection]
        if not np.isfinite(coordinates).all():
            raise ValueError(f"frame {i}: positions must be finite")

        if wrapped_state is not None:
            cell, inverse_cell, cumulative = wrapped_state
            if not np.allclose(frame.cell, cell, atol=1e-8, rtol=0):
                raise ValueError(f"frame {i}: wrapped reconstruction requires a fixed cell")
            fractional = coordinates @ inverse_cell
            if previous_fractional is not None:
                step = fractional - previous_fractional
                step -= np.rint(step)
                cumulative += step @ cell
            previous_fractional = fractional
            displacement = cumulative
        else:
            if origin is None:
                origin = coordinates.copy()
            displacement = coordinates - origin

        if weights is not None:
            displacement = (displacement - weights @ displacement)[selection]
        positions[i] = displacement
    return positions, frame_times


def _parse_timestep(value: str, frame_index: int) -> int | float:
    """Parse a timestamp, preserving integer step counters exactly."""
    try:
        return int(value)
    except ValueError:
        pass

    try:
        timestep = float(value)
    except ValueError as exc:
        raise ValueError(
            f"frame {frame_index}: timestep metadata must be numeric and finite"
        ) from exc
    if not np.isfinite(timestep):
        raise ValueError(f"frame {frame_index}: timestep metadata must be numeric and finite")
    return timestep


def _msd_fft(
    positions: NDArray[np.float64], lags: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Combine linear autocorrelation with the two endpoint sums of squares."""
    if not np.isfinite(positions).all():
        raise ValueError("displacements must be finite")
    if lags.size == 1:
        return np.zeros(1)

    n_frames, n_atoms, _ = positions.shape
    # Center each time series to reduce cancellation without changing the MSD.
    positions -= positions.mean(axis=0)
    squared = np.einsum("fac,fac->f", positions, positions)
    n_origins = n_frames - lags
    starts = np.cumsum(squared)[n_origins - 1]
    ends = np.cumsum(squared[::-1])[n_origins - 1]

    # Pad to at least 2*n_frames - 1 for a linear, not circular, correlation.
    fft_len = 1 << (2 * n_frames - 2).bit_length()
    transformed = np.fft.rfft(positions, n=fft_len, axis=0)
    power = np.einsum("fac,fac->f", transformed.real, transformed.real)
    power += np.einsum("fac,fac->f", transformed.imag, transformed.imag)
    correlation = np.fft.irfft(power, n=fft_len)[lags]
    msd = (starts + ends - 2 * correlation) / n_origins / n_atoms
    if not np.isfinite(msd).all():
        raise ValueError("mean squared displacement overflowed")
    msd[0] = 0.0
    return np.maximum(msd, 0.0)  # Negative values can only be FFT roundoff.
