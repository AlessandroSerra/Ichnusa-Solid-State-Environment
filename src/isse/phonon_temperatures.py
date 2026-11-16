from __future__ import annotations

from logging import getLogger
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .constants import AMU_A2_FS2_TO_EV, KB_EV_K
from .phonon_modes import PhononModes
from .project_velocities import project_velocities
from .structures import Atoms, Trajectory

logger = getLogger(__name__)


def calculate_temperature(
    trajectory: Trajectory,
    reference_atoms: Atoms,
    evec_filepath: str | Path | PhononModes | None = None,
    selected_iqs: NDArray[np.int32] | None = None,
    batch_size: int = 100,
    parseval_tolerance: float | None = 1e-6,
    *,
    modes: PhononModes | None = None,
) -> dict[str, NDArray[np.float64] | float]:
    """
    Project a trajectory onto phonon modes and compute modal temperatures.

    This is a standalone, end-to-end entry point: it runs the velocity
    projection and then derives all the thermodynamic quantities from it.

    Parameters
    ----------
    trajectory : Trajectory
        Lazy trajectory yielding one ``Atoms`` object per frame.
    reference_atoms : Atoms
        Reference structure corresponding to the supercell used for the
        molecular dynamics simulation.
    evec_filepath : str, pathlib.Path or PhononModes, optional
        Legacy file or source-independent phonons.
    modes : PhononModes, optional
        Explicit ordered phonon data, instead of evec_filepath. Partial
        projections require parseval_tolerance=None; their temperatures are
        kinetic-equivalent quantities, not a reconstructed system temperature.
    selected_iqs : numpy.ndarray, optional
        Indices of q-points for which the instantaneous modal temperature
        time series is also returned. The default is ``None`` (skipped).
    batch_size : int, optional
        Maximum number of trajectory frames processed in each batch.
        The default is ``100``.
    parseval_tolerance : float, optional
        If provided, raise a ``RuntimeError`` as soon as any frame's
        relative Parseval error exceeds this value. The default is
        ``1e-6``. Pass ``None`` to disable this check.

    Returns
    -------
    dict
        Dictionary with the following keys:
        - ``"qpoints"``: used qpoints, shape
          ``(nqpoints, 3)``.
        - ``"mode_temperatures"``: modal kinetic temperatures, shape
          ``(nqpoints, nmodes)``.
        - ``"mean_mode_temperature"``: modal temperature averaged over all
          modes except the acoustic modes at Gamma.
        - ``"selected_mode_temperatures"``: instantaneous modal
          temperature time series at ``selected_iqs``, shape
          ``(nframes, nselected, nmodes)``. Present only if
          ``selected_iqs`` is not ``None``.

        ``reconstructed_temperature`` is computed internally and logged, but
        is not currently included in the returned dictionary.
    """
    qpoints, qdot2, _, parseval_errors = project_velocities(
        trajectory,
        reference_atoms,
        evec_filepath,
        batch_size=batch_size,
        parseval_tolerance=parseval_tolerance,
        modes=modes,
    )

    natoms = len(reference_atoms)

    mean_qdot2 = qdot2.mean(axis=0)
    mode_temperatures = mean_qdot2 * AMU_A2_FS2_TO_EV / KB_EV_K

    thermal_mask = np.ones_like(mode_temperatures, dtype=bool)
    gamma = np.all(np.isclose(qpoints, np.rint(qpoints), atol=1e-8), axis=1)
    # As in the legacy workflow, the first three Gamma modes are assumed
    # acoustic. Arbitrarily selected mode subsets cannot use this average.
    thermal_mask[gamma, :3] = False
    mean_mode_temperature = (
        float(mode_temperatures[thermal_mask].mean())
        if thermal_mask.any()
        else float("nan")
    )

    ndof = 3 * natoms - 3
    reconstructed_temperature = float(
        mean_qdot2.sum() * AMU_A2_FS2_TO_EV / (ndof * KB_EV_K)
    )

    temperature_rtol = 1e-6

    relative_temperature_difference = (
        abs(mean_mode_temperature - reconstructed_temperature) / mean_mode_temperature
        if mean_mode_temperature > 0
        else float("nan")
    )

    if (
        parseval_tolerance is not None
        and relative_temperature_difference > temperature_rtol
    ):
        logger.warning(
            "Potential residual drift present in the trajectory. "
            "Mean thermal mode temperature and reconstructed temperature differ "
            f"by {relative_temperature_difference:.2e} relative "
            f"(tolerance: {temperature_rtol:.1e})."
        )

    results: dict[str, NDArray[np.float64] | float] = {
        "qpoints": qpoints,
        "mode_temperatures": mode_temperatures,
        "mean_mode_temperature": mean_mode_temperature,
    }

    if selected_iqs is not None:
        nframes, _, nmodes = qdot2.shape
        if selected_iqs.size == 0:
            results["selected_mode_temperatures"] = np.empty(
                (nframes, 0, nmodes), dtype=np.float64
            )
        else:
            results["selected_mode_temperatures"] = (
                qdot2[:, selected_iqs, :] * AMU_A2_FS2_TO_EV / KB_EV_K
            )

    label = (
        "Reconstructed temperature"
        if parseval_tolerance is not None
        else "Projected kinetic-equivalent temperature contribution (not a system temperature)"
    )
    finite_errors = parseval_errors[np.isfinite(parseval_errors)]
    max_error = float(finite_errors.max()) if finite_errors.size else float("nan")
    logger.info(
        f"{label}: {reconstructed_temperature:.2f} K (mean thermal mode: "
        f"{mean_mode_temperature:.2f} K, max Parseval error: {max_error:.3e})"
    )
    return results
