from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..constants import KCAL_MOL_TO_EV, PS_TO_FS, mass_from_symbol
from ..structures import Atoms, Trajectory


def write_lammps_data(
    filename: str | Path,
    atoms: Atoms,
    *,
    units: str,
) -> None:
    """
    Write a LAMMPS ``atom_style atomic`` data file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path of the output LAMMPS data file.
    atoms : isse.structures.Atoms
        Atomic configuration to write. Positions and cell vectors are expected
        in angstrom. Velocities, when present, are expected in angstrom/fs.
    units : {"metal", "real"}
        LAMMPS unit style for the written file. For ``"metal"``, velocities
        are converted from angstrom/fs to angstrom/ps. For ``"real"``,
        velocities are written in angstrom/fs. This argument is required to
        avoid silent unit assumptions.

    Returns
    -------
    None
        The function writes the file and returns ``None``.

    Raises
    ------
    ValueError
        If ``units`` is not supported or if the cell is not in restricted
        triclinic form.
    """
    _validate_lammps_units(units)
    type_ids, type_symbols = _type_ids_from_symbols(atoms.symbols)
    masses_by_type = _masses_by_type(atoms, type_symbols, type_ids)
    cell = atoms.cell
    positions = atoms.positions

    with Path(filename).open("w") as file:
        file.write(f"ISSE LAMMPS data file (units {units})\n\n")
        file.write(f"{len(atoms)} atoms\n")
        file.write(f"{len(type_symbols)} atom types\n\n")
        _write_lammps_data_box(file, cell)
        file.write("\nMasses\n\n")
        for atom_type, mass in masses_by_type.items():
            file.write(f"{atom_type} {mass:.16g}\n")
        file.write("\nAtoms # atomic\n\n")
        for index, (atom_type, position) in enumerate(
            zip(type_ids, positions), start=1
        ):
            file.write(
                f"{index} {atom_type} "
                f"{position[0]:.16g} {position[1]:.16g} {position[2]:.16g}\n"
            )

        if atoms.velocities is not None:
            velocities = _velocities_to_lammps(atoms.velocities, units)
            # Keep the section header immediately after the Atoms rows because
            # the current ISSE LAMMPS data parser looks at the next line only.
            file.write("Velocities\n\n")
            for index, velocity in enumerate(velocities, start=1):
                file.write(
                    f"{index} {velocity[0]:.16g} {velocity[1]:.16g} "
                    f"{velocity[2]:.16g}\n"
                )


def write_lammps_dump(
    filename: str | Path,
    trajectory: Atoms | Trajectory | list[Atoms],
    *,
    units: str,
    fractional: bool = False,
) -> None:
    """
    Write a LAMMPS dump trajectory.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path of the output LAMMPS dump file.
    trajectory : Atoms, Trajectory, or list of Atoms
        Atomic configuration or trajectory to write. A list of ``Atoms`` is
        accepted for in-memory trajectories. Frame positions and cell vectors
        are expected in angstrom. Velocities are expected in angstrom/fs and
        forces in eV/A.
    units : {"metal", "real"}
        LAMMPS unit style for the written dump. For ``"metal"``, velocities
        are converted from angstrom/fs to angstrom/ps and forces are written in
        eV/A. For ``"real"``, velocities are written in angstrom/fs and forces
        are converted to kcal/mol/A. This argument is required to avoid silent
        unit assumptions.
    fractional : bool, optional
        If ``True``, write positions as scaled ``xs ys zs``. If ``False``,
        write Cartesian ``x y z`` positions. The default is ``False``.

    Returns
    -------
    None
        The function writes the file and returns ``None``.

    Raises
    ------
    ValueError
        If ``units`` is not supported or if any frame cell is not in restricted
        triclinic form.
    """
    _validate_lammps_units(units)

    with Path(filename).open("w") as file:
        for iframe, atoms in enumerate(_iter_trajectory(trajectory)):
            timestep = _lammps_timestep(atoms.info.get("timestep"), iframe)
            file.write("ITEM: TIMESTEP\n")
            file.write(f"{timestep}\n")
            file.write("ITEM: NUMBER OF ATOMS\n")
            file.write(f"{len(atoms)}\n")
            _write_lammps_dump_box(file, atoms.cell)

            columns = ["id", "type", "mass"]
            position_values = (
                _scaled_positions(atoms) if fractional else atoms.positions
            )
            columns.extend(["xs", "ys", "zs"] if fractional else ["x", "y", "z"])

            velocities = None
            if atoms.velocities is not None:
                velocities = _velocities_to_lammps(atoms.velocities, units)
                columns.extend(["vx", "vy", "vz"])

            forces = None
            if atoms.forces is not None:
                forces = _forces_to_lammps(atoms.forces, units)
                columns.extend(["fx", "fy", "fz"])

            file.write("ITEM: ATOMS " + " ".join(columns) + "\n")

            type_ids, type_symbols = _type_ids_from_symbols(atoms.symbols)
            masses_by_type = _masses_by_type(atoms, type_symbols, type_ids)
            ids = atoms.arrays.get("id", np.arange(1, len(atoms) + 1))

            for iatom in range(len(atoms)):
                atom_type = int(type_ids[iatom])
                values: list[str] = [
                    str(int(ids[iatom])),
                    str(atom_type),
                    f"{masses_by_type[atom_type]:.16g}",
                    *(f"{value:.16g}" for value in position_values[iatom]),
                ]
                if velocities is not None:
                    values.extend(f"{value:.16g}" for value in velocities[iatom])
                if forces is not None:
                    values.extend(f"{value:.16g}" for value in forces[iatom])
                file.write(" ".join(values) + "\n")


def _iter_trajectory(
    trajectory: Atoms | Trajectory | list[Atoms],
) -> tuple[Atoms, ...] | list[Atoms] | Trajectory:
    if isinstance(trajectory, Atoms):
        return (trajectory,)
    return trajectory


def _scaled_positions(atoms: Atoms) -> np.ndarray:
    return np.linalg.solve(atoms.cell.T, atoms.positions.T).T


def _type_ids_from_symbols(symbols: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    type_symbols: list[str] = []
    type_ids = np.empty(len(symbols), dtype=np.int64)
    for index, symbol in enumerate(symbols):
        if symbol not in type_symbols:
            type_symbols.append(symbol)
        type_ids[index] = type_symbols.index(symbol) + 1
    return type_ids, type_symbols


def _masses_by_type(
    atoms: Atoms,
    type_symbols: Sequence[str],
    type_ids: np.ndarray,
) -> dict[int, float]:
    masses: dict[int, float] = {}
    for atom_type, symbol in enumerate(type_symbols, start=1):
        if atoms.masses is not None:
            selected = atoms.masses[type_ids == atom_type]
            masses[atom_type] = float(np.mean(selected))
        else:
            masses[atom_type] = mass_from_symbol(symbol)
    return masses


def _lammps_timestep(value: str | None, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        return int(value)
    except ValueError:
        return int(float(value))


def _validate_lammps_units(units: str) -> None:
    if units not in ("metal", "real"):
        raise ValueError("LAMMPS units must be either 'metal' or 'real'.")


def _velocities_to_lammps(velocities: np.ndarray, units: str) -> np.ndarray:
    if units == "metal":
        return velocities * PS_TO_FS
    if units == "real":
        return velocities
    raise ValueError("LAMMPS units must be either 'metal' or 'real'.")


def _forces_to_lammps(forces: np.ndarray, units: str) -> np.ndarray:
    if units == "metal":
        return forces
    if units == "real":
        return forces / KCAL_MOL_TO_EV
    raise ValueError("LAMMPS units must be either 'metal' or 'real'.")


def _ensure_restricted_triclinic(cell: np.ndarray, *, tolerance: float = 1e-6) -> None:
    if not np.allclose(cell[np.triu_indices(3, k=1)], 0.0, atol=tolerance):
        raise ValueError(
            "LAMMPS writers require cells in restricted-triclinic form "
            "with zero upper-triangular components."
        )


def _write_lammps_data_box(file, cell: np.ndarray) -> None:
    _ensure_restricted_triclinic(cell)
    lx = float(cell[0, 0])
    ly = float(cell[1, 1])
    lz = float(cell[2, 2])
    xy = float(cell[1, 0])
    xz = float(cell[2, 0])
    yz = float(cell[2, 1])

    file.write(f"0.0 {lx:.16g} xlo xhi\n")
    file.write(f"0.0 {ly:.16g} ylo yhi\n")
    file.write(f"0.0 {lz:.16g} zlo zhi\n")
    if any(abs(value) > 1e-14 for value in (xy, xz, yz)):
        file.write(f"{xy:.16g} {xz:.16g} {yz:.16g} xy xz yz\n")


def _write_lammps_dump_box(file, cell: np.ndarray) -> None:
    _ensure_restricted_triclinic(cell)
    lx = float(cell[0, 0])
    ly = float(cell[1, 1])
    lz = float(cell[2, 2])
    xy = float(cell[1, 0])
    xz = float(cell[2, 0])
    yz = float(cell[2, 1])

    if any(abs(value) > 1e-14 for value in (xy, xz, yz)):
        xlo_bound = min(0.0, xy, xz, xy + xz)
        xhi_bound = lx + max(0.0, xy, xz, xy + xz)
        ylo_bound = min(0.0, yz)
        yhi_bound = ly + max(0.0, yz)
        file.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
        file.write(f"{xlo_bound:.16g} {xhi_bound:.16g} {xy:.16g}\n")
        file.write(f"{ylo_bound:.16g} {yhi_bound:.16g} {xz:.16g}\n")
        file.write(f"0.0 {lz:.16g} {yz:.16g}\n")
    else:
        file.write("ITEM: BOX BOUNDS pp pp pp\n")
        file.write(f"0.0 {lx:.16g}\n")
        file.write(f"0.0 {ly:.16g}\n")
        file.write(f"0.0 {lz:.16g}\n")
