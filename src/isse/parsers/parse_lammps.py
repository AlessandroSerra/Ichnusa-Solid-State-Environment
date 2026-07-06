from __future__ import annotations

from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Sequence

import numpy as np

from ..constants import KCAL_MOL_TO_EV, PS_TO_FS, mass_from_symbol, symbol_from_mass
from ..structures import Atoms, Trajectory

logger = getLogger(__name__)

DUMP_HEADER_NLINES = 9


def parse_lammps(
    filename: str | Path,
    *,
    format: str,
    symbols: Sequence[str] | None = None,
    units: str | None = None,
) -> Atoms | Trajectory:
    """
    Parse a LAMMPS file by explicitly selecting the file format.

    This is a convenience dispatcher around ``parse_lammps_dump`` and
    ``parse_lammps_data``. LAMMPS dump files are returned as lazy
    trajectories, while LAMMPS data files are returned as single
    ``Atoms`` objects.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the LAMMPS file.
    format : {"dump", "data"}
        LAMMPS file format to parse. Use ``"dump"`` for LAMMPS dump
        trajectories and ``"data"`` for LAMMPS data files.
    symbols : Sequence[str] or None, optional
        Chemical symbols in LAMMPS atom-type order. For example,
        ``["Si", "O"]`` maps type 1 to Si and type 2 to O. This is required
        when chemical symbols cannot be inferred from masses.
    units : {"metal", "real"} or None, optional
        LAMMPS unit style. Required for ``format="dump"``. For
        ``format="data"``, the parser first attempts to infer the unit style
        from the first line of the data file. If inference fails, ``units`` is
        used as fallback.

    Returns
    -------
    Atoms or Trajectory
        ``Trajectory`` for ``format="dump"`` and ``Atoms`` for
        ``format="data"``.

    Raises
    ------
    ValueError
        If ``format`` is not ``"dump"`` or ``"data"``, or if ``units`` is
        missing when parsing a dump file.
    """

    if format == "data":
        return parse_lammps_data(filename, symbols=symbols, units=units)

    elif format == "dump":
        if units not in ("real", "metal"):
            raise ValueError(
                "Unrecognized or absent LAMMPS unit style. "
                "Specify either 'metal' or 'real'."
            )
        else:
            return parse_lammps_dump(filename, symbols=symbols, units=units)
    else:
        raise ValueError(
            f"{format} is not a supported LAMMPS format. Specify either 'dump' or 'data'"
        )


def parse_lammps_data(
    filename: str | Path,
    symbols: Sequence[str] | None = None,
    units: str | None = None,
) -> Atoms:
    """
    Parse a LAMMPS data file as a single atomistic configuration.

    The parser currently supports ``atom_style atomic`` data files, with
    optional ``Masses`` and ``Velocities`` sections. Atoms are sorted by
    their LAMMPS atom IDs before constructing the returned ``Atoms`` object.

    Parsed quantities are converted to the internal units used by the
    package:

        - positions and cell vectors: angstrom
        - velocities: angstrom / fs
        - masses: atomic mass units

    For ``units="metal"``, velocities are converted from angstrom / ps to
    angstrom / fs. For ``units="real"``, velocities are already interpreted
    as angstrom / fs.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the LAMMPS data file.
    symbols : Sequence[str] or None, optional
        Chemical symbols in LAMMPS atom-type order. For example,
        ``["Si", "O"]`` maps type 1 to Si and type 2 to O. Required if the
        data file does not contain a ``Masses`` section. If masses are present
        and ``symbols`` is not provided, atoms are labeled as ``"type_1"``,
        ``"type_2"``, etc.
    units : {"metal", "real"} or None, optional
        LAMMPS unit style. If ``None``, the parser attempts to infer the unit
        style from the first line of the data file. If the unit style cannot
        be inferred, it must be provided explicitly.

    Returns
    -------
    Atoms
        Atoms object containing the configuration read from the data file.

    Raises
    ------
    ValueError
        If the box section, ``Atoms`` section, unit style, or symbol/mass
        information is invalid or incomplete.
    NotImplementedError
        If the data file declares an atom style other than ``atomic``.
    """
    filepath = Path(filename)
    masses_by_type: dict[int, float] | None = None
    velocities: np.ndarray | None = None

    with filepath.open("r") as file:
        header = file.readline().strip()
        file.readline()  # blank line

        n_atoms = int(file.readline().split()[0])
        n_atom_types = int(file.readline().split()[0])

        file.readline()  # blank line

        box_lines = [file.readline() for _ in range(3)]
        next_line = file.readline()

        if "xy xz yz" in next_line:
            box_lines.append(next_line)
            file.readline()  # blank line
        else:
            if next_line.strip():
                raise ValueError(
                    f"Expected blank line after box bound, found {next_line!r}"
                )

        next_line = file.readline().strip()

        if next_line.lower() == "masses":
            masses_by_type = {}

            file.readline()  # blank line

            for _ in range(n_atom_types):
                fields = file.readline().split()
                atom_type = int(fields[0])
                mass = float(fields[1])
                masses_by_type[atom_type] = mass

            file.readline()  # blank line
            atoms_header = file.readline().strip()

        else:
            atoms_header = next_line
            logger.warning("No masses found in data file, using symbols.")

        if not atoms_header.lower().startswith("atoms"):
            raise ValueError(f"Expected 'Atoms' section, found {atoms_header!r}.")

        if "#" in atoms_header and "atomic" not in atoms_header.lower():
            raise NotImplementedError(
                f"{atoms_header!r} atom style not implemented. "
                "Only 'atomic' is currently supported."
            )

        file.readline()  # blank line

        atom_types: list[int] = []
        atom_ids: list[int] = []
        positions = np.empty((n_atoms, 3), dtype=np.float64)

        for atom_index in range(n_atoms):
            fields = file.readline().split()

            atom_id = int(fields[0])
            atom_type = int(fields[1])
            atom_ids.append(atom_id)
            atom_types.append(atom_type)

            positions[atom_index] = [
                float(fields[2]),
                float(fields[3]),
                float(fields[4]),
            ]

        next_line = file.readline()

        if "velocities" in next_line.strip().lower():
            file.readline()
            velocities = np.asarray(
                [
                    list(map(float, file.readline().split()[1:4]))
                    for _ in range(n_atoms)
                ],
                dtype=np.float64,
            )

    if "metal" in header.lower():
        time_factor = PS_TO_FS
    elif "real" in header.lower():
        time_factor = 1.0
    elif units == "metal":
        time_factor = PS_TO_FS
    elif units == "real":
        time_factor = 1.0
    else:
        raise ValueError(
            "No valid units reference found in data file. "
            "Please specify units='real' or units='metal'."
        )

    cell, origin = _parse_lammps_data_box(box_lines)

    order = np.argsort(np.asarray(atom_ids, dtype=np.int32))
    atom_types_ordered = np.asarray(atom_types, dtype=np.int32)[order]
    positions_ordered = np.ascontiguousarray(
        positions[order] - origin, dtype=np.float64
    )

    if velocities is not None:
        velocities_ordered = np.ascontiguousarray(velocities[order] / time_factor)
    else:
        velocities_ordered = None

    if symbols is not None:
        if len(symbols) != n_atom_types:
            raise ValueError(f"Expected {n_atom_types} symbols, found {len(symbols)}.")

        symbols_ordered = [
            symbols[int(atom_type) - 1] for atom_type in atom_types_ordered
        ]
    else:
        symbols_ordered = [f"type_{int(atom_type)}" for atom_type in atom_types_ordered]

    if masses_by_type is not None:
        masses = np.asarray(
            [masses_by_type[int(atom_type)] for atom_type in atom_types_ordered],
            dtype=np.float64,
        )
    else:
        if symbols is None:
            raise ValueError(
                "LAMMPS data file does not contain a Masses section. "
                "Please provide symbols in LAMMPS atom-type order."
            )

        masses = np.asarray(
            [mass_from_symbol(symbol) for symbol in symbols_ordered],
            dtype=np.float64,
        )
    return Atoms(
        symbols=symbols_ordered,
        cell=cell,
        positions=positions_ordered,
        velocities=velocities_ordered,
        masses=masses,
    )


def parse_lammps_dump(
    filename: str | Path, units: str, symbols: Sequence[str] | None = None
) -> Trajectory:
    """
    Parse a LAMMPS dump file as a lazy trajectory.

    LAMMPS dump quantities are interpreted using the following units:
        - time: fs
        - positions and cell vectors: angstrom
        - velocities: angstrom / fs
        - forces: eV / angstrom
        - masses: atomic mass units

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the LAMMPS dump file.
    units : {"metal", "real"}
        LAMMPS unit style. Required to avoid silent unit assumptions. For
        ``"metal"``, velocities are converted from angstrom / ps to angstrom / fs.
        For ``"real"``, velocities are interpreted as angstrom / fs and forces
        are converted from kcal / mol / angstrom to eV / angstrom.
    symbols : Sequence[str] or None, optional
        Chemical symbols in LAMMPS atom-type order. For example,
        ``["Si", "O"]`` maps type 1 to Si and type 2 to O. Required when
        the dump contains atom types but does not contain atomic masses.

    Returns
    -------
    Trajectory
        Lazy trajectory providing access to individual frames.

    Raises
    ------
    ValueError
        If a frame header is incomplete, if no valid frames are found, or if
        chemical symbols cannot be inferred from masses or atom types.
    """

    if units not in ("real", "metal"):
        raise ValueError(
            "Unrecognized or absent LAMMPS unit style. "
            "Specify either 'metal' or 'real'."
        )

    filepath = Path(filename)
    offsets: list[int] = []
    first_properties: list[str] | None = None

    with filepath.open("rb") as f:
        while True:
            offset = f.tell()
            first_line = f.readline()

            if not first_line:
                break

            if not first_line.startswith(b"ITEM: TIMESTEP"):
                raise ValueError(f"Invalid LAMMPS frame at byte offset {offset}")

            header = [first_line] + [
                f.readline() for _ in range(DUMP_HEADER_NLINES - 1)
            ]

            if any(not line for line in header):
                raise ValueError(f"Incomplete LAMMPS header at byte offset {offset}")

            natoms = int(header[3])
            offsets.append(offset)

            if first_properties is None:
                first_properties = header[8].decode("utf-8").split()[2:]

            for _ in range(natoms):
                if not f.readline():
                    raise ValueError(f"Incomplete LAMMPS frame at byte offset {offset}")

    if not offsets:
        raise ValueError("No LAMMPS dump frames found")

    assert first_properties is not None

    has_masses = "mass" in first_properties
    has_types = "type" in first_properties

    if not has_masses and not has_types:
        raise ValueError(
            "Cannot determine chemical symbols because neither "
            "'mass' nor 'type' is present"
        )

    if not has_masses and has_types and symbols is None:
        raise ValueError(
            "The dump contains atom types but no masses. "
            "Provide symbols in LAMMPS type order"
        )

    frame_reader = partial(_read_frame_lammps_dump, type_symbols=symbols, units=units)

    logger.info(f"Successfully loaded {len(offsets)} frames from {filepath}")

    return Trajectory(
        path=filepath,
        offsets=offsets,
        reader=frame_reader,
    )


def _read_frame_lammps_dump(
    filepath: Path,
    offset: int,
    *,
    units: str,
    type_symbols: Sequence[str] | None = None,
) -> Atoms:
    """
    Read one frame from a LAMMPS dump file.

    The frame is read from a byte offset previously collected by
    ``parse_lammps_dump``. Positions, velocities, forces, masses, atom IDs
    and atom types are parsed when present in the dump columns.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to the LAMMPS dump file.
    offset : int
        Byte offset marking the beginning of the frame.
    type_symbols : Sequence[str] or None, optional
        Chemical symbols in LAMMPS atom-type order. Required if the dump
        contains atom types but no masses.
    units : {"metal", "real"}
        LAMMPS unit style.

    Returns
    -------
    Atoms
        Parsed atomistic configuration for the requested frame.

    Raises
    ------
    ValueError
        If the frame header is invalid or incomplete, if no supported
        position fields are present, if the unit style is invalid, or if
        chemical symbols cannot be inferred.
    """

    if units == "real":
        time_factor = 1
        energy_factor = KCAL_MOL_TO_EV
    else:
        time_factor = PS_TO_FS
        energy_factor = 1

    with filepath.open("rb") as file:
        file.seek(offset)

        header = [file.readline() for _ in range(DUMP_HEADER_NLINES)]

        if any(not line for line in header):
            raise ValueError(f"Incomplete LAMMPS dump header at offset {offset}")

        if not header[0].startswith(b"ITEM: TIMESTEP"):
            raise ValueError(f"Invalid LAMMPS frame at offset {offset}")

        timestep = int(header[1])
        natoms = int(header[3])

        box_header = header[4].decode("utf-8").strip()
        box_lines = [line.decode("utf-8").strip() for line in header[5:8]]

        properties = header[8].decode("utf-8").split()[2:]

        atom_lines = [file.readline() for _ in range(natoms)]

    if any(not line for line in atom_lines):
        raise ValueError(f"Incomplete LAMMPS dump frame at timestep {timestep}")

    cell, origin = _parse_lammps_dump_box(
        box_header,
        box_lines,
    )

    columns = {name: index for index, name in enumerate(properties)}

    has_positions = all(name in columns for name in ("x", "y", "z"))
    has_scaled_positions = all(name in columns for name in ("xs", "ys", "zs"))
    has_unwrapped_positions = all(name in columns for name in ("xu", "yu", "zu"))
    has_velocities = all(name in columns for name in ("vx", "vy", "vz"))
    has_forces = all(name in columns for name in ("fx", "fy", "fz"))
    has_masses = "mass" in columns
    has_types = "type" in columns
    has_ids = "id" in columns

    if not any(
        (
            has_positions,
            has_scaled_positions,
            has_unwrapped_positions,
        )
    ):
        raise ValueError("No supported position fields found in LAMMPS dump")

    if not has_masses and not has_types:
        raise ValueError(
            "Cannot determine chemical symbols because neither "
            "'mass' nor 'type' is present"
        )

    if not has_masses and type_symbols is None:
        raise ValueError(
            "The dump contains atom types but no masses. "
            "Provide symbols in LAMMPS type order"
        )

    if has_positions:
        position_columns = tuple(columns[name] for name in ("x", "y", "z"))
        positions_raw = np.empty(
            (natoms, 3),
            dtype=np.float64,
        )

    if has_scaled_positions:
        scaled_position_columns = tuple(columns[name] for name in ("xs", "ys", "zs"))
        scaled_positions_raw = np.empty(
            (natoms, 3),
            dtype=np.float64,
        )

    if has_unwrapped_positions:
        unwrapped_position_columns = tuple(columns[name] for name in ("xu", "yu", "zu"))
        unwrapped_positions_raw = np.empty(
            (natoms, 3),
            dtype=np.float64,
        )

    if has_velocities:
        velocity_columns = tuple(columns[name] for name in ("vx", "vy", "vz"))
        velocities = np.empty(
            (natoms, 3),
            dtype=np.float64,
        )

    if has_forces:
        force_columns = tuple(columns[name] for name in ("fx", "fy", "fz"))
        forces = np.empty(
            (natoms, 3),
            dtype=np.float64,
        )

    if has_masses:
        masses = np.empty(
            natoms,
            dtype=np.float64,
        )

    if has_ids:
        ids = np.empty(
            natoms,
            dtype=np.int64,
        )

    if has_types:
        types = np.empty(
            natoms,
            dtype=np.int64,
        )

    atom_symbols: list[str] = []

    for atom_index, line in enumerate(atom_lines):
        values = line.split()

        if len(values) != len(properties):
            raise ValueError(
                "Unexpected number of columns at "
                f"timestep {timestep}, atom row {atom_index}"
            )

        if has_positions:
            positions_raw[atom_index] = [
                float(values[column]) for column in position_columns
            ]

        if has_scaled_positions:
            scaled_positions_raw[atom_index] = [
                float(values[column]) for column in scaled_position_columns
            ]

        if has_unwrapped_positions:
            unwrapped_positions_raw[atom_index] = [
                float(values[column]) for column in unwrapped_position_columns
            ]

        if has_velocities:
            velocities[atom_index] = [
                (float(values[column]) / time_factor) for column in velocity_columns
            ]

        if has_forces:
            forces[atom_index] = [
                (float(values[column]) * energy_factor) for column in force_columns
            ]

        if has_ids:
            ids[atom_index] = int(values[columns["id"]])

        if has_types:
            atom_type = int(values[columns["type"]])
            types[atom_index] = atom_type

        if has_masses:
            mass = float(values[columns["mass"]])
            masses[atom_index] = mass

            symbol = symbol_from_mass(mass)

            if symbol is None:
                raise ValueError(f"Could not infer chemical symbol from mass {mass}")

            atom_symbols.append(symbol)
        else:
            assert type_symbols is not None

            try:
                atom_symbols.append(type_symbols[atom_type - 1])
            except IndexError as error:
                raise ValueError(
                    f"No chemical symbol was provided for LAMMPS atom type {atom_type}"
                ) from error

    if has_positions:
        positions = positions_raw - origin
    elif has_scaled_positions:
        positions = scaled_positions_raw @ cell
    elif has_unwrapped_positions:
        unwrapped_positions = unwrapped_positions_raw - origin
        scaled_positions = np.linalg.solve(
            cell.T,
            unwrapped_positions.T,
        ).T
        positions = (scaled_positions % 1.0) @ cell
    else:
        raise ValueError(
            f"No supported position format found in {filepath} LAMMPS dump"
        )

    positions = np.asarray(positions, dtype=np.float64)

    if has_unwrapped_positions:
        unwrapped_positions = unwrapped_positions_raw - origin

    arrays: dict[str, np.ndarray] = {}

    if has_ids:
        arrays["id"] = ids

    if has_types:
        arrays["type"] = types

    if has_unwrapped_positions:
        return Atoms(
            symbols=atom_symbols,
            cell=cell,
            positions=positions,
            unwrapped_positions=unwrapped_positions,
            velocities=velocities if has_velocities else None,
            masses=masses if has_masses else None,
            forces=forces if has_forces else None,
            arrays=arrays,
            info={"timestep": str(timestep)},
        )

    return Atoms(
        symbols=atom_symbols,
        cell=cell,
        positions=positions,
        velocities=velocities if has_velocities else None,
        masses=masses if has_masses else None,
        forces=forces if has_forces else None,
        arrays=arrays,
        info={"timestep": str(timestep)},
    )


def _parse_lammps_dump_box(
    box_header: str,
    box_lines: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse orthogonal or triclinic box bounds from a LAMMPS dump frame.

    LAMMPS dump files report triclinic boxes using bounded box limits and
    tilt factors. For triclinic boxes, this function converts the bounded
    limits to the lower box origin and the corresponding cell matrix.

    Parameters
    ----------
    box_header : str
        ``ITEM: BOX BOUNDS`` header line. The presence of ``xy``, ``xz`` and
        ``yz`` is used to detect a triclinic box.
    box_lines : list of str
        Three box-bound lines from the dump frame. Orthogonal boxes contain
        ``lo hi`` values. Triclinic boxes contain ``lo_bound hi_bound tilt``
        values.

    Returns
    -------
    cell : numpy.ndarray
        Simulation cell with shape ``(3, 3)`` and lattice vectors stored by row.
    origin : numpy.ndarray
        Lower box origin with shape ``(3,)``.
    """

    values = [list(map(float, line.split())) for line in box_lines]

    if all(tilt in box_header.split() for tilt in ("xy", "xz", "yz")):
        xlo_bound, xhi_bound, xy = values[0]
        ylo_bound, yhi_bound, xz = values[1]
        zlo, zhi, yz = values[2]

        xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
        xhi = xhi_bound - max(0.0, xy, xz, xy + xz)

        ylo = ylo_bound - min(0.0, yz)
        yhi = yhi_bound - max(0.0, yz)
    else:
        xlo, xhi = values[0][:2]
        ylo, yhi = values[1][:2]
        zlo, zhi = values[2][:2]

        xy = 0.0
        xz = 0.0
        yz = 0.0

    cell = np.array(
        [
            [xhi - xlo, 0.0, 0.0],
            [xy, yhi - ylo, 0.0],
            [xz, yz, zhi - zlo],
        ],
        dtype=np.float64,
    )

    origin = np.array([xlo, ylo, zlo], dtype=np.float64)

    return cell, origin


def _parse_lammps_data_box(
    box_lines: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse orthogonal or triclinic box bounds from a LAMMPS data file.

    The first three lines define the lower and upper bounds along the
    Cartesian axes. If a fourth line is present and contains ``xy xz yz``,
    it is interpreted as the triclinic tilt-factor line.

    Parameters
    ----------
    box_lines : list of str
        Box-bound lines from a LAMMPS data file. The first three lines must
        contain ``xlo xhi``, ``ylo yhi`` and ``zlo zhi``. An optional fourth
        line may contain ``xy xz yz``.

    Returns
    -------
    cell : numpy.ndarray
        Simulation cell with shape ``(3, 3)`` and lattice vectors stored by
        row.
    origin : numpy.ndarray
        Lower box origin with shape ``(3,)``.
    """
    xlo, xhi = map(float, box_lines[0].split()[:2])
    ylo, yhi = map(float, box_lines[1].split()[:2])
    zlo, zhi = map(float, box_lines[2].split()[:2])

    xy = 0.0
    xz = 0.0
    yz = 0.0

    if len(box_lines) > 3:
        fields = box_lines[3].split()

        if len(fields) >= 6 and fields[3:6] == ["xy", "xz", "yz"]:
            xy, xz, yz = map(float, fields[:3])

    cell = np.array(
        [
            [xhi - xlo, 0.0, 0.0],
            [xy, yhi - ylo, 0.0],
            [xz, yz, zhi - zlo],
        ],
        dtype=np.float64,
    )

    origin = np.array([xlo, ylo, zlo], dtype=np.float64)

    return cell, origin
