from __future__ import annotations

from logging import getLogger
from pathlib import Path

import numpy as np

from isse.constants import mass_from_symbol
from isse.structures import Atoms

logger = getLogger(__name__)


def parse_poscar(filename: str | Path) -> Atoms:
    """
    Parse a POSCAR file.

    POSCAR quantities are interpreted using the following units:
        - cell vectors: angstrom
        - Cartesian positions: angstrom
        - masses: atomic mass units

    Direct coordinates are interpreted as fractional coordinates and are
    converted to Cartesian coordinates using the POSCAR cell.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the POSCAR file.

    Returns
    -------
    Atoms
        Atoms object representing the POSCAR configuration.

    Raises
    ------
    ValueError
        If an invalid coordinate type is encountered.
    """
    filepath = Path(filename)

    with filepath.open("r") as file:
        header = file.readline().strip()
        cell_scaling_factor = float(file.readline().strip())
        if cell_scaling_factor <= 0.0:
            raise NotImplementedError(
                "Only positive POSCAR scaling factors are currently supported."
            )

        cell = (
            np.asarray(
                [list(map(float, file.readline().split())) for _ in range(3)],
                dtype=np.float64,
            )
            * cell_scaling_factor
        )

        species = file.readline().split()
        atoms_per_species = list(map(int, file.readline().split()))
        n_atoms = int(sum(atoms_per_species))

        coords_type = file.readline().strip().lower()

        if coords_type.startswith("s"):
            coords_type = file.readline().strip().lower()

        raw_positions = np.asarray(
            [list(map(float, file.readline().split()[:3])) for _ in range(n_atoms)],
            dtype=np.float64,
        )

    symbols: list[str] = []

    for symbol, count in zip(species, atoms_per_species, strict=True):
        symbols.extend([symbol] * count)

    masses = np.asarray(
        [mass_from_symbol(symbol) for symbol in symbols],
        dtype=np.float64,
    )

    if coords_type.startswith("d"):
        positions = raw_positions @ cell

    elif coords_type.startswith("c") or coords_type.startswith("k"):
        positions = raw_positions * cell_scaling_factor

    elif coords_type.startswith("s"):
        logger.warning("Selective dynamics informations are currently ignored.")

    else:
        raise ValueError(f"{coords_type!r} is not a valid POSCAR coordinate format.")

    logger.info(f"Succesfully loaded atomic configuration from {filepath}")

    info = {"header": header}

    return Atoms(
        symbols=symbols,
        cell=cell,
        positions=positions,
        masses=masses,
        info=info,
    )
