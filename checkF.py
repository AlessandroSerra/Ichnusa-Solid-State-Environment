import numpy as np
from ase.io import read
from convert_file import _read_lammps_alamode

# Read LAMMPS trajectory converted with ALAMODE
cell_lmp = _read_lammps_alamode("../test/files/XFSET.lammpstrj")

# Read reference extxyz file
cell_xyz = read("../test/files/XFSET.xyz", format="extxyz")

# Extract forces
xyz_forces = cell_xyz.get_forces()
lmp_forces = cell_lmp.get_forces()

# Compare forces
print(np.allclose(xyz_forces, lmp_forces))
