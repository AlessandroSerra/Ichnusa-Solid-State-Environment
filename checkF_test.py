import numpy as np
from ase.io import read
from convert_file import _read_lammps_alamode

cell_lmp = _read_lammps_alamode("../test/files/XFSET.lammpstrj")
cell_xyz = read("../test/files/XFSET.xyz", format="extxyz")

xyz_forces = cell_xyz.get_forces()
lmp_forces = cell_lmp.get_forces()

print("XYZ forces:\n", xyz_forces[:3])
print("LMP forces:\n", lmp_forces[:3])
