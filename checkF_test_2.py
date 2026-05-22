import numpy as np
from convert_file import _read_lammps_alamode

cell_lmp = _read_lammps_alamode("../test/files/XFSET.lammpstrj")
print("Atom 0 force:", cell_lmp.get_forces()[0])
print("Atom 1023 force:", cell_lmp.get_forces()[1023])
