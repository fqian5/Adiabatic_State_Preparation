"""
Solving eigenvalue problem along the adiabatic path using pyscf FCI solver.

Author: Seunghoon Lee, Jan 17, 2022
"""

import numpy as np
from tools_io import dumpERIs, loadERIs, fcidump_interpolate 
from pyscf import gto, scf, fci
 
#==================================================================
# Molecule
#==================================================================
mol = gto.Mole()
mol.verbose = 5
mol.atom = '''
 Fe                 5.22000000    1.05000000   -7.95000000
 S                  3.86000000   -0.28000000   -9.06000000
 S                  5.00000000    0.95000000   -5.66000000
 S                  4.77000000    3.18000000   -8.74000000
 S                  7.23000000    0.28000000   -8.38000000
 Fe                 5.88000000   -1.05000000   -9.49000000
 S                  6.10000000   -0.95000000  -11.79000000
 S                  6.33000000   -3.18000000   -8.71000000
 C                  6.00000000    4.34000000   -8.17000000
 H                  6.46000000    4.81000000   -9.01000000
 H                  5.53000000    5.08000000   -7.55000000
 H                  6.74000000    3.82000000   -7.60000000
 C                  3.33000000    1.31000000   -5.18000000
 H                  2.71000000    0.46000000   -5.37000000
 H                  3.30000000    1.54000000   -4.13000000
 H                  2.97000000    2.15000000   -5.73000000
 C                  5.10000000   -4.34000000   -9.28000000
 H                  5.56000000   -5.05000000   -9.93000000
 H                  4.67000000   -4.84000000   -8.44000000
 H                  4.34000000   -3.81000000   -9.81000000
 C                  7.77000000   -1.31000000  -12.27000000
 H                  7.84000000   -1.35000000  -13.34000000
 H                  8.42000000   -0.54000000  -11.90000000
 H                  8.06000000   -2.25000000  -11.86000000
'''
mol.basis = 'tzp-dkh'
mol.charge = -2
mol.spin = 0
mol.build()
mol.symmetry = False
mol.build()

#==================================================================
# Linearly interpolated Hamiltonian
#==================================================================
# load final Hamiltonian
import sys
interpol = int(sys.argv[1])
ecore_i, h1e_i, g2e_i, norb, nelec, ms = loadERIs('../1_FCIDUMP/2_FCIDUMP_initial_meanfield/output/FCIDUMP0')
ecore_f, h1e_f, g2e_f, norb, nelec, ms = loadERIs('../1_FCIDUMP/1_FCIDUMP_final/output/FCIDUMP10000')
info_i = (ecore_i, h1e_i, g2e_i)
info_f = (ecore_f, h1e_f, g2e_f)
ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, interpol/10000.)

#==================================================================
# FCI calculation
#==================================================================
mf = scf.RHF(mol)
mci = fci.direct_spin1.FCI(mol)
mci.spin = 0 
mci.conv_tol = 1e-8
mci.max_cycle = 2000
mci = fci.addons.fix_spin_(mci, shift=0.05, ss=0.0)    
e, ci = mci.kernel(h1e, g2e, norb, nelec, nroots=10, max_space=30, max_cycle=2000)
e = np.array(e)
# save eigenvalues and eigenstates for the final Hamiltonian
np.save("./output/E%d.npy" % interpol, e+ecore)
np.save("./output/fcivec%d.npy" % interpol, ci)

