"""
Generating FCIDUMP for the initial interacting (Dyall) Hamiltonian of ASP using pyscf.

Author: Seunghoon Lee, Jan 17, 2022
"""

import sys, h5py
import numpy as np
from pyscf import gto, scf, fci
from tools_io import loadERIs, dumpERIs, gen_small_cas_Ham, gen_Dyall_Ham 

#==================================================================
# Generating effective Hamiltonian for small active space 
#==================================================================
n_small_cas = int(sys.argv[1]) - 2
info_cas = loadERIs('../1_FCIDUMP_final/output/FCIDUMP10000')
info_small_cas = gen_small_cas_Ham(info_cas, n_small_cas) 
ecore_sc, h1e_sc, g2e_sc, no_sc, ne_sc, twos_sc = info_small_cas

#==================================================================
# solving small cas problem to get 1rdm
#==================================================================
mol = gto.Mole()
mol.verbose = 5
mol.atom ='''
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
mol.symmetry = False #True
mol.build()

mf = scf.RHF(mol)
mci = fci.direct_spin1.FCI(mol)
mci.conv_tol = 1e-8
mci.max_cycle = 2000
mci = fci.addons.fix_spin_(mci, shift=0.05, ss=0.0)    
_, fcivec = mci.kernel(h1e_sc, g2e_sc, no_sc, ne_sc, nroots=10, max_space=30, max_cycle=2000)
rdm1 = mci.make_rdm1(fcivec[0], no_sc, ne_sc) 

#==================================================================
# FCIDUMP for Dyall Hamiltonian 
#==================================================================
info_dyall = gen_Dyall_Ham(info_cas, n_small_cas, rdm1) 
fcidump = './output/FCIDUMP0'
header =""" &FCI NORB=  12,NELEC=14,MS2=0,
  ORBSYM=1,1,1,1,1,1,1,1,1,1,1,1,
  ISYM=1,
 &END
"""
ecore_d, h1e_d, g2e_d, norb_d, nelec_d, twos_d = info_dyall
dumpERIs(fcidump, header, int1e=h1e_d, int2e=g2e_d, ecore=ecore_d)

#==================================================================
# FCI calculation for the initial Hamiltonian
#==================================================================
mci = fci.direct_spin1.FCI(mol)
mci.conv_tol = 1e-8
mci.max_cycle = 2000
mci = fci.addons.fix_spin_(mci, shift=0.05, ss=0.0)    
e, ci = mci.kernel(h1e_d, g2e_d, norb_d, nelec_d, nroots=7, max_space=30, max_cycle=2000)
e = np.array(e)
# save eigenvalues and eigenstates for the initial Hamiltonian
np.save("./output/E0.npy", e+ecore_d)
np.save("./output/fcivec0.npy", ci)

