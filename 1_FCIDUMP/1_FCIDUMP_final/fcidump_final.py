"""
Generating FCIDUMP for the final Hamiltonian of ASP using pyscf.

Author: Seunghoon Lee, Jan 17, 2022
"""

import numpy as np
from tools_io import dumpERIs
from pyscf import gto, scf, dft, ao2mo

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
# SCF
#==================================================================
mf = scf.sfx2c(scf.RKS(mol))
mf.chkfile = './output/hs_bp86.chk'
mf.max_cycle = 500
mf.conv_tol = 1.e-4
mf.xc = 'b88,p86' 
mf.scf()

mf2 = scf.newton(mf)
mf2.chkfile = './output/hs_bp86.chk'
mf2.conv_tol = 1.e-12
mf2.kernel()

#==================================================================
# Dump integrals
#==================================================================
mo = mf2.mo_coeff
norb = 12 
nelec = [7, 7]
from pyscf import mcscf
mc = mcscf.CASCI(mf, norb, nelec)
mc.mo_coeff = mo
act_cp_op = [78, 84]
act_d = [89, 90, 91, 92, 93, 94, 95, 96, 97, 98]
act_idx = act_cp_op + act_d
assert len(act_idx) == norb
mo = mc.sort_mo(act_idx)
mc.mo_coeff = mo
#from pyscf import tools
#tools.molden.from_mo(mol, 'fe2s2_actonly.molden', mo[:,86:98])

h1e, ecore = mc.get_h1eff()
g2e = mc.get_h2eff()
g2e = ao2mo.restore(1, g2e, norb)
header =""" &FCI NORB=  12,NELEC=14,MS2=0,
  ORBSYM=1,1,1,1,1,1,1,1,1,1,1,1,
  ISYM=1,
 &END
"""
dumpERIs('./output/FCIDUMP10000', header, int1e=h1e, int2e=g2e, ecore=ecore)       # Final Hamiltonian

