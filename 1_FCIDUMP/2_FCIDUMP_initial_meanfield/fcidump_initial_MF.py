"""
Generating FCIDUMP for the initial mean-field Hamiltonian of ASP using pyscf.
Numbering is based on the order of the MF states with the largest FCI amplitude.

Author: Seunghoon Lee, Jan 17, 2022
"""

import numpy as np
from tools_io import dumpERIs, loadERIs 
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
# FCI calculation for the final Hamiltonian
#==================================================================
mf = scf.RHF(mol)
# load final Hamiltonian
ecore, h1e, g2e, norb, nelec, ms = loadERIs('../1_FCIDUMP_final/output/FCIDUMP10000')
mci = fci.direct_spin1.FCI(mol)
mci.conv_tol = 1e-8
mci.max_cycle = 2000
mci = fci.addons.fix_spin_(mci, shift=0.05, ss=0.0)    
e, ci = mci.kernel(h1e, g2e, norb, nelec, nroots=10, max_space=30, max_cycle=2000)
e = np.array(e)
# save eigenvalues and eigenstates for the final Hamiltonian
np.save("./output/E10000.npy", e+ecore)
np.save("./output/fcivec10000.npy", ci)

#==================================================================
# FCIDUMP for initial mean-field Hamiltonian
#==================================================================
from pyscf.fci import cistring
import h5py
f = h5py.File('../1_FCIDUMP_final/output/hs_bp86.chk', 'r')
fock = np.array(f['scf']['mo_energy'])
header =""" &FCI NORB=  12,NELEC=14,MS2=0,
  ORBSYM=1,1,1,1,1,1,1,1,1,1,1,1,
  ISYM=1,
 &END
"""
norb   = 12 
neleca = 7 
nelecb = 7 
tol    = 0.
ncore = (mol.nelectron - neleca - nelecb) // 2
act_cp_op = [78, 84]
act_d = [89, 90, 91, 92, 93, 94, 95, 96, 97, 98]
act_idx = np.array(act_cp_op + act_d) - 1
core_idx = np.array(list(set(range(98)) - set(act_idx)))
assert len(act_idx) == norb
fock_cas = fock[act_idx]
fock_cas = np.diag(fock_cas)
fock_core = np.sum(fock[core_idx])
print(fock_core)

na = cistring.num_strings(norb, neleca)
nb = cistring.num_strings(norb, nelecb)
assert(ci[0].shape == (na, nb))
addra, addrb = np.where(abs(ci[0]) > tol)
strsa = cistring.addrs2str(norb, neleca, addra)
strsb = cistring.addrs2str(norb, nelecb, addrb)
occa = cistring._strs2occslst(strsa, norb)
occb = cistring._strs2occslst(strsb, norb)
ci_tol = ci[0][addra,addrb]

# sort occupation strings in the order of the largest FCI amplitude for ground state
weight = np.square(ci_tol)
idx = np.argsort(weight)
weight = weight[idx][::-1]
ci_tol = ci_tol[idx][::-1]
occa = occa[idx][::-1]
occb = occb[idx][::-1]

# in this example script, I just write FCIDUMP for an arbitrary mean-field state
samp = [45]   # []
sequence = [i for i in range(norb)]
epsilon = 0.5  # energy shift in Hartree
iit = 0
for i in range(occa.shape[0]):
    if np.array_equal(occa[i], occb[i]):
        iit += 1
        if len(samp) > 0 and iit not in samp: continue
        print(iit, occa[i], occb[i], ci_tol[i])
        fock_shift = fock_cas.copy()
        for i in list(set(sequence) - set(occa[i])):
            fock_shift[i,i] += epsilon 
        dumpERIs('./output/FCIDUMP0', header, int1e=fock_shift, ecore=fock_core)

#==================================================================
# FCI calculation for the initial Hamiltonian
#==================================================================
ecore, h1e, g2e, norb, nelec, ms = loadERIs('./output/FCIDUMP0')
mci = fci.direct_spin1.FCI(mol)
mci = fci.addons.fix_spin_(mci, shift=0.05, ss=0.0)    
e, ci = mci.kernel(h1e, g2e, norb, nelec, nroots=10, max_space=30, max_cycle=2000)
e = np.array(e)
# save eigenvalues and eigenstates for the initial Hamiltonian
np.save("./output/E0.npy", e+ecore)
np.save("./output/fcivec0.npy", ci)

