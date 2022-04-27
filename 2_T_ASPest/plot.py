import numpy as np
from tools_io import dumpERIs, loadERIs, fcidump_interpolate
from pyscf import gto,scf

def T_ASPest_weight(interpol, m_max):
    #==================================================================
    # Molecule
    #==================================================================
    mol = gto.Mole()
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
    mol.symmetry = False
    mol.build()
    
    #==================================================================
    # SCF
    #==================================================================
    mf = scf.RHF(mol)
    from pyscf import mcscf, fci
    from functools import reduce
    from pyscf.tools import fcidump
    from pyscf import ao2mo
    ecore_i, h1e_i, h2e_i, norb, nelec, ms = loadERIs('../1_FCIDUMP/2_FCIDUMP_initial_meanfield/output/FCIDUMP0')
    ecore_f, h1e_f, h2e_f, norb, nelec, ms = loadERIs('../1_FCIDUMP/1_FCIDUMP_final/output/FCIDUMP10000')
    ecore = ecore_f - ecore_i
    h1e = h1e_f - h1e_i
    h2e = h2e_f - h2e_i
    
    mci = fci.direct_spin1.FCI(mol)
    e = np.load("./output/E%d.npy" % interpol)
    fcivec = np.load("./output/fcivec%d.npy" % interpol)
    fcivec_exact = np.load("./output/fcivec10000.npy")[0]

    s0 = None 
    max_T_ASPest = 0.
    nroots = 10
    assert len(e) == len(fcivec) and len(fcivec) == nroots
    for i in range(nroots):
        mult = fci.spin_op.spin_square0(fcivec[i], norb, nelec)[1]
        if np.abs(mult - 1.) < 0.01:
            if s0 is None:
                s0 = i 
                e0 = e[i]
                continue
            rdm1, rdm2 = mci.trans_rdm12(fcivec[i], fcivec[s0], norb, nelec) 
            epsil = np.abs(np.einsum('pq,qp', h1e, rdm1) + 0.5 * np.einsum('pqrs,pqrs', h2e, rdm2))
            de = e[i] - e0 
            T_ASPest = epsil / (de * de)
            if T_ASPest > max_T_ASPest:
                max_T_ASPest = T_ASPest
    from pyscf.fci.addons import overlap
    ov0 = overlap(fcivec[s0], fcivec_exact, norb, nelec) 
    return max_T_ASPest, ov0 * ov0 
    
m_max = 10000
max_T_ASPest = 0.
s_l = [] 
ov2_l = [] 
ov2_init = 0.0
first = True 
for interpol in range(0, 10050, 50):
    T_ASPest, ov2 = T_ASPest_weight(interpol, m_max)
    if T_ASPest > max_T_ASPest: 
        max_T_ASPest = T_ASPest
    if first:
        ov2_init = ov2 
        first = False
    s_l.append(interpol / m_max)
    ov2_l.append(ov2)

np.save('./output/T_ASPest.npy', max_T_ASPest)
np.save('./output/init_weight.npy', ov2_init)
np.save('./output/weight_vs_s.npy', [s_l, ov2_l])

import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6,5))
ax.plot(s_l, ov2_l)
ax.set_xlabel('$\\mathrm{Adiabatic\\ transition\\ parameter\\ (s)}$')
ax.set_ylabel('$|\\langle \\Upsilon (s) | \\Psi_0 \\rangle|^2$')
plt.savefig("./output/weight_vs_s.png", dpi=200)

