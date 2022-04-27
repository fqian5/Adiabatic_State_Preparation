"""
Adiabatic state preparation (ASP) simulation using pyscf fci solver.

Author: Seunghoon Lee, Jan 17, 2022
"""

import time, os
import numpy as np
from pyscf import lib, fci, cc
from pyscf.fci import direct_spin1, cistring
einsum = lib.einsum

def make_hop(eris, norb, nelec):
    h2e = direct_spin1.absorb_h1e(eris.h1e, eris.g2e, norb, nelec,.5)
    def _hop(c):
        return direct_spin1.contract_2e(h2e, c, norb, nelec)
    return _hop

def compute_update(ci, eris, h, RK=4):
    hop = make_hop(eris, ci.norb, ci.nelec)
    dr1 =  hop(ci.i)
    di1 = -hop(ci.r)
    if RK == 1:
        return dr1, di1
    if RK == 4:
        r = ci.r+dr1*h*0.5
        i = ci.i+di1*h*0.5
        norm = np.linalg.norm(r + 1j*i)
        r /= norm
        i /= norm
        dr2 =  hop(i)
        di2 = -hop(r)

        r = ci.r+dr2*h*0.5
        i = ci.i+di2*h*0.5
        norm = np.linalg.norm(r + 1j*i)
        r /= norm
        i /= norm
        dr3 =  hop(i)
        di3 = -hop(r)

        r = ci.r+dr3*h
        i = ci.i+di3*h
        norm = np.linalg.norm(r + 1j*i)
        r /= norm
        i /= norm
        dr4 =  hop(i)
        di4 = -hop(r)

        dr = (dr1+2.0*dr2+2.0*dr3+dr4)/6.0
        di = (di1+2.0*di2+2.0*di3+di4)/6.0
        return dr, di

def compute_weight(ci, fci):
    norb = ci.norb
    nelec= ci.nelec

    from pyscf.fci.addons import overlap
    rr1 = overlap(ci.r, fci, norb, nelec)
    ir1 = overlap(ci.i, fci, norb, nelec)
    ov = rr1 + 1j*ir1
    return np.vdot(ov,ov).real

def compute_energy(d1, d2, eris, ecore, time=None):
    h1e_a, h1e_b = eris.h1e, eris.h1e
    g2e_aa, g2e_ab, g2e_bb = eris.g2e, eris.g2e, eris.g2e
    h1e_a = np.array(h1e_a,dtype=complex)
    h1e_b = np.array(h1e_b,dtype=complex)
    g2e_aa = np.array(g2e_aa,dtype=complex)
    g2e_ab = np.array(g2e_ab,dtype=complex)
    g2e_bb = np.array(g2e_bb,dtype=complex)
    d1a, d1b = d1
    d2aa, d2ab, d2bb = d2
    # to physicts notation
    g2e_aa = g2e_aa.transpose(0,2,1,3)
    g2e_ab = g2e_ab.transpose(0,2,1,3)
    g2e_bb = g2e_bb.transpose(0,2,1,3)
    d2aa = d2aa.transpose(0,2,1,3)
    d2ab = d2ab.transpose(0,2,1,3)
    d2bb = d2bb.transpose(0,2,1,3)
    # antisymmetrize integral
    g2e_aa -= g2e_aa.transpose(1,0,2,3)
    g2e_bb -= g2e_bb.transpose(1,0,2,3)

    e  = einsum('pq,qp',h1e_a,d1a)
    e += einsum('PQ,QP',h1e_b,d1b)
    e += 0.25 * einsum('pqrs,rspq',g2e_aa,d2aa)
    e += 0.25 * einsum('PQRS,RSPQ',g2e_bb,d2bb)
    e +=        einsum('pQrS,rSpQ',g2e_ab,d2ab)
    return e.real + ecore

def kernel(eris, ci, tf, dt, RK=4):
    N = int(tf/dt+1e-6)
    d1as = []
    d1bs = []
    d2aas = []
    d2abs = []
    d2bbs = []
    for i in range(N):
        (d1a, d1b), (d2aa, d2ab, d2bb) = ci.compute_rdm12()
        dr, di = compute_update(ci, eris, dt, RK)
        r = ci.r + dt*dr
        i = ci.i + dt*di
        norm = np.linalg.norm(r + 1j*i)
        ci.r = r/norm 
        ci.i = i/norm
    d1a = np.array(d1a,dtype=complex)
    d1b = np.array(d1b,dtype=complex)
    d2aa = np.array(d2aa,dtype=complex)
    d2ab = np.array(d2ab,dtype=complex)
    d2bb = np.array(d2bb,dtype=complex)
    return (d1a, d1b), (d2aa, d2ab, d2bb)

class ERIs():
    def __init__(self, h1e, g2e, mo_coeff=None):
        ''' 
            h1e: 1-elec Hamiltonian in site basis 
            g2e: 2-elec Hamiltonian in site basis
                 chemists notation (pr|qs)=<pq|rs>
            mo_coeff: moa, mob 
        '''
        if mo_coeff is not None:
            mo = mo_coeff
            
            h1e = einsum('uv,up,vq->pq',h1e,mo,mo)
            g2e = einsum('uvxy,up,vr->prxy',g2e,mo,mo)
            g2e = einsum('prxy,xq,ys->prqs',g2e,mo,mo)

        self.mo_coeff = mo_coeff
        self.h1e = h1e
        self.g2e = g2e

class CIObject():
    def __init__(self, fcivec, norb, nelec):
        '''
           fcivec: ground state spin1 fcivec
           norb: size of site basis
           nelec: nea, neb
        '''
        self.r = fcivec.copy()
        self.i = np.zeros_like(fcivec)
        self.norb = norb
        self.nelec = nelec

    def compute_rdm1(self):
        rr = direct_spin1.make_rdm1s(self.r, self.norb, self.nelec)
        ii = direct_spin1.make_rdm1s(self.i, self.norb, self.nelec)
        ri = direct_spin1.trans_rdm1s(self.r, self.i, self.norb, self.nelec)
        d1a = rr[0] + ii[0] + 1j*(ri[0]-ri[0].T)
        d1b = rr[1] + ii[1] + 1j*(ri[1]-ri[1].T)
        return d1a, d1b

    def compute_rdm12(self):
        # 1pdm[q,p] = \langle p^\dagger q\rangle
        # 2pdm[p,r,q,s] = \langle p^\dagger q^\dagger s r\rangle
        rr1, rr2 = direct_spin1.make_rdm12s(self.r, self.norb, self.nelec)
        ii1, ii2 = direct_spin1.make_rdm12s(self.i, self.norb, self.nelec)
        ri1, ri2 = direct_spin1.trans_rdm12s(self.r, self.i, self.norb, self.nelec)
        # make_rdm12s returns (d1a, d1b), (d2aa, d2ab, d2bb)
        # trans_rdm12s returns (d1a, d1b), (d2aa, d2ab, d2ba, d2bb)
        d1a = rr1[0] + ii1[0] + 1j*(ri1[0]-ri1[0].T)
        d1b = rr1[1] + ii1[1] + 1j*(ri1[1]-ri1[1].T)
        d2aa = rr2[0] + ii2[0] + 1j*(ri2[0]-ri2[0].transpose(1,0,3,2))
        d2ab = rr2[1] + ii2[1] + 1j*(ri2[1]-ri2[2].transpose(3,2,1,0))
        d2bb = rr2[2] + ii2[2] + 1j*(ri2[3]-ri2[3].transpose(1,0,3,2))
        # 2pdm[r,p,s,q] = \langle p^\dagger q^\dagger s r\rangle
        d2aa = d2aa.transpose(1,0,3,2) 
        d2ab = d2ab.transpose(1,0,3,2)
        d2bb = d2bb.transpose(1,0,3,2)
        return (d1a, d1b), (d2aa, d2ab, d2bb)

if __name__ == '__main__':
    from pyscf import gto,scf,ao2mo,symm 
    import math
    import numpy as np

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
    mol.spin = 0 # na - nb
    mol.build()
    mol.symmetry = False
    mol.build()

    #==================================================================
    # Load files 
    #==================================================================
    from tools_io import loadERIs, fcidump_interpolate
    ecore_i, h1e_i, g2e_i, norb, nelec, ms = loadERIs('../1_FCIDUMP/2_FCIDUMP_initial_meanfield/output/FCIDUMP0') # Initial Hamiltonian
    ecore_f, h1e_f, g2e_f, norb, nelec, ms = loadERIs('../1_FCIDUMP/1_FCIDUMP_final/output/FCIDUMP10000')         # Final Hamiltonian
    info_i = (ecore_i, h1e_i, g2e_i)
    info_f = (ecore_f, h1e_f, g2e_f)

    eci = np.load("../1_FCIDUMP/2_FCIDUMP_initial_meanfield/output/E0.npy")[0]                                    # Ground-state (GS) energy for initial Hamiltonian  
    fcivec = np.load("../1_FCIDUMP/2_FCIDUMP_initial_meanfield/output/fcivec0.npy")[0]                            # GS FCI amplitude for initial Hamiltonian  
    fcivec_exact = np.load("../1_FCIDUMP/2_FCIDUMP_initial_meanfield/output/fcivec10000.npy")[0]                  # GS FCI amplitude for final Hamiltonian 
    ci = CIObject(fcivec, norb, nelec)
    ci_exact = CIObject(fcivec_exact, norb, nelec)

    #==================================================================
    # Set parameters 
    #==================================================================
    tf = 660                                                                                # T_ASPest = 660 for this example initial Hamiltonian
    dt = 0.01                                                                               # time step size
    minN = 0                                                                                # Initial time step (!= 0 for restart) 
    maxN= int(tf/dt + 1e-6)                                                                 # Total number of time steps
    scr = "./output/te_results_%d_%.2f"%(int(tf), dt)                                       # scratch folder
    os.mkdir(scr)

    start = time.time()
    f = open('./output/T%d_%.2f.out'%(int(tf), dt),'w')
    f.write('it e ov2\n')
    f.write('%21.15f %21.15f %22.20f\n'%(minN/maxN, eci, compute_weight(ci, fcivec_exact)))

    sampN = tf 
    for interpol in range(minN, maxN+1, 1):
        if interpol == maxN: ecore, h1e, h2e = fcidump_interpolate(info_i, info_f, 1.)
        else: ecore, h1e, h2e = fcidump_interpolate(info_i, info_f, (interpol+0.5)/maxN)
        eris = ERIs(h1e, h2e)
        (d1as, d1bs), (d2aas, d2abs, d2bbs) = kernel(eris, ci, dt, dt)
        if interpol % sampN == 0:
            end = time.time()
            string_wr = '%21.15f %21.15f %22.20f %22.1f\n'%((interpol+0.5)/maxN,
                                                             compute_energy((d1as, d1bs), (d2aas, d2abs, d2bbs), eris, ecore),
                                                             compute_weight(ci, fcivec_exact),
                                                             end-start) 
            f.write(string_wr)
            f.flush()
            np.save('%s/te%d_%d_%.2f.npy'%(scr, interpol, int(tf), dt), (ci.r, ci.i))

