"""
Ground state eigenvalue solver for final Hamiltonian using sparse eigensolver.

Author: Generated for Fe2S2 ASP project
"""

import numpy as np
import scipy.sparse.linalg as sla
from scipy.sparse import csr_matrix
import os
from tools_io import loadERIs

def build_hamiltonian_matrix(h1e, g2e, norb, nelec):
    """
    Build full CI Hamiltonian matrix in determinant basis
    
    Args:
        h1e: one-electron integrals (norb x norb)
        g2e: two-electron integrals (norb x norb x norb x norb) 
        norb: number of orbitals
        nelec: number of electrons [na, nb]
    
    Returns:
        H: sparse Hamiltonian matrix
        ndets: number of determinants
    """
    from pyscf.fci import cistring
    
    na, nb = nelec
    if isinstance(nelec, int):
        na = nb = nelec // 2
    
    # Generate alpha and beta string lists
    strsa = cistring.gen_strings4orblist(range(norb), na)
    strsb = cistring.gen_strings4orblist(range(norb), nb) 
    
    ndeta = len(strsa)
    ndetb = len(strsb)
    ndets = ndeta * ndetb
    
    print(f"Building Hamiltonian matrix: {ndets} determinants")
    print(f"Alpha strings: {ndeta}, Beta strings: {ndetb}")
    
    # Use PySCF's direct_spin1 to build Hamiltonian
    from pyscf.fci import direct_spin1
    
    # Create CI vector space
    ci0 = np.zeros((ndeta, ndetb))
    ci0[0, 0] = 1.0  # Reference determinant
    
    # Use make_hdiag to get diagonal elements efficiently
    hdiag = direct_spin1.make_hdiag(h1e, g2e, norb, (na, nb))
    
    # For sparse eigenvalue solver, we need matrix-vector product function
    def matvec(c):
        c = c.reshape((ndeta, ndetb))
        hc = direct_spin1.contract_2e(direct_spin1.absorb_h1e(h1e, g2e, norb, (na, nb), 0.5), 
                                     c, norb, (na, nb))
        return hc.ravel()
    
    # Create linear operator
    from scipy.sparse.linalg import LinearOperator
    H_op = LinearOperator((ndets, ndets), matvec=matvec, dtype=float)
    
    return H_op, ndets, hdiag

def solve_ground_state(fcidump_path, nroots=1, max_space=20, conv_tol=1e-8):
    """
    Solve for ground state using sparse eigenvalue solver
    
    Args:
        fcidump_path: path to FCIDUMP file
        nroots: number of roots to compute
        max_space: maximum Krylov space dimension 
        conv_tol: convergence tolerance
    
    Returns:
        energies: ground state energies
        eigenvectors: ground state wavefunctions
    """
    print(f"Loading Hamiltonian from {fcidump_path}")
    
    # Load Hamiltonian
    ecore, h1e, g2e, norb, nelec, ms2 = loadERIs(fcidump_path)
    
    print(f"System: {norb} orbitals, {nelec} electrons, MS={ms2/2}")
    print(f"Core energy: {ecore:.10f}")
    
    # Build Hamiltonian matrix operator
    H_op, ndets, hdiag = build_hamiltonian_matrix(h1e, g2e, norb, [nelec//2, nelec//2])
    
    print(f"Solving eigenvalue problem for {nroots} roots...")
    
    # Use ARPACK to solve eigenvalue problem
    # Start with good initial guess using diagonal dominance
    v0 = np.zeros(ndets)
    min_idx = np.argmin(hdiag)
    v0[min_idx] = 1.0
    
    try:
        eigenvals, eigenvecs = sla.eigsh(H_op, k=nroots, which='SA', 
                                       v0=v0, maxiter=1000, tol=conv_tol,
                                       ncv=min(max_space, ndets-1))
        
        # Add core energy
        energies = eigenvals + ecore
        
        print(f"Converged eigenvalues:")
        for i, e in enumerate(energies):
            print(f"  State {i}: {e:.10f} Hartree")
        
        return energies, eigenvecs, norb, nelec
        
    except Exception as e:
        print(f"Eigenvalue solver failed: {e}")
        print("Trying with smaller convergence criteria...")
        
        # Fallback with relaxed criteria
        eigenvals, eigenvecs = sla.eigsh(H_op, k=nroots, which='SA',
                                       v0=v0, maxiter=2000, tol=1e-6,
                                       ncv=min(max_space*2, ndets-1))
        
        energies = eigenvals + ecore
        return energies, eigenvecs, norb, nelec

def save_ground_state(energies, eigenvecs, norb, nelec, output_dir="output"):
    """Save ground state results to files"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save energies
    np.save(os.path.join(output_dir, "ground_state_energies.npy"), energies)
    print(f"Saved energies to {output_dir}/ground_state_energies.npy")
    
    # Save eigenvectors  
    np.save(os.path.join(output_dir, "ground_state_vectors.npy"), eigenvecs)
    print(f"Saved eigenvectors to {output_dir}/ground_state_vectors.npy")
    
    # Save system info
    system_info = {
        'norb': norb,
        'nelec': nelec, 
        'ground_state_energy': energies[0]
    }
    np.save(os.path.join(output_dir, "system_info.npy"), system_info)
    print(f"Saved system info to {output_dir}/system_info.npy")
    
    # Save human-readable summary
    with open(os.path.join(output_dir, "ground_state_summary.txt"), 'w') as f:
        f.write("Fe2S2 Ground State Calculation Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"System: {norb} orbitals, {nelec} electrons\n")
        f.write(f"Ground state energy: {energies[0]:.10f} Hartree\n")
        f.write(f"Ground state energy: {energies[0] * 27.2114:.6f} eV\n\n")
        
        if len(energies) > 1:
            f.write("Excited state energies:\n")
            for i, e in enumerate(energies[1:], 1):
                gap_ev = (e - energies[0]) * 27.2114
                f.write(f"  State {i}: {e:.10f} Hartree ({gap_ev:.6f} eV gap)\n")

if __name__ == '__main__':
    # Path to final Hamiltonian FCIDUMP
    fcidump_path = "../1_FCIDUMP/1_FCIDUMP_final/output/FCIDUMP10000"
    
    print("Fe2S2 Ground State Eigenvalue Solver")
    print("=" * 40)
    
    try:
        # Solve for ground state (and a few excited states)
        energies, eigenvecs, norb, nelec = solve_ground_state(fcidump_path, nroots=3)
        
        # Save results
        save_ground_state(energies, eigenvecs, norb, nelec)
        
        print("\nGround state calculation completed successfully!")
        print(f"Ground state energy: {energies[0]:.10f} Hartree")
        
    except Exception as e:
        print(f"Calculation failed: {e}")
        import traceback
        traceback.print_exc()