#!/usr/bin/env python3
"""
FCI energy calculation using PySCF from FCIDUMP file
"""

import numpy as np
from pyscf import fci, gto, scf
from pyscf.tools import fcidump
import os
import sys

def calculate_fci_energy(fcidump_file):
    """
    Calculate FCI energy from FCIDUMP file using PySCF
    
    Args:
        fcidump_file (str): Path to FCIDUMP file
        
    Returns:
        float: FCI ground state energy
    """
    
    print(f"Reading FCIDUMP file: {fcidump_file}")
    
    # Read FCIDUMP file
    mol, h1e, eri, norb, nelec, _ = fcidump.read(fcidump_file)
    
    print(f"Number of orbitals: {norb}")
    print(f"Number of electrons: {nelec}")
    
    # Create a dummy molecule object if mol is None
    if mol is None:
        mol = gto.Mole()
        mol.nelectron = sum(nelec) if isinstance(nelec, (list, tuple)) else nelec
        mol.spin = abs(nelec[0] - nelec[1]) if isinstance(nelec, (list, tuple)) else 0
        mol.build(dump_input=False, verbose=0)
    
    # Set up FCI solver
    cisolver = fci.FCI(mol)
    
    print("Starting FCI calculation...")
    
    # Calculate FCI energy
    efci, civec = cisolver.kernel(h1e, eri, norb, nelec)
    
    print(f"FCI ground state energy: {efci:.10f} Ha")
    print(f"FCI ground state energy: {efci * 27.2114:.6f} eV")
    
    return efci, civec

def main():
    # Path to FCIDUMP file
    fcidump_path = "../1_FCIDUMP/1_FCIDUMP_final/output/FCIDUMP10000"
    
    if not os.path.exists(fcidump_path):
        print(f"Error: FCIDUMP file not found at {fcidump_path}")
        sys.exit(1)
    
    try:
        energy, civec = calculate_fci_energy(fcidump_path)
        
        # Save results to file
        with open("fci_results.txt", "w") as f:
            f.write(f"FCI Ground State Energy Results\n")
            f.write(f"================================\n")
            f.write(f"Energy (Ha): {energy:.10f}\n")
            f.write(f"Energy (eV): {energy * 27.2114:.6f}\n")
            f.write(f"FCIDUMP file: {fcidump_path}\n")
        
        print(f"\nResults saved to fci_results.txt")
        
    except Exception as e:
        print(f"Error during FCI calculation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()