#!/usr/bin/env python3
"""
Run script for Fe2S2 ground state calculation
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ground_state_solver import solve_ground_state, save_ground_state

def main():
    print("Fe2S2 Ground State Calculation")
    print("=" * 50)
    
    # Check if FCIDUMP file exists
    fcidump_path = "../1_FCIDUMP/1_FCIDUMP_final/output/FCIDUMP10000"
    
    if not os.path.exists(fcidump_path):
        print(f"Error: FCIDUMP file not found at {fcidump_path}")
        print("Please run the 1_FCIDUMP/1_FCIDUMP_final/fcidump_final.py script first")
        return 1
    
    try:
        print(f"Loading Hamiltonian from: {fcidump_path}")
        
        # Solve for ground state and a few excited states
        energies, eigenvecs, norb, nelec = solve_ground_state(
            fcidump_path, 
            nroots=5,          # Get ground state + 4 excited states
            max_space=30,      # Krylov subspace size
            conv_tol=1e-10     # Tight convergence
        )
        
        # Save all results
        save_ground_state(energies, eigenvecs, norb, nelec)
        
        print("\n" + "=" * 50)
        print("CALCULATION COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Ground state energy: {energies[0]:.10f} Hartree")
        print(f"Ground state energy: {energies[0] * 27.2114:.6f} eV")
        
        if len(energies) > 1:
            print("\nExcitation energies:")
            for i, e in enumerate(energies[1:], 1):
                gap_hartree = e - energies[0]
                gap_ev = gap_hartree * 27.2114
                gap_cm1 = gap_hartree * 219474.63
                print(f"  State {i}: {gap_hartree:.6f} H, {gap_ev:.4f} eV, {gap_cm1:.1f} cm⁻¹")
        
        print(f"\nResults saved in: ./output/")
        return 0
        
    except Exception as e:
        print(f"\nError during calculation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)