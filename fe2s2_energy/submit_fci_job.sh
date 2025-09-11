#!/bin/bash
#SBATCH -p batch
#SBATCH --job-name fe2s2_fci
#SBATCH -N 1 # nodes requested
#SBATCH -n 1 # tasks requested
#SBATCH -c 5 # cores requested
#SBATCH -t 3-00:00:00
#SBATCH --mem=500000 # memory in Mb
#SBATCH -o fci_output.out # send stdout to outfile
#SBATCH -e fci_error.out  # send stderr to errfile
module load miniforge/24.11.2-py312
source activate /cluster/tufts/lovelab/fqian03/condaenv/fe2s2
python fci_energy_calculation.py 1>fci_out 2>fci_error