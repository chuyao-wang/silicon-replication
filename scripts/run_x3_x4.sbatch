#!/bin/bash
#SBATCH --job-name=silicon_x34
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=x34_slurm.log
# One submission for both remaining analyses. X3 needs the ESS microdata, X4 needs
# the raw per-response files; both are already in ~/Winston_Code. Outputs land in
# out/ and are tarred at the end so there is a single file to bring home.
set -e
cd ~/Winston_Code
mkdir -p out
echo "########## X3  leakage over all eight backstory modes"
python3 -u x3_leakage_all_modes.py
echo
echo "########## X4  which way the out-of-range responses miss"
python3 -u x4_out_of_range.py llama_3p llama_1p
echo
tar czf silicon_x34_results.tar.gz out x34_slurm.log
echo "########## packed silicon_x34_results.tar.gz"
ls -lh silicon_x34_results.tar.gz
