#!/bin/bash
#SBATCH --job-name=silicon_x6
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=x6_slurm.log
# X6: is the Llama third-person arm echoing the persona instead of answering?
# Needs results/silicon_full_raw_llama_3p_seed888.csv and data/ESS Data/ESS11e04_1.csv,
# both already in ~/Winston_Code. Reads them in place and writes two small files.
set -e
cd ~/Winston_Code
mkdir -p out
python3 -u x6_echo_test.py
echo
echo "########## the file to read first"
head -80 out/echo_samples.txt
echo
tar czf silicon_x6_results.tar.gz out/echo_test.csv out/echo_samples.txt x6_slurm.log
ls -lh silicon_x6_results.tar.gz
