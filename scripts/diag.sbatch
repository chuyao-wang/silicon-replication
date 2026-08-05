#!/bin/bash
#SBATCH --job-name=diag
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=diag_slurm.log
cd ~/Winston_Code
python3 -u d.py llama_3p llama_1p
