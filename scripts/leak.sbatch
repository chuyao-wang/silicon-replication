#!/bin/bash
#SBATCH --job-name=leak
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=leak_slurm.log
cd ~/Winston_Code
python3 -u leak.py
