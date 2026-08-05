#!/bin/bash
#SBATCH --job-name=silicon_v4
#SBATCH --partition=winston-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/silicon_v4_%j.out
#SBATCH --error=logs/silicon_v4_%j.err

# ============================================================================
# Silicon Sampling v4 — Multi-Country ESS11 — Slurm Submission Script
# ============================================================================
#
# IMPORTANT: Place ESS11e04_1.csv in ~/Winston_Code/data/
#
# Usage:
#   sbatch submit_extended_v4.sh                                          # Social trust, all 30 countries
#   BLOCK=trust COUNTRIES="GB FR DE" sbatch submit_extended_v4.sh         # Trust, 3 countries
#   BLOCK=trust COUNTRIES="GB FR DE" SAMPLE_N=500 sbatch submit_extended_v4.sh  # Trust, 500 per country
#   BLOCK=full COUNTRIES="GB" sbatch submit_extended_v4.sh                # All vars, GB only
#   BLOCK=all COUNTRIES="GB FR DE NL SE" sbatch submit_extended_v4.sh     # All blocks, 5 countries
#   BLOCK=rq8 COUNTRIES="GB FR DE" sbatch submit_extended_v4.sh           # Model comparison
#   BLOCK=overtime sbatch submit_extended_v4.sh                           # Overtime (needs multi-round data)
#
# Estimated GPU hours (Qwen 7B, per country, social trust only):
#   ~1,500 respondents × 3 variables = 4,500 prompts ≈ 15-20 min
#   30 countries ≈ 8-10 hours
#   3 countries ≈ 45-60 min
#
# ============================================================================

echo "======================================================================"
echo "SILICON SAMPLING v4 — Multi-Country ESS11"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "======================================================================"

# Configuration (with defaults)
BLOCK=${BLOCK:-"trust"}
MODEL=${MODEL:-"qwen"}
SEED=${SEED:-888}
N_SIMILAR=${N_SIMILAR:-3}
COUNTRIES=${COUNTRIES:-""}  # Empty = all 30 countries
SAMPLE_N=${SAMPLE_N:-""}    # Empty = use all respondents

echo "Configuration:"
echo "  BLOCK:     $BLOCK"
echo "  MODEL:     $MODEL"
echo "  SEED:      $SEED"
echo "  COUNTRIES: ${COUNTRIES:-ALL (30 countries)}"
echo "  SAMPLE_N:  ${SAMPLE_N:-ALL (no sampling)}"
echo "  N_SIMILAR: $N_SIMILAR (for RQ11)"
echo "======================================================================"

# Setup environment
cd ~/Winston_Code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Create directories
mkdir -p logs results data

# Verify data file exists
if [ ! -f "data/ESS11e04_1.csv" ]; then
    echo "ERROR: ESS11 data file not found at data/ESS11e04_1.csv"
    echo "Please place the ESS11 CSV file in ~/Winston_Code/data/"
    exit 1
fi

# Verify GPU
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Build command
CMD="python silicon_sampling_extended_v4.py --block $BLOCK --model $MODEL --seed $SEED --n_similar $N_SIMILAR"

# Add country filter if specified
if [ -n "$COUNTRIES" ]; then
    CMD="$CMD --countries $COUNTRIES"
fi

# Add per-country sampling if specified
if [ -n "$SAMPLE_N" ]; then
    CMD="$CMD --sample_per_country $SAMPLE_N"
fi

echo "Running: $CMD"
echo "======================================================================"

eval $CMD

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "EXPERIMENT COMPLETE"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "======================================================================"

echo ""
echo "Results saved to:"
ls -la results/

exit $EXIT_CODE
