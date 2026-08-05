#!/bin/bash
#SBATCH --job-name=silicon_extended
#SBATCH --partition=winston-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/silicon_extended_%j.out
#SBATCH --error=logs/silicon_extended_%j.err

# ============================================================================
# Silicon Sampling Extended - Slurm Submission Script
# ============================================================================
# 
# Full Sample + RQ8 (Model Comparison) + RQ11 (Few-shot Anchoring)
#
# Usage:
#   sbatch submit_extended.sh                    # Run all with Qwen
#   BLOCK=1 sbatch submit_extended.sh            # Run Block 1 only
#   BLOCK=4 sbatch submit_extended.sh            # Run RQ11 Few-shot only
#   BLOCK=rq8 sbatch submit_extended.sh          # Run model comparison
#   MODEL=llama BLOCK=1 sbatch submit_extended.sh  # Run Block 1 with Llama
#
# ============================================================================

echo "======================================================================"
echo "SILICON SAMPLING EXTENDED"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "======================================================================"

# Configuration (with defaults)
BLOCK=${BLOCK:-"all"}
MODEL=${MODEL:-"qwen"}
SEED=${SEED:-888}
N_SIMILAR=${N_SIMILAR:-3}

echo "Configuration:"
echo "  BLOCK: $BLOCK"
echo "  MODEL: $MODEL"
echo "  SEED: $SEED"
echo "  N_SIMILAR: $N_SIMILAR (for RQ11)"
echo "======================================================================"

# Setup environment
cd ~/Winston_Code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Create directories
mkdir -p logs results

# Verify GPU
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run experiment
echo "Starting experiment..."
echo "======================================================================"

python silicon_sampling_extended.py \
    --block $BLOCK \
    --model $MODEL \
    --seed $SEED \
    --n_similar $N_SIMILAR

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "EXPERIMENT COMPLETE"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "======================================================================"

# List results
echo ""
echo "Results saved to:"
ls -la results/

exit $EXIT_CODE
