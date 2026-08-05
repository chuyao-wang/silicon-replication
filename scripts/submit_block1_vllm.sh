#!/bin/bash
#SBATCH --job-name=base
#SBATCH --partition=winston-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/block1_vllm_%j.out
#SBATCH --error=logs/block1_vllm_%j.err

# =============================================================================
# Silicon Sampling Block 1 - VLLM + Qwen (Quick Test n=50)
# =============================================================================
# Submit with: sbatch submit_block1_vllm.sh
# Monitor with: squeue -u $USER
# View output: tail -f logs/block1_vllm_*.out
# =============================================================================

echo "=============================================="
echo "Silicon Sampling - Block 1 (VLLM + Qwen)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p results

# Load conda
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source $(conda info --base)/etc/profile.d/conda.sh

# Activate environment
echo "Activating conda environment: base"
conda activate base

# Check if environment activated
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'base'"
    echo "Please run setup_vllm_environment.sh first"
    exit 1
fi

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Check data files exist
BACKSTORY_FILE="data/ess_uk_with_backstories_v1.csv"
ESS_FILE="data/ESS9e03_2_GBonly.dta"

if [ ! -f "$BACKSTORY_FILE" ]; then
    echo "ERROR: Backstory file not found: $BACKSTORY_FILE"
    exit 1
fi

if [ ! -f "$ESS_FILE" ]; then
    echo "ERROR: ESS file not found: $ESS_FILE"
    exit 1
fi

echo "Data files found:"
echo "  - $BACKSTORY_FILE"
echo "  - $ESS_FILE"
echo ""

# Run Block 1 (Quick Test: n=50)
echo "=============================================="
echo "Running Block 1: Core Validation (n=50)"
echo "=============================================="
echo ""

python block1_vllm_qwen.py \
    --backstory_file "$BACKSTORY_FILE" \
    --ess_file "$ESS_FILE" \
    --output_dir results/ \
    --sample_size 50

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✓ Block 1 completed successfully!"
    echo "=============================================="
    echo "Results saved to: results/"
    echo "End time: $(date)"
else
    echo ""
    echo "=============================================="
    echo "✗ Block 1 failed with errors"
    echo "=============================================="
    echo "Check logs/block1_vllm_${SLURM_JOB_ID}.err for details"
    exit 1
fi
