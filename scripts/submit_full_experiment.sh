#!/bin/bash
#SBATCH --job-name=silicon_full
#SBATCH --partition=winston-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/silicon_full_%j.out
#SBATCH --error=logs/silicon_full_%j.err

# =============================================================================
# Silicon Sampling Full Experiment
# All 3 Blocks, 7 RQs, 50 Variables, n=1000
# =============================================================================

echo "======================================================================"
echo "SILICON SAMPLING FULL EXPERIMENT"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "======================================================================"

# Environment setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Create directories
mkdir -p logs results

# Configuration
SAMPLE_SIZE=${SAMPLE_SIZE:-1000}
SEED=${SEED:-888}
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
BLOCK=${BLOCK:-"all"}

echo ""
echo "Configuration:"
echo "  Sample size: $SAMPLE_SIZE"
echo "  Random seed: $SEED"
echo "  Model: $MODEL"
echo "  Block: $BLOCK"
echo ""

# Run the experiment
cd ~/Winston_Code

python silicon_sampling_full_vllm.py \
    --block $BLOCK \
    --sample_size $SAMPLE_SIZE \
    --seed $SEED \
    --model "$MODEL" \
    --backstory_file data/ess_uk_with_backstories_v1.csv \
    --ess_file data/ESS9e03_2_GBonly.dta

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "EXPERIMENT COMPLETE"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "======================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Results saved to:"
    ls -la results/
else
    echo ""
    echo "✗ Experiment failed with errors"
    echo "Check logs/silicon_full_${SLURM_JOB_ID}.err for details"
fi
