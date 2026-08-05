#!/bin/bash
#SBATCH --job-name=silicon_v5
#SBATCH --partition=winston-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/silicon_v5_%j.out
#SBATCH --error=logs/silicon_v5_%j.err

# ============================================================================
# Silicon Sampling v5 — Slurm Submission Script (Final)
# ============================================================================
# Addresses supervisor feedback (Patrick Sturgis & Daniel De Kadt, Feb 2025):
#   1. FULL SAMPLE: No per-country cap (use all respondents)
#   2. EXPANDED BACKSTORY: 10 additional demographic variables
#   3. LLAMA MODEL: Comparison with Llama 3.1-8B
#   4. PLOTS: Auto-generate scatterplots after each analysis
#   5. ABLATION: v4 vs v5 backstory comparison (500/country, 3 countries)
#   6. ICC: Explains why ppltrst > pplfair/pplhlp (no GPU needed)
#
# ===== RECOMMENDED EXECUTION SEQUENCE =====
#
#   Step 0: ICC analysis (no GPU, run on login node, <1 min)
#   python silicon_sampling_extended_v5.py --block icc
#
#   Step 1: Backstory ablation (GB/DE/FI, 500/country, ~35 min GPU)
#   BLOCK=ablation sbatch submit_extended_v5.sh
#
#   Step 2: Cross-sectional, Qwen, FULL SAMPLE (~7-8h GPU)
#   BLOCK=trust sbatch submit_extended_v5.sh
#
#   Step 3: Cross-sectional, Llama, FULL SAMPLE (~7-8h GPU)
#   BLOCK=trust MODEL=llama sbatch submit_extended_v5.sh
#
#   Step 4: Generate all plots (on login node)
#   python plot_silicon_results.py --model qwen
#   python plot_silicon_results.py --model llama
#
#   Step 5 (optional): Overtime — highlight but don't expand
#   BLOCK=overtime SAMPLE_N=200 sbatch submit_extended_v5.sh
#
# ===== CUSTOM EXAMPLES =====
#
#   # Ablation with custom countries / sample
#   BLOCK=ablation COUNTRIES="GB FR PL" SAMPLE_N=300 sbatch submit_extended_v5.sh
#
#   # Trust with v4 backstory (for comparison)
#   BLOCK=trust BACKSTORY=v4 COUNTRIES="GB DE FI" SAMPLE_N=500 sbatch submit_extended_v5.sh
#
#   # Full variable set, single country
#   BLOCK=full COUNTRIES="GB" sbatch submit_extended_v5.sh
#
# ===== ESTIMATED GPU HOURS =====
#
#   Block       | Countries | Sample    | Prompts     | Time (Qwen 7B)
#   ------------|-----------|-----------|-------------|----------------
#   icc         | all 30    | full      | 0 (no GPU)  | <1 min (CPU)
#   ablation    | 3         | 500/cntry | ~9,000      | ~35 min
#   trust       | 30        | full      | ~135,000    | ~7-8 hours
#   trust       | 3         | full      | ~13,500     | ~45 min
#   rq8         | 30        | full      | ~270,000    | ~15 hours
#   overtime    | 15        | 200/cntry | ~54,000     | ~3 hours
#
# ============================================================================

echo "======================================================================"
echo "SILICON SAMPLING v5 — Multi-Country ESS11 (Final)"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "======================================================================"

# Configuration (with defaults)
BLOCK=${BLOCK:-"trust"}
MODEL=${MODEL:-"qwen"}
BACKSTORY=${BACKSTORY:-"v5"}
SEED=${SEED:-888}
N_SIMILAR=${N_SIMILAR:-3}
COUNTRIES=${COUNTRIES:-""}      # Empty = all 30 countries
SAMPLE_N=${SAMPLE_N:-""}        # Empty = FULL SAMPLE

echo "Configuration:"
echo "  BLOCK:     $BLOCK"
echo "  MODEL:     $MODEL"
echo "  BACKSTORY: $BACKSTORY"
echo "  SEED:      $SEED"
echo "  COUNTRIES: ${COUNTRIES:-ALL (30 countries)}"
echo "  SAMPLE_N:  ${SAMPLE_N:-FULL (no sampling)}"
echo "  N_SIMILAR: $N_SIMILAR (for RQ11)"
echo "======================================================================"

# Setup environment
cd ~/Winston_Code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Create directories
mkdir -p logs results data plots

# Verify data file exists
if [ ! -f "data/ESS11e04_1.csv" ]; then
    echo "ERROR: ESS11 data file not found at data/ESS11e04_1.csv"
    echo "Please place the ESS11 CSV file in ~/Winston_Code/data/"
    exit 1
fi

# ICC block needs no GPU — skip GPU check
if [ "$BLOCK" != "icc" ]; then
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ""
fi

# Build command
CMD="python silicon_sampling_extended_v5.py --block $BLOCK --model $MODEL --backstory $BACKSTORY --seed $SEED --n_similar $N_SIMILAR"

# Add country filter if specified
if [ -n "$COUNTRIES" ]; then
    CMD="$CMD --countries $COUNTRIES"
fi

# Add per-country sampling ONLY if explicitly set
if [ -n "$SAMPLE_N" ]; then
    CMD="$CMD --sample_per_country $SAMPLE_N"
fi

echo "Running: $CMD"
echo "======================================================================"

eval $CMD

EXIT_CODE=$?

echo ""
echo "======================================================================"

# Auto-generate plots if analysis blocks completed successfully
if [ $EXIT_CODE -eq 0 ]; then
    case $BLOCK in
        trust|rq8|all)
            echo "Generating cross-sectional plots..."
            pip install matplotlib --quiet --break-system-packages 2>/dev/null
            python plot_silicon_results.py --model $MODEL --plot scatter 2>&1 || true
            python plot_silicon_results.py --model $MODEL --plot forest 2>&1 || true
            python plot_silicon_results.py --model $MODEL --plot variance 2>&1 || true
            python plot_silicon_results.py --model $MODEL --plot heterogeneity 2>&1 || true
            echo "Plots saved to plots/"
            ;;
        overtime)
            echo "Generating overtime plots..."
            python plot_silicon_results.py --model $MODEL --plot overtime 2>&1 || true
            python plot_silicon_results.py --model $MODEL --plot overtime_scatter 2>&1 || true
            echo "Plots saved to plots/"
            ;;
    esac
fi

echo "======================================================================"
echo "EXPERIMENT COMPLETE"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "======================================================================"

echo ""
echo "Results saved to:"
ls -la results/
echo ""
echo "Plots saved to:"
ls -la plots/ 2>/dev/null || echo "(no plots yet)"

exit $EXIT_CODE
