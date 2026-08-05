#!/bin/bash
#SBATCH --job-name=silicon_pilot
#SBATCH --partition=winston-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/silicon_pilot_%j.out
#SBATCH --error=logs/silicon_pilot_%j.err

# ============================================================================
# Silicon Sampling v5 — PILOT RUN
# ============================================================================
# Runs three steps sequentially on a single GPU allocation:
#
#   Step 0: ICC analysis (no GPU, <1 min)
#   Step 1: Trust pilot — 5 countries × 300 respondents (~15 min)
#   Step 2: Backstory ablation — 3 countries × 500 respondents (~30 min)
#   Step 3: Generate all plots
#
# Total estimated time: ~50 min (3h wall time for safety margin)
#
# Countries:
#   Trust pilot:  FI (high), DE (mid-high), FR (mid), PL (mid-low), UA (low)
#   Ablation:     DE, FI, PL (subset with max trust variance)
#
# Usage:
#   sbatch submit_pilot.sh
#
# Checklist after completion:
#   [ ] results/silicon_icc_analysis.csv — ICC values for 3 trust vars
#   [ ] results/silicon_trust_raw_qwen_seed888.csv — 4,500 rows
#   [ ] results/silicon_trust_country_scatter_qwen.csv — 15 rows (5×3)
#   [ ] results/silicon_trust_individual_qwen.csv — per-country r values
#   [ ] results/silicon_ablation_comparison_qwen.csv — v4 vs v5 comparison
#   [ ] plots/country_scatterplots_qwen.png — 2×2 panel
#   [ ] plots/within_country_forest_qwen.png — forest plot
#   [ ] Parse rate >95% (check silicon_response non-null rate)
#   [ ] ppltrst country-level r is positive
# ============================================================================

echo "======================================================================"
echo "SILICON SAMPLING v5 — PILOT RUN"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "======================================================================"

# Setup
cd ~/Winston_Code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
mkdir -p logs results plots

# Verify data
if [ ! -f "data/ESS11e04_1.csv" ]; then
    echo "ERROR: ESS11 data not found at data/ESS11e04_1.csv"
    exit 1
fi

echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

PILOT_OK=true

# ============================================================================
# STEP 0: ICC Analysis (no GPU, <1 min)
# ============================================================================
echo ""
echo "======================================================================"
echo "[STEP 0/3] ICC ANALYSIS — Why does ppltrst correlate better?"
echo "======================================================================"

python silicon_sampling_extended_v5.py --block icc

if [ $? -eq 0 ]; then
    echo "[STEP 0] ✓ ICC analysis complete"
    echo "  Output: results/silicon_icc_analysis.csv"
else
    echo "[STEP 0] ✗ ICC analysis FAILED"
    PILOT_OK=false
fi

# ============================================================================
# STEP 1: Trust Pilot — 5 countries × 300 respondents (~15 min)
# ============================================================================
echo ""
echo "======================================================================"
echo "[STEP 1/3] TRUST PILOT — FI DE FR PL UA × 300 respondents"
echo "  Expected: 1,500 respondents × 3 variables = 4,500 prompts"
echo "  Estimated time: ~15 min"
echo "======================================================================"

python silicon_sampling_extended_v5.py \
    --block trust \
    --model qwen \
    --backstory v5 \
    --countries FI DE FR PL UA \
    --sample_per_country 300 \
    --seed 888

STEP1_EXIT=$?

if [ $STEP1_EXIT -eq 0 ]; then
    echo "[STEP 1] ✓ Trust pilot complete"

    # Quick validation
    echo ""
    echo "--- PILOT VALIDATION ---"

    # Check row count
    RAW_FILE="results/silicon_trust_raw_qwen_seed888.csv"
    if [ -f "$RAW_FILE" ]; then
        N_ROWS=$(wc -l < "$RAW_FILE")
        N_ROWS=$((N_ROWS - 1))  # subtract header
        echo "  Raw data rows: $N_ROWS (expected: ~4,500)"

        # Parse rate
        N_PARSED=$(awk -F',' 'NR>1 && $6!=""' "$RAW_FILE" | wc -l)
        if [ $N_ROWS -gt 0 ]; then
            PARSE_PCT=$((N_PARSED * 100 / N_ROWS))
            echo "  Parse rate: ${PARSE_PCT}% ($N_PARSED / $N_ROWS)"
            if [ $PARSE_PCT -lt 95 ]; then
                echo "  ⚠ WARNING: Parse rate below 95% — check raw responses"
            fi
        fi
    else
        echo "  ⚠ WARNING: Raw file not found"
    fi

    # Check scatter data
    SCATTER_FILE="results/silicon_trust_country_scatter_qwen.csv"
    if [ -f "$SCATTER_FILE" ]; then
        N_SCATTER=$(wc -l < "$SCATTER_FILE")
        N_SCATTER=$((N_SCATTER - 1))
        echo "  Scatter data rows: $N_SCATTER (expected: 15 = 5 countries × 3 vars)"
    fi

    echo "--- END VALIDATION ---"
else
    echo "[STEP 1] ✗ Trust pilot FAILED (exit code: $STEP1_EXIT)"
    echo "  Check logs/silicon_social_trust_qwen_*.log for details"
    PILOT_OK=false
fi

# ============================================================================
# STEP 2: Backstory Ablation — 3 countries × 500 respondents (~30 min)
# ============================================================================
echo ""
echo "======================================================================"
echo "[STEP 2/3] BACKSTORY ABLATION — DE FI PL × 500 respondents"
echo "  v4 (17 vars) vs v5 (27 vars), same respondents"
echo "  Expected: 2 conditions × 1,500 respondents × 3 variables = 9,000 prompts"
echo "  Estimated time: ~30 min"
echo "======================================================================"

python silicon_sampling_extended_v5.py \
    --block ablation \
    --model qwen \
    --countries DE FI PL \
    --sample_per_country 500 \
    --seed 888

STEP2_EXIT=$?

if [ $STEP2_EXIT -eq 0 ]; then
    echo "[STEP 2] ✓ Ablation complete"
    echo "  Output: results/silicon_ablation_comparison_qwen.csv"
else
    echo "[STEP 2] ✗ Ablation FAILED (exit code: $STEP2_EXIT)"
    PILOT_OK=false
fi

# ============================================================================
# STEP 3: Generate Plots
# ============================================================================
echo ""
echo "======================================================================"
echo "[STEP 3/3] GENERATING PLOTS"
echo "======================================================================"

pip install matplotlib --quiet --break-system-packages 2>/dev/null

python plot_silicon_results.py --model qwen --plot scatter 2>&1 && \
    echo "  ✓ Country scatterplots" || echo "  ✗ Country scatterplots failed"

python plot_silicon_results.py --model qwen --plot forest 2>&1 && \
    echo "  ✓ Forest plot" || echo "  ✗ Forest plot failed"

python plot_silicon_results.py --model qwen --plot variance 2>&1 && \
    echo "  ✓ Variance compression" || echo "  ✗ Variance compression failed"

python plot_silicon_results.py --model qwen --plot heterogeneity 2>&1 && \
    echo "  ✓ Heterogeneity plot" || echo "  ✗ Heterogeneity plot failed"

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "======================================================================"
echo "PILOT COMPLETE"
echo "======================================================================"
echo "End Time: $(date)"
echo ""

if [ "$PILOT_OK" = true ]; then
    echo "STATUS: ✓ ALL STEPS PASSED"
else
    echo "STATUS: ✗ SOME STEPS FAILED — check logs before proceeding"
fi

echo ""
echo "--- Output Files ---"
echo "Results:"
ls -lh results/silicon_icc_*.csv results/silicon_trust_*.csv \
       results/silicon_ablation_*.csv 2>/dev/null
echo ""
echo "Plots:"
ls -lh plots/*.png 2>/dev/null

echo ""
echo "======================================================================"
echo "NEXT STEPS (if pilot passed):"
echo "  1. Review plots/country_scatterplots_qwen.png"
echo "  2. Check results/silicon_icc_analysis.csv"
echo "  3. Check results/silicon_ablation_comparison_qwen.csv"
echo "  4. Pilot results backed up to results/pilot/ (full runs will overwrite)"
echo "  5. If all OK, submit full runs:"
echo "     BLOCK=trust sbatch submit_extended_v5.sh"
echo "     BLOCK=trust MODEL=llama sbatch submit_extended_v5.sh"
echo "======================================================================"

# Backup pilot results so full runs don't overwrite
mkdir -p results/pilot plots/pilot
cp results/silicon_trust_*.csv results/silicon_icc_*.csv \
   results/silicon_ablation_*.csv results/pilot/ 2>/dev/null
cp plots/*.png plots/pilot/ 2>/dev/null
echo "Pilot results backed up to results/pilot/ and plots/pilot/"

exit $( [ "$PILOT_OK" = true ] && echo 0 || echo 1 )
