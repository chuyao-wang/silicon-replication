#!/bin/bash
# ============================================================================
# Silicon Sampling v7 — MAIN ANALYSIS + ABLATION
# ============================================================================
#
# 9 jobs total:
#   Main analysis (v5_clean backstory, 20 vars):
#     1. Qwen  × 1P    2. Qwen  × 3P
#     3. Llama × 1P    4. Llama × 3P
#   Backstory ablation (Qwen 1P only):
#     5. demo_only (3 vars, NO country — country-label isolation test)
#     6. minimal (4 vars)    7. ses (14 vars)
#     8. political (16 vars) 9. full_clean (20 vars, = main Qwen 1P)
#
# Usage:
#   bash submit_v7.sh                       # all 9 jobs, 400/country
#   bash submit_v7.sh all 1                 # PILOT: 1/country (~30s each)
#   bash submit_v7.sh main 400             # main 4 jobs only
#   bash submit_v7.sh ablation 400         # ablation 5 jobs only
# ============================================================================

set -e
cd ~/Winston_Code
mkdir -p logs results plots

MODE=${1:-"all"}       # "all", "main", or "ablation"
SAMPLE_N=${2:-"400"}   # respondents per country

SAMPLE_ARG="--sample_per_country $SAMPLE_N"
SAMPLE_LABEL="${SAMPLE_N}pp"
TIME_LIMIT="240:00:00"

echo "======================================================================"
echo "SILICON SAMPLING v7"
echo "======================================================================"
echo "  Variables:    42 (all ESS R11)"
echo "  Countries:    30"
echo "  Sample:       ${SAMPLE_LABEL}"
echo "  Temperature:  0.7"
echo "  Mode:         ${MODE}"
echo "======================================================================"

sinfo -p winston-gpu --format="%n %G %t %C" 2>/dev/null | head -5 || true
echo ""

SUBMITTED=0

submit_job() {
    local MODEL=$1
    local PROMPT=$2
    local BACKSTORY=$3
    local JOB_NAME="v7_${MODEL}_${PROMPT}_${BACKSTORY}_${SAMPLE_LABEL}"
    
    local JOB_ID=$(sbatch --parsable \
        --job-name=${JOB_NAME} \
        --partition=winston-gpu \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=${TIME_LIMIT} \
        --output=logs/${JOB_NAME}_%j.out \
        --error=logs/${JOB_NAME}_%j.err \
        --wrap="bash -c '
cd ~/Winston_Code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
mkdir -p results plots logs

echo \"GPU Status:\"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo \"Starting: ${MODEL} ${PROMPT} backstory=${BACKSTORY} ${SAMPLE_LABEL}\"
echo \"Time: \$(date)\"

python silicon_sampling_extended_v7.py \
    --block full \
    --model ${MODEL} \
    --prompt ${PROMPT} \
    --temperature 0.7 \
    --backstory ${BACKSTORY} \
    --seed 888 \
    ${SAMPLE_ARG}

echo \"Completed: ${MODEL} ${PROMPT} ${BACKSTORY} at \$(date)\"
ls -lh results/silicon_full_*${MODEL}_${PROMPT}*.csv 2>/dev/null
'")
    
    echo "  ${JOB_NAME}: Job ${JOB_ID}"
    SUBMITTED=$((SUBMITTED + 1))
}

# --- MAIN ANALYSIS: 2 models × 2 prompts, backstory = v5_clean ---
if [ "$MODE" = "all" ] || [ "$MODE" = "main" ]; then
    echo "--- Main analysis (v5_clean, 20 backstory vars) ---"
    for MODEL in qwen llama; do
        for PROMPT in 1p 3p; do
            submit_job $MODEL $PROMPT v5_clean
        done
    done
fi

# --- ABLATION: Qwen 1P only, 5 backstory levels ---
if [ "$MODE" = "all" ] || [ "$MODE" = "ablation" ]; then
    echo "--- Backstory ablation (Qwen 1P) ---"
    for BS in demo_only minimal ses political full_clean; do
        submit_job qwen 1p $BS
    done
fi

echo ""
echo "======================================================================"
echo "SUBMITTED ${SUBMITTED} JOBS"
echo "======================================================================"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/v7_qwen_1p_v5_clean_*.out"
echo "======================================================================"
