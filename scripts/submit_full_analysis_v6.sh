#!/bin/bash
# ============================================================================
# Silicon Sampling v6 — FULL ANALYSIS: 42 vars × 30 countries × 2 models × 2 prompts
# ============================================================================
#
# Submits 4 independent GPU jobs (Slurm auto-schedules based on available GPUs):
#   1. Qwen  × 1P (role-play)     ~28h
#   2. Qwen  × 3P (third-person)  ~28h
#   3. Llama × 1P (role-play)     ~28h
#   4. Llama × 3P (third-person)  ~28h
#
# Wall time: ~28h if 4 GPUs available, ~56h with 2, ~112h with 1.
#
# Changes from v5:
#   - temperature=0.7 (stochastic sampling)
#   - 42 ESS variables (all domains)
#   - 3P prompt condition (Daniel's suggestion to reduce social desirability)
#   - Auto-parallel via Slurm (submit all, let scheduler decide)
#
# Usage:
#   bash submit_full_analysis.sh                    # all 4 jobs, full sample
#   bash submit_full_analysis.sh all all 10         # PILOT: 10/country, all jobs (~10 min each)
#   bash submit_full_analysis.sh qwen 1p            # qwen 1P only, full sample
#   bash submit_full_analysis.sh llama 3p 10        # llama 3P pilot
# ============================================================================

set -e
cd ~/Winston_Code
mkdir -p logs results plots

MODEL_FILTER=${1:-"all"}  # "all", "qwen", or "llama"
PROMPT_FILTER=${2:-"all"} # "all", "1p", or "3p"
SAMPLE_N=${3:-""}         # empty=full sample, or number e.g. "10" for pilot

SAMPLE_ARG=""
SAMPLE_LABEL="full"
TIME_LIMIT="36:00:00"
if [ -n "$SAMPLE_N" ]; then
    SAMPLE_ARG="--sample_per_country $SAMPLE_N"
    SAMPLE_LABEL="${SAMPLE_N}pp"
    TIME_LIMIT="36:00:00"  # pilot needs much less time
fi

echo "======================================================================"
echo "SILICON SAMPLING v6 — FULL ANALYSIS"
echo "======================================================================"
echo "  Variables:   42 (all ESS R11)"
echo "  Countries:   30 (all)"
echo "  Sample:      ${SAMPLE_LABEL} (${SAMPLE_N:-all respondents})"
echo "  Models:      ${MODEL_FILTER}"
echo "  Prompts:     ${PROMPT_FILTER}"
echo "  Temperature: 0.7 (stochastic)"
echo "======================================================================"

# Check available GPUs on cluster
echo ""
echo "Cluster status:"
sinfo -p winston-gpu --format="%n %G %t %C" 2>/dev/null | head -5 || true
echo ""

SUBMITTED=0
JOB_IDS=""

submit_job() {
    local MODEL=$1
    local PROMPT=$2
    local JOB_NAME="silicon_${MODEL}_${PROMPT}_${SAMPLE_LABEL}"
    
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

echo \"Starting: ${MODEL} ${PROMPT} ${SAMPLE_LABEL} (42 vars × 30 countries)\"
echo \"Time: \$(date)\"

python silicon_sampling_extended_v6.py \
    --block full \
    --model ${MODEL} \
    --prompt ${PROMPT} \
    --temperature 0.7 \
    --backstory v5 \
    --seed 888 \
    ${SAMPLE_ARG}

echo \"Completed: ${MODEL} ${PROMPT} at \$(date)\"
ls -lh results/silicon_full_*${MODEL}_${PROMPT}*.csv 2>/dev/null
'")
    
    echo "  ${JOB_NAME}: Job ${JOB_ID}"
    JOB_IDS="${JOB_IDS} ${JOB_ID}"
    SUBMITTED=$((SUBMITTED + 1))
}

# Submit jobs based on filters
for MODEL in qwen llama; do
    if [ "$MODEL_FILTER" != "all" ] && [ "$MODEL_FILTER" != "$MODEL" ]; then
        continue
    fi
    for PROMPT in 1p 3p; do
        if [ "$PROMPT_FILTER" != "all" ] && [ "$PROMPT_FILTER" != "$PROMPT" ]; then
            continue
        fi
        submit_job $MODEL $PROMPT
    done
done

# Also submit 3P trust-only for quick comparison (same 3 variables, ~2h)
if [ "$PROMPT_FILTER" = "all" ] || [ "$PROMPT_FILTER" = "3p" ]; then
    for MODEL in qwen llama; do
        if [ "$MODEL_FILTER" != "all" ] && [ "$MODEL_FILTER" != "$MODEL" ]; then
            continue
        fi
        JOB_NAME="silicon_${MODEL}_3p_trust_${SAMPLE_LABEL}"
        JOB_ID=$(sbatch --parsable \
            --job-name=${JOB_NAME} \
            --partition=winston-gpu \
            --gres=gpu:1 \
            --cpus-per-task=8 \
            --mem=64G \
            --time=04:00:00 \
            --output=logs/${JOB_NAME}_%j.out \
            --error=logs/${JOB_NAME}_%j.err \
            --wrap="bash -c '
cd ~/Winston_Code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
mkdir -p results plots logs

python silicon_sampling_extended_v6.py \
    --block trust \
    --model ${MODEL} \
    --prompt 3p \
    --temperature 0.7 \
    --backstory v5 \
    --seed 888 \
    ${SAMPLE_ARG}

echo \"Trust 3P complete: ${MODEL} at \$(date)\"
'")
        echo "  ${JOB_NAME} (quick, ~2h): Job ${JOB_ID}"
        JOB_IDS="${JOB_IDS} ${JOB_ID}"
        SUBMITTED=$((SUBMITTED + 1))
    done
fi

echo ""
echo "======================================================================"
echo "SUBMITTED ${SUBMITTED} JOBS"
echo "======================================================================"
echo ""
echo "Expected outputs:"
echo "  results/silicon_full_raw_{model}_{prompt}_seed888.csv  (42 vars)"
echo "  results/silicon_full_rq1_{model}_{prompt}.csv          (per-var r)"
echo "  results/silicon_full_country_scatter_{model}_{prompt}.csv"
echo "  results/silicon_trust_raw_{model}_3p_seed888.csv       (3P trust)"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/silicon_qwen_1p_full_*.out"
echo ""
echo "Estimated wall time (full sample):"
echo "  4 GPUs available: ~28h + 2h (trust 3P finishes first)"
echo "  2 GPUs available: ~56h"
echo "  1 GPU available:  ~112h"
echo ""
echo "Pilot (10/country): ~10 min per job"
echo "======================================================================"
