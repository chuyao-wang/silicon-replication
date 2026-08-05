#!/bin/bash
set -e
cd ~/Winston_Code
mkdir -p logs results plots

MODEL_FILTER=${1:-"all"}
PROMPT_FILTER=${2:-"all"}
SAMPLE_N=${3:-""}

SAMPLE_ARG=""
SAMPLE_LABEL="full"
TIME_LIMIT="36:00:00"
if [ -n "$SAMPLE_N" ]; then
    SAMPLE_ARG="--sample_per_country $SAMPLE_N"
    SAMPLE_LABEL="${SAMPLE_N}pp"
    TIME_LIMIT="04:00:00"
fi

echo "======================================================================"
echo "SILICON SAMPLING v6 — FULL ANALYSIS"
echo "======================================================================"
echo "  Variables:   42 (all ESS R11)"
echo "  Countries:   30 (all)"
echo "  Sample:      ${SAMPLE_LABEL}"
echo "  Models:      ${MODEL_FILTER}"
echo "  Prompts:     ${PROMPT_FILTER}"
echo "  Temperature: 0 (deterministic)"
echo "======================================================================"

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
        --mem=32G \
        --time=${TIME_LIMIT} \
        --output=logs/${JOB_NAME}_%j.out \
        --error=logs/${JOB_NAME}_%j.err \
        --wrap="bash -c 'cd ~/Winston_Code && source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && mkdir -p results plots logs && python silicon_sampling_extended_v5.py --block full --model ${MODEL} --prompt ${PROMPT} --temperature 0 --backstory v5 --seed 888 ${SAMPLE_ARG} && echo Done: ${MODEL} ${PROMPT} at \$(date)'")
    echo "  ${JOB_NAME}: Job ${JOB_ID}"
    JOB_IDS="${JOB_IDS} ${JOB_ID}"
    SUBMITTED=$((SUBMITTED + 1))
}

for MODEL in qwen llama; do
    if [ "$MODEL_FILTER" != "all" ] && [ "$MODEL_FILTER" != "$MODEL" ]; then continue; fi
    for PROMPT in 1p 3p; do
        if [ "$PROMPT_FILTER" != "all" ] && [ "$PROMPT_FILTER" != "$PROMPT" ]; then continue; fi
        submit_job $MODEL $PROMPT
    done
done

if [ "$PROMPT_FILTER" = "all" ] || [ "$PROMPT_FILTER" = "3p" ]; then
    for MODEL in qwen llama; do
        if [ "$MODEL_FILTER" != "all" ] && [ "$MODEL_FILTER" != "$MODEL" ]; then continue; fi
        JOB_NAME="silicon_${MODEL}_3p_trust_${SAMPLE_LABEL}"
        JOB_ID=$(sbatch --parsable --job-name=${JOB_NAME} --partition=winston-gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=04:00:00 --output=logs/${JOB_NAME}_%j.out --error=logs/${JOB_NAME}_%j.err \
            --wrap="bash -c 'cd ~/Winston_Code && source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && mkdir -p results plots logs && python silicon_sampling_extended_v5.py --block trust --model ${MODEL} --prompt 3p --temperature 0 --backstory v5 --seed 888 ${SAMPLE_ARG} && echo Trust 3P done: ${MODEL} at \$(date)'")
        echo "  ${JOB_NAME} (trust 3P quick): Job ${JOB_ID}"
        SUBMITTED=$((SUBMITTED + 1))
    done
fi

echo ""
echo "SUBMITTED ${SUBMITTED} JOBS. Monitor: squeue -u \$USER"
