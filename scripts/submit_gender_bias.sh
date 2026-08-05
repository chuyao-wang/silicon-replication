#!/bin/bash
# ============================================================================
# LLM Identity & Prompt Sensitivity Audit v4 — Slurm + API Submission
# ============================================================================
#
# Design v4 (combines Designs 1 + 3):
#   10 vignettes × 2 genders × 2 ethnicities × 4 prompt ops × K reps
#   = 160 cells per model; at K=100, that is 16,000 calls per model
#
# Workflow: pilot (5 reps) → inspect → full run (100 reps)
#
# Usage:
#   bash submit_gender_bias.sh pilot              # 5 reps, Qwen, 1 vignette (~1 min)
#   bash submit_gender_bias.sh main 100           # Qwen + Llama, 100 reps
#   bash submit_gender_bias.sh api 100            # GPT-4o only (needs OPENAI_API_KEY)
#   bash submit_gender_bias.sh all 100            # everything available
#   bash submit_gender_bias.sh robust 100         # temp=0.3 robustness (Qwen+Llama)
#   bash submit_gender_bias.sh reanalyze 100      # NEW: re-run analyses on
#                                                   existing raw CSVs (no LLM
#                                                   calls; updates outputs to
#                                                   include the new ADI metric)
#
# Cost guide for GPT-4o (gpt-4o-2024-11-20):
#   100 reps  ->  16,000 calls  ->  ~USD 35-45 per temperature
# ============================================================================

set -e
cd ~/Winston_Code
mkdir -p logs results

MODE=${1:-"all"}
REPS=${2:-100}
SEED=42
ETH_FLAG="--ethnicity anglo black"          # v4 default: intersectional design
NAME_FLAG="--rotate_names"                  # v4 default: decouple name from vignette/identity
COMMON_FLAGS="${ETH_FLAG} ${NAME_FLAG}"

# Pilot override
if [ "$MODE" = "pilot" ]; then
    REPS=5
fi

# Calls per model: 10 vig x 2 gen x 2 eth x 4 prompt x REPS
CALLS_PER_MODEL=$((10 * 2 * 2 * 4 * REPS))

echo "======================================================================"
echo "LLM IDENTITY & PROMPT SENSITIVITY AUDIT v4"
echo "======================================================================"
echo "  Vignettes:     10 matched-text pairs"
echo "  Conditions:    2 genders x 2 ethnicities x 4 prompt ops = 16 cells"
echo "  Reps/cell:     ${REPS}"
echo "  Calls/model:   ${CALLS_PER_MODEL}"
echo "  Mode:          ${MODE}"
echo "  Ethnicity:     anglo + black (B&M 2004 / Gaddis 2017)"
echo "  Names:         ROTATED across reps (decouples name from identity/vignette)"
echo "  GPT-4o key:    $(if [ -n "$OPENAI_API_KEY" ]; then echo "SET"; else echo "NOT SET (gpt4o jobs will skip)"; fi)"
echo "  Pre-req:       pip install statsmodels  (for mixed_effects_summary)"
echo "======================================================================"

sinfo -p winston-gpu --format="%n %G %t %C" 2>/dev/null | head -5 || true
echo ""

SUBMITTED=0

submit_gpu() {
    local MODEL=$1 TEMP=$2
    local JOB="audit_${MODEL}_t${TEMP//./_}_${REPS}reps"
    local JID=$(sbatch --parsable \
        --job-name=${JOB} \
        --partition=winston-gpu \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=48:00:00 \
        --output=logs/${JOB}_%j.out \
        --error=logs/${JOB}_%j.err \
        --wrap="bash -c '
cd ~/Winston_Code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
mkdir -p results logs
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo \"START: ${MODEL} temp=${TEMP} reps=${REPS} \$(date)\"
python gender_bias_experiment.py --model ${MODEL} --reps ${REPS} \\
    --temperature ${TEMP} --seed ${SEED} ${COMMON_FLAGS}
echo \"DONE: \$(date)\"
ls -lh results/gender_*${MODEL}*${REPS}reps*.csv 2>/dev/null
'")
    echo "  [GPU]  ${JOB}: Job ${JID}"
    SUBMITTED=$((SUBMITTED + 1))
}

run_api() {
    local TEMP=$1
    local TAG="gpt4o_t${TEMP//./_}_${REPS}reps"
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "  [API]  SKIP: OPENAI_API_KEY not set (gpt4o jobs skipped)"
        return
    fi
    # Cost estimate: input ~350 tok, output ~150 tok per call
    # gpt-4o-2024-11-20: \$2.50/1M input, \$10/1M output
    # Total per call ~\$0.0024  ->  16,000 calls ~ \$38
    local IN_COST_C=$((CALLS_PER_MODEL * 350 * 250 / 100000000))   # cents
    local OUT_COST_C=$((CALLS_PER_MODEL * 150 * 1000 / 100000000)) # cents
    local TOTAL_USD=$(( (IN_COST_C + OUT_COST_C) / 100 ))
    echo "  [API]  GPT-4o temp=${TEMP} (background, est. ~\$${TOTAL_USD})"
    nohup bash -c "
cd ~/Winston_Code
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate base 2>/dev/null || true
python gender_bias_experiment.py --model gpt4o --reps ${REPS} \\
    --temperature ${TEMP} --seed ${SEED} ${COMMON_FLAGS}
" > logs/${TAG}.out 2> logs/${TAG}.err &
    echo "         PID: $!  ->  logs/${TAG}.out"
    SUBMITTED=$((SUBMITTED + 1))
}

# --- PILOT: Qwen, 5 reps, 3 vignettes, no ethnicity (~1 min sanity check) ---
# 3 vignettes ensures mixed_effects_summary() can fit (needs >= 3 groups).
# No ethnicity keeps pilot fast; the parse-rate diagnostic is the main goal.
if [ "$MODE" = "pilot" ]; then
    echo "--- PILOT (5 reps, Qwen, vignettes 1-3, anglo only, ~1 min) ---"
    JOB="audit_pilot_qwen_${REPS}reps"
    JID=$(sbatch --parsable \
        --job-name=${JOB} \
        --partition=winston-gpu \
        --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:30:00 \
        --output=logs/${JOB}_%j.out --error=logs/${JOB}_%j.err \
        --wrap="bash -c '
cd ~/Winston_Code
source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
python gender_bias_experiment.py --model qwen --reps ${REPS} \\
    --temperature 0.7 --seed ${SEED} --vignettes 1 2 3 ${NAME_FLAG}
echo DONE
'")
    echo "  [GPU]  ${JOB}: Job ${JID}  (120 calls -- confirm parse rate per prompt_op)"
    SUBMITTED=$((SUBMITTED + 1))
fi

# --- MAIN: Qwen + Llama, temp=0.7, full v4 design ---
if [ "$MODE" = "all" ] || [ "$MODE" = "main" ]; then
    echo "--- Main (temp=0.7, anglo+black, 4 prompt ops) ---"
    for M in qwen llama; do submit_gpu $M 0.7; done
fi

# --- ROBUSTNESS: temp=0.3 ---
if [ "$MODE" = "all" ] || [ "$MODE" = "robust" ]; then
    echo "--- Robustness (temp=0.3) ---"
    for M in qwen llama; do submit_gpu $M 0.3; done
fi

# --- API: GPT-4o ---
if [ "$MODE" = "all" ] || [ "$MODE" = "api" ]; then
    echo "--- GPT-4o via API ---"
    run_api 0.7
    run_api 0.3
fi

# --- REANALYZE: re-run analyses on existing raw CSVs (no LLM calls) ---
# Use this after adding/changing analysis code (e.g., the new ADI metric)
# to update output CSVs without re-generating LLM responses.
# Runs in foreground (analyses are fast: ~1 min per model).
if [ "$MODE" = "reanalyze" ]; then
    echo "--- REANALYZE (no LLM calls; updates analysis CSVs in place) ---"
    for M in qwen llama gpt4o; do
        RAW="results/gender_raw_${M}_${REPS}reps_anglo+black_seed${SEED}.csv"
        if [ -f "$RAW" ]; then
            echo "  [LOCAL]  ${M}: re-running analyses on ${RAW}"
            python gender_bias_experiment.py --analyze_only \
                --model ${M} --reps ${REPS} --seed ${SEED} \
                ${ETH_FLAG} 2>&1 | tail -25
            SUBMITTED=$((SUBMITTED + 1))
        else
            echo "  [SKIP]   ${M}: no raw file at ${RAW}"
        fi
    done
fi

echo ""
echo "======================================================================"
echo "SUBMITTED ${SUBMITTED} JOB(S)"
echo "======================================================================"
echo ""
echo "Recommended workflow:"
echo "  1. pip install statsmodels --user                # one-time, for mixed_effects"
echo "  2. bash submit_gender_bias.sh pilot              # ~1-2 min sanity check"
echo "                                                     (3 vignettes, anglo only)"
echo "  3. cat results/gender_parse_diag_qwen_5reps.csv  # parse rate by prompt_op"
echo "                                                     (must be >= 0.95 per row)"
echo "  4. cat results/gender_ceiling_floor_qwen_5reps.csv | head -10"
echo "  5. head -20 results/gender_prompt_x_identity_qwen_5reps.csv"
echo "  6. head -20 results/gender_mixed_effects_qwen_5reps.csv"
echo "  7. bash submit_gender_bias.sh all 100            # full v4 run"
echo ""
echo "After full run, files are suffixed with '_anglo+black', e.g.:"
echo "  results/gender_prompt_x_identity_qwen_100reps_anglo+black.csv"
echo "  results/gender_mixed_effects_qwen_100reps_anglo+black.csv"
echo "  results/gender_mixed_effects_gpt4o_100reps_anglo+black.csv  (if API key set)"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "          tail -f logs/audit_qwen_*.out"
echo ""
echo "Key v4 output files (per model + reps):"
echo "  gender_raw_*.csv                  -- raw 16,000 rows"
echo "  gender_prompt_x_identity_*.csv    -- *core analysis*"
echo "                                       (cells, mains, gaps, interactions)"
echo "                                       VIGNETTE-LEVEL bootstrap CIs"
echo "  gender_audit_disagreement_*.csv   -- *NEW v4-ext* Audit Disagreement Index"
echo "                                       (SD of identity gap across prompts /"
echo "                                        mean|gap|; methodological contribution"
echo "                                        for AI audit literature)"
echo "  gender_mixed_effects_*.csv        -- per-DV mixed model fixed effects"
echo "                                       (Wald tests, statsmodels.mixedlm)"
echo "  gender_ceiling_floor_*.csv        -- scale-saturation diagnostic"
echo "  gender_parse_diag_*.csv           -- parse rate by prompt_op (pilot QA)"
echo "  gender_did_*.csv                  -- DiD per (vignette, dv, treatment)"
echo "  gender_pooled_*.csv               -- meta-analytic pooled gender gaps"
echo "  gender_refusal_*.csv              -- parse / refusal / disclaimer rates"
echo "  gender_lexical_*.csv              -- hedging / valence counts"
echo "  gender_variance_*.csv             -- variance ratio + Levene"
echo "  gender_domain_het_*.csv           -- domain x typicality heterogeneity"
echo "======================================================================"
