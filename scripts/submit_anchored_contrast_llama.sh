#!/bin/bash
# ============================================================================
# submit_anchored_contrast_llama.sh — the 2x2 in a second model.
#
#   bash submit_anchored_contrast_llama.sh
#
# WHY
# ---
# The chapter's strongest integrating claim is that the item-level failures and
# the weakness of individual recovery share a cause in the response format. It
# rests on one 2x2, in one model, on 22 items. Everything else a reader can
# challenge is answered in the text; this is the one thing only data can
# answer. Running it turns "single model" into "replicated in two
# independently trained models".
#
# WHAT IS ALREADY ON DISK
# -----------------------
# Half the Llama 2x2 exists. The numeric cells are the 42-item runs, and the
# analysis subsets them to the same 22 items:
#
#                     numeric scales                 anchored scales
#   with country      llama_1p_full_noregion  DONE   (new) ..._noregion_anchored
#   without country   llama_1p_full_nocountry DONE   (new) ..._nocountry_anchored
#
# So this submits TWO jobs, not four.
#
# EVERY SETTING MATCHES THE QWEN 2x2 AND THE EXISTING LLAMA ARMS
# --------------------------------------------------------------
#   n = 685 per country, sampling seed 888  -> the same respondents
#   temperature 0.7, max_model_len 1024, missing_rule range
#   human_weight pspwght, ESS11e04_1.csv
#   the same 22 items as the Qwen anchored arm
# Only two things differ from submit_anchored_contrast.sh: --model llama, and
# the job and tag names.
#
# COST
# ----
# 2 jobs x 30 countries x 685 respondents x 22 items = 904,200 prompts, the
# same count as the Qwen contrast. At the ~50 prompts/sec/GPU observed in the
# v13 runs, and with the two jobs running side by side on one GPU each, about
# 2.5 hours of wall clock. No code changes: --model llama, --scale_labels
# anchored and both backstory modes already exist and have been exercised.
#
# WHEN IT FINISHES
# ----------------
#   python analyze_2x2.py --model llama
# ============================================================================
set -euo pipefail
cd ~/Winston_Code
mkdir -p logs results

PY=silicon_sampling_extended_v12.py
MAX_LEN=${MAX_LEN:-1024}
TIME_LIMIT=${TIME_LIMIT:-"24:00:00"}
SAMPLE_N=${SAMPLE_N:-685}
PARTITION=${PARTITION:-winston-gpu}
ESS_FILE=${ESS_FILE:-"data/ESS Data/ESS11e04_1.csv"}

if [ ! -f "$PY" ]; then
    echo "ERROR: generator not found in $(pwd): $PY" >&2
    exit 1
fi
if [ ! -f "$ESS_FILE" ]; then
    echo "ERROR: ESS_FILE not found: $ESS_FILE" >&2
    exit 1
fi

# The same 22 items as the Qwen anchored arm, in the same order: 12
# reverse-coded (vote excluded, its labels carry no direction), 6 coarse
# forward controls, 4 long-scale forward placebos.
ANCHOR_SET="polintr imsmetn impcntr imdfetn gincdif freehms hmsfmlsh hmsacld \
health rlgatnd aesfdrk hincfel actrolga cptppola inprdsc psppipla psppsgva \
sclmeet trstplc stflife happy stfdem"

job() {   # NAME BACKSTORY
    local NAME=$1 BS=$2
    local ID
    ID=$(sbatch --parsable \
        --job-name="$NAME" \
        --partition="$PARTITION" \
        --gres=gpu:1 --cpus-per-task=8 --mem=64G \
        --time="$TIME_LIMIT" \
        --output="logs/${NAME}_%j.out" \
        --error="logs/${NAME}_%j.err" \
        --wrap="bash -c '
cd ~/Winston_Code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
echo \"START \$(date)  ${NAME}\"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python ${PY} --block full \
    --model llama --prompt 1p \
    --backstory ${BS} --scale_labels anchored \
    --sample_per_country ${SAMPLE_N} \
    --temperature 0.7 --seed 888 \
    --missing_rule range --human_weight pspwght \
    --max_model_len ${MAX_LEN} \
    --ess_file \"${ESS_FILE}\" \
    --variables ${ANCHOR_SET}
echo \"END \$(date)  ${NAME}\"
'")
    [ -n "$ID" ] || { echo "  ERROR: sbatch failed for $NAME" >&2; exit 1; }
    printf "  %-34s job %s\n" "$NAME" "$ID"
}

TOTAL=$(( 2 * 30 * SAMPLE_N * 22 ))
echo "=================================================================="
echo "ANCHORED x COUNTRY CONTRAST, LLAMA — replicates the 2x2"
echo "  n=${SAMPLE_N}, 22 items, 2 jobs, $(( TOTAL / 1000 ))k prompts total"
echo "  partition: ${PARTITION}"
echo "  ESS file:  ${ESS_FILE}"
echo "=================================================================="
job anch_llama_noregion  full_noregion
job anch_llama_nocountry full_nocountry
cat <<'EOM'

  Expected output tags:
    llama_1p_full_noregion_anchored
    llama_1p_full_nocountry_anchored

  Expected files, one per tag:
    results/silicon_full_country_scatter_<tag>.csv
    results/manifest_<tag>.json

  Monitor:      squeue --me
  Tail a log:   tail -f logs/anch_llama_noregion_*.out
  Then:         python analyze_2x2.py --model llama
EOM
echo "=================================================================="
