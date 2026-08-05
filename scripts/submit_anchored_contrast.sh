#!/bin/bash
# ============================================================================
# submit_anchored_contrast.sh — the highest-value follow-up run.
#
#   bash submit_anchored_contrast.sh
#
# WHAT QUESTION THIS ANSWERS
# --------------------------
# The v13 results show that the country label and the response-scale direction
# are not two independent mechanisms but one signal times one sign:
#
#   forward-coded items:  country label helps    (27/29 and 28/29 improve)
#   reverse-coded items:  country label HARMS    (1/13 improve, both baselines)
#   verbal anchors:       flip reverse items from about -0.43 to about +0.55,
#                         with a placebo signal of ~0.03 on forward long scales
#
# The natural reading is that the label supplies cross-national ordering
# information which the model then expresses along whichever scale direction it
# has adopted; where that direction is inverted, better country information
# produces a more confidently wrong answer.
#
# That reading makes a sharp, falsifiable prediction which the existing runs
# cannot test, because the anchored arm was only ever run with the full
# backstory (which contains the country label). The prediction is:
#
#   ONCE THE SCALE DIRECTION IS DISAMBIGUATED BY VERBAL ANCHORS, THE COUNTRY
#   LABEL SHOULD HELP REVERSE-CODED ITEMS AS MUCH AS IT HELPS FORWARD ONES.
#
# If confirmed, the paper reports a single interacting mechanism with a clean
# 2x2 experimental demonstration. If the negative country effect on reverse
# items SURVIVES anchoring, the two-mechanism account is wrong in a way worth
# knowing about, and something beyond scale direction is involved.
#
# THE DESIGN
# ----------
# Two new arms, both anchored, both on the same 22 items as the existing
# anchored arm, both at n=685 with seed 888 (hence the same respondents):
#
#   full_noregion  + anchored   -> qwen_1p_full_noregion_anchored
#   full_nocountry + anchored   -> qwen_1p_full_nocountry_anchored
#
# Combined with the runs already on disk this gives a complete 2x2:
#
#                     numeric scales              anchored scales
#   with country      qwen_1p_full_noregion       (new) ..._noregion_anchored
#   without country   qwen_1p_full_nocountry      (new) ..._nocountry_anchored
#
# The quantity of interest is the difference-in-difference-in-differences:
# the country-label effect under anchoring minus the country-label effect under
# numeric scales, computed separately for forward and reverse items. Because all
# four cells share seed 888 and n=685, every comparison is paired at the
# respondent level.
#
# COST
# ----
# 2 jobs x 30 countries x 685 respondents x 22 items = 904,200 prompts.
# At the ~50 prompts/sec/GPU observed in the v13 runs, on two GPUs, about
# 2.5 hours. No code changes are required: both backstory modes and the
# anchored scale mode already exist and were exercised in v13.
# ============================================================================
set -euo pipefail
cd ~/Winston_Code
mkdir -p logs results

PY=silicon_sampling_extended_v12.py
MAX_LEN=${MAX_LEN:-1024}
TIME_LIMIT=${TIME_LIMIT:-"24:00:00"}
SAMPLE_N=${SAMPLE_N:-685}
ESS_FILE=${ESS_FILE:-"data/ESS Data/ESS11e04_1.csv"}

if [ ! -f "$ESS_FILE" ]; then
    echo "ERROR: ESS_FILE not found: $ESS_FILE" >&2
    exit 1
fi

# Same 22 items as the existing anchored arm: 12 reverse-coded (vote excluded,
# its labels carry no latent direction), 6 dosage-matched coarse forward
# controls, 4 long-scale forward placebos.
ANCHOR_SET="polintr imsmetn impcntr imdfetn gincdif freehms hmsfmlsh hmsacld \
health rlgatnd aesfdrk hincfel actrolga cptppola inprdsc psppipla psppsgva \
sclmeet trstplc stflife happy stfdem"

job() {   # NAME BACKSTORY
    local NAME=$1 BS=$2
    local ID
    ID=$(sbatch --parsable \
        --job-name="$NAME" \
        --partition=winston-gpu \
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
    --model qwen --prompt 1p \
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
echo "ANCHORED x COUNTRY CONTRAST — completes the 2x2"
echo "  n=${SAMPLE_N}, 22 items, 2 jobs, $(( TOTAL / 1000 ))k prompts total"
echo "  ESS file: ${ESS_FILE}"
echo "=================================================================="
job anch_noregion  full_noregion
job anch_nocountry full_nocountry
cat <<'EOM'

  Expected output tags:
    qwen_1p_full_noregion_anchored
    qwen_1p_full_nocountry_anchored

  Monitor with:  squeue --me
  When both finish, analyse with:  python analyze_2x2.py
EOM
echo "=================================================================="
