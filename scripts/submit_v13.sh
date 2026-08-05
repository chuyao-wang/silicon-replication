#!/bin/bash
# ============================================================================
# submit_v13.sh — clean full rerun. Four subcommands.
#
#   bash submit_v13.sh pilot     # 3 countries x 2 respondents, ~4 minutes
#   bash submit_v13.sh all       # everything, n=SAMPLE_N per country (default 685)
#   bash submit_v13.sh rungs     # ONLY the cumulative backstory ladder (5 jobs),
#                                # independent of the other 8 jobs in `all`
#   bash submit_v13.sh status    # what has finished
#
# WHY A CLEAN RERUN RATHER THAN NEW ARMS BOLTED ONTO THE MARCH FILES
# ------------------------------------------------------------------
# The March runs were produced under a different pandas and a different vLLM
# from the ones now installed (pandas 3.0.0, vLLM 0.15.0). Two consequences:
#
#   * DataFrame.sample(random_state=888) is what selects the 500 respondents per
#     country. If pandas changed that code path, seed 888 no longer reproduces
#     the March sample, and no new file can be compared with an old one.
#   * Even if it does reproduce, the generation library differs, so any small
#     discrepancy between a new arm and an old one is uninterpretable: library
#     drift and a genuine effect look the same.
#
# Rerunning everything in one environment removes that entire class of problem
# instead of testing for it. It also lets every arm run at n = 500, so all
# contrasts are attenuated identically, and it means the corrected human-side
# missingness rule is applied uniformly rather than to some files and not others.
#
# WHAT A CLEAN RERUN MAKES UNNECESSARY
# ------------------------------------
# The reproduction gate, the verifyA/verifyB arms, the nested-250 argument, the
# max_model_len confound and the engine-seed question all existed only to make
# new files comparable with old ones. None of them is needed here. The one thing
# worth keeping from that machinery is a single repeated-run replicate, because
# the manuscript lists run-to-run stability as an untested limitation.
#
# max_model_len is 1024, not the 2048 of v11/v12. A CPU audit of the actual
# prompts shows the longest anchored prompt is about 340 tokens, so 1024 is
# ample; the smaller value leaves more of the GPU free for KV cache, which
# raises batch size and therefore throughput.
# ============================================================================
set -euo pipefail
cd ~/Winston_Code
mkdir -p logs results plots

if [ -z "${1:-}" ]; then
    echo "usage: submit_v13.sh (pilot|all|rungs|status)" >&2
    exit 1
fi
CMD=$1
PY=silicon_sampling_extended_v12.py
MAX_LEN=${MAX_LEN:-1024}

# v13.1: n per country. Default is 685 -- not a round number chosen for
# convenience, but the FULL available ESS11 edition 4.1 sample size of Cyprus,
# the smallest of the 30 countries. Every country is therefore sampled down to
# exactly the same n, so the design stays perfectly balanced across countries
# at the maximum n that balance allows. Any --sample_per_country above this
# would force Cyprus (and, above 842/906, Iceland and Israel) into "keep all"
# rather than "sample", breaking that balance -- see the per-country counts
# printed by inspect_prompts.py or:
#   python -c "import pandas as pd; d=pd.read_csv('<ess file>', usecols=['cntry']); print(d.cntry.value_counts().sort_values().head(8))"
SAMPLE_N=${SAMPLE_N:-685}

# v13.1 FIX: the pipeline's built-in default (data/ESS11e04_2.csv, edition 4.2)
# does not exist on this cluster; only edition 4.1 is present, under a path
# containing a space. Both pilot jobs failed at the data-loading step before
# touching the GPU, which is why they finished in under a second. Every job
# below now passes --ess_file explicitly, quoted, so this cannot recur.
#
# Record the edition used: this run is on ESS11 EDITION 4.1, not 4.2. If the
# March runs used 4.2, that is itself part of why this is a clean rerun rather
# than a patch to old files, and it belongs in the methods section.
ESS_FILE=${ESS_FILE:-"data/ESS Data/ESS11e04_1.csv"}
if [ ! -f "$ESS_FILE" ]; then
    echo "ERROR: ESS_FILE not found: $ESS_FILE" >&2
    echo "  set it explicitly, e.g.: ESS_FILE=\"data/ESS Data/ESS11e04_1.csv\" bash submit_v13.sh pilot" >&2
    exit 1
fi
echo "Using ESS file: $ESS_FILE"
TIME_LIMIT=${TIME_LIMIT:-"24:00:00"}

# 12 reverse-coded items (vote excluded: Yes/No/Not eligible carries no latent
# direction) + 6 dosage-matched coarse forward controls + 4 long-scale forward
# placebos. Used only by the anchored arm and its stability replicate.
ANCHOR_SET="polintr imsmetn impcntr imdfetn gincdif freehms hmsfmlsh hmsacld \
health rlgatnd aesfdrk hincfel actrolga cptppola inprdsc psppipla psppsgva \
sclmeet trstplc stflife happy stfdem"

job() {   # NAME MODEL PROMPT BACKSTORY SCALES N EXTRA...
    local NAME=$1 MODEL=$2 PROMPT=$3 BS=$4 SC=$5 N=$6; shift 6
    local EXTRA="$*"
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
    --model ${MODEL} --prompt ${PROMPT} \
    --backstory ${BS} --scale_labels ${SC} \
    --sample_per_country ${N} \
    --temperature 0.7 --seed 888 \
    --missing_rule range --human_weight pspwght \
    --max_model_len ${MAX_LEN} \
    --ess_file \"${ESS_FILE}\" ${EXTRA}
echo \"END \$(date)  ${NAME}\"
'")
    [ -n "$ID" ] || { echo "  ERROR: sbatch failed for $NAME" >&2; exit 1; }
    printf "  %-34s job %s\n" "$NAME" "$ID"
}

echo "=================================================================="
echo "submit_v13.sh  mode=$CMD  max_model_len=$MAX_LEN  time=$TIME_LIMIT"
echo "=================================================================="

case "$CMD" in

pilot)
    # Deliberately tiny: 3 countries x 2 respondents. Enough to see raw model
    # output and compare parse rates between the two scale conditions, and small
    # enough that the whole thing is dominated by model loading. Not a
    # scientific arm; nothing from it is reported.
    echo "  264 prompts per arm (22 items x 3 countries x 2 respondents)"
    TIME_LIMIT="00:30:00"
    job pilot_numeric  qwen 1p v5_clean numeric  2 \
        --countries AT DE FI --variables $ANCHOR_SET --tag_suffix _pilotnum
    TIME_LIMIT="00:30:00"
    job pilot_anchored qwen 1p v5_clean anchored 2 \
        --countries AT DE FI --variables $ANCHOR_SET --tag_suffix _pilotanch
    cat <<'EOM'

  When both finish (a few minutes), check them with:
      python check_pilot.py
  Proceed to `all` only if that script prints PROCEED.
EOM
    ;;

rungs)
    # Only the cumulative-ladder jobs: demo_only, minimal, ses, political, and
    # the top rung (full_clean = main_qwen_1p, backstory v5_clean). Independent
    # of the other 8 jobs in `all` — nothing else needs to be resubmitted to
    # redo or extend the ladder.
    TOTAL_PROMPTS=$(( SAMPLE_N * 30 * 42 * 5 ))
    read -r -p "Submit 5 jobs (ladder only), sample_per_country=${SAMPLE_N}, about $(( TOTAL_PROMPTS / 1000000 )).$(( (TOTAL_PROMPTS / 100000) % 10 ))M prompts. Type YES: " c || true
    [ "$c" = "YES" ] || { echo "aborted"; exit 1; }

    echo
    echo "-- cumulative backstory ladder, Qwen 1P ---------------------------"
    for BS in demo_only minimal ses political; do
        job "rung_${BS}" qwen 1p "$BS" numeric ${SAMPLE_N}
    done
    echo
    echo "-- top rung (full_clean = main_qwen_1p) ---------------------------"
    # Resubmitting this regenerates the SAME condition (same tag, same seed),
    # so it is safe whether or not main_qwen_1p already exists: if it does,
    # this run just reproduces and overwrites it with a fresh, independently
    # generated copy (useful as an extra repeated-run check on this condition).
    # If you already trust the existing main_qwen_1p and want to skip
    # regenerating it, comment out the next line.
    job "main_qwen_1p" qwen 1p v5_clean numeric ${SAMPLE_N}
    ;;

newrung)
    # v12.2 (2 Aug 2026, reviewer-requested regrouped ladder): one job, the
    # minimal_politics rung -- gender, age, birth year, country, plus the
    # complete political-identity block (lrscale + clsprty). Together with the
    # existing demo_only, minimal and main_qwen_1p arms this yields the
    # four-level ladder demographics -> +country -> +politics -> full.
    echo
    echo "-- regrouped ladder rung: minimal_politics (6 vars), Qwen 1P ------"
    job "rung_minimal_politics" qwen 1p minimal_politics numeric ${SAMPLE_N}
    ;;

logo)
    # v12.4 (2 Aug 2026): six leave-one-group-out arms against full_clean,
    # for the per-block contrast figure (country label vs every other
    # backstory group). Country and the NUTS code stay in every arm; each
    # arm removes exactly one G-group of sentences (regression-tested:
    # logo_regression.py, 15 pre-existing modes byte-identical).
    echo
    echo "-- six LOGO arms against full_clean, Qwen 1P ----------------------"
    for BS in full_noascriptive full_nosocioecon full_nohousehold full_nocivic full_nomembership full_nominority full_nodomicil; do
        job "logo_${BS#full_no}" qwen 1p "$BS" numeric ${SAMPLE_N}
    done
    ;;

logo2)
    # v12.5: the two arms that replace the cancelled full_nomarkers (30821),
    # aligning the LOGO partition with the ladder tiers.
    echo
    echo "-- tier-nested replacements for the markers arm --------------------"
    for BS in full_nomembership full_nominority; do
        job "logo_${BS#full_no}" qwen 1p "$BS" numeric ${SAMPLE_N}
    done
    ;;

logo_rest)
    # the five LOO arms cancelled on 2 Aug to let the design discussion
    # finish; identical specs to the logo target.
    for BS in full_nohousehold full_nocivic full_nominority full_nomembership full_nodomicil; do
        job "logo_${BS#full_no}" qwen 1p "$BS" numeric ${SAMPLE_N}
    done
    ;;

addone)
    # v12.6 add-one arms: minimal (country + demographics) plus exactly one
    # block. With the LOO arms these give every block its two marginal
    # contributions (sparse and saturated context), an order-free design.
    for BS in minimal_ses minimal_membership minimal_household minimal_civic minimal_minority minimal_domicil minimal_region; do
        job "add_${BS#minimal_}" qwen 1p "$BS" numeric ${SAMPLE_N}
    done
    ;;

probes)
    # the age/birth-year twins (surface-form sensitivity), the country-only
    # arm (label with no demographics at all), and the single-variable
    # income probe (single removal vs its block).
    for BS in minimal_ageonly minimal_yrbrnonly country_only full_noincome; do
        job "probe_${BS}" qwen 1p "$BS" numeric ${SAMPLE_N}
    done
    ;;

swap)
    # v12.6 falsification arm: every backstory carries a WRONG country under
    # a fixed seed-888 derangement; recovery scored against the labeled and
    # the true country separates the label prior from composition.
    job "swap_country" qwen 1p full_swapcountry numeric ${SAMPLE_N}
    ;;

llamapair)
    # the primary country contrast replicated on Llama 1P.
    job "llama_full_noregion"  llama 1p full_noregion  numeric ${SAMPLE_N}
    job "llama_full_nocountry" llama 1p full_nocountry numeric ${SAMPLE_N}
    ;;

memsplit)
    # v12.7: single-variable add arms decomposing the membership block.
    job "add_union"    qwen 1p minimal_union    numeric ${SAMPLE_N}
    job "add_internet" qwen 1p minimal_internet numeric ${SAMPLE_N}
    ;;

memrep)
    # replicate of the one surprising positive (add_membership), so the
    # 0.571 median is read against its own repeat, not only the generic
    # noise floor.
    job "add_membership_rep" qwen 1p minimal_membership numeric ${SAMPLE_N} --tag_suffix _rep
    ;;

newrung3)
    # v12.3 final design (five-level ladder): the economic-position-and-
    # resources tier -- SES triad plus union membership and internet use,
    # 12 variables over the politics rung. Level 5 is full_clean itself.
    echo
    echo "-- regrouped ladder rung: econ tier (12 vars), Qwen 1P ------------"
    job "rung_minimal_politics_econ" qwen 1p minimal_politics_econ numeric ${SAMPLE_N}
    ;;

all)
    # total = 30 countries x n x (11 arms x 42 items + 2 arms x 22 items)
    #       = 30 x n x 506 = 15180 x n
    TOTAL_PROMPTS=$(( 15180 * SAMPLE_N ))
    read -r -p "Submit 13 jobs, sample_per_country=${SAMPLE_N}, about $(( TOTAL_PROMPTS / 1000000 )).$(( (TOTAL_PROMPTS / 100000) % 10 ))M prompts. Type YES: " c || true
    [ "$c" = "YES" ] || { echo "aborted"; exit 1; }

    echo
    echo "-- 4 main conditions (2 models x 2 framings), 42 items ------------"
    for M in qwen llama; do for P in 1p 3p; do
        job "main_${M}_${P}" "$M" "$P" v5_clean numeric ${SAMPLE_N}
    done; done

    echo
    echo "-- 4 cumulative backstory rungs, Qwen 1P --------------------------"
    # The top rung is main_qwen_1p, so it is not repeated. 'political' is
    # included: it was run in March, its summaries shipped in the replication
    # package, and its median country correlation is the lowest of any condition
    # containing the country label. It belongs in Figure 4.
    for BS in demo_only minimal ses political; do
        job "rung_${BS}" qwen 1p "$BS" numeric ${SAMPLE_N}
    done

    echo
    echo "-- primary country contrast (paired; both arms region-free) -------"
    # These two differ by exactly one sentence, 'I live in <country>.', which is
    # verified textually by inspect_prompts.py. Both drop the NUTS region code,
    # because its first two characters are the country ISO code and an arm
    # keeping it would not be a no-country arm.
    job contrast_noregion  qwen 1p full_noregion  numeric ${SAMPLE_N}
    job contrast_nocountry qwen 1p full_nocountry numeric ${SAMPLE_N}

    echo
    echo "-- leave-one-out of the political-identity group ------------------"
    job logo_nopolitical qwen 1p full_nopolitical numeric ${SAMPLE_N}

    echo
    echo "-- scale-anchoring experiment, 22 items --------------------------"
    # Comparator is main_qwen_1p, same code, same library, same seed, same
    # respondents. No separate numeric arm is needed.
    job anchored qwen 1p v5_clean anchored ${SAMPLE_N} --variables $ANCHOR_SET

    echo
    echo "-- repeated-run replicate, 22 items ------------------------------"
    # Closes the manuscript limitation that run-to-run stability was untested,
    # and supplies the empirical noise scale for every |delta r| comparison.
    job replicate qwen 1p v5_clean numeric ${SAMPLE_N} \
        --variables $ANCHOR_SET --tag_suffix _rep

    cat <<'EOM'

  13 jobs queued. Two GPUs, so about two at a time; expect roughly one to two
  days of wall clock. Monitor with:
      bash submit_v13.sh status
EOM
    ;;

status)
    echo
    squeue --me -o '%.10i %.22j %.9P %.2t %.11M %R' 2>/dev/null || true
    echo
    printf "  %-40s %10s %6s\n" FILE SIZE ROWS
    for f in results/silicon_full_raw_*_seed888.csv; do
        [ -e "$f" ] || continue
        printf "  %-40s %10s %6s\n" "$(basename "$f")" \
               "$(du -h "$f" | cut -f1)" "$(( $(wc -l < "$f") - 1 ))"
    done
    ;;

*) echo "Unknown mode: $CMD"; exit 1 ;;
esac
echo "=================================================================="
