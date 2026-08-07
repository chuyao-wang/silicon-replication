#!/bin/bash
# ============================================================================
# watch_llama_anchored.sh — poll the two Llama anchored jobs, then finish.
#
#   bash scripts/watch_llama_anchored.sh              # checks every 10 minutes
#   INTERVAL=3600 bash scripts/watch_llama_anchored.sh   # hourly
#
# RUNS ON THE MAC. Each tick opens one ssh, asks squeue, closes it. Nothing
# stays connected, so a dropped VPN or a closed lid costs one tick, not the
# run. The cluster jobs are independent of this script: killing it does not
# kill them.
#
# When both jobs have left the queue it does the three remaining steps by
# itself:
#   1. checks the four output files exist and are the right shape
#   2. runs `python analyze_2x2.py --model llama` on the cluster
#   3. copies the csv, the manifests and the analysis output down to the Mac
# then raises a macOS notification and exits. If the jobs left the queue
# without producing files, it copies the tail of the error logs down instead
# and says so.
#
# Everything it prints also goes to ~/llama_anchored_watch.log, so you can
# close the terminal and read the log later.
# ============================================================================
set -uo pipefail

CLUSTER=${CLUSTER:-winston}                       # your ~/.ssh/config alias
REMOTE=${REMOTE:-Winston_Code}                    # path under $HOME on the cluster
PKG=${PKG:-$HOME/silicon_pkg/silicon_chapter}     # the package on the Mac
INTERVAL=${INTERVAL:-600}                         # seconds between checks
LOG=${LOG:-$HOME/llama_anchored_watch.log}
# A non-interactive ssh does not read .bashrc, so conda is never initialized and
# bare `python` does not exist. Pick the interpreter on the remote side.
PYSEL='PY=$HOME/miniconda3/bin/python; [ -x "$PY" ] || PY=$(command -v python3 || command -v python); [ -n "$PY" ] || { echo "no python on the cluster PATH" >&2; exit 127; }'

TAGS="llama_1p_full_noregion_anchored llama_1p_full_nocountry_anchored"
NAMES="anch_llama_noregion anch_llama_nocountry"

say()  { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
raw()  { printf '%s\n' "$*" | tee -a "$LOG"; }
ping_user() {   # TITLE MESSAGE
    osascript -e "display notification \"$2\" with title \"$1\" sound name \"Glass\"" 2>/dev/null
    printf '\a'
}
# Quiet: for the squeue poll, where a banner on stderr is just noise.
rsh()  { ssh -T -o BatchMode=yes -o ConnectTimeout=20 "$CLUSTER" "$@" 2>/dev/null; }
# Loud: for anything whose failure must not be silent. The first version of this
# script ran the two python steps through rsh(); the interpreter was missing and
# the error went to /dev/null, so the log recorded two blank lines where the
# result should have been.
rshv() { ssh -T -o BatchMode=yes -o ConnectTimeout=20 "$CLUSTER" "$@" 2>&1; }

say "watching $NAMES on $CLUSTER, every ${INTERVAL}s"
say "log: $LOG"

# ---------------------------------------------------------------- 1. the wait
seen=0
while :; do
    q=$(rsh "squeue --me --noheader --format='%i %j %T %M'")
    if [ $? -ne 0 ]; then
        say "  ssh did not answer; trying again in ${INTERVAL}s"
        sleep "$INTERVAL"; continue
    fi
    live=$(printf '%s\n' "$q" | grep -E 'anch_llama_(noregion|nocountry)')
    if [ -n "$live" ]; then
        seen=1
        say "  still running:"
        raw "$live"
        sleep "$INTERVAL"; continue
    fi
    if [ "$seen" -eq 0 ]; then
        # First tick and nothing in the queue. Either they finished before the
        # watcher started, or they were never submitted. The file check below
        # settles it, so fall through rather than guess.
        say "  no matching job in the queue on the first check; checking output"
    else
        say "  both jobs have left the queue"
    fi
    break
done

# --------------------------------------------------- 2. did they produce files
missing=$(rsh "cd ~/$REMOTE 2>/dev/null || { echo '  missing: the ~/$REMOTE directory itself'; exit 0; }
for t in $TAGS; do
  [ -s results/silicon_full_country_scatter_\$t.csv ] || echo \"  missing: silicon_full_country_scatter_\$t.csv\"
  [ -s results/manifest_\$t.json ]                     || echo \"  missing: manifest_\$t.json\"
done")

if [ -n "$missing" ]; then
    say "THE RUN DID NOT COMPLETE"
    raw "$missing"
    say "last 40 lines of each error log:"
    raw "$(rsh "cd ~/$REMOTE && for n in $NAMES; do
        f=\$(ls -t logs/\${n}_*.err 2>/dev/null | head -1)
        [ -n \"\$f\" ] && { echo \"=== \$f\"; tail -40 \"\$f\"; }
    done")"
    say "also worth checking:  ssh $CLUSTER 'sacct -X --format=JobID,JobName%22,State,Elapsed,ExitCode -S today'"
    ping_user "Llama anchored run FAILED" "No output files. See $LOG"
    exit 1
fi

# ------------------------------------------------------- 3. shape of the files
say "all four files present; checking their shape"
raw "$(rshv "$PYSEL; cd ~/$REMOTE && \$PY - <<'PY'
import json, pandas as pd
for tag in ('llama_1p_full_noregion_anchored', 'llama_1p_full_nocountry_anchored'):
    m = json.load(open(f'results/manifest_{tag}.json'))
    d = pd.read_csv(f'results/silicon_full_country_scatter_{tag}.csv')
    print(' ', tag)
    print('    items', m['variables_n'], '| n/country', m['sample_per_country'],
          '| seed', m['sampling_seed'], '| scale', m['scale_labels'])
    print('    rows', len(d), '| countries', d.cntry.nunique(),
          '| variables', d.variable.nunique())
PY")"
say "expected: items 22, n/country 685, seed 888, scale anchored, 660 rows"

# ------------------------------------------------- 4. the number that matters
say "running analyze_2x2.py --model llama"
result=$(rshv "$PYSEL; cd ~/$REMOTE && \$PY analyze_2x2.py --model llama")
raw "$result"

# --------------------------------------------------------- 5. pull everything
say "copying results to $PKG"
mkdir -p "$PKG/data/summary" "$PKG/results/tables"
for t in $TAGS; do
    scp -q "$CLUSTER:~/$REMOTE/results/silicon_full_country_scatter_$t.csv" "$PKG/data/summary/" \
        && say "  data/summary/silicon_full_country_scatter_$t.csv"
    scp -q "$CLUSTER:~/$REMOTE/results/manifest_$t.json" "$PKG/data/summary/" \
        && say "  data/summary/manifest_$t.json"
done
scp -q "$CLUSTER:~/$REMOTE/results/analysis/twoxtwo_*llama*" "$PKG/results/tables/" 2>/dev/null \
    && say "  results/tables/twoxtwo_*llama*"

headline=$(printf '%s\n' "$result" | grep -iE 'triple|difference' | head -2)
say "DONE"
ping_user "Llama anchored 2x2 finished" "${headline:-see $LOG}"

cat <<EOM

--------------------------------------------------------------------
Next, by hand:
  1. read the triple difference above
  2. RUN_LLAMA_ANCHORED.md section 7 says which sentence to write in 4.2
     and what happens to limitation four in 5.4, under each of the three
     outcomes
  3. optional appendix figure:
       python3 code/figures/fig_2x2.py --model llama \\
               --data data/summary --figdir results/figures \\
               --name fig_2x2_country_by_scale_llama
  4. rebuild and verify on the Mac as usual
--------------------------------------------------------------------
EOM
