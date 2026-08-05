#!/bin/bash
# ============================================================================
# monitor_v13.sh — poll the queue every 300 seconds until all jobs finish.
#
#   Foreground (stops if you close the terminal):
#       bash monitor_v13.sh
#
#   Background (keeps running after you log out; recommended):
#       nohup bash monitor_v13.sh > /dev/null 2>&1 &
#       disown
#
#   Check on it later, from a fresh login:
#       tail -f monitor.log          # live tail, Ctrl+C to stop watching
#       tail -20 monitor.log         # just the last few checks
#
# It writes one timestamped line every 300 seconds to monitor.log, and a
# final summary (from `submit_v13.sh status`) once squeue --me is empty.
# ============================================================================
set -uo pipefail
cd ~/Winston_Code

INTERVAL=${INTERVAL:-300}
LOG=monitor.log

echo "=== monitor started $(date) — checking every ${INTERVAL}s ===" | tee -a "$LOG"

while true; do
    N=$(squeue --me -h 2>/dev/null | wc -l)
    TS=$(date '+%Y-%m-%d %H:%M:%S')

    if [ "$N" -eq 0 ]; then
        echo "[$TS] queue empty — all jobs finished" | tee -a "$LOG"
        break
    fi

    RUN=$(squeue --me -h -t RUNNING 2>/dev/null | wc -l)
    PEND=$(squeue --me -h -t PENDING 2>/dev/null | wc -l)
    echo "[$TS] $N job(s) left: $RUN running, $PEND pending" | tee -a "$LOG"
    squeue --me -o '    %.10i %.20j %.2t %.10M %R' 2>/dev/null | tail -n +1 >> "$LOG"

    sleep "$INTERVAL"
done

echo "" | tee -a "$LOG"
echo "=== FINAL STATUS $(date) ===" | tee -a "$LOG"
bash submit_v13.sh status 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== monitor finished. See monitor.log for the full history. ===" | tee -a "$LOG"
