#!/usr/bin/env python3
"""
eta.py — aggregate progress and ETA across all 13 jobs from submit_v13.sh all.

WHY THIS EXISTS
---------------
`squeue --me` truncates job names to 8 characters (contrast_noregion and
contrast_nocountry both show as "contrast"; logo_nopolitical shows as
"logo_nop"), so it cannot tell you which specific job is which, or how far
each one has actually gotten. The log FILES, by contrast, are written with the
full untruncated name (submit_v13.sh passes --job-name="$NAME" and
--output="logs/${NAME}_%j.out", which SLURM does not truncate), so this script
reads those directly instead of relying on squeue's display.

For each of the 13 jobs it determines one of three states:
  DONE     the final raw response CSV exists with >= the expected row count
  RUNNING  a log file exists and its last "Processed prompts" line is parsed
           for done/total and tqdm's own elapsed/remaining estimate
  PENDING  no log file yet (job has not been dispatched to a GPU)

It then reports a per-job table and an aggregate ETA: for jobs currently
running it trusts tqdm's own remaining-time estimate (most reliable, since it
already reflects this run's real throughput); for jobs not yet started it
estimates their duration from the observed steady-state rate of whichever job
has processed the most prompts so far, then divides total remaining GPU-time
by 2 (two GPUs on this partition).

Usage (from ~/Winston_Code):
    python eta.py                  # SAMPLE_N=685 default
    python eta.py --sample-n 685
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess

RESULTS = "results"
LOGS = "logs"

# (job name as passed to --job-name, output tag, item count)
JOBS = [
    ("main_qwen_1p",        "qwen_1p",                 42),
    ("main_qwen_3p",        "qwen_3p",                 42),
    ("main_llama_1p",       "llama_1p",                42),
    ("main_llama_3p",       "llama_3p",                42),
    ("rung_demo_only",      "qwen_1p_demo_only",       42),
    ("rung_minimal",        "qwen_1p_minimal",         42),
    ("rung_ses",            "qwen_1p_ses",             42),
    ("rung_political",      "qwen_1p_political",       42),
    ("contrast_noregion",   "qwen_1p_full_noregion",   42),
    ("contrast_nocountry",  "qwen_1p_full_nocountry",  42),
    ("logo_nopolitical",    "qwen_1p_full_nopolitical",42),
    ("anchored",            "qwen_1p_anchored",        22),
    ("replicate",           "qwen_1p_rep",             22),
]

TQDM_RE = re.compile(
    r"(\d+)/(\d+)\s*\[(\d+):(\d+):(\d+)<(?:(\d+):)?(\d+):(\d+)"
)


def hms_to_sec(h, m, s):
    return int(h or 0) * 3600 + int(m) * 60 + int(s)


def latest_log(name: str) -> str | None:
    cands = sorted(glob.glob(os.path.join(LOGS, f"{name}_*.err")),
                   key=os.path.getmtime, reverse=True)
    return cands[0] if cands else None


def parse_progress(path: str):
    """Return (done, total, elapsed_sec, remaining_sec) from the last tqdm line,
    or None if no such line is present yet (still loading the model)."""
    last = None
    # tqdm overwrites with \r; files may store it as one giant line or many.
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    for chunk in text.replace("\r", "\n").split("\n"):
        if "Processed prompts:" in chunk:
            last = chunk
    if last is None:
        return None
    m = TQDM_RE.search(last)
    if not m:
        return None
    done, total = int(m.group(1)), int(m.group(2))
    elapsed = hms_to_sec(m.group(3), m.group(4), m.group(5))
    remaining = hms_to_sec(m.group(6), m.group(7), m.group(8))
    return done, total, elapsed, remaining


def check_errors(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    if "Traceback (most recent call last)" in text:
        # Only flag it if it's NOT the known-benign empty-rq1_df summary crash,
        # which is already patched out in this version but may appear in logs
        # from an unpatched run.
        if "KeyError: 'r_pearson'" in text:
            return ["non-fatal: empty rq1_df summary crash (raw data already "
                   "written; safe to ignore)"]
        return ["UNEXPECTED TRACEBACK — inspect this file directly"]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-n", type=int, default=685)
    args = ap.parse_args()
    n = args.sample_n

    rows = []
    total_target = 0
    total_done = 0
    running_rates = []       # (jobs/sec) for jobs currently in progress
    running_remaining = []   # seconds, tqdm's own estimate
    pending = []

    for name, tag, items in JOBS:
        expected = 30 * n * items
        total_target += expected
        raw_path = os.path.join(RESULTS, f"silicon_full_raw_{tag}_seed888.csv")

        if os.path.exists(raw_path):
            with open(raw_path) as f:
                nrows = sum(1 for _ in f) - 1
            if nrows >= expected:
                rows.append((name, "DONE", expected, expected, "-", []))
                total_done += expected
                continue
            # file exists but short — being written or truncated; treat as running
            status_note = f"raw file present but only {nrows}/{expected} rows"
        else:
            status_note = ""

        log = latest_log(name)
        errs = check_errors(log) if log else []
        if log is None:
            rows.append((name, "PENDING", 0, expected, "not started yet", errs))
            pending.append((name, expected))
            continue

        prog = parse_progress(log)
        if prog is None:
            rows.append((name, "LOADING", 0, expected,
                        "log exists, model still loading / no progress line yet",
                        errs))
            pending.append((name, expected))  # treat like pending for ETA purposes
            continue

        done, total, elapsed, remaining = prog
        total_done += done
        rate = done / elapsed if elapsed > 0 else 0
        running_rates.append(rate)
        running_remaining.append(remaining)
        note = f"{done}/{total} ({100*done/total:.1f}%), elapsed {elapsed//3600}h{(elapsed%3600)//60}m, tqdm ETA {remaining//3600}h{(remaining%3600)//60}m"
        if status_note:
            note += f"  [{status_note}]"
        rows.append((name, "RUNNING", done, total, note, errs))

    # ---- report -------------------------------------------------------
    print("=" * 78)
    print(f"PROGRESS  (sample_per_country={n}, expecting {total_target:,} prompts total)")
    print("=" * 78)
    for name, status, done, total, note, errs in rows:
        flag = "  <-- CHECK THIS" if any("UNEXPECTED" in e for e in errs) else ""
        print(f"  [{status:7s}] {name:20s} {note}{flag}")
        for e in errs:
            print(f"             {e}")

    n_done = sum(1 for r in rows if r[1] == "DONE")
    n_running = sum(1 for r in rows if r[1] == "RUNNING")
    n_pending = sum(1 for r in rows if r[1] in ("PENDING", "LOADING"))
    print()
    print(f"  {n_done} done, {n_running} running, {n_pending} pending/loading "
          f"(out of {len(JOBS)})")
    print(f"  overall: {total_done:,} / {total_target:,} prompts "
          f"({100*total_done/total_target:.1f}%)")

    # ---- ETA ------------------------------------------------------------
    print()
    print("=" * 78)
    print("ESTIMATED TIME TO COMPLETION")
    print("=" * 78)
    if n_pending == 0 and n_running == 0:
        print("  All jobs done." if n_done == len(JOBS) else
              "  Nothing running or pending, but not all done — check manually.")
        return

    if not running_rates:
        print("  No job has produced a progress line yet; cannot estimate rate. "
              "Check back once at least one job is past the model-loading stage.")
        return

    avg_rate = sum(running_rates) / len(running_rates)   # prompts/sec per GPU
    pending_prompt_total = sum(exp for _, exp in pending)
    # crude model: 2 GPUs total; currently-running jobs occupy up to 2 slots and
    # will free up in `max(running_remaining)`; pending jobs then run 2 at a time
    # at the same average per-GPU rate observed so far.
    pending_gpu_seconds = pending_prompt_total / avg_rate if avg_rate > 0 else float("inf")
    pending_wall_seconds = pending_gpu_seconds / 2  # two GPUs

    current_tail = max(running_remaining) if running_remaining else 0
    total_wall_seconds = current_tail + pending_wall_seconds

    print(f"  observed steady-state rate (this environment): {avg_rate:.1f} prompts/sec/GPU")
    print(f"  currently running jobs finish in:               ~{current_tail/3600:.1f}h "
          f"(tqdm's own estimate, most reliable)")
    print(f"  {n_pending} not-yet-started job(s), {pending_prompt_total:,} prompts,")
    print(f"  estimated at {avg_rate:.1f} prompts/sec/GPU x 2 GPUs:  ~{pending_wall_seconds/3600:.1f}h")
    print(f"  ESTIMATED TOTAL REMAINING WALL-CLOCK TIME:       ~{total_wall_seconds/3600:.1f}h "
          f"(rough — depends on job scheduling order)")
    print()
    print("  Caveats: this assumes remaining jobs run 2-at-a-time back-to-back with no")
    print("  scheduling gaps, and that all jobs share roughly the same per-GPU rate")
    print("  regardless of item count (42-item vs 22-item jobs). Re-run this script")
    print("  periodically; the estimate tightens as more jobs report real progress.")


if __name__ == "__main__":
    main()
