#!/usr/bin/env python3
"""
check_pilot.py — read the two pilot files and print PROCEED or STOP.

The pilot exists to answer one question that cannot be answered without running
the model: does attaching the ESS showcard labels to the scale line change how
often the model returns a parsable number? If it does, the anchored arm's
difference-in-differences would be contaminated by differential missingness
rather than measuring the scale-direction effect.

Everything else the pilot could check is a property of the prompt text and is
verified by inspect_prompts.py on CPU, without a queue slot.

Run from ~/Winston_Code after both pilot jobs finish:
    python check_pilot.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

R = "results"
PAIRS = [("numeric", f"{R}/silicon_full_raw_qwen_1p_pilotnum_seed888.csv"),
         ("anchored", f"{R}/silicon_full_raw_qwen_1p_anchored_pilotanch_seed888.csv")]
# NOTE: the anchored filename has "anchored" inserted before the tag suffix.
# main() builds the tag by first appending "_anchored" when --scale_labels
# anchored is set, and only then appending --tag_suffix, so
# (backstory=v5_clean, scale_labels=anchored, tag_suffix=_pilotanch) produces
# qwen_1p_anchored_pilotanch, not qwen_1p_pilotanch. The real anchored arm (no
# tag_suffix) is unaffected and produces the clean tag qwen_1p_anchored.

REVERSE = set("polintr imsmetn impcntr imdfetn gincdif freehms hmsfmlsh hmsacld "
              "health rlgatnd aesfdrk hincfel".split())

problems, notes = [], []
print("=" * 72)
print("PILOT CHECK")
print("=" * 72)

dfs = {}
for name, path in PAIRS:
    if not os.path.exists(path):
        problems.append(f"{path} is missing — did that job finish? check logs/pilot_*.err")
        continue
    d = pd.read_csv(path)
    d["silicon_response"] = pd.to_numeric(d["silicon_response"], errors="coerce")
    dfs[name] = d

if len(dfs) < 2:
    for p in problems:
        print(f"  STOP: {p}")
    sys.exit(1)

# --- 1. parse rate, the gate ------------------------------------------------
print("\n1. Parse rate — can the output be read as a number?")
rates = {}
for name, d in dfs.items():
    ok = d["silicon_response"].notna()
    rates[name] = ok.mean()
    print(f"   {name:9s} {ok.sum():5d}/{len(d):5d} = {ok.mean():7.2%}")
gap = abs(rates["anchored"] - rates["numeric"]) * 100
n = min(len(d) for d in dfs.values())
se = np.sqrt(2 * 0.985 * 0.015 / n) * 100
print(f"   gap {gap:.2f} percentage points (standard error about {se:.2f}pp at this size)")
if gap > 2 + 2 * se:
    problems.append(f"parse-rate gap of {gap:.2f}pp is too large. Anchoring is changing "
                    f"missingness, not only direction. Inspect the failures below before "
                    f"paying for the full arm.")
else:
    print("   -> acceptable")

# --- 2. what the failures actually look like --------------------------------
print("\n2. Unparsable outputs (verbatim, up to 8 per condition)")
for name, d in dfs.items():
    bad = d.loc[d["silicon_response"].isna(), "raw_response"].astype(str)
    print(f"   {name}: {len(bad)} failures")
    for s in bad.head(8):
        print(f"      {s[:90]!r}")

# --- 3. the dangerous case: a number, but the wrong one ---------------------
# parse_response takes the FIRST number in the output. An output such as
# "On the 1-4 scale, 4" parses as 1. Anchoring makes label echo more likely, so
# this is the failure mode that would silently corrupt the arm.
print("\n3. Outputs that are not a bare number (these parse to the FIRST number "
      "found, which may be wrong)")
for name, d in dfs.items():
    s = d["raw_response"].astype(str).str.strip()
    messy = d[~s.str.fullmatch(r"-?\d+\.?\d*", na=False)]
    print(f"   {name}: {len(messy)} of {len(d)} = {len(messy)/len(d):.2%}")
    for _, r in messy.head(6).iterrows():
        first = re.findall(r"-?\d+\.?\d*", str(r["raw_response"]))
        last = first[-1] if first else None
        flag = "  <-- first != last" if first and first[0] != last else ""
        print(f"      {str(r['raw_response'])[:70]!r} -> parsed {r['silicon_response']}{flag}")
    n_amb = sum(1 for x in messy["raw_response"].astype(str)
                if (f := re.findall(r"-?\d+\.?\d*", x)) and f[0] != f[-1])
    if n_amb:
        notes.append(f"{name}: {n_amb} outputs where the first and last number differ. "
                     f"Do not change parse_response — that would break comparability. "
                     f"Report a last-number sensitivity check instead; raw_response is "
                     f"saved, so it costs nothing.")

# --- 4. is the anchoring visibly doing anything at all? --------------------
# Not a gate. At two respondents per country the means are far too noisy for
# inference; the point is only to confirm the manipulation reaches the model.
print("\n4. Mean response by item, numeric vs anchored (indicative only, n is tiny)")
mn = dfs["numeric"].groupby("variable")["silicon_response"].mean()
ma = dfs["anchored"].groupby("variable")["silicon_response"].mean()
both = sorted(set(mn.index) & set(ma.index))
moved = [(v, ma[v] - mn[v]) for v in both]
rev_shift = np.mean([d for v, d in moved if v in REVERSE])
fwd_shift = np.mean([d for v, d in moved if v not in REVERSE])
print(f"   reverse-coded items: mean shift {rev_shift:+.3f}")
print(f"   forward items:       mean shift {fwd_shift:+.3f}")
print("   A larger shift on reverse items is the direction the mechanism predicts,")
print("   but at this sample size the numbers are noise. Do not read them as a result.")

# --- 5. prompt length -------------------------------------------------------
print("\n5. Prompt length recorded in the job logs")
for f in sorted(glob.glob("logs/pilot_*.out")):
    for line in open(f, errors="ignore"):
        if "Prompt tokens" in line or "Max prompt length" in line:
            print(f"   {os.path.basename(f)}: {line.strip()}")

# --- verdict ----------------------------------------------------------------
print("\n" + "=" * 72)
for x in notes:
    print(f"  NOTE:  {x}")
if problems:
    for x in problems:
        print(f"  STOP:  {x}")
    print("=" * 72)
    sys.exit(1)
print("  PROCEED — run: bash submit_v13.sh all")
print("=" * 72)
