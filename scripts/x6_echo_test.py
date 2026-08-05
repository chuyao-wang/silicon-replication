#!/usr/bin/env python3
"""
X6 -- what is the Llama third-person arm actually emitting?

Run on Winston from ~/Winston_Code. Writes out/echo_test.csv and out/echo_samples.txt.

    python3 x6_echo_test.py

WHY THIS RUN EXISTS
X4 established what the failures are not. Of 863,100 responses, 5.75 per cent carry a
value outside the declared scale and 8.25 per cent carry a value above 1000. Truncation
and infrastructure are excluded, and the first-person control fails on 64 responses that
are all benign fractional midpoints. But neither pre-specified account of the mechanism
fits: 93.52 per cent of misses are above the top, which overshoot predicts, yet the
median excess is 39 scale points, 6.7 times the whole width, over 2,963 distinct values
whose most frequent members are 42, 24, 11, 625, 35 and 0 -- valid on no ESS item.

The leading hypothesis is BACKSTORY ECHO: third-person framing makes the model restate
the persona instead of answering, and the parser takes the first number in the
restatement. 42 and 24 are plausible ages, 11 a plausible income decile or years of
education, and the 71,224 values above 1000 are plausibly years of birth, which the
demo_only leakage n-grams confirm are rendered as bare four-digit numbers.

ONE VALUE ALREADY ARGUES AGAINST A PURE ECHO ACCOUNT. 625 appears 1,170 times and
matches no backstory field: not age, not year of birth, not years of education (0 to
56), not the income decile (1 to 10), not household size, not the left-right placement
(0 to 10). So the script tests three accounts rather than one, and reports how much each
leaves unexplained.

WHAT IT DOES, IN ORDER OF EVIDENTIAL VALUE
1. READS THE TEXT. For each of the twenty most frequent out-of-range values it prints
   up to eight raw responses verbatim and untruncated. This is the cheapest and most
   decisive step and X4 under-used it: statistics were computed where the text should
   have been read first. If the model is restating the persona, one look settles it.
2. THE ECHO JOIN. Every out-of-range response is joined on idno to the respondent's own
   ESS row, and the emitted value is compared against that respondent's agea, yrbrn,
   eduyrs, hinctnta, hhmmb, lrscale and gndr. A per-field match rate follows, together
   with the rate for any field at all. Because a value can coincide with a backstory
   number by chance, the same match rate is computed against a SHUFFLED respondent
   assignment, which is the null this test needs; the echo claim rests on the gap
   between the two, not on the raw rate.
3. WHAT NEITHER ACCOUNT EXPLAINS. Values matching no field of their own respondent are
   tabulated separately with their verbatim text, so that a third mechanism, if there is
   one, is visible rather than buried in a residual.

HOW TO READ THE RESULT
  observed match rate far above the shuffled rate, and the verbatim text shows the
  persona being restated
        -> backstory echo. The failure is instruction following, not scale
           comprehension, and the arm belongs in the limitations rather than in the
           results as evidence for mechanism two.

  match rate near the shuffled rate, and the text shows the model reasoning about
  magnitude before naming a number
        -> not echo. Whatever it is, mechanism two remains available as an account and
           the residual table is where to look next.

Either answer settles where the arm goes in the chapter, which is why this is worth one
job.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

RAW = "results/silicon_full_raw_llama_3p_seed888.csv"
ESS = "data/ESS Data/ESS11e04_1.csv"
OUT = "out"
CH = 200_000
# Backstory numbers a restatement could plausibly surface, with the ESS column name.
FIELDS = ["agea", "yrbrn", "eduyrs", "hinctnta", "hhmmb", "lrscale", "gndr"]
WANT = ["idno", "cntry", "variable", "scale", "silicon_response", "raw_response"]

os.makedirs(OUT, exist_ok=True)
if not os.path.exists(RAW):
    sys.exit(f"absent: {RAW}")

# ---------------------------------------------------------------- collect the misses
frames = []
n_total = 0
for d in pd.read_csv(RAW, usecols=lambda c: c in WANT, chunksize=CH, low_memory=False):
    n_total += len(d)
    raw = d.raw_response.astype(str)
    fail = pd.to_numeric(d.silicon_response, errors="coerce").isna()
    val = pd.to_numeric(raw.str.extract(r"(-?\d+\.?\d*)", expand=False), errors="coerce")
    sc = d.scale.astype(str)
    lo = pd.to_numeric(sc.str.split("-").str[0], errors="coerce")
    hi = pd.to_numeric(sc.str.split("-").str[-1], errors="coerce")
    # Both rejected-numeric branches, because the >1000 group is the one most likely to
    # be years of birth and X4 excluded it.
    bad = fail & val.notna() & ((val < lo) | (val > hi) | (val.abs() > 1000))
    if bad.any():
        frames.append(pd.DataFrame(dict(
            idno=d.idno[bad] if "idno" in d.columns else np.nan,
            cntry=d.cntry[bad], variable=d.variable[bad], val=val[bad],
            lo=lo[bad], hi=hi[bad], raw=raw[bad])))
    print(f"  read {n_total:,}", flush=True)

if not frames:
    sys.exit("no rejected numeric responses found")
a = pd.concat(frames, ignore_index=True)
print(f"\n{len(a):,} rejected numeric responses of {n_total:,} "
      f"({100*len(a)/n_total:.2f} per cent)", flush=True)
print(f"  outside the scale, |value| <= 1000 : "
      f"{int(((a.val.abs() <= 1000) & ((a.val < a.lo) | (a.val > a.hi))).sum()):,}",
      flush=True)
print(f"  |value| > 1000                     : {int((a.val.abs() > 1000).sum()):,}",
      flush=True)

# ---------------------------------------------------------------- 1. read the text
common = [v for v, _ in Counter(a.val.round(2)).most_common(20)]
with open(os.path.join(OUT, "echo_samples.txt"), "w") as fh:
    fh.write(f"{len(a)} rejected numeric responses of {n_total}\n")
    fh.write("Verbatim raw_response for the twenty most frequent rejected values.\n")
    fh.write("Read this before any statistic below it.\n")
    for v in common:
        sub = a[a.val.round(2) == v]
        fh.write(f"\n{'=' * 76}\nvalue {v:g}  ({len(sub):,} occurrences, "
                 f"{sub.variable.nunique()} items, {sub.cntry.nunique()} countries)\n"
                 f"{'=' * 76}\n")
        for _, r in sub.head(8).iterrows():
            fh.write(f"[{r.variable} {r.cntry} scale {r.lo:g}-{r.hi:g}] "
                     f"{r.raw!r}\n")
print(f"\nwrote {OUT}/echo_samples.txt -- READ THIS FIRST", flush=True)
print("  the twenty most frequent rejected values, eight verbatim responses each",
      flush=True)

# ---------------------------------------------------------------- 2. the echo join
rows = []
if a.idno.notna().any() and os.path.exists(ESS):
    ess = pd.read_csv(ESS, usecols=lambda c: c in (["idno", "cntry"] + FIELDS),
                      low_memory=False)
    have = [f for f in FIELDS if f in ess.columns]
    for f in have:
        ess[f] = pd.to_numeric(ess[f], errors="coerce")
    j = a.merge(ess, on=["idno", "cntry"], how="left", suffixes=("", "_ess"))
    matched = j[have].notna().any(axis=1)
    print(f"\njoined {int(matched.sum()):,} of {len(j):,} to their own ESS row",
          flush=True)

    def rates(frame, label):
        out = {}
        anyf = pd.Series(False, index=frame.index)
        for f in have:
            m = np.isclose(frame.val, frame[f], equal_nan=False)
            out[f] = float(np.nanmean(m))
            anyf = anyf | m
        out["ANY_FIELD"] = float(anyf.mean())
        rows.append(dict(comparison=label, n=len(frame),
                         **{k: round(100 * v, 2) for k, v in out.items()}))
        return out

    obs = rates(j[matched], "observed: value against the respondent's own numbers")
    # The null: the same values against a shuffled respondent assignment.
    rng = np.random.default_rng(888)
    sh = j[matched].copy()
    for f in have:
        sh[f] = sh[f].to_numpy()[rng.permutation(len(sh))]
    nul = rates(sh, "null: value against a shuffled respondent")

    print(f"\n{'field':12s} {'observed':>10s} {'shuffled':>10s} {'gap':>8s}", flush=True)
    for f in have + ["ANY_FIELD"]:
        print(f"{f:12s} {100*obs[f]:9.2f}% {100*nul[f]:9.2f}% "
              f"{100*(obs[f]-nul[f]):+7.2f}", flush=True)

    # ------------------------------------------------ 3. what nothing explains
    anym = pd.Series(False, index=j[matched].index)
    for f in have:
        anym = anym | np.isclose(j[matched].val, j[matched][f], equal_nan=False)
        
    resid = j[matched][~anym]
    print(f"\nunexplained by any backstory field: {len(resid):,} "
          f"({100*len(resid)/max(int(matched.sum()), 1):.1f} per cent)", flush=True)
    if len(resid):
        top = Counter(resid.val.round(2)).most_common(10)
        print("  most frequent unexplained values: " +
              "; ".join(f"{v:g} x{c:,}" for v, c in top), flush=True)
        with open(os.path.join(OUT, "echo_samples.txt"), "a") as fh:
            fh.write(f"\n\n{'#' * 76}\nUNEXPLAINED BY ANY BACKSTORY FIELD: "
                     f"{len(resid)} responses\n{'#' * 76}\n")
            for v, c in top:
                sub = resid[resid.val.round(2) == v]
                fh.write(f"\nvalue {v:g} ({c:,} occurrences)\n")
                for _, r in sub.head(6).iterrows():
                    fh.write(f"  [{r.variable} {r.cntry} scale {r.lo:g}-{r.hi:g}] "
                             f"{r.raw!r}\n")
else:
    print("\nno idno column or no ESS file: the echo join is skipped and only the "
          "verbatim samples are produced", flush=True)

if rows:
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "echo_test.csv"), index=False)
    print(f"\nwrote {OUT}/echo_test.csv", flush=True)
print("\nREAD out/echo_samples.txt FIRST. If the responses restate the persona, the "
      "gap in the table above is confirmation and not discovery.", flush=True)
print("DONE", flush=True)
