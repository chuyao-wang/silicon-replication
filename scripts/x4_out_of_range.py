#!/usr/bin/env python3
"""
X4 -- which way does the Llama third-person arm miss the scale?

Run on Winston from ~/Winston_Code. Writes out/out_of_range_profile.csv.

WHAT IS ALREADY KNOWN
14.00 per cent of the arm's 863,100 responses fall outside the declared response
scale, 21.15 per cent of those on eleven-point items against 5.35 per cent on the
short scales, while the first-person control has 64 such responses in the whole arm.
Truncation and infrastructure are excluded. So the loss is scale comprehension.

WHAT IS NOT KNOWN, AND WHY IT MATTERS
The out-of-range values were discarded at parse time, so nobody has looked at WHICH
WAY they miss. Mechanism two says the model treats a numeric response scale as a
magnitude rather than as a bounded set of labelled options. That predicts a specific
signature: the misses should be systematically ABOVE the top of the scale, because a
model reading "0 to 10" as a magnitude and wanting to express a strong response has
nowhere to go but up, and it should be worse on wider scales because the top is
further from the anchors it does understand.

The competing explanation is scale confusion rather than magnitude reasoning: the
model answers a 0-10 item on a 1-5 scale or the reverse, in which case the misses
cluster at values that are valid on some OTHER item in the battery, and they will be
symmetric or even biased downward.

These make opposite predictions and the raw file settles it. Reported here:

  direction   the share above the maximum against below the minimum
  magnitude   how far outside, in scale points and as a share of the scale width
  clustering  whether the out-of-range values pile up on a small number of
              distinct values, which is the signature of the competing account
  by width    all of the above split by eleven-point against short scales

If the misses are overwhelmingly above the top and the overshoot grows with scale
width, mechanism two gains a direct behavioural measurement rather than an inference
from correlations. If they cluster on a handful of values valid elsewhere in the
battery, the finding is scale confusion and the chapter must say so instead.
"""
import os, sys
import numpy as np, pandas as pd
from collections import Counter

COLS = ["variable", "cntry", "scale", "silicon_response", "raw_response"]
CH, OUT = 200_000, "out"
os.makedirs(OUT, exist_ok=True)
rows, val_counts = [], {}

for tag in (sys.argv[1:] or ["llama_3p", "llama_1p"]):
    f = f"results/silicon_full_raw_{tag}_seed888.csv"
    if not os.path.exists(f):
        print(f"{tag}: absent at {f}", flush=True); continue
    acc, n = [], 0
    vc = Counter()
    for d in pd.read_csv(f, usecols=lambda c: c in COLS, chunksize=CH,
                         low_memory=False):
        n += len(d)
        raw = d.raw_response.astype(str)
        fail = pd.to_numeric(d.silicon_response, errors="coerce").isna()
        val = pd.to_numeric(raw.str.extract(r"(-?\d+\.?\d*)", expand=False),
                            errors="coerce")
        sc = d.scale.astype(str)
        lo = pd.to_numeric(sc.str.split("-").str[0], errors="coerce")
        hi = pd.to_numeric(sc.str.split("-").str[-1], errors="coerce")
        oor = fail & val.notna() & (val.abs() <= 1000) & ((val < lo) | (val > hi))
        if not oor.any():
            continue
        sub = pd.DataFrame(dict(variable=d.variable[oor], cntry=d.cntry[oor],
                                val=val[oor], lo=lo[oor], hi=hi[oor]))
        sub["width"] = sub.hi - sub.lo + 1
        sub["above"] = sub.val > sub.hi
        sub["excess"] = np.where(sub.above, sub.val - sub.hi, sub.lo - sub.val)
        sub["excess_rel"] = sub.excess / (sub.width - 1)
        acc.append(sub)
        vc.update(sub.val.astype(float).round(2).tolist())
        print(f"  {tag} {n:,}", flush=True)
    if not acc:
        print(f"\n=== {tag}: no out-of-range responses ===", flush=True)
        rows.append(dict(arm=tag, n_out_of_range=0)); continue
    a = pd.concat(acc, ignore_index=True)
    val_counts[tag] = vc

    def block(sel, label):
        x = a[sel]
        if not len(x):
            return None
        return dict(arm=tag, subset=label, n=len(x),
                    pct_above_top=round(100 * x.above.mean(), 2),
                    median_excess_points=round(float(x.excess.median()), 3),
                    median_excess_rel_width=round(float(x.excess_rel.median()), 4),
                    p90_excess_points=round(float(x.excess.quantile(.9)), 3),
                    n_distinct_values=int(x.val.nunique()),
                    top5_values="; ".join(
                        f"{v:g} x{c:,}" for v, c in
                        Counter(x.val.astype(float).round(2)).most_common(5)))

    for sel, label in [(a.index == a.index, "all out-of-range"),
                       (a.width >= 11, "eleven-point items"),
                       (a.width < 11, "short-scale items")]:
        b = block(sel, label)
        if b:
            rows.append(b)

    print(f"\n=== {tag}: {len(a):,} out-of-range of {n:,} responses ===", flush=True)
    for lab, sel in [("all", a.index == a.index), ("11pt", a.width >= 11),
                     ("short", a.width < 11)]:
        x = a[sel]
        if not len(x):
            continue
        print(f"  {lab:6s} n={len(x):8,d}  above top {100*x.above.mean():6.2f}%  "
              f"median excess {x.excess.median():6.2f} pts "
              f"({x.excess_rel.median():.3f} of width)  "
              f"distinct values {x.val.nunique()}", flush=True)
        print(f"         most common: " + "; ".join(
            f"{v:g} x{c:,}" for v, c in
            Counter(x.val.astype(float).round(2)).most_common(6)), flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "out_of_range_profile.csv"), index=False)
print(f"\nwrote {OUT}/out_of_range_profile.csv", flush=True)
print("READING IT: pct_above_top near 100 with excess growing in scale width supports",
      flush=True)
print("mechanism two. A small n_distinct_values clustering on values valid elsewhere",
      flush=True)
print("in the battery supports scale confusion instead.", flush=True)
print("DONE", flush=True)
