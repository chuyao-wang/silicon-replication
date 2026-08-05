#!/usr/bin/env python3
"""
compare_rounds.py — how stable are the results across the March and v13 rounds?

Run from ~/Winston_Code, where the March files sit in results_march2026/ and the
v13 files in results/.

    python compare_rounds.py

WHY THIS IS THE RIGHT STABILITY TEST
------------------------------------
The two rounds differ in four ways at once: ESS edition (4.2 vs 4.1), sample
size (500 vs 685 per country), library versions (older pandas/vLLM vs pandas
3.0.0 / vLLM 0.15.0), and the fact that generation is unseeded so every run is
an independent draw. If the results agree across all four differences
simultaneously, they are stable in the only sense that matters for the thesis.

The comparison is anchored on SILICON country means rather than on correlations.
That choice is deliberate: the March scatter files carry a human benchmark
computed under the pre-v12 missingness rule (which retained ESS refusal /
don't-know / no-answer codes 7/8/9 on sixteen coarse-scale items), so comparing
March correlations against v13 correlations would confound the round difference
with that correction. Silicon means involve no human data at all, so they
isolate model-output stability cleanly. Correlations are reported afterwards as
a secondary check, with the caveat stated.

THE NOISE FLOOR
---------------
Any comparison needs a yardstick for "how different is different". The v13 round
provides one directly: qwen_1p and qwen_1p_rep are two independent unseeded
generations of the same 22 items over the same 685 respondents, so the spread
between them is pure run-to-run stochasticity with everything else held fixed.
Cross-round differences are judged against that floor. A cross-round difference
at or below the floor means the edition, the sample size and the library
versions contribute nothing detectable.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

MARCH = "results_march2026"
NEW = "results"
OUT = os.path.join(NEW, "analysis")
os.makedirs(OUT, exist_ok=True)

# March tag -> v13 tag. The March package used slightly different names for the
# ladder rungs' scatter files (no _seed888 suffix on the summary tables).
PAIRS = [
    ("qwen_1p", "qwen_1p", "main Qwen 1P (full_clean)"),
    ("qwen_3p", "qwen_3p", "main Qwen 3P"),
    ("llama_1p", "llama_1p", "main Llama 1P"),
    ("llama_3p", "llama_3p", "main Llama 3P"),
    ("qwen_1p_demo_only", "qwen_1p_demo_only", "rung demo_only"),
    ("qwen_1p_minimal", "qwen_1p_minimal", "rung minimal"),
    ("qwen_1p_ses", "qwen_1p_ses", "rung ses"),
    ("qwen_1p_political", "qwen_1p_political", "rung political"),
]

REVERSE = {"polintr", "vote", "imsmetn", "impcntr", "imdfetn", "gincdif", "freehms",
           "hmsfmlsh", "hmsacld", "health", "rlgatnd", "aesfdrk", "hincfel"}


def scatter(base: str, tag: str) -> pd.DataFrame | None:
    for name in (f"silicon_full_country_scatter_{tag}.csv",):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def rbc(sc: pd.DataFrame, hcol: str = "survey_mean") -> pd.Series:
    out = {}
    for v, g in sc.groupby("variable"):
        g = g.dropna(subset=[hcol, "silicon_mean"])
        if len(g) >= 10 and g[hcol].std() > 0 and g.silicon_mean.std() > 0:
            out[v] = stats.pearsonr(g[hcol], g.silicon_mean)[0]
    return pd.Series(out, dtype=float)


# ---------------------------------------------------------------------------
# 0. establish the within-round noise floor from the repeated-run pair
# ---------------------------------------------------------------------------
print("=" * 78)
print("0. WITHIN-ROUND NOISE FLOOR (v13: qwen_1p vs qwen_1p_rep)")
print("=" * 78)
floor_mean = floor_r = None
sc_a, sc_b = scatter(NEW, "qwen_1p"), scatter(NEW, "qwen_1p_rep")
if sc_a is not None and sc_b is not None:
    ma = sc_a.set_index(["variable", "cntry"]).silicon_mean
    mb = sc_b.set_index(["variable", "cntry"]).silicon_mean
    idx = ma.index.intersection(mb.index)
    dm = (ma[idx] - mb[idx]).abs()
    ra, rb = rbc(sc_a), rbc(sc_b)
    ci = sorted(set(ra.index) & set(rb.index))
    dr = (ra[ci] - rb[ci]).abs()
    floor_mean, floor_r = dm.median(), dr.median()
    print(f"  cells compared: {len(idx)} ({len(ci)} items x 30 countries)")
    print(f"  |delta silicon country mean|: median {dm.median():.4f}  p95 {dm.quantile(.95):.4f}")
    print(f"  |delta r_bc| per item:        median {dr.median():.4f}  max {dr.max():.4f}")
    print("  This is the irreducible run-to-run jitter: same data, same libraries,")
    print("  same respondents, unseeded generation. Nothing below this level is")
    print("  interpretable as an effect of anything else.")
else:
    print("  qwen_1p_rep not found — cannot establish a noise floor.")

# ---------------------------------------------------------------------------
# 1. cross-round comparison of silicon country means (no human data involved)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("1. CROSS-ROUND: SILICON COUNTRY MEANS  (March 4.2/n=500 vs v13 4.1/n=685)")
print("=" * 78)
rows = []
hdr = f"  {'condition':28s} {'cells':>6s} {'pearson':>8s} {'med|d|':>8s} {'p95|d|':>8s} {'vs floor':>9s}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for m_tag, n_tag, label in PAIRS:
    sm, sn = scatter(MARCH, m_tag), scatter(NEW, n_tag)
    if sm is None or sn is None:
        print(f"  {label:28s} missing ({'March' if sm is None else 'v13'})")
        continue
    a = sm.set_index(["variable", "cntry"]).silicon_mean
    b = sn.set_index(["variable", "cntry"]).silicon_mean
    idx = a.index.intersection(b.index)
    if len(idx) < 100:
        print(f"  {label:28s} only {len(idx)} overlapping cells — skipped")
        continue
    d = (a[idx] - b[idx]).abs()
    pr = stats.pearsonr(a[idx], b[idx])[0]
    ratio = (d.median() / floor_mean) if floor_mean else np.nan
    print(f"  {label:28s} {len(idx):6d} {pr:8.4f} {d.median():8.4f} "
          f"{d.quantile(.95):8.4f} {ratio:8.2f}x")
    rows.append(dict(condition=label, n_cells=len(idx), pearson=pr,
                     median_abs_delta=d.median(), p95_abs_delta=d.quantile(.95),
                     ratio_to_noise_floor=ratio))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "cross_round_silicon_means.csv"), index=False)
print("\n  'vs floor' below about 1.0 means the cross-round difference is no larger")
print("  than running the same condition twice, i.e. edition, sample size and")
print("  library versions have no detectable effect on the model's outputs.")

# ---------------------------------------------------------------------------
# 2. cross-round comparison of r_bc, per item (secondary, with caveat)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("2. CROSS-ROUND: ITEM-LEVEL r_bc  (secondary — see caveat)")
print("=" * 78)
print("  CAVEAT: the March human benchmark was computed under the pre-v12")
print("  missingness rule, so part of any difference below is that correction")
print("  rather than a round effect. Section 1 is the clean comparison.")
sm, sn = scatter(MARCH, "qwen_1p"), scatter(NEW, "qwen_1p")
if sm is not None and sn is not None:
    rm, rn = rbc(sm), rbc(sn)
    common = sorted(set(rm.index) & set(rn.index))
    a, b = rm[common], rn[common]
    d = (b - a).abs()
    print(f"\n  items compared: {len(common)}")
    print(f"  Pearson {stats.pearsonr(a, b)[0]:.4f}   Spearman {stats.spearmanr(a, b)[0]:.4f}")
    print(f"  |delta r|: median {d.median():.4f}  mean {d.mean():.4f}  max {d.max():.4f} ({d.idxmax()})")
    print(f"  sign agreement: {int((np.sign(a) == np.sign(b)).sum())}/{len(common)}")
    if floor_r:
        print(f"  median |delta| vs within-round floor ({floor_r:.4f}): "
              f"{d.median()/floor_r:.2f}x")

    unstable = d[d > (floor_r if floor_r else 0.10) * 2].sort_values(ascending=False)
    if len(unstable):
        print(f"\n  items exceeding twice the noise floor — do not interpret these:")
        for v in unstable.index:
            print(f"    {v:10s} March {a[v]:+.4f}  v13 {b[v]:+.4f}  delta {b[v]-a[v]:+.4f}"
                  f"  [{'reverse' if v in REVERSE else 'forward'}]")
    pd.DataFrame({"r_bc_march": a, "r_bc_v13": b, "abs_delta": d}).to_csv(
        os.path.join(OUT, "cross_round_item_r_bc.csv"))

# ---------------------------------------------------------------------------
# 3. verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("3. WHAT THIS IMPLIES FOR RERUNNING")
print("=" * 78)
if rows:
    med_ratio = np.nanmedian([r["ratio_to_noise_floor"] for r in rows])
    print(f"  median cross-round difference across conditions = {med_ratio:.2f}x the")
    print(f"  within-round noise floor.")
    if med_ratio <= 1.5:
        print("""
  A third round would not make the numbers cleaner. The residual jitter is
  generation stochasticity, which is irreducible by rerunning: a new round is
  simply another draw with the same spread. If tighter estimates are wanted, the
  correct move is to AVERAGE over the rounds already in hand, or to report both,
  not to generate a third.

  Items whose value moves by more than about twice the floor do so because their
  true correlation is near zero and their silicon between-country dispersion is
  tiny; that is a property of those items and no amount of rerunning fixes it.
  They should be pre-declared uninterpretable, which the revision plan already
  does for hmsfmlsh and rlgatnd.""")
    else:
        print("""
  Cross-round differences exceed the run-to-run floor appreciably, so something
  other than generation noise differs between the rounds. Investigate before
  pooling: check the manifests for sample_per_country, missing_rule,
  human_weight, ESS edition and library versions.""")
print("=" * 78)
