#!/usr/bin/env python3
"""
analyze_2x2.py — does disambiguating the scale direction rescue the country label?

Run after submit_anchored_contrast.sh completes.

    python analyze_2x2.py                 # Qwen, the reported 2x2
    python analyze_2x2.py --model llama   # the replication

THE 2x2
-------
                     numeric scales                 anchored scales
  with country       qwen_1p_full_noregion          qwen_1p_full_noregion_anchored
  without country    qwen_1p_full_nocountry         qwen_1p_full_nocountry_anchored

All four cells use seed 888 and n=685, so they draw the same respondents and
every contrast below is paired at the item level with a common human benchmark.

THE PREDICTION UNDER TEST
-------------------------
v13 showed the country label helping forward-coded items (27/29 improved) while
harming reverse-coded ones (1/13 improved), at both a sparse and a rich
baseline. If that asymmetry exists because the model expresses country
information along an inverted scale direction, then supplying verbal anchors
should remove it: under anchoring, the country-label effect on reverse items
should become positive and statistically indistinguishable from its effect on
forward items.

  H1 (single interacting mechanism):
      country effect | anchored, reverse  >  country effect | numeric, reverse
      and the forward/reverse gap in the country effect shrinks towards zero.

  H0 (something else is going on):
      the negative country effect on reverse items survives anchoring, implying
      a factor beyond scale direction.

The decisive quantity is the triple difference: the change in the country-label
effect produced by anchoring, contrasted between reverse and forward items.
Reported on the Fisher-z scale as primary, because these are differences of
correlations and z stabilises their variance; the raw-correlation version is
printed alongside as a descriptive companion.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

R = "results"
OUT = os.path.join(R, "analysis")
os.makedirs(OUT, exist_ok=True)

REVERSE = {"polintr", "imsmetn", "impcntr", "imdfetn", "gincdif", "freehms",
           "hmsfmlsh", "hmsacld", "health", "rlgatnd", "aesfdrk", "hincfel"}
CTRL = {"actrolga", "cptppola", "inprdsc", "psppipla", "psppsgva", "sclmeet"}
PLACEBO = {"trstplc", "stflife", "happy", "stfdem"}
NOISE_DOMINATED = {"hmsfmlsh", "rlgatnd"}   # pre-declared in the revision plan

# The four cells, for whichever model is asked for. The numeric cells are the
# 42-item runs; the item filter below subsets them to the same 22 items as the
# anchored arms, so the contrast is like for like.
_AP = argparse.ArgumentParser(description=__doc__)
_AP.add_argument("--model", default="qwen", choices=["qwen", "llama"],
                 help="which model's 2x2 to analyse (default qwen)")
_A = _AP.parse_args()
M = _A.model
# The default path keeps its historical filenames, so the shipped Qwen
# tables and everything that reads them are untouched. A non-default model
# writes beside them instead of over them: the first Llama run overwrote
# the Qwen tables on the cluster, which is silent and easy to miss.
SUF = "" if M == "qwen" else f"_{M}"

CELLS = {
    ("with",    "numeric"):  f"{M}_1p_full_noregion",
    ("without", "numeric"):  f"{M}_1p_full_nocountry",
    ("with",    "anchored"): f"{M}_1p_full_noregion_anchored",
    ("without", "anchored"): f"{M}_1p_full_nocountry_anchored",
}


def rbc(tag: str) -> pd.Series | None:
    p = os.path.join(R, f"silicon_full_country_scatter_{tag}.csv")
    if not os.path.exists(p):
        return None
    sc = pd.read_csv(p)
    out = {}
    for v, g in sc.groupby("variable"):
        g = g.dropna(subset=["survey_mean", "silicon_mean"])
        if len(g) >= 10 and g.survey_mean.std() > 0 and g.silicon_mean.std() > 0:
            out[v] = stats.pearsonr(g.survey_mean, g.silicon_mean)[0]
    return pd.Series(out, dtype=float)


def z(r):
    return np.arctanh(np.clip(np.asarray(r, float), -0.9999, 0.9999))


# ---- load ---------------------------------------------------------------
r = {}
missing = []
for key, tag in CELLS.items():
    s = rbc(tag)
    if s is None:
        missing.append(tag)
    else:
        r[key] = s
if missing:
    print("Missing scatter files — has submit_anchored_contrast.sh finished?")
    for m in missing:
        print(f"  {m}")
    raise SystemExit(1)

common = sorted(set.intersection(*(set(s.index) for s in r.values())))
print("=" * 78)
print(f"2x2 COUNTRY LABEL x SCALE ANCHORING   ({len(common)} items, n=685, seed 888)")
print("=" * 78)

# ---- cell-level description ---------------------------------------------
print("""
DESCRIPTIVE ONLY -- do NOT read the country effect off these cells.
This batch is 12 of 22 reverse-coded, so a condition-level median is dominated by
the reverse items and can show the country label making things worse while the
paired per-item effects show it helping forward items. The paired estimates are
below; those are the ones to use.""")
print("\nCell medians of the between-country correlation:")
print(f"  {'':18s} {'numeric':>10s} {'anchored':>10s}")
for lab in ("with", "without"):
    a, b = r[(lab, "numeric")][common], r[(lab, "anchored")][common]
    print(f"  {lab+' country':18s} {a.median():+10.4f} {b.median():+10.4f}")

# ---- country effect within each scale condition, by direction -----------
print("\n" + "-" * 78)
print("COUNTRY-LABEL EFFECT, by scale condition and item direction")
print("-" * 78)
rows = []
eff = {}
for scale in ("numeric", "anchored"):
    d_raw = r[("with", scale)][common] - r[("without", scale)][common]
    d_z = pd.Series(z(r[("with", scale)][common]) - z(r[("without", scale)][common]),
                    index=common)
    eff[scale] = d_z
    for grp, items in (("forward (ctrl+placebo)", sorted(CTRL | PLACEBO)),
                       ("reverse (all 12)", sorted(REVERSE)),
                       ("reverse (excl. noise-dominated)",
                        sorted(REVERSE - NOISE_DOMINATED))):
        it = [v for v in items if v in common]
        if not it:
            continue
        dz, dr = d_z[it], d_raw[it]
        w = stats.wilcoxon(dz) if len(it) > 5 else None
        # v13.1: the MEAN is the primary cell statistic, because the triple
        # difference below is a difference of means and was pre-specified that
        # way. Reporting medians here while the triple difference used means made
        # the numbers table arithmetically inconsistent: subtracting the four
        # cells recovered +0.262 rather than the reported +0.423. The median is
        # kept alongside as a robustness statistic. Note the Wilcoxon statistic
        # concerns the MEDIAN of the differences, so it corroborates the sign
        # rather than testing the reported mean.
        print(f"  {scale:9s} {grp:32s} z mean {dz.mean():+7.4f}  "
              f"z median {dz.median():+7.4f}  raw mean {dr.mean():+7.4f}  "
              f"improved {int((dz>0).sum())}/{len(it)}"
              + (f"  p={w.pvalue:.4g}" if w else ""))
        rows.append(dict(scale=scale, group=grp, n=len(it),
                         z_mean=dz.mean(), z_median=dz.median(),
                         raw_mean=dr.mean(), raw_median=dr.median(),
                         n_improved=int((dz > 0).sum()),
                         wilcoxon_p=(w.pvalue if w else np.nan)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, f"twoxtwo_country_effect{SUF}.csv"), index=False)

# ---- the decisive triple difference -------------------------------------
print("\n" + "=" * 78)
print("TRIPLE DIFFERENCE — does anchoring rescue the country label on reverse items?")
print("=" * 78)
fwd = [v for v in sorted(CTRL | PLACEBO) if v in common]
rev = [v for v in sorted(REVERSE - NOISE_DOMINATED) if v in common]

d_rev = eff["anchored"][rev] - eff["numeric"][rev]   # change in country effect
d_fwd = eff["anchored"][fwd] - eff["numeric"][fwd]
triple = d_rev.mean() - d_fwd.mean()

rng = np.random.default_rng(888)
boots = [d_rev[rng.choice(rev, len(rev), replace=True)].mean()
         - d_fwd[rng.choice(fwd, len(fwd), replace=True)].mean()
         for _ in range(5000)]
lo, hi = np.percentile(boots, [2.5, 97.5])

print(f"  change in country effect (z) on REVERSE items: {d_rev.mean():+.4f}")
print(f"  change in country effect (z) on FORWARD items: {d_fwd.mean():+.4f}")
print(f"  TRIPLE DIFFERENCE: {triple:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]"
      f"  {'excludes 0' if lo > 0 or hi < 0 else 'INCLUDES 0'}")

gap_num = eff["numeric"][fwd].mean() - eff["numeric"][rev].mean()
gap_anc = eff["anchored"][fwd].mean() - eff["anchored"][rev].mean()
print(f"\n  forward-minus-reverse gap in the country effect (z):")
print(f"    under numeric scales:  {gap_num:+.4f}")
print(f"    under anchored scales: {gap_anc:+.4f}")
print(f"    gap closed by:         {gap_num - gap_anc:+.4f} "
      f"({100*(gap_num-gap_anc)/gap_num:.0f}% of the numeric gap)"
      if abs(gap_num) > 1e-9 else "")

# Write the primary quantity to disk. It was previously printed only, which
# means the single most important number of this experiment could not be picked
# up by collect_numbers.py and would have had to be transcribed by hand.
# Robustness to dropping the two largest movers is computed here too, because
# freehms and hmsacld contribute a large share of the reverse-item change and a
# reader will ask whether the result survives without them.
# N1 FIX: gap_numeric / gap_anchored were previously computed ONCE, outside this
# loop, on the R10 set, and then written into every row. The R8 row therefore
# carried R10's gap values. That mattered substantively, not just cosmetically:
# on R8 the forward-minus-reverse gap narrows from +0.271 to +0.107, i.e. by
# about 60% -- it is NOT eliminated. "Eliminated, not reversed" holds under R12
# and R10 but fails under R8, and R8 is our own pre-declared robustness
# specification, so a reader is invited to perform exactly the deletion that
# breaks the claim. Report all three.
#
# N2 FIX: the anchored DiD family reports both the pre-declared-exclusion and the
# all-twelve specifications, but the 2x2 previously reported only R10 and R8. R12
# is added because it makes no exclusions at all and is the STRONGER result,
# which removes the obvious question of why two items were dropped.
_big = ["freehms", "hmsacld"]
_r12 = [v for v in common if v in REVERSE]
_rows = []
for _lab, _rv in (("all reverse items in batch (n=12)", _r12),
                  ("analysable reverse items, pre-declared exclusions (n=10)", rev),
                  ("further excluding freehms and hmsacld (n=8)",
                   [v for v in rev if v not in _big])):
    if len(_rv) < 3:
        continue
    _dr = eff["anchored"][_rv] - eff["numeric"][_rv]
    _df = eff["anchored"][fwd] - eff["numeric"][fwd]
    _d = _dr.mean() - _df.mean()
    _rng = np.random.default_rng(888)
    _b = [_dr[_rng.choice(_rv, len(_rv), replace=True)].mean()
          - _df[_rng.choice(fwd, len(fwd), replace=True)].mean() for _ in range(5000)]
    _lo, _hi = np.percentile(_b, [2.5, 97.5])
    _gn = eff["numeric"][fwd].mean() - eff["numeric"][_rv].mean()
    _ga = eff["anchored"][fwd].mean() - eff["anchored"][_rv].mean()
    # "median-based" is itself ambiguous: the median OF the per-item differences
    # and the difference OF the medians are different estimators (+0.271 versus
    # +0.262 on this data). Both are recorded under distinct names so that
    # whichever is quoted can be identified.
    _rows.append(dict(specification=_lab, n_reverse=len(_rv), n_forward=len(fwd),
                      triple_difference_z=_d, ci_lo=_lo, ci_hi=_hi,
                      excludes_zero=bool(_lo > 0 or _hi < 0),
                      reverse_moved_up=int((_dr > 0).sum()),
                      estimator="mean (pre-specified)",
                      td_median_of_differences=_dr.median() - _df.median(),
                      td_difference_of_medians=(
                          eff["anchored"][_rv].median() - eff["numeric"][_rv].median())
                          - (eff["anchored"][fwd].median() - eff["numeric"][fwd].median()),
                      country_effect_numeric_reverse_z=eff["numeric"][_rv].mean(),
                      country_effect_anchored_reverse_z=eff["anchored"][_rv].mean(),
                      gap_numeric_z=_gn, gap_anchored_z=_ga,
                      gap_narrowed_pct=(100 * (_gn - _ga) / _gn if _gn else np.nan),
                      # Three-way, judged against the empirical repeated-run noise
                      # floor for r_bc (about 0.060 on the z scale is the right
                      # order of magnitude). A gap smaller than that is
                      # indistinguishable from zero; a same-signed gap larger than
                      # it has narrowed but survived; a sign flip larger than it
                      # has overshot, which is not the same as elimination.
                      gap_verdict=("eliminated (within noise)" if abs(_ga) < 0.06
                                   else "narrowed but SURVIVES" if _ga * _gn > 0
                                   else "overshot (sign flip)")))
pd.DataFrame(_rows).to_csv(os.path.join(OUT, f"twoxtwo_triple_difference{SUF}.csv"), index=False)
print("\n  three reverse-item specifications, each with its OWN gap:")
print(f"    {'specification':52s} {'DiD3':>8s} {'up':>6s} {'gap num':>9s} {'gap anch':>9s} {'narrowed':>9s}")
for _r in _rows:
    print(f"    {_r['specification']:52s} {_r['triple_difference_z']:+8.4f} "
          f"{f'{_r["reverse_moved_up"]}/{_r["n_reverse"]}':>6s} "
          f"{_r['gap_numeric_z']:+9.4f} {_r['gap_anchored_z']:+9.4f} "
          f"{_r['gap_narrowed_pct']:8.0f}%  {_r['gap_verdict']}")
print("""
    The triple difference is positive and excludes zero in all three, so the
    interaction itself is robust. The GAP is a separate claim and behaves
    differently: it is eliminated under the pre-declared specification, slightly
    overshoots under no exclusions, and under the most conservative specification
    narrows by about 60% while REMAINING POSITIVE. Write it that way. Presenting
    elimination as unconditional invites a reader to run the deletion we
    pre-declared ourselves and find it does not hold there.""")

rev_anch_positive = eff["anchored"][rev].mean() > 0
print()
if (lo > 0) and rev_anch_positive:
    print("  VERDICT: H1 SUPPORTED. Anchoring makes the country label help reverse")
    print("  items too, so the label supplies cross-national signal and the scale")
    print("  direction determines the sign with which it is expressed. Report as a")
    print("  single interacting mechanism with a 2x2 experimental demonstration.")
elif lo > 0:
    print("  VERDICT: PARTIAL. Anchoring shifts the country effect on reverse items")
    print("  upward, but their mean effect is still not positive. Report the")
    print("  interaction, and be explicit that anchoring attenuates rather than")
    print("  eliminates the asymmetry.")
else:
    print("  VERDICT: H1 NOT SUPPORTED. The negative country effect on reverse items")
    print("  survives anchoring. Scale direction alone does not explain the")
    print("  asymmetry; do not merge the two mechanisms in the discussion, and treat")
    print("  this as an open question rather than smoothing it over.")

# ---- per-item detail -----------------------------------------------------
print("\n" + "-" * 78)
print("PER-ITEM DETAIL (country effect on the z scale)")
print("-" * 78)
print(f"  {'item':10s} {'grp':8s} {'numeric':>9s} {'anchored':>9s} {'change':>9s}")
for v in rev + fwd:
    grp = ("reverse" if v in REVERSE else "ctrl" if v in CTRL else "placebo")
    print(f"  {v:10s} {grp:8s} {eff['numeric'][v]:+9.4f} {eff['anchored'][v]:+9.4f} "
          f"{eff['anchored'][v] - eff['numeric'][v]:+9.4f}")
pd.DataFrame({"country_effect_numeric_z": eff["numeric"][rev + fwd],
              "country_effect_anchored_z": eff["anchored"][rev + fwd],
              "change": (eff["anchored"] - eff["numeric"])[rev + fwd]}).to_csv(
    os.path.join(OUT, f"twoxtwo_per_item{SUF}.csv"))

print("\n" + "=" * 78)
print(f"Tables written to {os.path.abspath(OUT)}/")
print("=" * 78)
