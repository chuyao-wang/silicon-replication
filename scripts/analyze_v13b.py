#!/usr/bin/env python3
"""
analyze_v13b.py — follow-up analyses that analyze_v13.py's headline numbers
showed to be necessary. Run after analyze_v13.py.

    python analyze_v13b.py

WHY EACH OF THESE EXISTS
------------------------
A. Fisher-z difference-in-differences.
   The raw-correlation DiD is +0.846, but the reverse-coded items start at a
   mean r of -0.432 (1.43 units of headroom to +1) while the forward controls
   start at +0.517 (0.48 units). That is a 3x asymmetry, so part of the DiD
   magnitude is mechanical rather than substantive: a bounded quantity moves
   further when it starts further from the bound, and the sampling variance of r
   itself depends on |r|. Fisher's z is the standard remedy and is the
   specification that should be reported as primary. The raw-scale version stays
   as a descriptive companion.

B. Primary contrast and LOGO split by response direction.
   The pooled country-label effect (median +0.091) hides two opposite effects:
   every one of the five items the label helps most is forward-coded, and every
   one of the five it hurts most is reverse-coded. For a reverse-coded item the
   label pushes an already-wrong-signed correlation further wrong, so pooling
   forward and reverse cancels a real effect against an artefactual one. The
   country effect must therefore be reported separately by direction, and the
   headline should rest on the forward items or on sign-aligned values.

C. A better LOGO summary.
   analyze_v13.py reported "mean delta (full - nopolitical) = +0.008", which is
   near zero only because item-level deltas cancel. The condition-level medians
   tell a different story: full_clean +0.443 against full_nopolitical +0.523.
   Removing the political-identity variables IMPROVES aggregate recovery. This
   script reports the paired comparison properly, split by direction.

D. Coverage and parse rates by condition.
   main_llama_3p has a median r of +0.173 with a within-country SD ratio of 1.41
   and 15 negative items, i.e. it is barely recovering anything while producing
   more dispersion than the humans. The March data had an unexplained 76.3%
   coverage figure for the Llama third-person condition. This checks parse rates
   and per-item coverage directly from the raw files so the anomaly is diagnosed
   rather than only flagged, as the manuscript's limitations section promises.

E. Anchored-arm robustness to the choice of numeric baseline.
   Section 8 used the 42-item main run as the numeric baseline. The replicate is
   also numeric and was generated as a 22-item batch, matching the anchored
   arm's batch composition, so re-running the DiD against it removes batch
   composition as a difference between treatment and baseline.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

R = "results"
OUT = os.path.join(R, "analysis")
os.makedirs(OUT, exist_ok=True)

REVERSE = {"polintr", "vote", "imsmetn", "impcntr", "imdfetn", "gincdif", "freehms",
           "hmsfmlsh", "hmsacld", "health", "rlgatnd", "aesfdrk", "hincfel"}
ANCHOR_REVERSE = REVERSE - {"vote"}
ANCHOR_CTRL = {"actrolga", "cptppola", "inprdsc", "psppipla", "psppsgva", "sclmeet"}
ANCHOR_PLACEBO = {"trstplc", "stflife", "happy", "stfdem"}
PREDECLARED_EXCL = {"hmsfmlsh", "rlgatnd"}


def load(kind, tag):
    p = os.path.join(R, f"silicon_full_{kind}_{tag}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def rbc(scatter):
    out = {}
    for v, g in scatter.groupby("variable"):
        g = g.dropna(subset=["survey_mean", "silicon_mean"])
        if len(g) >= 10 and g.survey_mean.std() > 0 and g.silicon_mean.std() > 0:
            out[v] = stats.pearsonr(g.survey_mean, g.silicon_mean)[0]
    return pd.Series(out, dtype=float)


def fisher_z(r):
    r = np.clip(np.asarray(r, dtype=float), -0.9999, 0.9999)
    return np.arctanh(r)


def split_report(delta: pd.Series, label_a: str, label_b: str, name: str,
                 dz: pd.Series | None = None):
    """Report a paired item-level difference split by response direction."""
    is_rev = delta.index.isin(REVERSE)
    fwd, rev = delta[~is_rev], delta[is_rev]
    print(f"  {name}: {label_a} minus {label_b}")
    print(f"    all items      median {delta.median():+.4f}  mean {delta.mean():+.4f}  "
          f"n={len(delta)}  improved {int((delta>0).sum())}/{len(delta)}")
    print(f"    FORWARD items  median {fwd.median():+.4f}  mean {fwd.mean():+.4f}  "
          f"n={len(fwd)}  improved {int((fwd>0).sum())}/{len(fwd)}")
    print(f"    REVERSE items  median {rev.median():+.4f}  mean {rev.mean():+.4f}  "
          f"n={len(rev)}  improved {int((rev>0).sum())}/{len(rev)}")
    t = stats.wilcoxon(fwd) if len(fwd) > 5 else None
    if t:
        print(f"    forward-only Wilcoxon signed-rank p = {t.pvalue:.4g}")
    # The CSV previously carried medians only, while the printed output also
    # showed means and Fisher-z values. Anything needed for a figure must be in
    # the file, not only on screen.
    # AUDIT FIX (P1). This line previously read
    #     dz = pd.Series(fisher_z(delta.values), index=delta.index)
    # which applies arctanh to a DIFFERENCE of correlations. The Fisher
    # transform is defined on a correlation, so the transforms must be
    # differenced: dz_i = z(r_a,i) - z(r_b,i). The correct quantity was
    # already computed at the call site for printing but never written to the
    # CSV. The wrong version understated the forward country-label effect as
    # +0.1360 where the correct value is +0.1967, and was undefined wherever
    # |delta| >= 1, which occurs under the sparse baseline.
    if dz is None:
        raise ValueError("split_report requires dz = z(r_a) - z(r_b) per "
                         "item; arctanh of a difference is not a Fisher "
                         "transform")
    return dict(name=name,
                all_median=delta.median(), all_mean=delta.mean(),
                fwd_median=fwd.median(), fwd_mean=fwd.mean(),
                rev_median=rev.median(), rev_mean=rev.mean(),
                all_median_z=dz.median(), all_mean_z=dz.mean(),
                fwd_median_z=dz[~is_rev].median(), fwd_mean_z=dz[~is_rev].mean(),
                rev_median_z=dz[is_rev].median(), rev_mean_z=dz[is_rev].mean(),
                n_fwd=len(fwd), n_rev=len(rev),
                fwd_wilcoxon_p=(t.pvalue if t else np.nan))


print("=" * 78)
print("A. ANCHORED DiD ON FISHER-z SCALE (primary specification)")
print("=" * 78)
sc_A = load("country_scatter", "qwen_1p")
sc_B = load("country_scatter", "qwen_1p_rep")
sc_T = load("country_scatter", "qwen_1p_anchored")
did_rows = []
if all(x is not None for x in (sc_A, sc_B, sc_T)):
    anchor_items = ANCHOR_REVERSE | ANCHOR_CTRL | ANCHOR_PLACEBO
    rA_all, rB_all, rT_all = rbc(sc_A), rbc(sc_B), rbc(sc_T)
    common = sorted(set(rA_all.index) & set(rB_all.index) & set(rT_all.index)
                    & anchor_items)

    for baseline_name, rBase in (("main_qwen_1p (42-item batch)", rA_all),
                                 ("replicate (22-item batch, matches anchored)", rB_all)):
        print(f"\n  baseline = {baseline_name}")
        zBase, zT = fisher_z(rBase[common]), fisher_z(rT_all[common])
        sig_z = pd.Series(zT - zBase, index=common)
        # empirical repeated-run noise on the z scale
        noise_z = pd.Series(np.abs(fisher_z(rA_all[common]) - fisher_z(rB_all[common])),
                            index=common)

        for excl_label, excl in (("pre-declared exclusions", PREDECLARED_EXCL),
                                 ("all 12 reverse items", set())):
            rev_items = [v for v in common if v in ANCHOR_REVERSE and v not in excl]
            ctl_items = [v for v in common if v in ANCHOR_CTRL]
            plc_items = [v for v in common if v in ANCHOR_PLACEBO]
            did = sig_z[rev_items].mean() - sig_z[ctl_items].mean()
            rng = np.random.default_rng(888)
            boots = [sig_z[rng.choice(rev_items, len(rev_items), replace=True)].mean()
                     - sig_z[rng.choice(ctl_items, len(ctl_items), replace=True)].mean()
                     for _ in range(5000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            n_up = int((sig_z[rev_items] > 0).sum())
            print(f"    [{excl_label}] DiD(z) = {did:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
                  f"  {'excludes 0' if lo > 0 or hi < 0 else 'INCLUDES 0'}"
                  f"  up {n_up}/{len(rev_items)}")
            print(f"      placebo mean signal(z) = {sig_z[plc_items].mean():+.4f}"
                  f"   (should be ~0 if anchoring per se does not inflate)")
            did_rows.append(dict(baseline=baseline_name, exclusions=excl_label,
                                 did_z=did, ci_lo=lo, ci_hi=hi,
                                 n_up=n_up, n_rev=len(rev_items),
                                 placebo_mean_z=sig_z[plc_items].mean()))
    pd.DataFrame(did_rows).to_csv(os.path.join(OUT, "anchored_did_fisherz.csv"), index=False)
    print("\n  Interpretation note: report the Fisher-z DiD as primary and the")
    print("  raw-correlation DiD as descriptive. The raw-scale magnitude is inflated")
    print("  by the reverse items' 3x greater headroom to the +1 bound.")
else:
    print("  required scatter files not all present — skipping.")

print("\n" + "=" * 78)
print("B. PRIMARY CONTRAST AND LOGO, SPLIT BY RESPONSE DIRECTION")
print("=" * 78)
split_rows = []
pairs = [
    ("qwen_1p_full_noregion", "qwen_1p_full_nocountry",
     "country label effect (rich baseline)"),
    ("qwen_1p_minimal", "qwen_1p_demo_only",
     "country label effect (sparse baseline)"),
    ("qwen_1p", "qwen_1p_full_nopolitical",
     "political-identity variables (full minus nopolitical)"),
    ("qwen_1p", "qwen_1p_minimal",
     "all enrichment beyond country (full minus minimal)"),
]
for tag_a, tag_b, name in pairs:
    sa, sb = load("country_scatter", tag_a), load("country_scatter", tag_b)
    if sa is None or sb is None:
        print(f"  {name}: files missing — skipped")
        continue
    ra, rb = rbc(sa), rbc(sb)
    common = sorted(set(ra.index) & set(rb.index))
    delta = (ra[common] - rb[common])
    # AUDIT FIX (P1): difference the transforms, never transform the difference.
    dz = pd.Series(fisher_z(ra[common]) - fisher_z(rb[common]), index=common)
    print()
    split_rows.append(split_report(delta, tag_a, tag_b, name, dz=dz))
    is_rev = dz.index.isin(REVERSE)
    print(f"    Fisher-z: all {dz.median():+.4f} | forward {dz[~is_rev].median():+.4f} "
          f"| reverse {dz[is_rev].median():+.4f}")
pd.DataFrame(split_rows).to_csv(os.path.join(OUT, "contrasts_by_direction.csv"), index=False)

print("\n" + "=" * 78)
print("C. COVERAGE AND PARSE RATES BY CONDITION")
print("=" * 78)
cov_rows = []
raws = sorted(glob.glob(os.path.join(R, "silicon_full_raw_*_seed888.csv")))
print(f"  {'condition':30s} {'rows':>10s} {'silicon ok':>11s} {'human ok':>10s} {'both':>8s}")
for p in raws:
    tag = os.path.basename(p).replace("silicon_full_raw_", "").replace("_seed888.csv", "")
    try:
        d = pd.read_csv(p, usecols=["variable", "cntry", "human_response",
                                    "silicon_response"], low_memory=False)
    except Exception as e:
        print(f"  {tag:30s} could not read: {e}")
        continue
    s = pd.to_numeric(d.silicon_response, errors="coerce")
    h = pd.to_numeric(d.human_response, errors="coerce")
    both = (s.notna() & h.notna()).mean()
    print(f"  {tag:30s} {len(d):10,} {s.notna().mean():10.2%} {h.notna().mean():9.2%} "
          f"{both:7.2%}")
    cov_rows.append(dict(condition=tag, n_rows=len(d), silicon_ok=s.notna().mean(),
                         human_ok=h.notna().mean(), both_ok=both))
    # diagnose the worst offenders by item and country
    if s.notna().mean() < 0.95:
        by_item = s.groupby(d.variable).apply(lambda x: x.notna().mean()).sort_values()
        by_cntry = s.groupby(d.cntry).apply(lambda x: x.notna().mean()).sort_values()
        print(f"      LOW COVERAGE — worst 5 items:   "
              + ", ".join(f"{k} {v:.1%}" for k, v in by_item.head(5).items()))
        print(f"      worst 5 countries: "
              + ", ".join(f"{k} {v:.1%}" for k, v in by_cntry.head(5).items()))
        by_item.to_csv(os.path.join(OUT, f"coverage_by_item_{tag}.csv"))
pd.DataFrame(cov_rows).to_csv(os.path.join(OUT, "coverage_by_condition.csv"), index=False)

print("\n" + "=" * 78)
print(f"Tables written to {os.path.abspath(OUT)}/")
print("=" * 78)
