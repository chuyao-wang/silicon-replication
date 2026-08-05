#!/usr/bin/env python3
"""
analyze_v13c.py — everything in the supervisor's letter that still lacks numbers.

Run from ~/Winston_Code:

    python analyze_v13c.py --ess "data/ESS Data/ESS11e04_1.csv"

WHY THIS SCRIPT EXISTS
----------------------
analyze_v13.py and analyze_v13b.py cover the mechanism argument, but a check
against the letter shows they compute none of the following, each of which
answers a specific criticism:

  "Figure 2 ... pooled correlations are extremely high, around 0.8, while the
   typical item-level correlations are much lower"
      -> needs the pooled correlation itself. Section 1 below.

  "country-level profile correlations are discussed without being shown
   graphically. I found myself looking for a figure that was not there"
      -> needs per-country profile correlations, and a second quantity to pair
         them with so the figure says something a scatter cannot. Section 2.

  "I also struggled to follow Section 4.4 and Figure 6"
      -> the replacement argument needs the country-level correlation between
         aggregate and individual recovery WITH its confidence interval, so the
         claim can be stated as a bound rather than a relationship. Section 3.

  "Section 4.5 also feels rather thin. It should describe the robustness checks
   in much more detail, explain why they were undertaken, and summarise what
   they show"
      -> the March replication package had six robustness scripts. None had been
         ported to the v13 data. Sections 4 to 7 port the five that can be
         recomputed without a new inference run.

  "the paper should say more explicitly what constitutes good performance"
      -> needs the demographic-explainable ceiling recomputed on ESS edition
         4.1, since the individual-level benchmark is expressed relative to it.
         Section 5.

Implementations follow the March scripts (compute_ceiling.py,
compute_normalized_mae.py, compute_concordance.py, compute_weight_robustness.py,
compute_vote_robustness.py) so the numbers remain comparable across rounds. The
leakage check is NOT ported: it requires a backstory arm containing the leaked
variables, which the v13 round deliberately did not run. The March leakage
result therefore has to be cited as-is, and that limitation is stated here
rather than left implicit.
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

RANGE = {
    "health": (1, 5), "imbgeco": (0, 10), "imdfetn": (1, 4), "impcntr": (1, 4),
    "imsmetn": (1, 4), "imueclt": (0, 10), "imwbcnt": (0, 10), "hincfel": (1, 4),
    "trstep": (0, 10), "trstlgl": (0, 10), "trstplc": (0, 10), "trstplt": (0, 10),
    "trstprl": (0, 10), "trstprt": (0, 10), "trstun": (0, 10), "atchctr": (0, 10),
    "atcherp": (0, 10), "polintr": (1, 4), "psppipla": (1, 5), "psppsgva": (1, 5),
    "stfdem": (0, 10), "vote": (1, 3), "actrolga": (1, 5), "cptppola": (1, 5),
    "rlgatnd": (1, 7), "aesfdrk": (1, 4), "inprdsc": (0, 6), "pplfair": (0, 10),
    "pplhlp": (0, 10), "ppltrst": (0, 10), "sclmeet": (1, 7), "euftf": (0, 10),
    "freehms": (1, 5), "gincdif": (1, 5), "hmsacld": (1, 5), "hmsfmlsh": (1, 5),
    "happy": (0, 10), "stfeco": (0, 10), "stfedu": (0, 10), "stfgov": (0, 10),
    "stfhlth": (0, 10), "stflife": (0, 10),
}
# Clean numeric backstory predictors, as in compute_ceiling.py.
DEMO = {"agea": (14, 123), "gndr": (1, 2), "eduyrs": (0, 56),
        "hinctnta": (1, 10), "lrscale": (0, 10)}
REVERSE = {"polintr", "vote", "imsmetn", "impcntr", "imdfetn", "gincdif", "freehms",
           "hmsfmlsh", "hmsacld", "health", "rlgatnd", "aesfdrk", "hincfel"}
MAIN = ["qwen_1p", "qwen_3p", "llama_1p", "llama_3p"]


def sc(tag):
    p = os.path.join(R, f"silicon_full_country_scatter_{tag}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def rq3(tag):
    p = os.path.join(R, f"silicon_full_rq3_{tag}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return np.nan
    r = stats.pearsonr(x, y)[0]
    return 2 * r * x.std() * y.std() / (x.std() ** 2 + y.std() ** 2 + (x.mean() - y.mean()) ** 2)


def fisher_ci(r, n, conf=0.90):
    if not np.isfinite(r) or n < 4:
        return np.nan, np.nan
    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1 / np.sqrt(n - 3)
    crit = stats.norm.ppf(0.5 + conf / 2)
    return np.tanh(z - crit * se), np.tanh(z + crit * se)


ap = argparse.ArgumentParser()
ap.add_argument("--ess", default="data/ESS Data/ESS11e04_1.csv")
args = ap.parse_args()

# ---------------------------------------------------------------------------
print("=" * 78)
print("1. POOLED vs ITEM-LEVEL CORRELATION — the Figure 2 point")
print("=" * 78)
print("  The pooled correlation treats every item x country cell as one")
print("  observation, so items measured on 0-10 scales and items on 1-4 scales")
print("  are mixed. Most of the covariance it picks up is between-ITEM (a 0-10")
print("  trust item sits near 5, a 1-4 attitude item sits near 2), not")
print("  between-COUNTRY. That is why it is high while item-level correlations")
print("  are not, and why it must not be read as evidence of recovery.\n")
rows = []
print(f"  {'condition':12s} {'r_pooled':>9s} {'median item r':>14s} {'inflation':>10s}")
for tag in MAIN:
    d = sc(tag)
    if d is None:
        continue
    dd = d.dropna(subset=["survey_mean", "silicon_mean"])
    r_pool = stats.pearsonr(dd.survey_mean, dd.silicon_mean)[0]
    per = {}
    for v, g in dd.groupby("variable"):
        if len(g) >= 10 and g.survey_mean.std() > 0 and g.silicon_mean.std() > 0:
            per[v] = stats.pearsonr(g.survey_mean, g.silicon_mean)[0]
    med = np.median(list(per.values()))
    print(f"  {tag:12s} {r_pool:+9.4f} {med:+14.4f} {r_pool - med:+10.4f}")
    rows.append(dict(condition=tag, r_pooled=r_pool, median_item_r=med,
                     inflation=r_pool - med))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "pooled_vs_item.csv"), index=False)

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("2. COUNTRY PROFILE CORRELATION AND NORMALIZED MAE — the missing figure")
print("=" * 78)
print("  Profile r asks whether the model gets a country's item PROFILE right;")
print("  normalized MAE asks whether it gets the LEVELS right, on a scale made")
print("  comparable across items by dividing by each item's response range.")
print("  Pairing them is what makes a figure worth showing: a country can rank")
print("  items correctly while being systematically displaced.\n")
prof_rows = []
for tag in MAIN:
    d = sc(tag)
    if d is None:
        continue
    dd = d.dropna(subset=["survey_mean", "silicon_mean"]).copy()
    span = dd.variable.map(lambda v: RANGE[v][1] - RANGE[v][0]).astype(float)
    dd["norm_ae"] = (dd.survey_mean - dd.silicon_mean).abs() / span
    for cn, g in dd.groupby("cntry"):
        if len(g) >= 10 and g.survey_mean.std() > 0 and g.silicon_mean.std() > 0:
            prof_rows.append(dict(condition=tag, cntry=cn,
                                  profile_r=stats.pearsonr(g.survey_mean, g.silicon_mean)[0],
                                  norm_mae=g.norm_ae.mean()))
prof = pd.DataFrame(prof_rows)
prof.to_csv(os.path.join(OUT, "country_profile_and_mae.csv"), index=False)
for tag in MAIN:
    p = prof[prof.condition == tag]
    if len(p) == 0:
        continue
    print(f"  {tag:12s} profile r {p.profile_r.min():.3f} ({p.loc[p.profile_r.idxmin(),'cntry']}) "
          f"to {p.profile_r.max():.3f} ({p.loc[p.profile_r.idxmax(),'cntry']}), "
          f"mean {p.profile_r.mean():.3f} SD {p.profile_r.std():.3f}")
    print(f"  {'':12s} norm MAE  {p.norm_mae.min():.3f} to {p.norm_mae.max():.3f}, "
          f"mean {p.norm_mae.mean():.3f}")
q, l = prof[prof.condition == "qwen_1p"], prof[prof.condition == "llama_1p"]
if len(q) and len(l):
    m = q.merge(l, on="cntry", suffixes=("_q", "_l"))
    print(f"\n  Qwen vs Llama per-country normalized MAE correlate at "
          f"r = {stats.pearsonr(m.norm_mae_q, m.norm_mae_l)[0]:.3f}")
    print(f"  Qwen profile r vs Qwen normalized MAE: "
          f"r = {stats.pearsonr(q.profile_r, q.norm_mae)[0]:.3f}")
    print("  (a strong negative value would mean the two metrics rank countries")
    print("   the same way; a weak one means they are separate facts and both")
    print("   belong in the figure)")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("3. AGGREGATE-TO-INDIVIDUAL SELECTION — the Section 4.4 replacement")
print("=" * 78)
r3 = rq3("qwen_1p")
if r3 is not None and len(prof):
    wc = r3.groupby("cntry").r_pearson.mean()
    pr = prof[prof.condition == "qwen_1p"].set_index("cntry").profile_r
    idx = pr.index.intersection(wc.index)
    r_val = stats.pearsonr(pr[idx], wc[idx])[0]
    lo, hi = fisher_ci(r_val, len(idx), 0.90)
    print(f"  Across {len(idx)} countries, correlation between aggregate profile")
    print(f"  recovery and mean individual-level recovery:")
    print(f"    r = {r_val:+.4f}   90% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  The interval is what carries the argument: with n = {len(idx)} countries")
    print("  the data cannot distinguish a null from a moderate association, so")
    print("  aggregate performance cannot be used to select countries for")
    print("  individual-level use. State this as a bound, not as a relationship.")
    # decision-relevant counterexample
    best = pr[idx].nlargest(5).index.tolist()
    worst = pr[idx].nsmallest(5).index.tolist()
    print(f"\n  counterexample: the 5 countries with the best profile recovery")
    print(f"    {best}  mean individual r = {wc[best].mean():+.4f}")
    print(f"  the 5 worst: {worst}  mean individual r = {wc[worst].mean():+.4f}")
    pd.DataFrame({"profile_r": pr[idx], "mean_within_r": wc[idx]}).to_csv(
        os.path.join(OUT, "aggregate_vs_individual_by_country.csv"))

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("4. CONCORDANCE AND COMPRESSION — r is blind to calibration")
print("=" * 78)
print("  Pearson r is invariant to location and scale, so a model that recovers")
print("  the ordering of countries perfectly while compressing the spread and")
print("  shifting the level scores r = 1. Lin's concordance coefficient rewards")
print("  agreement with the 45-degree line instead, and the SD ratio measures")
print("  the compression directly.\n")
crows = []
print(f"  {'condition':12s} {'median r':>9s} {'median CCC':>11s} {'median SD ratio':>16s} {'median Cb':>11s}")
for tag in MAIN:
    d = sc(tag)
    if d is None:
        continue
    per = []
    for v, g in d.dropna(subset=["survey_mean", "silicon_mean"]).groupby("variable"):
        if len(g) < 10:
            continue
        x, y = g.survey_mean.values, g.silicon_mean.values
        rr = stats.pearsonr(x, y)[0]
        # Standard decomposition CCC = r * Cb (Lin 1989). r is PRECISION, i.e.
        # whether the model ranks countries correctly; Cb is ACCURACY, i.e. how
        # much agreement is lost to miscalibration. Cb itself splits into a
        # scale shift v = sd_human/sd_silicon and a location shift u, the mean
        # difference standardised by sqrt(sd_human*sd_silicon). Reporting the
        # decomposition rather than CCC alone is what makes the number
        # interpretable: it says WHY concordance is low.
        sx, sy = x.std(), y.std()
        if sx > 0 and sy > 0:
            vv = sx / sy
            uu = (x.mean() - y.mean()) / np.sqrt(sx * sy)
            Cb = 2 / (vv + 1 / vv + uu ** 2)
        else:
            vv = uu = Cb = np.nan
        per.append(dict(variable=v, r=rr, ccc=ccc(x, y), Cb=Cb,
                        loc_shift_u=uu, scale_shift_v=vv,
                        bias=y.mean() - x.mean(),
                        sd_ratio=(sy / sx if sx > 0 else np.nan)))
    p = pd.DataFrame(per).assign(condition=tag)
    crows.append(p)
    print(f"  {tag:12s} {p.r.median():+9.4f} {p.ccc.median():+11.4f} "
          f"{p.sd_ratio.median():16.4f} {p.Cb.median():11.4f}")
if crows:
    allc = pd.concat(crows)
    allc.to_csv(os.path.join(OUT, "concordance_per_variable.csv"), index=False)
    q = allc[allc.condition == "qwen_1p"]
    print("\n  DECOMPOSITION for qwen_1p, CCC = r x Cb:")
    print(f"    r  (precision: does it rank countries correctly?)   {q.r.median():+.4f}")
    print(f"    Cb (accuracy: is the number itself usable?)         {q.Cb.median():.4f}")
    print(f"    -> {100*(1-q.Cb.median()):.0f}% of attainable agreement is lost to miscalibration")
    print(f"    median |bias|            {q.bias.abs().median():.3f} scale points")
    print(f"    median SD ratio          {q.sd_ratio.median():.3f} (compression)")
    print(f"    median |location shift|  {q.loc_shift_u.abs().median():.3f} "
          f"(mean gap in units of sqrt(sd_h*sd_s))")
    neg = int((q.ccc < 0).sum())
    print(f"    items with NEGATIVE concordance: {neg}/{len(q)} "
          f"(worse than no agreement at all)")
    print("\n  This is the sharpest available answer to what counts as good")
    print("  performance. A reader who only wants to RANK countries gets moderate")
    print("  performance; a reader who wants to USE the model's number as an")
    print("  estimate of a country's value gets essentially nothing. Report both,")
    print("  and say which question each metric answers.")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("5. DEMOGRAPHIC CEILING — what counts as good at the individual level")
print("=" * 78)
if not os.path.exists(args.ess):
    print(f"  ESS file not found at {args.ess} — skipping.")
else:
    outs, preds = list(RANGE), list(DEMO)
    ess = pd.read_csv(args.ess, usecols=["cntry"] + outs + preds, low_memory=False)
    for v, (lo, hi) in {**RANGE, **DEMO}.items():
        ess[v] = pd.to_numeric(ess[v], errors="coerce")
        ess.loc[(ess[v] < lo) | (ess[v] > hi), v] = np.nan

    def within_R(y):
        d = ess[["cntry", y] + preds].dropna()
        if d.cntry.nunique() < 2 or len(d) < 200:
            return np.nan
        dw = d.copy()
        for c in [y] + preds:
            dw[c] = dw[c] - dw.groupby("cntry")[c].transform("mean")
        X = np.column_stack([np.ones(len(dw)), dw[preds].values])
        yv = dw[y].values
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        ss = float((yv ** 2).sum())
        r2 = 1 - float(((yv - X @ beta) ** 2).sum()) / ss if ss > 0 else np.nan
        return float(np.sqrt(max(r2, 0.0)))

    ceil = pd.Series({y: within_R(y) for y in outs}).dropna().sort_values(ascending=False)
    ceil.rename("within_country_ceiling_R").to_csv(os.path.join(OUT, "ceiling_by_variable.csv"))
    print(f"  {len(preds)} predictors after country fixed effects, {len(ceil)} items")
    print(f"  median ceiling R = {ceil.median():.4f}  "
          f"IQR [{ceil.quantile(.25):.3f}, {ceil.quantile(.75):.3f}]  max {ceil.max():.3f}")
    if r3 is not None:
        # v13.1 FIX: the numerator must be the same estimator as the ceiling.
        # The ceiling is a POOLED within-country multiple R -- each item's
        # responses are demeaned within country, all countries are then pooled,
        # and one R is fitted -- with the median taken over items. Averaging the
        # per-country correlations instead (which is what this printed before)
        # is a different estimator and attenuates: it gave 12.9%, and two other
        # plausible aggregation orders gave 12.0% and 19.5%. The like-for-like
        # numerator is the pooled within-country correlation per item, median
        # over items, which gives about 20%.
        rows_pw = []
        for v, g in r3.groupby("variable"):
            g = g.dropna(subset=["n_valid", "human_sd", "silicon_sd", "r_pearson"])
            if len(g) < 3:
                continue
            w = g.n_valid.values.astype(float) - 1
            cov = (w * g.r_pearson.values * g.human_sd.values * g.silicon_sd.values).sum() / w.sum()
            vh = (w * g.human_sd.values ** 2).sum() / w.sum()
            vs = (w * g.silicon_sd.values ** 2).sum() / w.sum()
            if vh > 0 and vs > 0:
                rows_pw.append(dict(variable=v, r_pooled_within=cov / np.sqrt(vh * vs)))
        pw = pd.DataFrame(rows_pw).set_index("variable").r_pooled_within
        pw.to_csv(os.path.join(OUT, "pooled_within_r_by_item.csv"))
        print(f"  achieved, like-for-like (pooled within-country r per item, "
              f"median over items) = {pw.median():+.4f}")
        print(f"  achieved as a share of the ceiling = {100*pw.median()/ceil.median():.1f}%"
              f"   <- report THIS")
        print(f"  for comparison, other aggregations of the same data:")
        print(f"    mean of all per-country r_wc cells      {r3.r_pearson.mean():+.4f}"
              f"  ({100*r3.r_pearson.mean()/ceil.median():.1f}%)")
        print(f"    per-country means, median over countries {r3.groupby('cntry').r_pearson.mean().median():+.4f}"
              f"  ({100*r3.groupby('cntry').r_pearson.mean().median()/ceil.median():.1f}%)")
        print("  These differ because averaging correlations attenuates relative to")
        print("  pooling. State the aggregation order wherever the ratio is quoted.")
    print("  This is a LOWER bound on the ceiling: only five linear predictors are")
    print("  used out of roughly twenty backstory variables, so the true")
    print("  demographic ceiling is higher and the achieved share correspondingly")
    print("  smaller. Report it that way.")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("6. WEIGHTING ROBUSTNESS")
print("=" * 78)
print("  The human benchmark is design-weighted by pspwght throughout, while the")
print("  silicon side is necessarily unweighted. This check asks how much the")
print("  per-item country correlations move if the human side is left unweighted.")
if not os.path.exists(args.ess):
    print("  ESS file not found — skipping.")
else:
    d = sc("qwen_1p")
    if d is not None:
        V = list(RANGE)
        e = pd.read_csv(args.ess, usecols=["cntry", "pspwght"] + V, low_memory=False)
        sub = pd.concat([g.sample(n=685, random_state=888) if len(g) >= 685 else g
                         for _, g in e.groupby("cntry", sort=True)], ignore_index=True)
        sub["pspwght"] = pd.to_numeric(sub.pspwght, errors="coerce").fillna(1.0)
        rec = []
        for v in V:
            lo, hi = RANGE[v]
            x = pd.to_numeric(sub[v], errors="coerce").where(lambda s: (s >= lo) & (s <= hi))
            for cn, ix in sub.groupby("cntry", sort=True).groups.items():
                xi, m = x.loc[ix], x.loc[ix].notna()
                if m.any():
                    rec.append(dict(variable=v, cntry=cn, human_unw=float(xi[m].mean())))
        u = pd.DataFrame(rec)
        mg = d.merge(u, on=["variable", "cntry"])
        rows = []
        for v, g in mg.groupby("variable"):
            g = g.dropna(subset=["survey_mean", "human_unw", "silicon_mean"])
            if len(g) >= 10:
                rows.append(dict(variable=v,
                                 r_weighted=stats.pearsonr(g.survey_mean, g.silicon_mean)[0],
                                 r_unweighted=stats.pearsonr(g.human_unw, g.silicon_mean)[0]))
        w = pd.DataFrame(rows)
        w["delta"] = w.r_unweighted - w.r_weighted
        w.to_csv(os.path.join(OUT, "weight_robustness.csv"), index=False)
        print(f"  items: {len(w)}   correlation between the two sets of r values: "
              f"{stats.pearsonr(w.r_weighted, w.r_unweighted)[0]:.4f}")
        print(f"  max |delta r| at the ITEM level: {w.delta.abs().max():.4f} "
              f"({w.loc[w.delta.abs().idxmax(),'variable']})")
        print(f"  median |delta r|: {w.delta.abs().median():.4f}")
        print("  Report the item-level maximum, not only the cross-item correlation:")
        print("  a correlation near 1 between two vectors of coefficients is")
        print("  compatible with individual items moving appreciably.")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("7. VOTE ROBUSTNESS AND THE UNSTABLE-ITEM SET")
print("=" * 78)
d = sc("qwen_1p")
if d is not None:
    per = {}
    for v, g in d.dropna(subset=["survey_mean", "silicon_mean"]).groupby("variable"):
        if len(g) >= 10:
            per[v] = stats.pearsonr(g.survey_mean, g.silicon_mean)[0]
    s = pd.Series(per)
    for drop, label in (([], "all 42 items"),
                        (["vote"], "excluding vote"),
                        (["vote", "hmsfmlsh"], "excluding vote + hmsfmlsh"),
                        (["vote", "hmsfmlsh", "rlgatnd"],
                         "excluding vote + hmsfmlsh + rlgatnd")):
        k = s.drop(labels=[x for x in drop if x in s.index])
        is_rev = k.index.isin(REVERSE)
        x = is_rev.astype(float)
        r2 = stats.pearsonr(x, k.values)[0] ** 2 if len(set(x)) > 1 else np.nan
        print(f"  {label:36s} n={len(k):2d}  median {k.median():+.4f}  "
              f"neg {int((k<0).sum()):2d}  direction R2 {r2:.4f}")
    print("\n  vote has a 1-3 response range, is near zero in every round, and")
    print("  changed sign between rounds; hmsfmlsh is the only item that moved")
    print("  more than twice the repeated-run noise floor across rounds. The")
    print("  direction result should be shown to survive dropping all three.")

print("\n" + "=" * 78)
print("NOT PORTED, AND WHY")
print("=" * 78)
print("  Leakage robustness. The March round included a backstory arm containing")
print("  the leaked variables (items later used as outcomes), which is what makes")
print("  a leakage check possible. The v13 round deliberately did not run it, so")
print("  this check cannot be recomputed on the new data and the March result")
print("  must be cited as evidence from the earlier round. If it needs to be")
print("  recomputed on v13, that requires one additional inference arm.")
print()
print(f"All tables written to {os.path.abspath(OUT)}/")
print("=" * 78)
