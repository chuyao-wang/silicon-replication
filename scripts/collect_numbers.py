#!/usr/bin/env python3
"""
collect_numbers.py — one authoritative table of every number the manuscript cites.

Run from ~/Winston_Code, after any of the analysis scripts have run:

    python collect_numbers.py

WHY THIS EXISTS
---------------
The analysis is spread across four scripts (analyze_v13.py, analyze_v13b.py,
analyze_v13c.py, analyze_2x2.py) which between them write eighteen CSVs with no
overlapping filenames. Nothing conflicts and nothing needs redoing. The problem
is different: several of the numbers that will actually appear in the manuscript
are only ever printed to the terminal and never written to a file — the
forward/reverse means, the direction R-squared, the sign-aligned median, the
raw-scale difference-in-differences, the reliability and power figures, and the
vote robustness table. Recovering those means scrolling back through four
terminal sessions, which is how a paper ends up citing two slightly different
values for the same quantity.

This script therefore does not recompute anything from the raw response files.
It reads the item-level CSVs the analysis scripts already wrote and derives every
summary from them, so there is exactly one source of truth per number. If a
quantity here disagrees with something in a terminal log, this file is the one to
trust, because the terminal log may predate a later correction.

Output: results/analysis/manuscript_numbers.csv, one row per quantity, with the
manuscript section it belongs to and the file it came from. Missing inputs are
reported rather than silently skipped, so an absent 2x2 (for instance) is visible
as a gap rather than as a number that quietly failed to appear.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

A = os.path.join("results", "analysis")
REVERSE = {"polintr", "vote", "imsmetn", "impcntr", "imdfetn", "gincdif", "freehms",
           "hmsfmlsh", "hmsacld", "health", "rlgatnd", "aesfdrk", "hincfel"}
ANCHOR_REVERSE = REVERSE - {"vote"}
ANCHOR_CTRL = {"actrolga", "cptppola", "inprdsc", "psppipla", "psppsgva", "sclmeet"}
NOISE_DOMINATED = {"hmsfmlsh", "rlgatnd"}

rows: list[dict] = []
missing: list[str] = []


def add(section, claim, quantity, value, source, note=""):
    rows.append(dict(section=section, claim=claim, quantity=quantity,
                     value=value, source=source, note=note))


def read(name):
    p = os.path.join(A, name)
    if not os.path.exists(p):
        missing.append(name)
        return None
    return pd.read_csv(p)


def z(r):
    return np.arctanh(np.clip(np.asarray(r, float), -0.9999, 0.9999))


# --- 4.1 item-level heterogeneity ----------------------------------------
d = read("qwen_1p_item_r_bc.csv")
if d is not None:
    d = d.rename(columns={d.columns[0]: "variable"}).set_index("variable")
    r = d.r_bc
    is_rev = r.index.isin(REVERSE)
    S = "4.1 item heterogeneity"
    add(S, "aggregate recovery is moderate on the item-level metric",
        "median between-country r", round(r.median(), 4), "qwen_1p_item_r_bc.csv")
    add(S, "threshold counts", "items with r > 0.50", int((r > .5).sum()),
        "qwen_1p_item_r_bc.csv")
    add(S, "threshold counts", "items with r > 0.70", int((r > .7).sum()),
        "qwen_1p_item_r_bc.csv")
    add(S, "a subset of items is inverted", "items with r < 0",
        int((r < 0).sum()), "qwen_1p_item_r_bc.csv")
    add(S, "response direction explains most item heterogeneity",
        "forward mean r", round(r[~is_rev].mean(), 4), "qwen_1p_item_r_bc.csv",
        f"n={int((~is_rev).sum())}")
    add(S, "response direction explains most item heterogeneity",
        "reverse mean r", round(r[is_rev].mean(), 4), "qwen_1p_item_r_bc.csv",
        f"n={int(is_rev.sum())}")
    r2 = stats.pearsonr(is_rev.astype(float), r.values)[0] ** 2
    add(S, "response direction explains most item heterogeneity",
        "R^2 of r on direction", round(r2, 4), "qwen_1p_item_r_bc.csv")
    add(S, "recovery is substantial once direction is accounted for",
        "sign-aligned median r",
        round(float(np.median(np.where(is_rev, -r.values, r.values))), 4),
        "qwen_1p_item_r_bc.csv")
    neg = r[r < 0].sort_values()
    add(S, "the inverted items are almost all reverse-coded",
        "inverted items that are reverse-coded",
        f"{int(neg.index.isin(REVERSE).sum())}/{len(neg)}", "qwen_1p_item_r_bc.csv",
        "the exception is euftf, a bipolar policy item")
    add(S, "most severe inversions", "two most negative items",
        ", ".join(f"{k} {v:+.3f}" for k, v in neg.head(2).items()),
        "qwen_1p_item_r_bc.csv",
        "third and fourth ranks swap between rounds; do not rank beyond the top two")
    # robustness of the direction result to dropping unstable items
    for drop, lab in ((["vote"], "excl. vote"),
                      (["vote", "hmsfmlsh", "rlgatnd"], "excl. vote+hmsfmlsh+rlgatnd")):
        k = r.drop(labels=[x for x in drop if x in r.index])
        kr = k.index.isin(REVERSE).astype(float)
        add(S, "the direction result survives dropping unstable items",
            f"direction R^2 ({lab})",
            round(stats.pearsonr(kr, k.values)[0] ** 2, 4), "qwen_1p_item_r_bc.csv",
            f"n={len(k)}")

# --- 4.1 pooled inflation -------------------------------------------------
d = read("pooled_vs_item.csv")
if d is not None:
    q = d[d.condition == "qwen_1p"]
    if len(q):
        S = "4.1 pooled inflation"
        add(S, "pooled correlations are mechanically inflated and must not be read as recovery",
            "pooled r (all item x country cells)", round(float(q.r_pooled.iloc[0]), 4),
            "pooled_vs_item.csv")
        add(S, "pooled correlations are mechanically inflated",
            "inflation over the median item-level r",
            round(float(q.inflation.iloc[0]), 4), "pooled_vs_item.csv",
            "the gap is between-item scale variation, not between-country signal")

# --- 4.2 country label and enrichment ------------------------------------
d = read("contrasts_by_direction.csv")
if d is not None:
    S = "4.2 country label"
    for _, x in d.iterrows():
        nm = str(x["name"])
        if "rich baseline" in nm:
            add(S, "against a rich profile the country label helps forward items and harms reverse ones",
                "country effect, forward items (median dr)", round(x.fwd_median, 4),
                "contrasts_by_direction.csv", f"n={int(x.n_fwd)}")
            add(S, "against a rich profile the country label helps forward items and harms reverse ones",
                "country effect, reverse items (median dr)", round(x.rev_median, 4),
                "contrasts_by_direction.csv", f"n={int(x.n_rev)}")
        elif "sparse baseline" in nm:
            add(S, "against a sparse profile the country effect is much larger",
                "country effect, forward items (median dr)", round(x.fwd_median, 4),
                "contrasts_by_direction.csv")
        elif "political-identity" in nm:
            add(S, "removing the political-identity variables changes nothing",
                "LOGO effect, forward items (median dr)", round(x.fwd_median, 4),
                "contrasts_by_direction.csv",
                "paired item-level test; the condition-level median difference is misleading")
        elif "enrichment beyond country" in nm:
            add(S, "demographic enrichment beyond the country label contributes nothing",
                "enrichment effect, forward items (median dr)", round(x.fwd_median, 4),
                "contrasts_by_direction.csv", "Wilcoxon null; see analyze_v13b output")

d = read("ladder_summary.csv")
if d is not None:
    S = "4.2 ladder"
    add(S, "the cumulative curve is non-monotonic, which rules out a compositional account",
        "median r by rung",
        " -> ".join(f"{x.rung} {x.median_r_bc:+.3f}" for _, x in d.iterrows()),
        "ladder_summary.csv")
    add(S, "the model reproduces only a fraction of true cross-national spread",
        "between-country SD ratio by rung",
        " -> ".join(f"{x.rung} {x.between_sd_ratio:.3f}" for _, x in d.iterrows()),
        "ladder_summary.csv",
        "peaks at ses: compositional detail increases differentiation but misdirects it")

# --- 4.3 distributional ---------------------------------------------------
d = read("condition_summary.csv")
if d is not None:
    q = d[d.condition == "qwen_1p"]
    if len(q):
        S = "4.3 distributional"
        add(S, "the model compresses within-country dispersion",
            "within-country SD ratio", round(float(q.within_sd_ratio.iloc[0]), 4),
            "condition_summary.csv")
        add(S, "the model compresses between-country dispersion",
            "between-country SD ratio", round(float(q.between_sd_ratio.iloc[0]), 4),
            "condition_summary.csv")

d = read("concordance_per_variable.csv")
if d is not None:
    q = d[d.condition == "qwen_1p"]
    if len(q):
        add("4.3 distributional",
            "correlation is blind to calibration; concordance is not",
            "median Lin CCC", round(float(q.ccc.median()), 4),
            "concordance_per_variable.csv",
            f"against a median r of {q.r.median():.4f}; CCC = r x Cb")
        if "Cb" in q.columns:
            add("4.3 distributional",
                "almost all attainable agreement is lost to miscalibration, not to mis-ranking",
                "median Cb (accuracy factor)", round(float(q.Cb.median()), 4),
                "concordance_per_variable.csv",
                f"{100*(1-float(q.Cb.median())):.0f}% of attainable agreement lost; "
                f"ranking is the r term, calibration is the Cb term")
            add("4.3 distributional",
                "the model's country estimates are not usable as level estimates",
                "items with negative concordance",
                f"{int((q.ccc < 0).sum())}/{len(q)}",
                "concordance_per_variable.csv", "worse than no agreement at all")
            if "bias" in q.columns:
                add("4.3 distributional", "size of the level displacement",
                    "median |bias| (scale points)",
                    round(float(q.bias.abs().median()), 4),
                    "concordance_per_variable.csv")

# --- 4.3 country profile and error ---------------------------------------
d = read("country_profile_and_mae.csv")
if d is not None:
    q = d[d.condition == "qwen_1p"]
    if len(q):
        S = "4.3 country profile"
        add(S, "country profiles are recovered well on a scale-free measure",
            "profile r range",
            f"{q.profile_r.min():.3f} ({q.loc[q.profile_r.idxmin(),'cntry']}) to "
            f"{q.profile_r.max():.3f} ({q.loc[q.profile_r.idxmax(),'cntry']})",
            "country_profile_and_mae.csv", f"mean {q.profile_r.mean():.3f}")
        add(S, "but levels are systematically off",
            "mean range-normalized MAE", round(float(q.norm_mae.mean()), 4),
            "country_profile_and_mae.csv",
            f"range {q.norm_mae.min():.3f} to {q.norm_mae.max():.3f}")

# --- 4.4 cross-level ------------------------------------------------------
d = read("cross_level_agreement.csv")
if d is not None:
    d = d.rename(columns={d.columns[0]: "variable"}).set_index("variable")
    S = "4.4 cross-level"
    is_rev = d.index.isin(REVERSE)
    fwd = d[~is_rev]
    add(S, "item-level cross-level agreement is strong",
        "Pearson r(r_bc, r_ind), as-coded",
        round(stats.pearsonr(d.r_bc, d.r_ind)[0], 4), "cross_level_agreement.csv",
        f"n={len(d)}")
    add(S, "but roughly a third of that magnitude is the shared direction failure",
        "Pearson, forward items only",
        round(stats.pearsonr(fwd.r_bc, fwd.r_ind)[0], 4), "cross_level_agreement.csv",
        f"n={len(fwd)}")
    add(S, "but roughly a third of that magnitude is the shared direction failure",
        "Pearson, sign-aligned",
        round(stats.pearsonr(np.where(is_rev, -d.r_bc, d.r_bc),
                             np.where(is_rev, -d.r_ind, d.r_ind))[0], 4),
        "cross_level_agreement.csv")

d = read("aggregate_vs_individual_by_country.csv")
if d is not None:
    n = len(d)
    rv = stats.pearsonr(d.profile_r, d.mean_within_r)[0]
    se = 1 / np.sqrt(n - 3)
    zz = np.arctanh(np.clip(rv, -0.9999, 0.9999))
    lo, hi = np.tanh(zz - 1.645 * se), np.tanh(zz + 1.645 * se)
    add("4.4 cross-level",
        "aggregate performance cannot be used to select countries for individual use",
        "country-level r (aggregate vs individual)", round(rv, 4),
        "aggregate_vs_individual_by_country.csv",
        f"n={n}, 90% CI [{lo:+.3f}, {hi:+.3f}] — state as a bound, not a relationship")

# --- 5.6 aggregate benchmark ---------------------------------------------
d = read("cross_level_specifications.csv")
if d is not None:
    S = "4.4 cross-level"
    for _, x in d.iterrows():
        add(S, "cross-level agreement, with and without the shared component",
            f"r(r_bc, .) — {x.specification}",
            f"{x.vs_r_ind:+.4f} (r_ind) / {x.vs_pooled_within:+.4f} (pooled-within)",
            "cross_level_specifications.csv",
            "r_ind shares a between-country component with r_bc, so the second "
            "figure is the defensible one; the forward-coded-only pooled-within "
            "value is the most conservative specification")

d = read("aggregate_benchmark.csv")
if d is not None:
    S = "5.6 benchmarks"
    add(S, "a regional average that knows only the coarse region outranks the model",
        "median r: model vs LOO regional average",
        f"{d.r_model.median():+.4f} vs {d.r_region.median():+.4f}",
        "aggregate_benchmark.csv",
        f"the regional average wins on {int((d.r_region > d.r_model).sum())}/{len(d)} "
        f"items, so report this as a median shift, NOT as an item-level sweep")
    add(S, "and beats it far more decisively on level accuracy",
        "median normalised MAE: model vs LOO regional average",
        f"{d.mae_model.median():.4f} vs {d.mae_region.median():.4f}",
        "aggregate_benchmark.csv",
        f"the regional average wins on {int((d.mae_region < d.mae_model).sum())}/{len(d)} items")
    add(S, "the model's level estimates are worse than the cross-country average",
        "items where the model beats the grand mean on nMAE",
        f"{int((d.mae_model < d.mae_grand).sum())}/{len(d)}",
        "aggregate_benchmark.csv",
        f"median nMAE {d.mae_model.median():.4f} against {d.mae_grand.median():.4f}; "
        f"the grand mean cannot rank countries at all")
    red = 1 - d.mae_debias.median() / d.mae_model.median()
    add(S, "almost all of the level error is a single per-item offset",
        "error reduction from removing a per-item offset", f"{100*red:.1f}%",
        "aggregate_benchmark.csv",
        "requires knowing the truth, so this is an optimistic upper bound, not "
        "achievable performance")
    add(S, "MANDATORY CAVEAT for every comparison above", "framing",
        "both comparators require survey data from other countries",
        "aggregate_benchmark.csv",
        "the comparison answers 'should a researcher who already holds real data "
        "use the model to extend it', not 'is the model better than nothing'")

d = read("country_bloc_assignment.csv")
if d is not None:
    add("5.6 benchmarks", "the bloc definition behind the regional comparator",
        "blocs and their sizes",
        "; ".join(f"{b} n={len(g)}" for b, g in d.groupby("bloc")),
        "country_bloc_assignment.csv",
        "a bloc of one falls back to the mean of all other countries")

# --- 4.3 / 5.6 benchmarks -------------------------------------------------
d = read("ceiling_by_variable.csv")
if d is not None:
    col = d.columns[-1]
    S = "5.6 benchmarks"
    add(S, "individual-level recovery is near zero against its own ceiling",
        "median demographic ceiling R", round(float(d[col].median()), 4),
        "ceiling_by_variable.csv",
        "lower bound: five linear predictors after country fixed effects")
    pw = read("pooled_within_r_by_item.csv")
    if pw is not None:
        pwc = pw.columns[-1]
        num, den = float(pw[pwc].median()), float(d[col].median())
        add(S, "achieved individual-level recovery as a share of the ceiling",
            "achieved / ceiling", f"{100*num/den:.1f}%",
            "pooled_within_r_by_item.csv + ceiling_by_variable.csv",
            f"numerator {num:.4f} = pooled within-country r per item, median over "
            f"items — the SAME estimator as the ceiling. Averaging the "
            f"per-country correlations instead is not like-for-like, because "
            f"averaging correlations attenuates relative to pooling; see "
            f"analyze_v13c output for the alternative aggregations. Never use "
            f"r_ind as the numerator: it carries between-country covariance.")

# --- 4.5 robustness -------------------------------------------------------
d = read("weight_robustness.csv")
if d is not None:
    S = "4.5 robustness"
    add(S, "the weighting of the human benchmark does not change conclusions",
        "max item-level |dr| weighted vs unweighted",
        round(float(d.delta.abs().max()), 4), "weight_robustness.csv",
        f"largest mover: {d.loc[d.delta.abs().idxmax(),'variable']}; "
        f"report this, not only the cross-item correlation")

d = read("coverage_by_condition.csv")
if d is not None:
    S = "4.5 robustness"
    q = d[d.condition == "qwen_1p"]
    if len(q):
        add(S, "coverage of scheduled responses",
            "qwen_1p valid human-model intersection",
            f"{100*float(q.both_ok.iloc[0]):.2f}%", "coverage_by_condition.csv")
    l3 = d[d.condition == "llama_3p"]
    if len(l3):
        add(S, "third-person framing fails on long scales for Llama",
            "llama_3p silicon parse rate",
            f"{100*float(l3.silicon_ok.iloc[0]):.2f}%", "coverage_by_condition.csv",
            "concentrated on 0-10 items; treat llama_3p estimates as provisional")

# --- anchored experiment --------------------------------------------------
d = read("anchored_did_real_noise.csv")
if d is not None:
    d = d.rename(columns={d.columns[0]: "variable"}).set_index("variable")
    rev = [v for v in d.index if v in ANCHOR_REVERSE and v not in NOISE_DOMINATED]
    ctl = [v for v in d.index if v in ANCHOR_CTRL]
    S = "4.x anchoring experiment"
    add(S, "verbal anchors reverse the inverted items",
        "DiD, raw correlation scale",
        round(float(d.signal[rev].mean() - d.signal[ctl].mean()), 4),
        "anchored_did_real_noise.csv", f"reverse n={len(rev)}, control n={len(ctl)}")
    add(S, "empirical repeated-run noise floor",
        "median |dr| between two unseeded runs",
        round(float(d.empirical_noise.median()), 4), "anchored_did_real_noise.csv",
        "closes the manuscript limitation that run-to-run stability was untested")
    add(S, "empirical repeated-run noise floor", "max |dr| between two unseeded runs",
        round(float(d.empirical_noise.max()), 4), "anchored_did_real_noise.csv")

d = read("anchored_did_fisherz.csv")
if d is not None:
    pri = d[(d.baseline.str.contains("replicate")) &
            (d.exclusions.str.contains("pre-declared"))]
    if len(pri) == 0:
        pri = d.head(1)
    x = pri.iloc[0]
    S = "4.x anchoring experiment"
    add(S, "verbal anchors reverse the inverted items (primary specification)",
        "DiD, Fisher z scale", round(float(x.did_z), 4), "anchored_did_fisherz.csv",
        f"95% CI [{x.ci_lo:+.4f}, {x.ci_hi:+.4f}]; {int(x.n_up)}/{int(x.n_rev)} up; "
        f"baseline = {x.baseline}")
    add(S, "anchoring does not inflate correlations in general",
        "placebo mean signal (z)", round(float(x.placebo_mean_z), 4),
        "anchored_did_fisherz.csv",
        "long-scale forward items with endpoint-only anchors")
    # v13.1 FIX: this previously took the FIRST row containing "all 12", which
    # was the 42-item-batch baseline, while the primary row above uses the
    # replicate baseline. The two therefore differed in BOTH baseline and
    # exclusion set and were not a clean robustness pair. Match the baseline.
    base_pri = str(x.baseline)
    same = d[(d.baseline == base_pri) & (d.exclusions.str.contains("all 12"))]
    if len(same):
        y = same.iloc[0]
        add(S, "the DiD survives including the pre-excluded items",
            "DiD (z), all 12 reverse items", round(float(y.did_z), 4),
            "anchored_did_fisherz.csv",
            f"95% CI [{y.ci_lo:+.4f}, {y.ci_hi:+.4f}]; SAME baseline as the "
            f"primary ({base_pri}), so this varies only the exclusion set")
    other = d[(d.baseline != base_pri)]
    if len(other):
        add(S, "sensitivity to the choice of numeric baseline",
            "DiD (z), 42-item-batch baseline, pre-declared exclusions",
            round(float(other[other.exclusions.str.contains("pre-declared")].did_z.iloc[0]), 4),
            "anchored_did_fisherz.csv",
            "the raw-scale DiD reported above uses this same baseline, so "
            "compare raw and z within a baseline, never across")

# --- 2x2 ------------------------------------------------------------------
d = read("twoxtwo_country_effect.csv")
if d is not None:
    S = "4.x 2x2 country x anchoring"
    for _, x in d.iterrows():
        grp = "reverse" if "reverse" in str(x.group) else "forward"
        if grp == "reverse" and "excl. noise" not in str(x.group):
            continue
        # The scale condition MUST be in the quantity name. Two rows previously
        # shared the label "median country effect (z)", so the csv could not be
        # read without going back to the terminal to see which was which.
        est = "z_mean" if "z_mean" in d.columns else "z_median"
        add(S, f"country effect on {grp} items, {x.scale} scales",
            f"MEAN country effect (z), {grp}, {x.scale}"
            if est == "z_mean" else f"median country effect (z), {grp}, {x.scale}",
            round(float(x[est]), 4), "twoxtwo_country_effect.csv",
            f"{int(x.n_improved)}/{int(x.n)} improved; estimator = "
            f"{'mean, matching the triple difference so the four cells reconstruct it' if est=='z_mean' else 'MEDIAN — does NOT reconstruct the mean-based triple difference'}")
        if est == "z_mean" and "z_median" in d.columns:
            add(S, f"country effect on {grp} items, {x.scale} scales (robustness)",
                f"median country effect (z), {grp}, {x.scale}",
                round(float(x.z_median), 4), "twoxtwo_country_effect.csv",
                "robust alternative to the mean above")

d = read("twoxtwo_triple_difference.csv")
if d is not None:
    S = "4.x 2x2 country x anchoring"
    # Match on the specification NAME, never on row position: adding the
    # twelve-item specification made it row 0 and silently relabelled the
    # pre-declared primary as a robustness row.
    for _, x in d.iterrows():
        spec = str(x.specification)
        tag = ("primary (pre-declared, n=10)" if "pre-declared" in spec
               else "no exclusions (n=12)" if "in batch" in spec
               else "most conservative (n=8)")
        add(S, "anchoring removes the country label's asymmetry between forward "
               "and reverse items (single interacting mechanism)",
            f"triple difference (z), {tag}", round(float(x.triple_difference_z), 4),
            "twoxtwo_triple_difference.csv",
            f"{spec}; 95% CI [{x.ci_lo:+.4f}, {x.ci_hi:+.4f}]; "
            f"{int(x.reverse_moved_up)}/{int(x.n_reverse)} reverse items moved up")
        add(S, "the gap behaves differently from the interaction",
            f"gap (z), {tag}",
            f"{x.gap_numeric_z:+.4f} -> {x.gap_anchored_z:+.4f}",
            "twoxtwo_triple_difference.csv",
            f"{x.gap_verdict}; narrowed {x.gap_narrowed_pct:.0f}%")
    x0 = d[d.specification.str.contains("pre-declared")].iloc[0]
    for col, lab in (("td_median_of_differences", "median of the per-item differences"),
                     ("td_difference_of_medians", "difference of the medians")):
        if col in d.columns:
            add(S, "the triple difference is same-signed under robust estimators",
                f"triple difference (z), {lab}", round(float(x0[col]), 4),
                "twoxtwo_triple_difference.csv",
                "'median-based' is ambiguous; these two are different estimators "
                "and both are reported so a quoted figure can be identified")
    add(S, "the forward-minus-reverse gap in the country effect is eliminated",
        "gap (z): numeric -> anchored",
        f"{x0.gap_numeric_z:+.4f} -> {x0.gap_anchored_z:+.4f}",
        "twoxtwo_triple_difference.csv",
        "the anchored value is within noise of zero; read as eliminated, "
        "not reversed")

# --- write ---------------------------------------------------------------
out = pd.DataFrame(rows)
if len(out):
    p = os.path.join(A, "manuscript_numbers.csv")
    out.to_csv(p, index=False)
    print("=" * 100)
    print("MANUSCRIPT NUMBERS — one authoritative value per quantity")
    print("=" * 100)
    cur = None
    for _, x in out.iterrows():
        if x.section != cur:
            cur = x.section
            print(f"\n[{cur}]")
        print(f"  {str(x.quantity):48s} {str(x.value):>26s}")
        if x.note:
            print(f"    {'':48s} note: {x.note}")
    print("\n" + "=" * 100)
    print(f"{len(out)} quantities written to {os.path.abspath(p)}")
else:
    print("No analysis CSVs found under results/analysis/. Run the analyze scripts first.")

if missing:
    print("\nMISSING INPUTS (the corresponding numbers are absent above):")
    for m in sorted(set(missing)):
        why = ""
        if m.startswith("twoxtwo"):
            why = "  <- run submit_anchored_contrast.sh, then analyze_2x2.py"
        elif m in ("pooled_vs_item.csv", "ceiling_by_variable.csv",
                   "weight_robustness.csv", "concordance_per_variable.csv",
                   "country_profile_and_mae.csv",
                   "aggregate_vs_individual_by_country.csv"):
            why = "  <- run analyze_v13c.py"
        elif m in ("contrasts_by_direction.csv", "coverage_by_condition.csv",
                   "anchored_did_fisherz.csv"):
            why = "  <- run analyze_v13b.py"
        print(f"  {m}{why}")
print("=" * 100)
