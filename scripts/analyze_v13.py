#!/usr/bin/env python3
"""
analyze_v13.py — full headline analysis of the v13 clean rerun.

Run this from ~/Winston_Code once `python eta.py` reports all 13 jobs DONE.

    python analyze_v13.py

Reads only files the pipeline already wrote (country_scatter, rq1, rq3, and
manifest JSONs for each of the 13 conditions) — no ESS microdata access is
needed, because run_full_variables() in v12 already applies the corrected
missingness rule and pspwght weighting at generation time, unlike the March
files which needed retrospective correction in reanalyse_v12.py.

WHAT THIS PRODUCES, mapped to the revision plan
------------------------------------------------
  environment metadata   -> methods section: ESS edition, package versions,
                             confirmation that missing_rule=range and
                             human_weight=pspwght were used throughout, and
                             that every arm was generated with gen_seed=None
                             (independent, unseeded) so main_qwen_1p and
                             replicate constitute a genuine repeated-run pair
  condition summary       -> revision plan 3.1/3.4, corrected_condition_summary
  item direction decomp    -> revision plan 3.1 (forward/reverse means, R^2,
                             sign-aligned median, most severe inversions)
  primary contrast         -> the country-label effect: full_noregion vs
                             full_nocountry, paired per item (same respondents)
  ladder + dispersion      -> revision plan 3.5 (five rungs, between-country
                             SD ratio)
  LOGO                     -> full_nopolitical vs full_clean
  cross-level agreement    -> revision plan 3.3 (three robustness versions)
  reliability/attenuation  -> revision plan 3.2, recomputed at n=685
  anchored DiD             -> revision plan section 4/8, using REAL repeated-
                             run noise (replicate vs main_qwen_1p's 22-item
                             subset) instead of a parametric estimate

Output: prints headlines to the terminal and writes CSVs to
results/analysis/.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

R = "results"
OUT = os.path.join(R, "analysis")
os.makedirs(OUT, exist_ok=True)

REVERSE = {"polintr", "vote", "imsmetn", "impcntr", "imdfetn", "gincdif", "freehms",
           "hmsfmlsh", "hmsacld", "health", "rlgatnd", "aesfdrk", "hincfel"}
ANCHOR_REVERSE = REVERSE - {"vote"}  # vote is excluded from the anchored arm
ANCHOR_CTRL = {"actrolga", "cptppola", "inprdsc", "psppipla", "psppsgva", "sclmeet"}
ANCHOR_PLACEBO = {"trstplc", "stflife", "happy", "stfdem"}
# Pre-declared before results returned, on the grounds of near-zero silicon
# between-country dispersion. AUDIT NOTE: in v13 raw dispersion does NOT separate
# these two cleanly -- the forward control inprdsc sits between them at 0.0313,
# against hmsfmlsh 0.0201 and rlgatnd 0.0316 -- so the manuscript must not
# justify the screen on dispersion alone. Country-mean RELIABILITY does separate
# them: hmsfmlsh 0.25 and rlgatnd 0.44 are the only two items below 0.50, while
# inprdsc is 0.615 (see section 7). State the screen on reliability, and
# attribute the originally quoted 0.017 and 0.044 to the round on which the
# declaration was made rather than to v13. The screen was applied to the reverse
# arm only; dropping inprdsc from the forward controls moves the primary triple
# difference by -0.0004, and all ten forward leave-one-out variants keep the
# triple difference positive in [0.403, 0.455].
UNINTERPRETABLE = {"hmsfmlsh", "rlgatnd"}

TAGS = {
    "qwen_1p":                 "main_qwen_1p (full_clean, top rung)",
    "qwen_3p":                 "main_qwen_3p",
    "llama_1p":                "main_llama_1p",
    "llama_3p":                "main_llama_3p",
    "qwen_1p_demo_only":       "rung: demo_only",
    "qwen_1p_minimal":         "rung: minimal",
    "qwen_1p_ses":             "rung: ses",
    "qwen_1p_political":       "rung: political",
    "qwen_1p_full_noregion":   "contrast: full_noregion",
    "qwen_1p_full_nocountry":  "contrast: full_nocountry",
    "qwen_1p_full_nopolitical":"LOGO: full_nopolitical",
    "qwen_1p_anchored":        "anchored (22 items)",
    "qwen_1p_rep":             "replicate (22 items, numeric)",
    "qwen_1p_full_noregion_anchored":  "2x2: full_noregion + anchored",
    "qwen_1p_full_nocountry_anchored": "2x2: full_nocountry + anchored",
}


def load(kind: str, tag: str) -> pd.DataFrame | None:
    p = os.path.join(R, f"silicon_full_{kind}_{tag}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def rbc_by_item(scatter: pd.DataFrame) -> pd.Series:
    out = {}
    for v, g in scatter.groupby("variable"):
        g = g.dropna(subset=["survey_mean", "silicon_mean"])
        if len(g) >= 10 and g.survey_mean.std() > 0 and g.silicon_mean.std() > 0:
            out[v] = stats.pearsonr(g.survey_mean, g.silicon_mean)[0]
    return pd.Series(out, dtype=float)


def between_country_sd_ratio(scatter: pd.DataFrame) -> float:
    ratios = []
    for v, g in scatter.groupby("variable"):
        hs, ss = g.survey_mean.std(), g.silicon_mean.std()
        if hs and hs > 0:
            ratios.append(ss / hs)
    return float(np.median(ratios)) if ratios else np.nan


def within_country_sd_ratio(scatter: pd.DataFrame) -> float:
    r = (scatter.silicon_sd / scatter.survey_sd).replace([np.inf, -np.inf], np.nan)
    return float(r.mean(skipna=True))


# AUDIT FIX (P2). The two dispersion ratios reported side by side use DIFFERENT
# aggregators: within_country_sd_ratio is a MEAN over item x country cells,
# between_country_sd_ratio is a MEDIAN over items. Neither was recorded anywhere,
# and the thesis statement reports the pair in one sentence. Ratios are
# right-skewed, so for qwen_1p the same within-country claim can be stated as
# 0.605 (mean of ratios), 0.573 (median of ratios) or 0.586 (ratio of means).
# All estimators are now emitted so the manuscript can name the one it uses. The
# two original functions are left untouched, so no released figure moves.
def within_country_sd_ratio_median(scatter: pd.DataFrame) -> float:
    r = (scatter.silicon_sd / scatter.survey_sd).replace([np.inf, -np.inf], np.nan)
    return float(r.median(skipna=True))


def within_country_sd_ratio_of_means(scatter: pd.DataFrame) -> float:
    return float(scatter.silicon_sd.mean(skipna=True) /
                 scatter.survey_sd.mean(skipna=True))


def between_country_sd_ratio_mean(scatter: pd.DataFrame) -> float:
    ratios = []
    for v, g in scatter.groupby("variable"):
        hs, ss = g.survey_mean.std(), g.silicon_mean.std()
        if hs and hs > 0:
            ratios.append(ss / hs)
    return float(np.mean(ratios)) if ratios else np.nan


# ---------------------------------------------------------------------------
# 0. environment metadata
# ---------------------------------------------------------------------------
print("=" * 78)
print("0. ENVIRONMENT / METHODS METADATA")
print("=" * 78)
manifests = {}
for tag in TAGS:
    p = os.path.join(R, f"manifest_{tag}.json")
    if os.path.exists(p):
        with open(p) as f:
            manifests[tag] = json.load(f)
if manifests:
    m0 = next(iter(manifests.values()))
    print(f"  ESS file:        {m0.get('ess_file')}")
    print(f"  pandas / numpy:  {m0.get('pandas_version')} / {m0.get('numpy_version')}")
    print(f"  vLLM:            {m0.get('vllm_version')}")
    print(f"  missing_rule:    {set(m.get('missing_rule') for m in manifests.values())}")
    print(f"  human_weight:    {set(m.get('human_weight') for m in manifests.values())}")
    gen_seeds = {tag: m.get("gen_seed") for tag, m in manifests.items()}
    print(f"  gen_seed by condition (None = independent unseeded run):")
    for tag, gs in gen_seeds.items():
        print(f"    {tag:28s} {gs}")
    if all(gs is None for gs in gen_seeds.values()):
        print("  -> ALL conditions unseeded: main_qwen_1p and replicate are a genuine")
        print("     repeated-run pair (see section 8 below), not a simulated estimate.")
else:
    print("  WARNING: no manifest_*.json files found — cannot confirm environment.")

# ---------------------------------------------------------------------------
# 1. condition summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("1. CONDITION SUMMARY")
print("=" * 78)
scatters = {}
for tag in TAGS:
    sc = load("country_scatter", tag)
    if sc is not None:
        scatters[tag] = sc

summary_rows = []
hdr = f"{'condition':32s} {'median r':>9s} {'>.5':>4s} {'>.7':>4s} {'neg':>4s} {'wSD':>7s} {'bSD':>7s}"
print(hdr)
print("-" * len(hdr))
for tag, label in TAGS.items():
    if tag not in scatters:
        continue
    r = rbc_by_item(scatters[tag])
    wsd = within_country_sd_ratio(scatters[tag])
    bsd = between_country_sd_ratio(scatters[tag])
    wsd_med = within_country_sd_ratio_median(scatters[tag])
    wsd_rom = within_country_sd_ratio_of_means(scatters[tag])
    bsd_mean = between_country_sd_ratio_mean(scatters[tag])
    print(f"{label:32s} {r.median():+9.4f} {int((r>.5).sum()):4d} {int((r>.7).sum()):4d} "
          f"{int((r<0).sum()):4d} {wsd:7.4f} {bsd:7.4f}")
    summary_rows.append(dict(condition=tag, label=label, median_r_bc=r.median(),
                             n_gt50=int((r > .5).sum()), n_gt70=int((r > .7).sum()),
                             n_negative=int((r < 0).sum()), within_sd_ratio=wsd,
                             between_sd_ratio=bsd, n_items=len(r),
                             # P2: estimator made explicit in the column names.
                             # within_sd_ratio IS within_sd_ratio_mean and
                             # between_sd_ratio IS between_sd_ratio_median; the
                             # duplicates exist so downstream code need not change.
                             within_sd_ratio_mean=wsd,
                             within_sd_ratio_median=wsd_med,
                             within_sd_ratio_of_means=wsd_rom,
                             between_sd_ratio_median=bsd,
                             between_sd_ratio_mean=bsd_mean))
pd.DataFrame(summary_rows).to_csv(os.path.join(OUT, "condition_summary.csv"), index=False)

# ---------------------------------------------------------------------------
# 2. item-level direction decomposition (main condition)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("2. ITEM-LEVEL DIRECTION DECOMPOSITION — qwen_1p (full_clean)")
print("=" * 78)
if "qwen_1p" in scatters:
    r_main = rbc_by_item(scatters["qwen_1p"])
    is_rev = r_main.index.isin(REVERSE)
    fwd, rev = r_main[~is_rev], r_main[is_rev]
    print(f"  forward mean r_bc:  {fwd.mean():+.4f}  (n={len(fwd)})")
    print(f"  reverse mean r_bc:  {rev.mean():+.4f}  (n={len(rev)})")
    x = is_rev.astype(float)
    r2 = stats.pearsonr(x, r_main.values)[0] ** 2
    print(f"  R^2 (r_bc ~ reverse):  {r2:.4f}")
    aligned = np.where(is_rev, -r_main.values, r_main.values)
    print(f"  sign-aligned median:  {np.median(aligned):+.4f}")
    neg = r_main[r_main < 0].sort_values()
    print(f"\n  negative items ({len(neg)}), most severe first:")
    for v, val in neg.items():
        tag = "reverse" if v in REVERSE else "FORWARD (unexplained by direction)"
        print(f"    {v:10s} {val:+.4f}  [{tag}]")
    r_main.to_frame("r_bc").assign(reverse=is_rev).to_csv(
        os.path.join(OUT, "qwen_1p_item_r_bc.csv"))
else:
    print("  qwen_1p country_scatter file not found — skipping.")

# ---------------------------------------------------------------------------
# 3. PRIMARY CONTRAST: country-label effect
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("3. PRIMARY CONTRAST — country label effect (full_noregion vs full_nocountry)")
print("=" * 78)
if "qwen_1p_full_noregion" in scatters and "qwen_1p_full_nocountry" in scatters:
    r_with = rbc_by_item(scatters["qwen_1p_full_noregion"])
    r_without = rbc_by_item(scatters["qwen_1p_full_nocountry"])
    common = sorted(set(r_with.index) & set(r_without.index))
    delta = (r_with[common] - r_without[common]).sort_values(ascending=False)
    print(f"  n items compared: {len(common)}")
    print(f"  median country-label effect (r_with - r_without): {delta.median():+.4f}")
    print(f"  mean:  {delta.mean():+.4f}   items where delta > 0: {(delta>0).sum()}/{len(delta)}")
    print(f"\n  top 5 items most helped by the country label:")
    for v, d in delta.head(5).items():
        print(f"    {v:10s} with={r_with[v]:+.3f}  without={r_without[v]:+.3f}  delta={d:+.3f}")
    print(f"\n  bottom 5 (country label helps least / hurts):")
    for v, d in delta.tail(5).items():
        print(f"    {v:10s} with={r_with[v]:+.3f}  without={r_without[v]:+.3f}  delta={d:+.3f}")
    pd.DataFrame({"r_with_country": r_with[common], "r_without_country": r_without[common],
                 "delta": delta[common]}).to_csv(
        os.path.join(OUT, "primary_contrast_country_effect.csv"))
    print("\n  NOTE: both arms share seed 888 and n=685, drawing the SAME respondents,")
    print("  so delta reflects only the silicon side (paired design).")
else:
    print("  contrast files not found — skipping.")

# ---------------------------------------------------------------------------
# 4. ladder + between-country dispersion (revision plan 3.5)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("4. CUMULATIVE LADDER (revision plan Figure 4 / section 3.5)")
print("=" * 78)
ladder_tags = [("qwen_1p_demo_only", "demo_only", 3),
              ("qwen_1p_minimal", "minimal", 4),
              ("qwen_1p_ses", "ses", 13),
              ("qwen_1p_political", "political", 14),
              ("qwen_1p", "full_clean", 20)]
ladder_rows = []
print(f"  {'rung':12s} {'k':>3s} {'median r':>9s} {'between-SD ratio':>17s}")
for tag, name, k in ladder_tags:
    if tag in scatters:
        r = rbc_by_item(scatters[tag])
        bsd = between_country_sd_ratio(scatters[tag])
        print(f"  {name:12s} {k:3d} {r.median():+9.4f} {bsd:17.4f}")
        ladder_rows.append(dict(rung=name, k=k, median_r_bc=r.median(), between_sd_ratio=bsd))
pd.DataFrame(ladder_rows).to_csv(os.path.join(OUT, "ladder_summary.csv"), index=False)

# ---------------------------------------------------------------------------
# 5. LOGO — political-identity leave-one-out
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("5. LOGO — full_nopolitical vs full_clean")
print("=" * 78)
if "qwen_1p_full_nopolitical" in scatters and "qwen_1p" in scatters:
    r_no = rbc_by_item(scatters["qwen_1p_full_nopolitical"])
    r_full = rbc_by_item(scatters["qwen_1p"])
    common = sorted(set(r_no.index) & set(r_full.index))
    d = (r_full[common] - r_no[common])
    print(f"  median |delta| removing political-identity group: {d.abs().median():.4f}")
    print(f"  mean delta (full - nopolitical): {d.mean():+.4f}")
    pd.DataFrame({"r_full_clean": r_full[common], "r_nopolitical": r_no[common],
                 "delta": d}).to_csv(os.path.join(OUT, "logo_political.csv"))
else:
    print("  files not found — skipping.")

# ---------------------------------------------------------------------------
# 6. cross-level agreement (revision plan 3.3)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("6. CROSS-LEVEL AGREEMENT — qwen_1p (revision plan 3.3)")
print("=" * 78)
rq3_main = load("rq3", "qwen_1p")
if rq3_main is not None and "qwen_1p" in scatters:
    rows = []
    for v, g in rq3_main.groupby("variable"):
        n = g.n_valid.values.astype(float)
        hm, sm = g.human_mean.values, g.silicon_mean.values
        hs, ss = g.human_sd.values, g.silicon_sd.values
        r = g.r_pearson.values
        ok = np.isfinite(n) & np.isfinite(hm) & np.isfinite(sm) & np.isfinite(hs) & \
             np.isfinite(ss) & np.isfinite(r)
        n, hm, sm, hs, ss, r = [a[ok] for a in (n, hm, sm, hs, ss, r)]
        if n.sum() < 30:
            continue
        N = n.sum()
        Hm, Sm = (n * hm).sum() / N, (n * sm).sum() / N
        cov = ((n - 1) * r * hs * ss).sum() / (N - 1) + (n * (hm - Hm) * (sm - Sm)).sum() / (N - 1)
        vh = ((n - 1) * hs ** 2).sum() / (N - 1) + (n * (hm - Hm) ** 2).sum() / (N - 1)
        vs = ((n - 1) * ss ** 2).sum() / (N - 1) + (n * (sm - Sm) ** 2).sum() / (N - 1)
        if vh > 0 and vs > 0:
            rows.append((v, cov / np.sqrt(vh * vs)))
    r_ind = pd.Series(dict(rows))

    # v13.1: r_ind is a POOLED individual correlation, so its covariance contains
    # a between-country term -- and that term is r_bc. Correlating r_bc against
    # r_ind therefore partly correlates r_bc with itself. On this data the
    # between-country term is a median 36% of r_ind's total covariance (IQR
    # 26-57%), so the shared component is substantial rather than incidental.
    # The component-free analogue is the POOLED WITHIN-COUNTRY correlation, which
    # by construction has no between-country contribution. Both are reported;
    # the component-free version is the one that supports a claim of
    # semi-independent cross-level agreement.
    pw_rows, share_rows = [], []
    for v, g in rq3_main.groupby("variable"):
        g = g.dropna(subset=["n_valid", "human_mean", "silicon_mean",
                             "human_sd", "silicon_sd", "r_pearson"])
        if len(g) < 3:
            continue
        n = g.n_valid.values.astype(float); N = n.sum()
        hm, sm = g.human_mean.values, g.silicon_mean.values
        hs, ss = g.human_sd.values, g.silicon_sd.values
        r = g.r_pearson.values
        Hm, Sm = (n * hm).sum() / N, (n * sm).sum() / N
        cov_w = ((n - 1) * r * hs * ss).sum() / (N - 1)
        cov_b = (n * (hm - Hm) * (sm - Sm)).sum() / (N - 1)
        vh_w = ((n - 1) * hs ** 2).sum() / (N - 1)
        vs_w = ((n - 1) * ss ** 2).sum() / (N - 1)
        if vh_w > 0 and vs_w > 0:
            pw_rows.append((v, cov_w / np.sqrt(vh_w * vs_w)))
            if (cov_w + cov_b) != 0:
                share_rows.append(cov_b / (cov_w + cov_b))
    r_pw = pd.Series(dict(pw_rows))
    r_bc = rbc_by_item(scatters["qwen_1p"])
    common = [v for v in r_bc.index if v in r_ind.index]
    is_rev = pd.Series({v: v in REVERSE for v in common})
    a_full, b_full = r_bc[common], r_ind[common]
    fwd_only = [v for v in common if v not in REVERSE]
    aligned_bc = np.where(is_rev[common], -a_full, a_full)
    aligned_ind = np.where(is_rev[common], -b_full, b_full)
    print(f"  {'specification':28s} {'r_ind':>9s} {'pooled-within':>14s}")
    print("  " + "-" * 53)
    cp = [v for v in common if v in r_pw.index]
    cpf = [v for v in fwd_only if v in r_pw.index]
    sgn = np.array([-1.0 if v in REVERSE else 1.0 for v in cp])
    rows_cl = [
        ("as-coded", stats.pearsonr(a_full, b_full)[0],
         stats.pearsonr(r_bc[cp], r_pw[cp])[0]),
        ("forward-coded only", stats.pearsonr(r_bc[fwd_only], r_ind[fwd_only])[0],
         stats.pearsonr(r_bc[cpf], r_pw[cpf])[0]),
        ("sign-aligned", stats.pearsonr(aligned_bc, aligned_ind)[0],
         stats.pearsonr(r_bc[cp] * sgn, r_pw[cp] * sgn)[0]),
    ]
    for lab, v1, v2 in rows_cl:
        print(f"  {lab:28s} {v1:+9.4f} {v2:+14.4f}")
    if share_rows:
        sr = np.array(share_rows)
        print(f"\n  between-country share of r_ind's covariance: median "
              f"{np.median(sr):.1%}  IQR [{np.percentile(sr,25):.1%}, "
              f"{np.percentile(sr,75):.1%}]")
    print("""
  The r_ind column shares a component with r_bc and is therefore inflated. Lead
  with the pooled-within column. Its forward-coded-only value is the most
  conservative specification available -- it removes both the direction confound
  and the shared component -- and it is roughly half the as-coded headline. The
  agreement is real and significant at that level, but it should be described as
  moderate rather than strong.""")
    pd.DataFrame({"r_bc": a_full, "r_ind": b_full}).join(
        r_pw.rename("r_pooled_within")).to_csv(
        os.path.join(OUT, "cross_level_agreement.csv"))
    pd.DataFrame(rows_cl, columns=["specification", "vs_r_ind",
                                   "vs_pooled_within"]).to_csv(
        os.path.join(OUT, "cross_level_specifications.csv"), index=False)
else:
    print("  rq3 or scatter file for qwen_1p not found — skipping.")

# ---------------------------------------------------------------------------
# 7. reliability / attenuation at n=685
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("7. RELIABILITY / ATTENUATION at n=685")
print("=" * 78)
# AUDIT FIX (E2). This section previously printed only the detectable
# correlation and the observed mean, although the module docstring and this
# section's own header promise reliability and attenuation recomputed at n=685.
# No reliability, no attenuation factor and no CSV were produced, which is why
# the released package contains no such table and the revision plan's figures
# (0.955 at n=500, 0.909 at n=250, silicon reliability 0.924) still describe the
# March round rather than any arm of this one.
#
# The section also reproduced a logical error in revision plan 3.2. Printing a
# within-one-country detectable r of about 0.107 next to an observed 0.027
# supports "underpowered", not "genuine near-zero": the observation lies BELOW
# the detection threshold. The argument only closes on the POOLED within-country
# estimator, where the detectable r is about 0.020 and the observed r_pw of 0.043
# exceeds it. Both levels are now reported and the distinction is stated.
if rq3_main is not None and "qwen_1p" in scatters:
    from scipy.stats import norm

    sc7 = scatters["qwen_1p"]
    npc = manifests.get("qwen_1p", {}).get("sample_per_country")
    n_med = float(rq3_main.n_valid.median())
    n_pooled = float(rq3_main.groupby("variable").n_valid.sum().median())
    zc = norm.ppf(0.975) + norm.ppf(0.80)

    def _detect(nn):
        return float(np.tanh(zc / np.sqrt(nn - 3))) if nn and nn > 4 else np.nan

    # Country-mean reliability. The observed variance of the thirty country means
    # is true between-country variance plus sampling error; subtracting the mean
    # sampling variance of a country mean leaves the true part. Two biases run in
    # opposite directions and are stated rather than adjusted for: n_valid is the
    # human-silicon intersection, which is at or below the full-valid n and so
    # inflates the sampling term; and the human means are design-weighted, whose
    # design effect exceeds one and so deflates it.
    nv = rq3_main.set_index(["variable", "cntry"]).n_valid
    rel_rows = []
    for v, g in sc7.groupby("variable"):
        g = g.dropna(subset=["survey_mean", "silicon_mean", "survey_sd", "silicon_sd"])
        if len(g) < 10:
            continue
        n_i = nv.reindex([(v, c) for c in g.cntry]).to_numpy(float)
        if not np.isfinite(n_i).all():
            continue
        rec = dict(variable=v, n_countries=len(g), n_median=float(np.median(n_i)))
        for side, mcol, scol in (("human", "survey_mean", "survey_sd"),
                                 ("silicon", "silicon_mean", "silicon_sd")):
            S2 = float(g[mcol].var(ddof=1))
            E = float(np.mean(g[scol].to_numpy(float) ** 2 / n_i))
            rec[f"{side}_var_between"] = S2
            rec[f"{side}_var_sampling"] = E
            rec[f"{side}_reliability"] = (max(0.0, S2 - E) / S2) if S2 > 0 else np.nan
        rec["attenuation_factor"] = float(np.sqrt(rec["human_reliability"] *
                                                  rec["silicon_reliability"]))
        rel_rows.append(rec)
    rel = pd.DataFrame(rel_rows).set_index("variable")
    rel.to_csv(os.path.join(OUT, "reliability_attenuation_n685.csv"))

    print(f"  respondents per country (manifest):        {npc}")
    print(f"  median n_valid per item-country cell:      {n_med:.0f}")
    print(f"  median pooled N per item (30 countries):   {n_pooled:.0f}")
    print()
    print(f"  median country-mean reliability, human:    "
          f"{rel.human_reliability.median():.4f}")
    print(f"  median country-mean reliability, silicon:  "
          f"{rel.silicon_reliability.median():.4f}")
    print(f"  median attenuation factor on r_bc:         "
          f"{rel.attenuation_factor.median():.4f}  "
          f"({100 * (1 - rel.attenuation_factor.median()):.1f}% attenuation)")
    print(f"  items with silicon-side reliability < .50: "
          f"{int((rel.silicon_reliability < .5).sum())}  "
          f"{sorted(rel.index[rel.silicon_reliability < .5])}")
    print(f"  items with human-side reliability < .50:   "
          f"{int((rel.human_reliability < .5).sum())}")
    print()
    print(f"  r detectable at 80% power, WITHIN one country (n={n_med:.0f}): "
          f"{_detect(n_med):.4f}")
    print(f"  r detectable at 80% power, POOLED within-country (N={n_pooled:.0f}): "
          f"{_detect(n_pooled):.4f}")
    print(f"  observed mean r_wc (per country-item cell):  "
          f"{rq3_main.r_pearson.mean():+.4f}  (SD {rq3_main.r_pearson.std():.4f})")
    try:
        _pw_med = float(r_pw.median())
        print(f"  observed median r_pw (pooled within-country): {_pw_med:+.4f}")
        _verdict = ("ABOVE the pooled threshold: the near-zero is estimated "
                    "precisely, not merely undetected"
                    if _pw_med > _detect(n_pooled) else
                    "BELOW the pooled threshold: the null is underpowered")
        print(f"  -> {_verdict}")
    except NameError:
        _pw_med = np.nan
        print("  r_pw unavailable (section 6 did not run); pooled comparison skipped.")
    print()
    print("  REPORTING RULE. The within-country figure alone cannot support a")
    print("  'genuine near-zero' claim, because the observed value lies below its")
    print("  own detection threshold. Quote the pooled comparison for that claim,")
    print("  and quote the within-country threshold only to say that no single")
    print("  country's estimate is individually informative. Note also that a")
    print("  larger sample cannot rescue the aggregate conclusions: the median")
    print("  attenuation on r_bc is only a few per cent.")

    pd.DataFrame([
        dict(quantity="respondents per country (all arms)", value=npc),
        dict(quantity="median country-mean reliability, human side",
             value=rel.human_reliability.median()),
        dict(quantity="median country-mean reliability, silicon side",
             value=rel.silicon_reliability.median()),
        dict(quantity="median attenuation factor on r_bc",
             value=rel.attenuation_factor.median()),
        dict(quantity="items with silicon-side reliability below 0.50",
             value=int((rel.silicon_reliability < .5).sum())),
        dict(quantity="items with human-side reliability below 0.50",
             value=int((rel.human_reliability < .5).sum())),
        dict(quantity="detectable r, 80% power, within one country",
             value=_detect(n_med)),
        dict(quantity="detectable r, 80% power, pooled within-country",
             value=_detect(n_pooled)),
        dict(quantity="observed mean r_wc per cell", value=rq3_main.r_pearson.mean()),
        dict(quantity="observed median r_pw", value=_pw_med),
    ]).to_csv(os.path.join(OUT, "reliability_attenuation_summary.csv"), index=False)
else:
    print("  rq3 or scatter file for qwen_1p not found — skipping.")

# ---------------------------------------------------------------------------
# 8. anchored DiD with REAL repeated-run noise floor
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("8. ANCHORED SCALE EXPERIMENT — using REAL repeated-run noise")
print("=" * 78)
if all(t in scatters for t in ("qwen_1p", "qwen_1p_rep", "qwen_1p_anchored")):
    anchor_items = ANCHOR_REVERSE | ANCHOR_CTRL | ANCHOR_PLACEBO
    rA = rbc_by_item(scatters["qwen_1p"])           # numeric, run A (main_qwen_1p, 42-item scatter)
    rA = rA[rA.index.isin(anchor_items)]
    rB = rbc_by_item(scatters["qwen_1p_rep"])       # numeric, run B (independent, same 22 items)
    rT = rbc_by_item(scatters["qwen_1p_anchored"])  # anchored (treatment)
    common = sorted(set(rA.index) & set(rB.index) & set(rT.index))
    noise = (rA[common] - rB[common]).abs()
    signal = (rT[common] - rA[common])
    print(f"  n items with all three arms present: {len(common)}")
    print(f"  {'item':10s} {'grp':9s} {'r_A(num)':>9s} {'r_B(num)':>9s} {'r_T(anch)':>9s} "
          f"{'noise':>7s} {'signal':>8s}")
    for v in sorted(common, key=lambda v: (v in ANCHOR_PLACEBO, v in ANCHOR_CTRL, v)):
        grp = ("reverse" if v in ANCHOR_REVERSE else
              "ctrl" if v in ANCHOR_CTRL else "placebo")
        flag = "  <-- excluded (noise-dominated)" if v in UNINTERPRETABLE else \
               ("  <-- |signal|<noise" if abs(signal[v]) < noise[v] else "")
        print(f"  {v:10s} {grp:9s} {rA[v]:+9.3f} {rB[v]:+9.3f} {rT[v]:+9.3f} "
              f"{noise[v]:7.3f} {signal[v]:+8.3f}{flag}")

    usable_rev = [v for v in common if v in ANCHOR_REVERSE and v not in UNINTERPRETABLE]
    ctrl_items = [v for v in common if v in ANCHOR_CTRL]
    did = signal[usable_rev].mean() - signal[ctrl_items].mean()
    rng = np.random.default_rng(888)
    boots = []
    for _ in range(5000):
        rs = rng.choice(usable_rev, size=len(usable_rev), replace=True)
        cs = rng.choice(ctrl_items, size=len(ctrl_items), replace=True)
        boots.append(signal[rs].mean() - signal[cs].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    n_up = int((signal[usable_rev] > 0).sum())
    print(f"\n  PRE-REGISTERED PRIMARY RESULT (revision plan section 4):")
    print(f"    DiD (reverse minus forward-control mean signal): {did:+.4f}")
    print(f"    95% bootstrap CI: [{lo:+.4f}, {hi:+.4f}]  {'excludes 0' if lo>0 or hi<0 else 'INCLUDES 0'}")
    print(f"    reverse items moving upward: {n_up}/{len(usable_rev)}  "
          f"(success criterion: >= 8/{len(usable_rev)})")
    verdict = (lo > 0 or hi < 0) and n_up >= 8
    print(f"    VERDICT: {'SUCCESS — report as primary experimental result' if verdict else 'MIXED — report observational decomposition (section 3.1) as primary, this as secondary'}")
    pd.DataFrame({"r_A_numeric": rA[common], "r_B_numeric_repeat": rB[common],
                 "r_T_anchored": rT[common], "empirical_noise": noise,
                 "signal": signal}).to_csv(os.path.join(OUT, "anchored_did_real_noise.csv"))
else:
    print("  qwen_1p / qwen_1p_rep / qwen_1p_anchored scatter files not all found — skipping.")

print("\n" + "=" * 78)
print(f"All tables written to {os.path.abspath(OUT)}/")
print("=" * 78)
