#!/usr/bin/env python3
"""
silicon_v13.py -- v13-native metric library and verification suite for the
cross-national silicon sampling chapter.

This module supersedes the March round's ``silicon_lib.py`` for the v13 data.
Three things changed and each is reflected below.

1.  Naming. The March library exposed ``pooled_individual_r`` (which is r_ind)
    and ``pooled_country_r`` (which is r_pool). Both were called "pooled" and
    they are the two most confusable measures in the project. The functions
    here are named for the symbols used in the manuscript: ``r_bc``, ``r_wc``,
    ``r_pw``, ``r_ind``, ``r_pool``, ``r_prof``.

2.  Level of aggregation. Every country-level quantity is computed from the
    full-valid country means in the ``*_country_scatter_*`` files, never from
    the human-silicon intersection means stored in the ``*_rq3_*`` files.
    Using the latter shifts the median r_bc of the main condition from 0.4433
    to 0.4425 and moves individual items by as much as 0.109.

3.  Condition inventory. Fifteen conditions, including the three anchored arms
    and the repeated-run replicate, none of which exist in the March library.

Two estimator conventions in the released summaries are made explicit here
because they are not symmetric and are not documented elsewhere:
``within_sd_ratio`` is a MEAN over item x country cells, whereas
``between_sd_ratio`` is a MEDIAN over items.

Run ``python silicon_v13.py --data DIR`` to execute the verification suite.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Condition inventory
# --------------------------------------------------------------------------

MAIN_CONDITIONS = {
    "qwen_1p": "Qwen 1P", "qwen_3p": "Qwen 3P",
    "llama_1p": "Llama 1P", "llama_3p": "Llama 3P",
}

# Cumulative backstory ladder. The top rung is the main condition; the labels
# are the corrected ones (the manuscript's Table 1 category names are wrong,
# although its counts are right).
LADDER = {
    "qwen_1p_demo_only": ("demo_only", 3),
    "qwen_1p_minimal": ("minimal", 4),
    "qwen_1p_ses": ("ses", 13),
    "qwen_1p_political": ("political", 14),
    "qwen_1p": ("full_clean", 20),
}

# Clean leave-one-out contrasts. Each pair differs in exactly one prompt
# element. Both country arms also drop the NUTS region code, because that code
# begins with the country's ISO characters.
CONTRASTS = {
    "country label": ("qwen_1p_full_noregion", "qwen_1p_full_nocountry"),
    "NUTS region code": ("qwen_1p", "qwen_1p_full_noregion"),
    "political identity": ("qwen_1p", "qwen_1p_full_nopolitical"),
}

# The 2 x 2: country label present or absent, crossed with numeric or verbally
# anchored response scales, on the 22-item batch.
TWOXTWO = {
    ("numeric", "with"): "qwen_1p_full_noregion",
    ("numeric", "without"): "qwen_1p_full_nocountry",
    ("anchored", "with"): "qwen_1p_full_noregion_anchored",
    ("anchored", "without"): "qwen_1p_full_nocountry_anchored",
}

# Anchoring experiment. Baseline A is the 42-item batch, baseline B the paired
# same-batch replicate; the primary specification uses B.
ANCHORED = {"target": "qwen_1p_anchored", "baseline_A": "qwen_1p", "baseline_B": "qwen_1p_rep"}

# Pre-declared as uninterpretable on the round in which the declaration was
# made, on the grounds of near-zero silicon between-country dispersion. Note
# that in v13 the forward item ``inprdsc`` has a lower silicon between-country
# SD (0.0313) than ``rlgatnd`` (0.0316); the screen was applied to the reverse
# arm only. Dropping inprdsc from the forward controls moves the primary
# triple difference by -0.0004, so the asymmetry is immaterial.
PREDECLARED_EXCLUSIONS = ("hmsfmlsh", "rlgatnd")
UNINTERPRETABLE = ("hmsfmlsh", "rlgatnd", "vote")

DOMAIN_LABELS = {
    "social_trust": "Social trust", "inst_trust": "Institutional trust",
    "political": "Political attitudes", "efficacy": "Political efficacy",
    "wellbeing": "Subjective well-being", "immigration": "Immigration attitudes",
    "values": "Social values", "attachment": "National attachment",
    "health": "Health", "religion": "Religion", "safety": "Safety",
    "income": "Income",
}

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def conditions(data_dir: str) -> list[str]:
    """Every condition tag present in the bundle, from the run manifests."""
    return sorted(os.path.basename(p)[len("manifest_"):-len(".json")]
                  for p in glob.glob(os.path.join(data_dir, "manifest_*.json")))


def load_manifest(tag: str, data_dir: str) -> dict:
    with open(os.path.join(data_dir, f"manifest_{tag}.json")) as fh:
        return json.load(fh)


def load_scatter(tag: str, data_dir: str) -> pd.DataFrame:
    """Full-valid country means and SDs. The input to every aggregate quantity.

    Human means are design-weighted by ``pspwght``; silicon means are
    unweighted. The two sides are therefore not the same estimand, which is a
    limitation to state rather than a defect to correct here.
    """
    df = pd.read_csv(os.path.join(data_dir, f"silicon_full_country_scatter_{tag}.csv"))
    df = df.rename(columns={"survey_mean": "human_mean", "survey_sd": "human_sd"})
    for col in ("human_mean", "human_sd", "silicon_mean", "silicon_sd"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_rq3(tag: str, data_dir: str) -> pd.DataFrame:
    """Per item x country individual-level correlations and paired moments.

    The means stored here are computed on the human-silicon intersection and
    are unweighted. Do not use them for country-level results.
    """
    df = pd.read_csv(os.path.join(data_dir, f"silicon_full_rq3_{tag}.csv"))
    for col in ("r_pearson", "p_pearson", "human_mean", "silicon_mean",
                "human_sd", "silicon_sd", "n_valid"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_direction(data_dir: str) -> pd.DataFrame:
    """Frozen direction classification, all 42 items, 13 reverse-coded."""
    df = pd.read_csv(os.path.join(data_dir, "item_direction_table.csv")).set_index("variable")
    df["reverse"] = df["direction"].eq("reverse")
    df["range"] = df["high_value"] - df["low_value"]
    return df


# --------------------------------------------------------------------------
# Metrics. One function per symbol in the manuscript's metrics table.
# --------------------------------------------------------------------------


def r_bc(tag: str, data_dir: str) -> pd.Series:
    """PRIMARY AGGREGATE METRIC. Per item, the correlation of the 30 human
    country means against the 30 silicon country means."""
    d = load_scatter(tag, data_dir)
    return d.groupby("variable").apply(
        lambda g: stats.pearsonr(g.human_mean, g.silicon_mean)[0], include_groups=False)


def r_wc(tag: str, data_dir: str) -> pd.DataFrame:
    """Per item x country, the paired individual-level correlation."""
    return load_rq3(tag, data_dir).pivot(index="variable", columns="cntry", values="r_pearson")


def _total_covariance_parts(d: pd.DataFrame) -> dict:
    """Within- and between-country sums of squares and cross-products,
    recovered from the per-cell moments by the law of total covariance."""
    d = d.dropna(subset=["r_pearson", "human_sd", "silicon_sd",
                         "human_mean", "silicon_mean", "n_valid"])
    n = d["n_valid"].to_numpy(float)
    N = n.sum()
    if N < 3 or len(d) == 0:
        return {}
    mu_h = (n * d["human_mean"]).sum() / N
    mu_s = (n * d["silicon_mean"]).sum() / N
    return dict(
        N=N,
        w_h=((n - 1) * d["human_sd"] ** 2).sum(),
        w_s=((n - 1) * d["silicon_sd"] ** 2).sum(),
        w_hs=((n - 1) * d["r_pearson"] * d["human_sd"] * d["silicon_sd"]).sum(),
        b_h=(n * (d["human_mean"] - mu_h) ** 2).sum(),
        b_s=(n * (d["silicon_mean"] - mu_s) ** 2).sum(),
        b_hs=(n * (d["human_mean"] - mu_h) * (d["silicon_mean"] - mu_s)).sum(),
    )


def r_pw(tag: str, data_dir: str) -> pd.Series:
    """PRIMARY INDIVIDUAL METRIC. Per item, the pooled WITHIN-country
    correlation: covariances and variances pooled over countries after
    within-country demeaning. Component-free with respect to r_bc, and the
    correct numerator for the demographic-ceiling ratio."""
    out = {}
    for var, d in load_rq3(tag, data_dir).groupby("variable"):
        p = _total_covariance_parts(d)
        out[var] = (p["w_hs"] / np.sqrt(p["w_h"] * p["w_s"])) if p else np.nan
    return pd.Series(out)


def r_ind(tag: str, data_dir: str) -> pd.Series:
    """APPENDIX ONLY. Per item, the individual-level correlation with all
    countries stacked. Its covariance contains a between-country term that IS
    r_bc (median 36 per cent), so it must not be set against r_bc."""
    out = {}
    for var, d in load_rq3(tag, data_dir).groupby("variable"):
        p = _total_covariance_parts(d)
        out[var] = ((p["w_hs"] + p["b_hs"]) /
                    np.sqrt((p["w_h"] + p["b_h"]) * (p["w_s"] + p["b_s"]))) if p else np.nan
    return pd.Series(out)


def between_country_share_of_r_ind(tag: str, data_dir: str) -> pd.Series:
    """Diagnostic: the share of r_ind's covariance that is between-country."""
    out = {}
    for var, d in load_rq3(tag, data_dir).groupby("variable"):
        p = _total_covariance_parts(d)
        out[var] = p["b_hs"] / (p["w_hs"] + p["b_hs"]) if p else np.nan
    return pd.Series(out)


def r_pool(tag: str, data_dir: str) -> float:
    """APPENDIX ONLY, AS A COUNTEREXAMPLE. All item x country cells at once.
    Most of its covariance is between-item scale variation, not
    between-country signal. Never present as evidence of recovery."""
    d = load_scatter(tag, data_dir)
    return float(stats.pearsonr(d.human_mean, d.silicon_mean)[0])


def r_prof(tag: str, data_dir: str) -> pd.Series:
    """Descriptive. Per country, the correlation across item means."""
    d = load_scatter(tag, data_dir)
    return d.groupby("cntry").apply(
        lambda g: stats.pearsonr(g.human_mean, g.silicon_mean)[0], include_groups=False)


def normalised_mae(tag: str, data_dir: str, by: str = "cntry") -> pd.Series:
    """Mean over the other axis of |human - silicon| divided by the item range."""
    d = load_scatter(tag, data_dir)
    rng = load_direction(data_dir)["range"]
    d = d.assign(nae=(d.silicon_mean - d.human_mean).abs() / d.variable.map(rng))
    return d.groupby(by).nae.mean()


def within_sd_ratio(tag: str, data_dir: str) -> float:
    """MEAN over item x country cells of the full-valid SD ratio. This is the
    estimator behind the released 0.605; the median is 0.573 and the ratio of
    means is 0.587, so the estimator must be named wherever it is reported."""
    d = load_scatter(tag, data_dir)
    return float((d.silicon_sd / d.human_sd).replace([np.inf, -np.inf], np.nan).mean())


def between_sd_ratio(tag: str, data_dir: str) -> float:
    """MEDIAN over items of SD(silicon country means) / SD(human country
    means). Note the asymmetry with within_sd_ratio, which is a mean."""
    d = load_scatter(tag, data_dir)
    g = d.groupby("variable")
    return float((g.silicon_mean.std(ddof=1) / g.human_mean.std(ddof=1)).median())


def fisher_z_difference(tag_with: str, tag_without: str, data_dir: str) -> pd.Series:
    """Per item, z(r_bc of the first arm) - z(r_bc of the second arm).

    The transform is applied to each CORRELATION and the transforms are then
    differenced. Applying arctanh to a difference of correlations is not a
    Fisher transformation of anything and is undefined whenever the difference
    exceeds unity in absolute value.
    """
    a, b = r_bc(tag_with, data_dir), r_bc(tag_without, data_dir)
    idx = a.index.intersection(b.index)
    z = lambda s: np.arctanh(np.clip(s.reindex(idx).to_numpy(float), -0.999999, 0.999999))
    return pd.Series(z(a) - z(b), index=idx)


# --------------------------------------------------------------------------
# Verification suite
# --------------------------------------------------------------------------


def verify(data_dir: str) -> pd.DataFrame:
    """Recompute every quantity in the bundle that is recomputable from the
    bundle, and compare against the released value."""
    rows = []

    def chk(section, name, rep, ref, tol=5e-4):
        passed = pd.notna(rep) and pd.notna(ref) and abs(float(rep) - float(ref)) < tol
        rows.append(dict(section=section, check=name, recomputed=rep, released=ref,
                         diff=(float(rep) - float(ref)) if passed or (pd.notna(rep) and pd.notna(ref)) else None,
                         status="pass" if passed else "FAIL"))

    tags = conditions(data_dir)
    direction = load_direction(data_dir)
    rev = direction["reverse"]

    # ---- structural integrity
    for tag in tags:
        man = load_manifest(tag, data_dir)
        k, npc = man["variables_n"], man["sample_per_country"]
        rq3, sca = load_rq3(tag, data_dir), load_scatter(tag, data_dir)
        chk("structure", f"{tag}: rq3 rows", len(rq3), k * 30, 0.5)
        chk("structure", f"{tag}: scatter rows", len(sca), k * 30, 0.5)
        chk("structure", f"{tag}: duplicate item x country cells",
            rq3.duplicated(["variable", "cntry"]).sum() + sca.duplicated(["variable", "cntry"]).sum(), 0, 0.5)
        chk("structure", f"{tag}: n_valid never exceeds n per country",
            int(rq3.n_valid.max() > npc), 0, 0.5)
        oob = 0
        for var, g in sca.groupby("variable"):
            lo, hi = direction.loc[var, "low_value"], direction.loc[var, "high_value"]
            oob += int(((g[["human_mean", "silicon_mean"]] < lo - 1e-9) |
                        (g[["human_mean", "silicon_mean"]] > hi + 1e-9)).to_numpy().sum())
        chk("structure", f"{tag}: country means outside the item range", oob, 0, 0.5)
        chk("structure", f"{tag}: undefined individual-level r (zero-variance cells)",
            rq3.r_pearson.isna().sum(), 0, 0.5)

    # ---- condition_summary
    cs = pd.read_csv(os.path.join(data_dir, "condition_summary.csv")).set_index("condition")
    for tag in tags:
        r = r_bc(tag, data_dir)
        chk("condition_summary", f"{tag}: median r_bc", r.median(), cs.loc[tag, "median_r_bc"])
        chk("condition_summary", f"{tag}: items r_bc > 0.50", (r > 0.5).sum(), cs.loc[tag, "n_gt50"], 0.5)
        chk("condition_summary", f"{tag}: items r_bc > 0.70", (r > 0.7).sum(), cs.loc[tag, "n_gt70"], 0.5)
        chk("condition_summary", f"{tag}: items r_bc < 0", (r < 0).sum(), cs.loc[tag, "n_negative"], 0.5)
        chk("condition_summary", f"{tag}: within-country SD ratio (mean)",
            within_sd_ratio(tag, data_dir), cs.loc[tag, "within_sd_ratio"])
        chk("condition_summary", f"{tag}: between-country SD ratio (median)",
            between_sd_ratio(tag, data_dir), cs.loc[tag, "between_sd_ratio"])

    # ---- direction decomposition
    itab = (pd.read_csv(os.path.join(data_dir, "qwen_1p_item_r_bc.csv"))
            .rename(columns={"Unnamed: 0": "variable"}).set_index("variable"))
    r = r_bc("qwen_1p", data_dir).reindex(itab.index)
    chk("direction", "item table r_bc equals recomputed (max abs deviation)",
        (itab.r_bc - r).abs().max(), 0, 1e-9)
    chk("direction", "reverse flags agree with the frozen direction table",
        (itab.reverse != rev.reindex(itab.index)).sum(), 0, 0.5)
    chk("direction", "forward mean r_bc", itab.loc[~itab.reverse, "r_bc"].mean(), 0.5509)
    chk("direction", "reverse mean r_bc", itab.loc[itab.reverse, "r_bc"].mean(), -0.2979)
    chk("direction", "R^2 of r_bc on direction",
        stats.pearsonr(itab.reverse.astype(float), itab.r_bc)[0] ** 2, 0.6576)
    chk("direction", "sign-aligned median r_bc",
        (itab.r_bc * np.where(itab.reverse, -1, 1)).median(), 0.5353)
    sub = itab.drop(index=list(UNINTERPRETABLE))
    chk("direction", "R^2 excluding the three uninterpretable items",
        stats.pearsonr(sub.reverse.astype(float), sub.r_bc)[0] ** 2, 0.7455)

    # ---- pooled inflation
    pvi = pd.read_csv(os.path.join(data_dir, "pooled_vs_item.csv")).set_index("condition")
    for tag in pvi.index:
        chk("pooled inflation", f"{tag}: r_pool", r_pool(tag, data_dir), pvi.loc[tag, "r_pooled"])

    # ---- individual-level measures reconstructed from the per-cell moments
    cla = (pd.read_csv(os.path.join(data_dir, "cross_level_agreement.csv"))
           .rename(columns={"Unnamed: 0": "variable"}).set_index("variable"))
    chk("individual", "r_pw reconstructed from rq3 moments (max abs deviation)",
        (cla.r_pooled_within - r_pw("qwen_1p", data_dir).reindex(cla.index)).abs().max(), 0, 1e-9)
    chk("individual", "r_ind reconstructed from rq3 moments (max abs deviation)",
        (cla.r_ind - r_ind("qwen_1p", data_dir).reindex(cla.index)).abs().max(), 0, 1e-9)
    cls = pd.read_csv(os.path.join(data_dir, "cross_level_specifications.csv")).set_index("specification")
    rv = rev.reindex(cla.index)
    sg = np.where(rv, -1, 1)
    for label, mask, sign in [("as-coded", slice(None), 1), ("forward-coded only", ~rv, 1), ("sign-aligned", slice(None), sg)]:
        a, b = (cla.r_bc * sign)[mask], (cla.r_pooled_within * sign)[mask]
        chk("individual", f"cross-level agreement, {label}, against r_pw",
            stats.pearsonr(a, b)[0], cls.loc[label, "vs_pooled_within"])

    # ---- the country contrast, on both scales
    pcc = (pd.read_csv(os.path.join(data_dir, "primary_contrast_country_effect.csv"))
           .rename(columns={"Unnamed: 0": "variable"}).set_index("variable"))
    tw, tn = CONTRASTS["country label"]
    dr = (r_bc(tw, data_dir) - r_bc(tn, data_dir)).reindex(pcc.index)
    dz = fisher_z_difference(tw, tn, data_dir).reindex(pcc.index)
    cbd = pd.read_csv(os.path.join(data_dir, "contrasts_by_direction.csv")).set_index("name")
    row = cbd.loc["country label effect (rich baseline)"]
    rvp = rev.reindex(pcc.index)
    chk("country contrast", "forward median raw dr", dr[~rvp].median(), row.fwd_median)
    chk("country contrast", "reverse median raw dr", dr[rvp].median(), row.rev_median)
    chk("country contrast", "forward Wilcoxon p", stats.wilcoxon(dr[~rvp])[1], row.fwd_wilcoxon_p, 1e-6)
    chk("country contrast", "forward median Fisher-z dr (contrasts_by_direction)",
        dz[~rvp].median(), row.fwd_median_z)
    cab = pd.read_csv(os.path.join(data_dir, "clean_ablations.csv")).set_index("block")
    chk("country contrast", "forward median Fisher-z dr (clean_ablations)",
        dz[~rvp].median(), cab.loc["country label", "fwd_median_dz"])

    # ---- the 2 x 2 and the triple difference
    items22 = sorted(r_bc(ANCHORED["target"], data_dir).index)
    rev12 = [v for v in items22 if rev[v]]
    rev10 = [v for v in rev12 if v not in PREDECLARED_EXCLUSIONS]
    rev8 = [v for v in rev10 if v not in ("freehms", "hmsacld")]
    fwd10 = [v for v in items22 if not rev[v]]
    ce = {sc: fisher_z_difference(TWOXTWO[(sc, "with")], TWOXTWO[(sc, "without")], data_dir).reindex(items22)
          for sc in ("numeric", "anchored")}
    tdf = pd.read_csv(os.path.join(data_dir, "twoxtwo_triple_difference.csv"))
    for spec, rset in [("all reverse items in batch (n=12)", rev12),
                       ("analysable reverse items, pre-declared exclusions (n=10)", rev10),
                       ("further excluding freehms and hmsacld (n=8)", rev8)]:
        ref = tdf[tdf.specification == spec].iloc[0]
        gap_n = ce["numeric"][fwd10].mean() - ce["numeric"][rset].mean()
        gap_a = ce["anchored"][fwd10].mean() - ce["anchored"][rset].mean()
        chk("2x2", f"triple difference [{spec}]", gap_n - gap_a, ref.triple_difference_z)
        chk("2x2", f"cells reconstruct the triple difference [{spec}]",
            gap_n - gap_a, ref.gap_numeric_z - ref.gap_anchored_z)

    # ---- the anchoring experiment
    long_fwd = [v for v in fwd10 if direction.loc[v, "scale"] == "0-10"]
    coarse_fwd = [v for v in fwd10 if v not in long_fwd]
    rT = r_bc(ANCHORED["target"], data_dir).reindex(items22)
    rA = r_bc(ANCHORED["baseline_A"], data_dir).reindex(items22)
    rB = r_bc(ANCHORED["baseline_B"], data_dir).reindex(items22)
    chk("anchoring", "empirical noise floor, median |r_A - r_B|", (rA - rB).abs().median(), 0.0602)
    chk("anchoring", "empirical noise floor, max |r_A - r_B|", (rA - rB).abs().max(), 0.1854)
    chk("anchoring", "raw-scale DiD, 42-item baseline, 10 reverse minus 6 coarse controls",
        (rT - rA)[rev10].mean() - (rT - rA)[coarse_fwd].mean(), 0.8458)
    adf = pd.read_csv(os.path.join(data_dir, "anchored_did_fisherz.csv"))
    for _, ref in adf.iterrows():
        base = rA if "42-item" in ref.baseline else rB
        rset = rev10 if "pre-declared" in ref.exclusions else rev12
        z = lambda s: np.arctanh(np.clip(s.to_numpy(float), -0.999999, 0.999999))
        dz2 = pd.Series(z(rT) - z(base), index=items22)
        chk("anchoring", f"Fisher-z DiD [{ref.baseline} | {ref.exclusions}]",
            dz2[rset].mean() - dz2[coarse_fwd].mean(), ref.did_z)
        chk("anchoring", f"placebo mean z [{ref.baseline}]", dz2[long_fwd].mean(), ref.placebo_mean_z)

    # ---- benchmarks
    ab = pd.read_csv(os.path.join(data_dir, "aggregate_benchmark.csv")).set_index("variable")
    bloc = pd.read_csv(os.path.join(data_dir, "country_bloc_assignment.csv")).set_index("cntry").bloc
    sca = load_scatter("qwen_1p", data_dir)
    sca["bloc"] = sca.cntry.map(bloc)
    reg_r, reg_m, gr_m = {}, {}, {}
    for var, d in sca.groupby("variable"):
        d = d.set_index("cntry")
        rng = direction.loc[var, "range"]
        pred = {}
        for c in d.index:
            peers = d.index[(d.bloc == d.loc[c, "bloc"]) & (d.index != c)]
            pred[c] = d.loc[peers, "human_mean"].mean() if len(peers) else d.loc[d.index != c, "human_mean"].mean()
        pred = pd.Series(pred).reindex(d.index)
        reg_r[var] = stats.pearsonr(d.human_mean, pred)[0]
        reg_m[var] = (d.human_mean - pred).abs().mean() / rng
        gr_m[var] = (d.human_mean - d.human_mean.mean()).abs().mean() / rng   # IN-SAMPLE, as released
    chk("benchmark", "leave-one-out regional r (max abs deviation)",
        (ab.r_region - pd.Series(reg_r).reindex(ab.index)).abs().max(), 0, 1e-9)
    chk("benchmark", "leave-one-out regional nMAE (max abs deviation)",
        (ab.mae_region - pd.Series(reg_m).reindex(ab.index)).abs().max(), 0, 1e-9)
    chk("benchmark", "grand-mean nMAE is IN-SAMPLE (max abs deviation)",
        (ab.mae_grand - pd.Series(gr_m).reindex(ab.index)).abs().max(), 0, 1e-9)
    chk("benchmark", "model nMAE equals the released mae_model column (max abs deviation)",
        (ab.mae_model - normalised_mae("qwen_1p", data_dir, by="variable").reindex(ab.index)).abs().max(), 0, 1e-9)
    pw = pd.read_csv(os.path.join(data_dir, "pooled_within_r_by_item.csv")).set_index("variable")
    ceil = (pd.read_csv(os.path.join(data_dir, "ceiling_by_variable.csv"))
            .rename(columns={"Unnamed: 0": "variable"}).set_index("variable"))
    chk("benchmark", "achieved individual recovery as a share of the ceiling",
        pw.r_pooled_within.median() / ceil.within_country_ceiling_R.median(), 0.204, 1e-3)

    # ---- concordance
    cpv = pd.read_csv(os.path.join(data_dir, "concordance_per_variable.csv"))
    q = cpv[cpv.condition == "qwen_1p"].set_index("variable")
    chk("concordance", "CCC equals r x Cb (max abs deviation)", (q.ccc - q.r * q.Cb).abs().max(), 0, 1e-9)
    chk("concordance", "median CCC", q.ccc.median(), 0.0357)
    chk("concordance", "median Cb", q.Cb.median(), 0.1319)
    chk("concordance", "median absolute bias", q.bias.abs().median(), 1.1268)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="handoff_upload", help="directory holding the handoff bundle")
    ap.add_argument("--out", default="verification_v13.csv")
    args = ap.parse_args()
    res = verify(args.data)
    fails = res[res.status == "FAIL"]
    print(f"{len(res)} checks, {len(fails)} failures\n")
    if len(fails):
        with pd.option_context("display.width", 250, "display.max_colwidth", 70):
            print(fails.to_string(index=False))
    res.to_csv(args.out, index=False)
    print(f"\nfull log written to {args.out}")


if __name__ == "__main__":
    main()
