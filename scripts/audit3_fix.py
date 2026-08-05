#!/usr/bin/env python3
"""
audit3_fix.py -- third-round corrections and the three supervisor-driven
extensions, computed entirely from the released v13 bundle.

Run from ~/Winston_Code:

    python3 audit3_fix.py --data handoff_upload --out results/analysis/audit3

Nothing is overwritten. Every corrected table is written with a ``_v2`` suffix
alongside the original so that the two can be diffed before the package is
rebuilt. The upstream scripts still need the patches listed in
``audit3_patch_spec.csv``; this file exists so that the figure work is not
blocked on a regeneration round trip.

CORRECTIONS
  C1  contrasts_by_direction: the Fisher-z columns applied arctanh to a
      DIFFERENCE of correlations. Recomputed as a difference of transforms and
      re-emitted in long format, which removes the wide-layout hazard that
      produced the gap-column error in the 2 x 2 audit.
  C2  within- and between-country dispersion ratios use different aggregators
      (mean and median respectively) and neither was documented. All estimators
      are emitted for every condition so the manuscript can name one.
  C3  aggregate_benchmark: the mae_* columns are already range-normalised
      despite their names; the grand-mean comparator was in-sample while the
      regional comparator was leave-one-out; r_grand was released as 0.0 when
      it is undefined; and the per-item-offset reduction was a ratio of medians.
  C4  undefined individual-level correlations arising from zero-variance cells
      are counted and listed explicitly, per the project's own rule that any
      comparison which can produce NaN needs an explicit NaN report.
  C5  the 2 x 2 gains a forward-control robustness row, because the
      noise-domination screen behind the pre-declared exclusions was applied to
      the reverse arm only.

EXTENSIONS (each closes an item in the supervisor's letter that is still open
at the data layer, not at the writing layer)
  E1  paired item-level test of the first- versus third-person framing
      manipulation. The letter requires the manipulation to be part of the
      argument; the bundle contained only condition-level medians, which the
      project brief forbids comparing.
  E2  country-mean reliability, the implied attenuation factor and the
      detectable correlation at 80 per cent power, recomputed at n = 685. The
      released figures are for n = 500 and n = 250 and no longer describe any
      arm of this round.
  E3  the log GDP per capita plus region-dummy comparator. BLOCKED: needs a
      two-column country file. The specification is written to disk.
  E4  explicit counterexample pairs for the cross-level selection argument.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

Z = lambda x: np.arctanh(np.clip(np.asarray(x, dtype=float), -0.999999, 0.999999))


# --------------------------------------------------------------------------
# loading helpers (duplicated from silicon_v13.py so this file stands alone)
# --------------------------------------------------------------------------

def conditions(d):
    return sorted(os.path.basename(p)[9:-5] for p in glob.glob(os.path.join(d, "manifest_*.json")))


def scatter(tag, d):
    df = pd.read_csv(os.path.join(d, f"silicon_full_country_scatter_{tag}.csv"))
    return df.rename(columns={"survey_mean": "human_mean", "survey_sd": "human_sd"})


def rq3(tag, d):
    return pd.read_csv(os.path.join(d, f"silicon_full_rq3_{tag}.csv"))


def direction(d):
    t = pd.read_csv(os.path.join(d, "item_direction_table.csv")).set_index("variable")
    t["reverse"] = t["direction"].eq("reverse")
    t["range"] = t["high_value"] - t["low_value"]
    return t


def r_bc(tag, d):
    out = {}
    for var, g in scatter(tag, d).groupby("variable"):
        out[var] = stats.pearsonr(g.human_mean, g.silicon_mean)[0]
    return pd.Series(out)


def r_pw(tag, d):
    """Pooled within-country correlation per item, from the per-cell moments."""
    out = {}
    for var, g in rq3(tag, d).groupby("variable"):
        g = g.dropna(subset=["r_pearson", "human_sd", "silicon_sd", "n_valid"])
        n = g.n_valid.to_numpy(float)
        wh = ((n - 1) * g.human_sd ** 2).sum()
        ws = ((n - 1) * g.silicon_sd ** 2).sum()
        whs = ((n - 1) * g.r_pearson * g.human_sd * g.silicon_sd).sum()
        out[var] = whs / np.sqrt(wh * ws) if wh > 0 and ws > 0 else np.nan
    return pd.Series(out)


def paired(a, b, rev, label_a, label_b, level):
    """Paired item-level contrast of two conditions, split by scale direction."""
    idx = a.index.intersection(b.index)
    a, b, rv = a.reindex(idx), b.reindex(idx), rev.reindex(idx)
    dr, dz = a - b, pd.Series(Z(a) - Z(b), index=idx)
    rows = []
    for grp, mask in (("all", pd.Series(True, index=idx)), ("forward", ~rv), ("reverse", rv)):
        x, xz = dr[mask].dropna(), dz[mask].dropna()
        rows.append(dict(
            level=level, arm_a=label_a, arm_b=label_b, group=grp, n=len(x),
            median_dr=x.median(), mean_dr=x.mean(),
            median_dz=xz.median(), mean_dz=xz.mean(),
            n_a_higher=int((x > 0).sum()),
            wilcoxon_p=float(stats.wilcoxon(x).pvalue) if len(x) >= 6 else np.nan,
            n_undefined=int(dr[mask].isna().sum())))
    return rows


# --------------------------------------------------------------------------
# C1  contrasts by direction, with the Fisher transform applied correctly
# --------------------------------------------------------------------------

def c1(d, out):
    rev = direction(d)["reverse"]
    pairs = {
        "country label effect (rich baseline)": ("qwen_1p_full_noregion", "qwen_1p_full_nocountry"),
        "country label effect (sparse baseline)": ("qwen_1p_minimal", "qwen_1p_demo_only"),
        "NUTS region code effect": ("qwen_1p", "qwen_1p_full_noregion"),
        "political-identity variables (full minus nopolitical)": ("qwen_1p", "qwen_1p_full_nopolitical"),
        "all enrichment beyond country (full minus minimal)": ("qwen_1p", "qwen_1p_minimal"),
    }
    rows = []
    for name, (a, b) in pairs.items():
        for r in paired(r_bc(a, d), r_bc(b, d), rev, a, b, "r_bc"):
            r["contrast"] = name
            rows.append(r)
    t = pd.DataFrame(rows)[["contrast", "level", "arm_a", "arm_b", "group", "n",
                            "median_dr", "mean_dr", "median_dz", "mean_dz",
                            "n_a_higher", "wilcoxon_p", "n_undefined"]]
    t.to_csv(os.path.join(out, "contrasts_by_direction_v2.csv"), index=False)

    old = pd.read_csv(os.path.join(d, "contrasts_by_direction.csv")).set_index("name")
    cmp = []
    for name in pairs:
        if name not in old.index:
            continue
        for grp, key in (("all", "all"), ("forward", "fwd"), ("reverse", "rev")):
            new = t[(t.contrast == name) & (t.group == grp)].iloc[0]
            cmp.append(dict(contrast=name, group=grp,
                            released_median_z=old.loc[name, f"{key}_median_z"],
                            corrected_median_z=new.median_dz,
                            released_mean_z=old.loc[name, f"{key}_mean_z"],
                            corrected_mean_z=new.mean_dz))
    c = pd.DataFrame(cmp)
    c["median_z_shift"] = c.corrected_median_z - c.released_median_z
    c.to_csv(os.path.join(out, "c1_fisher_z_before_after.csv"), index=False)
    return t, c


# --------------------------------------------------------------------------
# C2  dispersion-ratio estimators
# --------------------------------------------------------------------------

def c2(d, out):
    rows = []
    for tag in conditions(d):
        s = scatter(tag, d)
        cell = (s.silicon_sd / s.human_sd).replace([np.inf, -np.inf], np.nan)
        bw = {}
        for var, g in s.groupby("variable"):
            bw[var] = g.silicon_mean.std(ddof=1) / g.human_mean.std(ddof=1)
        bw = pd.Series(bw)
        rows.append(dict(condition=tag,
                         within_mean_of_ratios=cell.mean(),          # the released figure
                         within_median_of_ratios=cell.median(),
                         within_ratio_of_means=s.silicon_sd.mean() / s.human_sd.mean(),
                         within_iqr_lo=cell.quantile(.25), within_iqr_hi=cell.quantile(.75),
                         between_median_of_ratios=bw.median(),       # the released figure
                         between_mean_of_ratios=bw.mean(),
                         between_iqr_lo=bw.quantile(.25), between_iqr_hi=bw.quantile(.75)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "sd_ratio_estimators.csv"), index=False)
    return t


# --------------------------------------------------------------------------
# C3  aggregate benchmark
# --------------------------------------------------------------------------

def c3(d, out):
    dirt = direction(d)
    bloc = pd.read_csv(os.path.join(d, "country_bloc_assignment.csv")).set_index("cntry").bloc
    s = scatter("qwen_1p", d)
    s["bloc"] = s.cntry.map(bloc)
    rows = []
    for var, g in s.groupby("variable"):
        g = g.set_index("cntry")
        rng = dirt.loc[var, "range"]
        h, si = g.human_mean, g.silicon_mean
        reg, gr_loo = {}, {}
        for c in g.index:
            peers = g.index[(g.bloc == g.loc[c, "bloc"]) & (g.index != c)]
            reg[c] = h[peers].mean() if len(peers) else h[h.index != c].mean()
            gr_loo[c] = h[h.index != c].mean()
        reg, gr_loo = pd.Series(reg).reindex(g.index), pd.Series(gr_loo).reindex(g.index)
        nmae_model = (si - h).abs().mean() / rng
        nmae_debias = (si - (si - h).mean() - h).abs().mean() / rng
        rows.append(dict(
            variable=var, item_range=rng,
            r_model=stats.pearsonr(h, si)[0], nmae_model=nmae_model,
            r_debias=stats.pearsonr(h, si)[0], nmae_debias=nmae_debias,
            offset_reduction_item=1 - nmae_debias / nmae_model,
            r_region_loo=stats.pearsonr(h, reg)[0], nmae_region_loo=(h - reg).abs().mean() / rng,
            r_grand=np.nan,                     # a constant predictor has zero variance
            nmae_grand_insample=(h - h.mean()).abs().mean() / rng,   # as released
            nmae_grand_loo=(h - gr_loo).abs().mean() / rng))         # matched to the regional arm
    t = pd.DataFrame(rows).set_index("variable")
    t.to_csv(os.path.join(out, "aggregate_benchmark_v2.csv"))

    summ = pd.DataFrame([
        dict(quantity="median r, model", value=t.r_model.median()),
        dict(quantity="median r, leave-one-out regional average", value=t.r_region_loo.median()),
        dict(quantity="items on which the regional average outranks the model",
             value=int((t.r_region_loo > t.r_model).sum())),
        dict(quantity="median nMAE, model", value=t.nmae_model.median()),
        dict(quantity="median nMAE, leave-one-out regional average", value=t.nmae_region_loo.median()),
        dict(quantity="items on which the regional average beats the model on nMAE",
             value=int((t.nmae_region_loo < t.nmae_model).sum())),
        dict(quantity="median nMAE, grand mean (in-sample, as released)", value=t.nmae_grand_insample.median()),
        dict(quantity="median nMAE, grand mean (leave-one-out, matched)", value=t.nmae_grand_loo.median()),
        dict(quantity="items on which the model beats the in-sample grand mean on nMAE",
             value=int((t.nmae_model < t.nmae_grand_insample).sum())),
        dict(quantity="items on which the model beats the leave-one-out grand mean on nMAE",
             value=int((t.nmae_model < t.nmae_grand_loo).sum())),
        dict(quantity="offset reduction, ratio of medians (as released, 64.2%)",
             value=1 - t.nmae_debias.median() / t.nmae_model.median()),
        dict(quantity="offset reduction, ratio of means", value=1 - t.nmae_debias.mean() / t.nmae_model.mean()),
        dict(quantity="offset reduction, MEDIAN of the per-item reductions (recommended)",
             value=t.offset_reduction_item.median()),
        dict(quantity="offset reduction, mean of the per-item reductions",
             value=t.offset_reduction_item.mean()),
    ])
    summ.to_csv(os.path.join(out, "aggregate_benchmark_estimators.csv"), index=False)
    return t, summ


# --------------------------------------------------------------------------
# C4  undefined cells
# --------------------------------------------------------------------------

def c4(d, out):
    rows, cells = [], []
    for tag in conditions(d):
        r = rq3(tag, d)
        bad = r[(r.r_pearson.isna()) | (r.human_sd.abs() < 1e-12) | (r.silicon_sd.abs() < 1e-12)]
        rows.append(dict(condition=tag, n_cells=len(r), n_undefined=len(bad),
                         share_undefined=len(bad) / len(r)))
        for _, x in bad.iterrows():
            cells.append(dict(condition=tag, variable=x.variable, cntry=x.cntry,
                              n_valid=x.n_valid, human_sd=x.human_sd, silicon_sd=x.silicon_sd,
                              reason="constant silicon response" if abs(x.silicon_sd) < 1e-12
                              else "constant human response"))
    pd.DataFrame(rows).to_csv(os.path.join(out, "undefined_cells_by_condition.csv"), index=False)
    cl = pd.DataFrame(cells)
    cl.to_csv(os.path.join(out, "undefined_cells_list.csv"), index=False)
    return cl


# --------------------------------------------------------------------------
# C5  2 x 2 forward-control robustness
# --------------------------------------------------------------------------

def c5(d, out):
    dirt = direction(d)
    rev = dirt["reverse"]
    items = sorted(r_bc("qwen_1p_anchored", d).index)
    rev12 = [v for v in items if rev[v]]
    rev10 = [v for v in rev12 if v not in ("hmsfmlsh", "rlgatnd")]
    rev8 = [v for v in rev10 if v not in ("freehms", "hmsacld")]
    fwd = [v for v in items if not rev[v]]
    ce = {}
    for sc, (tw, tn) in {"numeric": ("qwen_1p_full_noregion", "qwen_1p_full_nocountry"),
                         "anchored": ("qwen_1p_full_noregion_anchored", "qwen_1p_full_nocountry_anchored")}.items():
        a, b = r_bc(tw, d).reindex(items), r_bc(tn, d).reindex(items)
        ce[sc] = pd.Series(Z(a) - Z(b), index=items)

    def td(fset, rset):
        return ((ce["numeric"][fset].mean() - ce["numeric"][rset].mean())
                - (ce["anchored"][fset].mean() - ce["anchored"][rset].mean()))

    # v13 silicon between-country dispersion, which is what the pre-declared
    # exclusion was reasoned from. In v13 the FORWARD item inprdsc sits below
    # rlgatnd on this criterion, so the screen is asymmetric.
    s = scatter("qwen_1p", d)
    disp = {}
    for var, g in s.groupby("variable"):
        if var in items:
            disp[var] = g.silicon_mean.std(ddof=1)
    disp = pd.Series(disp).sort_values()
    dt = pd.DataFrame(dict(silicon_between_country_sd=disp))
    dt["reverse"] = rev.reindex(dt.index)
    dt["screened_out"] = dt.index.isin(["hmsfmlsh", "rlgatnd"])
    # Raw dispersion places the forward control inprdsc BETWEEN the two
    # screened-out items, which makes the screen look asymmetric. Country-mean
    # reliability does not: it separates the two excluded items cleanly from
    # every retained one. Report the screen on reliability.
    rel = {}
    sr = scatter("qwen_1p", d).set_index(["variable", "cntry"])
    nr = rq3("qwen_1p", d).set_index(["variable", "cntry"])
    for var in dt.index:
        gs, gn = sr.loc[var], nr.loc[var].n_valid
        S2 = gs.silicon_mean.var(ddof=1)
        E = float(np.mean(gs.silicon_sd.to_numpy(float) ** 2 / gn.reindex(gs.index).to_numpy(float)))
        rel[var] = max(0.0, S2 - E) / S2 if S2 > 0 else np.nan
    dt["silicon_country_mean_reliability"] = pd.Series(rel)
    dt.to_csv(os.path.join(out, "c5_dispersion_screen.csv"))

    rows = [dict(reverse_set="all reverse items in batch (n=12)", forward_controls="all 10", n_reverse=12,
                 triple_difference_z=td(fwd, rev12)),
            dict(reverse_set="pre-declared exclusions (n=10), PRIMARY", forward_controls="all 10", n_reverse=10,
                 triple_difference_z=td(fwd, rev10)),
            dict(reverse_set="further excluding freehms and hmsacld (n=8)", forward_controls="all 10", n_reverse=8,
                 triple_difference_z=td(fwd, rev8))]
    for drop in fwd:
        rows.append(dict(reverse_set="pre-declared exclusions (n=10), PRIMARY",
                         forward_controls=f"dropping {drop} (9 controls)", n_reverse=10,
                         triple_difference_z=td([v for v in fwd if v != drop], rev10)))
    t = pd.DataFrame(rows)
    t["shift_from_primary"] = t.triple_difference_z - td(fwd, rev10)
    t.to_csv(os.path.join(out, "twoxtwo_forward_control_robustness.csv"), index=False)
    return t, dt


# --------------------------------------------------------------------------
# E1  the framing manipulation, paired at the item level
# --------------------------------------------------------------------------

def e1(d, out):
    rev = direction(d)["reverse"]
    rows = []
    for model, (a, b) in {"qwen": ("qwen_1p", "qwen_3p"), "llama": ("llama_1p", "llama_3p")}.items():
        for level, fn in (("r_bc", r_bc), ("r_pw", r_pw)):
            for r in paired(fn(a, d), fn(b, d), rev, a, b, level):
                r["model"] = model
                r["caveat"] = ("llama_3p parses only 77.28 per cent of scheduled responses, "
                               "concentrated on the 0-10 items; treat as provisional"
                               if model == "llama" else "")
                rows.append(r)
    t = pd.DataFrame(rows)[["model", "level", "arm_a", "arm_b", "group", "n", "median_dr",
                            "mean_dr", "median_dz", "mean_dz", "n_a_higher", "wilcoxon_p",
                            "n_undefined", "caveat"]]
    t.to_csv(os.path.join(out, "framing_contrast_paired.csv"), index=False)
    return t


# --------------------------------------------------------------------------
# E2  reliability, attenuation and power at n = 685
# --------------------------------------------------------------------------

def e2(d, out, tag="qwen_1p"):
    s = scatter(tag, d).set_index(["variable", "cntry"])
    r = rq3(tag, d).set_index(["variable", "cntry"])
    rows = []
    for var in s.index.get_level_values(0).unique():
        gs, gr = s.loc[var], r.loc[var]
        n = gr.n_valid.reindex(gs.index).to_numpy(float)
        out_row = dict(variable=var)
        for side, mcol, scol in (("human", "human_mean", "human_sd"), ("silicon", "silicon_mean", "silicon_sd")):
            S2 = gs[mcol].var(ddof=1)
            E = float(np.mean(gs[scol].to_numpy(float) ** 2 / n))
            out_row[f"{side}_var_between"] = S2
            out_row[f"{side}_var_sampling"] = E
            out_row[f"{side}_reliability"] = max(0.0, (S2 - E)) / S2 if S2 > 0 else np.nan
        out_row["attenuation_factor"] = np.sqrt(out_row["human_reliability"] * out_row["silicon_reliability"])
        rows.append(out_row)
    t = pd.DataFrame(rows).set_index("variable")
    t.to_csv(os.path.join(out, "reliability_attenuation_n685.csv"))

    def detectable_r(n):
        return float(np.tanh((stats.norm.ppf(.975) + stats.norm.ppf(.80)) / np.sqrt(n - 3)))

    man = json.load(open(os.path.join(d, f"manifest_{tag}.json")))
    npc, N = man["sample_per_country"], man["n_respondents"]
    pw = pd.read_csv(os.path.join(d, "pooled_within_r_by_item.csv")).set_index("variable").r_pooled_within
    avi = pd.read_csv(os.path.join(d, "aggregate_vs_individual_by_country.csv"))
    summ = pd.DataFrame([
        dict(quantity="respondents per country (all arms)", value=npc),
        dict(quantity="median country-mean reliability, human side", value=t.human_reliability.median()),
        dict(quantity="median country-mean reliability, silicon side", value=t.silicon_reliability.median()),
        dict(quantity="median attenuation factor on r_bc", value=t.attenuation_factor.median()),
        dict(quantity="implied median attenuation, per cent", value=100 * (1 - t.attenuation_factor.median())),
        dict(quantity="items with human-side reliability below 0.50", value=int((t.human_reliability < .5).sum())),
        dict(quantity="items with silicon-side reliability below 0.50", value=int((t.silicon_reliability < .5).sum())),
        dict(quantity=f"detectable r, 80 per cent power, within one country (n={npc})", value=detectable_r(npc)),
        dict(quantity="detectable r, 80 per cent power, at n=500 (the figure the plan quotes)",
             value=detectable_r(500)),
        dict(quantity=f"detectable r, 80 per cent power, pooled within-country (N={N})", value=detectable_r(N)),
        dict(quantity="observed mean within-country individual r", value=avi.mean_within_r.mean()),
        dict(quantity="observed median r_pw", value=pw.median()),
    ])
    summ.to_csv(os.path.join(out, "reliability_attenuation_summary.csv"), index=False)
    return t, summ


# --------------------------------------------------------------------------
# E3  specification only: the GDP comparator is blocked on an input file
# --------------------------------------------------------------------------

def e3(out):
    spec = (
        "E3  log GDP per capita plus region dummies, as the aggregate comparator that\n"
        "    needs no survey data from other countries.\n\n"
        "STATUS: blocked. benchmark_v13.py --gdp is coded but was never run because the\n"
        "input file does not exist. This is the only comparator that speaks to a\n"
        "researcher holding no survey data at all, which is the silicon-sampling use\n"
        "case, so the gap is currently acknowledged in the memo rather than closed.\n\n"
        "REQUIRED INPUT: gdp_per_capita.csv, two columns, thirty rows.\n"
        "    cntry   ISO-2 code, matching country_bloc_assignment.csv exactly\n"
        "    gdp_pc  GDP per capita, current US dollars, one reference year\n\n"
        "Round 11 fieldwork ran across 2023 and 2024, so use a single year and record\n"
        "it. World Bank NY.GDP.PCAP.CD covers twenty-nine of the thirty; Montenegro is\n"
        "present, so no substitution is needed. Do not mix sources across countries.\n\n"
        "SPECIFICATION once the file exists, per item:\n"
        "    regress the human country mean on log(gdp_pc) and bloc dummies, thirty\n"
        "    observations, leave-one-out predicted values; report r against the human\n"
        "    mean and range-normalised MAE, on the same two axes as the model and the\n"
        "    regional average. Leave-one-out is required for comparability with the\n"
        "    regional arm; with five blocs and one covariate the in-sample fit would\n"
        "    otherwise carry six of thirty degrees of freedom.\n\n"
        "REPORTING: this comparator does not need survey data from other countries, so\n"
        "it is the only row of the benchmark table exempt from the mandatory caveat.\n"
    )
    with open(os.path.join(out, "e3_gdp_comparator_spec.txt"), "w") as fh:
        fh.write(spec)


# --------------------------------------------------------------------------
# E4  counterexample pairs for the cross-level selection argument
# --------------------------------------------------------------------------

def e4(d, out):
    avi = pd.read_csv(os.path.join(d, "aggregate_vs_individual_by_country.csv")).set_index("cntry")
    rows = []
    for a in avi.index:
        for b in avi.index:
            if a >= b:
                continue
            da = avi.loc[a, "profile_r"] - avi.loc[b, "profile_r"]
            di = avi.loc[a, "mean_within_r"] - avi.loc[b, "mean_within_r"]
            if da * di < 0:
                rows.append(dict(country_a=a, country_b=b,
                                 profile_r_a=avi.loc[a, "profile_r"], profile_r_b=avi.loc[b, "profile_r"],
                                 within_r_a=avi.loc[a, "mean_within_r"], within_r_b=avi.loc[b, "mean_within_r"],
                                 profile_gap=da, within_gap=di,
                                 severity=abs(da) * abs(di)))
    t = pd.DataFrame(rows).sort_values("severity", ascending=False)
    t.to_csv(os.path.join(out, "cross_level_counterexamples.csv"), index=False)
    n_pairs = len(avi) * (len(avi) - 1) // 2
    summ = pd.DataFrame([
        dict(quantity="country pairs in total", value=n_pairs),
        dict(quantity="pairs whose aggregate and individual ordering disagree", value=len(t)),
        dict(quantity="share of pairs that invert", value=len(t) / n_pairs),
        dict(quantity="sharpest inversion",
             value=f"{t.iloc[0].country_a} vs {t.iloc[0].country_b}: profile r "
                   f"{t.iloc[0].profile_r_a:.3f} vs {t.iloc[0].profile_r_b:.3f} but within-country r "
                   f"{t.iloc[0].within_r_a:.4f} vs {t.iloc[0].within_r_b:.4f}"),
    ])
    summ.to_csv(os.path.join(out, "cross_level_counterexamples_summary.csv"), index=False)
    return t, summ


# --------------------------------------------------------------------------
# patch specification for the upstream scripts
# --------------------------------------------------------------------------

PATCHES = [
    dict(id="P1", script="analyze_v13b.py", table="contrasts_by_direction.csv", severity="must fix",
         defect="the *_median_z and *_mean_z columns apply arctanh to a difference of correlations",
         fix="difference the transforms: dz = arctanh(r_with) - arctanh(r_without) per item, then take "
             "the median or mean over items; emit in long format",
         verifier="the forward median must equal clean_ablations.csv fwd_median_dz (0.196699), not 0.136016"),
    dict(id="P2", script="analyze_v13c.py", table="condition_summary.csv", severity="must document",
         defect="within_sd_ratio is a mean over cells while between_sd_ratio is a median over items; "
                "neither estimator is recorded",
         fix="rename to within_sd_ratio_mean and between_sd_ratio_median, or emit both estimators",
         verifier="within = mean of scatter silicon_sd/human_sd over cells; between = median over items"),
    dict(id="P3", script="benchmark_v13.py", table="aggregate_benchmark.csv", severity="must fix",
         defect="mae_grand is in-sample while mae_region is leave-one-out",
         fix="compute the grand mean leave-one-out as well, or label the column in_sample explicitly",
         verifier="median rises from 0.0715 to 0.0740 under leave-one-out"),
    dict(id="P4", script="benchmark_v13.py", table="aggregate_benchmark.csv", severity="must fix",
         defect="the mae_* columns are already range-normalised but are not named so",
         fix="rename to nmae_*, matching norm_mae in country_profile_and_mae.csv",
         verifier="median mae_model is 0.1796, which is already a normalised quantity"),
    dict(id="P5", script="benchmark_v13.py", table="aggregate_benchmark.csv", severity="must fix",
         defect="r_grand is written as 0.0 for all items although a constant predictor makes it undefined",
         fix="write NaN and report the count of undefined correlations explicitly",
         verifier="all 42 values are exactly 0.0 in the released file"),
    dict(id="P6", script="collect_numbers.py", table="manuscript_numbers.csv", severity="must fix",
         defect="the 64.2 per cent offset reduction is a ratio of medians, the largest of four "
                "defensible definitions",
         fix="report the median of the per-item reductions and name the estimator in the note field",
         verifier="the four definitions give 0.642, 0.629, 0.599 and 0.537"),
    dict(id="P7", script="analyze_2x2.py", table="twoxtwo_triple_difference.csv", severity="must document",
         defect="the noise-domination screen was applied to the reverse arm only; in v13 the forward "
                "control inprdsc has a lower silicon between-country SD than the screened-out rlgatnd",
         fix="add the forward-control leave-one-out row and record the v13 dispersion values, "
             "attributing the quoted 0.017 and 0.044 to the round of declaration",
         verifier="dropping inprdsc moves the primary triple difference by -0.0004"),
    dict(id="P8", script="analyze_v13.py", table="all rq3-derived tables", severity="must report",
         defect="one item x country cell is undefined (hmsfmlsh in Switzerland, anchored arm, "
                "constant silicon response) and no NaN count is emitted",
         fix="emit a per-condition undefined-cell count alongside every correlation summary",
         verifier="one cell in qwen_1p_anchored, zero elsewhere"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="handoff_upload")
    ap.add_argument("--out", default="audit3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 220)

    print("=" * 78, "\nCORRECTIONS\n", "=" * 78)
    _, c1cmp = c1(a.data, a.out)
    print("\nC1  Fisher-z columns, released against corrected")
    print(c1cmp.round(4).to_string(index=False))
    sd = c2(a.data, a.out)
    print("\nC2  dispersion-ratio estimators, main condition")
    print(sd[sd.condition == "qwen_1p"].T.round(4).to_string())
    _, bsum = c3(a.data, a.out)
    print("\nC3  aggregate benchmark under matched estimators")
    print(bsum.to_string(index=False))
    und = c4(a.data, a.out)
    print("\nC4  undefined cells")
    print(und.to_string(index=False) if len(und) else "  none")
    fc, dsp = c5(a.data, a.out)
    print("\nC5  2x2 forward-control robustness (largest three shifts)")
    print(fc.reindex(fc.shift_from_primary.abs().sort_values(ascending=False).index)
          .head(4).round(5).to_string(index=False))

    print("\n" + "=" * 78, "\nEXTENSIONS\n", "=" * 78)
    fr = e1(a.data, a.out)
    print("\nE1  framing manipulation, paired item-level")
    print(fr.drop(columns=["caveat", "arm_a", "arm_b"]).round(4).to_string(index=False))
    _, rsum = e2(a.data, a.out)
    print("\nE2  reliability, attenuation and power at n=685")
    print(rsum.to_string(index=False))
    e3(a.out)
    print("\nE3  written to e3_gdp_comparator_spec.txt (blocked on gdp_per_capita.csv)")
    _, csum = e4(a.data, a.out)
    print("\nE4  cross-level counterexamples")
    print(csum.to_string(index=False))

    pd.DataFrame(PATCHES).to_csv(os.path.join(a.out, "audit3_patch_spec.csv"), index=False)
    print(f"\n{len(PATCHES)} upstream patches specified in {a.out}/audit3_patch_spec.csv")
    print(f"all tables written to {a.out}/")


if __name__ == "__main__":
    main()
