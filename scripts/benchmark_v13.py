#!/usr/bin/env python3
"""
benchmark_v13.py — the three analyses that remained open, none of which needs the 2x2.

    python benchmark_v13.py
    python benchmark_v13.py --gdp gdp_per_capita.csv     # optional, see below

1. AGGREGATE BENCHMARK: what counts as good performance at the country level?
   The revision plan's section 5.6 specifies an individual-level benchmark (the
   demographic ceiling) and a distributional one (pre-declared dispersion bands),
   but the aggregate benchmark — "better than what alternative?" — was never
   built. Without it, "moderate recovery" has no referent, which is the specific
   thing the supervisor's letter says the paper never defines.

   Two comparators are computed here, both requiring no model at all:

     grand mean      predict every country's value on an item as the mean of
                     that item across countries. By construction its
                     between-country correlation is zero, so it cannot rank
                     countries at all, but it can still be close in absolute
                     terms.
     region mean     predict a country's value as the leave-one-out mean of the
                     other countries in its region. This is the cheapest
                     possible cross-national predictor: it knows only which
                     coarse region a country belongs to.

   A third row reports the model after per-item mean-bias removal. That
   correction requires knowing the true country means, so it is NOT available to
   a real user; it is included as an OPTIMISTIC UPPER BOUND that separates "the
   levels are displaced" from "the country-to-country pattern is wrong".

   Caveat to state in the paper: both comparators need survey data from other
   countries, so this is the comparison facing a researcher who already holds
   real data and is considering the model as an extension. A researcher with no
   survey data anywhere has neither comparator available, and for them the
   relevant benchmark is the GDP-based one below.

   --gdp accepts a two-column CSV (cntry,gdp_pc) and adds a regression
   predictor using log GDP per capita plus region dummies. No GDP figures are
   hard-coded here: supplying wrong macro data silently would be worse than
   omitting the comparator, so the file must come from the user (World Bank
   indicator NY.GDP.PCAP.CD or equivalent).

2. LLAMA THIRD-PERSON COVERAGE SENSITIVITY. Its parse rate is 77.28 per cent,
   concentrated on the 0-10 items, and missingness of that size is unlikely to be
   random. The manuscript's limitations section promises this be examined. Here
   the condition's headline numbers are recomputed on progressively stricter
   coverage thresholds, so the reader can see whether the conclusions about that
   condition depend on the poorly covered items.

3. THE euftf ANOMALY. It is the one forward-coded item with a negative
   between-country correlation, so the response-direction account cannot explain
   it, and the revision plan flags it as needing substantive treatment rather
   than being left for a reader to find. This asks an answerable question: across
   countries, which HUMAN item do the model's euftf answers actually track? If
   they track a different construct positively, that identifies what the model
   appears to have been answering.
"""
from __future__ import annotations

import argparse
import os

import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", message="An input array is constant")

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
# Conventional coarse grouping of the ESS11 countries. Deliberately simple and
# not tuned: the point of the comparator is that it uses almost no information.
REGION = {
    **{c: "Nordic" for c in ["FI", "IS", "NO", "SE"]},
    **{c: "Western" for c in ["AT", "BE", "CH", "DE", "FR", "GB", "IE", "NL"]},
    **{c: "Southern" for c in ["CY", "ES", "GR", "IT", "PT"]},
    **{c: "CentralEast" for c in ["BG", "CZ", "EE", "HR", "HU", "LT", "LV", "ME",
                                  "PL", "RS", "SI", "SK", "UA"]},
    "IL": "Other",
}

ap = argparse.ArgumentParser()
ap.add_argument("--gdp", default=None,
                help="optional CSV with columns cntry,gdp_pc")
args = ap.parse_args()


def scatter(tag):
    p = os.path.join(R, f"silicon_full_country_scatter_{tag}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


# ===========================================================================
print("=" * 78)
print("1. AGGREGATE BENCHMARK — better than what alternative?")
print("=" * 78)
sc = scatter("qwen_1p")
if sc is None:
    print("  qwen_1p scatter file not found — skipping.")
else:
    sc = sc.dropna(subset=["survey_mean", "silicon_mean"]).copy()
    sc["region"] = sc.cntry.map(REGION)
    unmapped = sorted(sc.loc[sc.region.isna(), "cntry"].unique())
    if unmapped:
        print(f"  WARNING: countries with no region assigned: {unmapped}")
        print("  They fall back to the all-other-countries mean.")

    gdp = None
    if args.gdp and os.path.exists(args.gdp):
        gdp = pd.read_csv(args.gdp)
        gdp.columns = [c.strip().lower() for c in gdp.columns]
        if not {"cntry", "gdp_pc"} <= set(gdp.columns):
            print(f"  --gdp file needs columns cntry,gdp_pc; found "
                  f"{list(gdp.columns)}. Ignoring it.")
            gdp = None
        else:
            gdp["log_gdp"] = np.log(pd.to_numeric(gdp.gdp_pc, errors="coerce"))
    elif args.gdp:
        print(f"  --gdp file not found: {args.gdp}")

    # Export the bloc assignment. Without it the comparator that carries the
    # sharpest practical claim cannot be reproduced from the package, and it is
    # NOT the same as the grouping in the March round's silicon_lib (which
    # deliberately left Israel unclassified). Note that "Other" is a bloc of one,
    # so Israel's leave-one-out prediction falls back to the mean of all other
    # countries; that behaviour is part of the definition.
    # N4b FIX: n_peers previously counted over the whole REGION dict, which
    # contains CZ while this ESS edition's 30 countries include UA and not CZ.
    # Every CentralEast row was therefore one too high. mae_region was always
    # computed on the in-sample frame and is unaffected, but this table is the
    # reproducibility artefact, so its columns must be right.
    _in = sorted(sc.cntry.unique())
    _blocs = {c: REGION.get(c, "UNASSIGNED") for c in _in}
    pd.DataFrame([dict(cntry=c, bloc=_blocs[c],
                       n_peers_in_sample=sum(1 for k in _in
                                             if _blocs[k] == _blocs[c] and k != c))
                  for c in _in]
                 ).to_csv(os.path.join(OUT, "country_bloc_assignment.csv"), index=False)
    _unused = sorted(set(REGION) - set(_in))
    if _unused:
        print(f"  note: REGION contains {_unused}, absent from this ESS edition; "
              f"ignored")
    print("  bloc assignment written to country_bloc_assignment.csv "
          "(required for reproducibility; Israel is a bloc of one and therefore "
          "falls back to the all-other-countries mean)\n")

    rows = []
    for v, g in sc.groupby("variable"):
        sp = RANGE[v][1] - RANGE[v][0]
        g = g.copy()
        # comparator A: grand mean.
        # AUDIT FIX (P3): the in-sample grand mean is given an advantage that the
        # region comparator is denied, since pred_region below is leave-one-out.
        # The asymmetry favours the comparator over the model, so the released
        # claim was conservative rather than inflated, but the two are not
        # like-for-like. Both are now computed. pred_grand_loo is the matched
        # comparator and is the one to report; pred_grand is retained because
        # every released figure was computed from it.
        g["pred_grand"] = g.survey_mean.mean()
        _gs, _gk = g.survey_mean.sum(), int(g.survey_mean.notna().sum())
        g["pred_grand_loo"] = ((_gs - g.survey_mean) / (_gk - 1)) if _gk > 1 else np.nan
        # comparator B: leave-one-out region mean
        loo = []
        for _, r in g.iterrows():
            peers = g[(g.region == r.region) & (g.cntry != r.cntry)]
            if len(peers) == 0:
                peers = g[g.cntry != r.cntry]
            loo.append(peers.survey_mean.mean())
        g["pred_region"] = loo
        # the model, and the model after removing its per-item mean offset
        g["pred_model"] = g.silicon_mean
        g["pred_debias"] = g.silicon_mean - (g.silicon_mean.mean() - g.survey_mean.mean())
        # optional comparator C: log GDP per capita + region dummies, leave-one-out
        if gdp is not None:
            gg = g.merge(gdp[["cntry", "log_gdp"]], on="cntry", how="left")
            preds = []
            for i in range(len(gg)):
                tr = gg.drop(gg.index[i]).dropna(subset=["log_gdp"])
                te = gg.iloc[[i]]
                if len(tr) < 8 or te.log_gdp.isna().all():
                    preds.append(np.nan)
                    continue
                D = pd.get_dummies(tr.region, drop_first=True).astype(float)
                X = np.column_stack([np.ones(len(tr)), tr.log_gdp.values, D.values])
                beta, *_ = np.linalg.lstsq(X, tr.survey_mean.values, rcond=None)
                dte = pd.get_dummies(te.region).reindex(columns=D.columns,
                                                        fill_value=0).astype(float)
                xte = np.concatenate([[1.0], [float(te.log_gdp.iloc[0])],
                                      dte.values.ravel()])
                preds.append(float(xte @ beta))
            g["pred_gdp"] = preds

        rec = dict(variable=v)
        for name in ["model", "debias", "region", "grand", "grand_loo"] + \
                (["gdp"] if gdp is not None else []):
            p = g[f"pred_{name}"]
            ok = p.notna() & g.survey_mean.notna()
            # AUDIT FIX (P4): this quantity is already divided by the item range
            # sp, so it is a range-normalised MAE, and the print header below
            # already calls it nMAE. The column name mae_* invites a reader or a
            # downstream script to normalise a second time; dividing by sp again
            # turns the median 0.180 into 0.019. country_profile_and_mae.csv names
            # the same quantity norm_mae. The honest name is emitted alongside.
            # mae_* is retained for compatibility with collect_numbers.py and the
            # released tables, and is DEPRECATED: prefer nmae_*.
            _nmae = float(((g.survey_mean - p).abs() / sp)[ok].mean())
            rec[f"mae_{name}"] = _nmae
            rec[f"nmae_{name}"] = _nmae
            # A constant predictor (the grand mean) has an undefined
            # correlation. Zero is the correct substantive reading: it has no
            # cross-national discrimination whatsoever. Recording it as 0
            # rather than NaN keeps the head-to-head counts meaningful.
            if name == "grand_loo":
                # AUDIT NOTE (P3 corollary). The leave-one-out grand mean is
                # (S - x_c)/(k-1), a strictly DECREASING affine function of the
                # very value it predicts, so its correlation with the truth is
                # exactly -1 for every item, by construction. That is an artefact
                # of leaving one out of a constant predictor, not a finding. It
                # must never be reported as the grand mean inverting country
                # rankings, and "the model beats it on r on 42 of 42 items" is
                # vacuous. This arm is a LEVEL comparator only: quote
                # nmae_grand_loo, and take the ranking comparison from the
                # in-sample arm, whose r of 0 encodes the substantive point that
                # a constant predictor cannot rank countries at all.
                rec[f"r_{name}"] = np.nan
            elif ok.sum() < 10:
                rec[f"r_{name}"] = np.nan
            elif p[ok].nunique() <= 1:
                # BUG FIX (retained): testing std() == 0 exactly does not work. A
                # column holding one repeated value returns a variance of order
                # 1e-16 rather than exactly zero, so the equality test failed, the
                # constant array reached pearsonr, and the result was NaN. NaN then
                # loses every ">" comparison silently, which understated the
                # head-to-head count against the grand-mean predictor. nunique()
                # is exact for this case.
                #
                # An EXACTLY constant predictor has an undefined correlation, and
                # zero is the correct substantive reading: no cross-national
                # discrimination whatsoever. It is kept at 0.0 so the head-to-head
                # counts stay meaningful. The manuscript table must nevertheless
                # print this cell as "undefined", not as 0.00, because a reader
                # scanning the r column will otherwise read 0.00 as an estimate.
                rec[f"r_{name}"] = 0.0
            elif p[ok].std() < 1e-10:
                # AUDIT FIX (P5'): a NEAR-constant predictor is a different case.
                # Its correlation is defined but numerically unstable, so
                # recording 0.0 would invent a number rather than report an
                # undefined one. This branch is dormant for the grand mean, which
                # is exactly constant, but becomes live as soon as --gdp is
                # supplied and a leave-one-out fit degenerates. The n_nan warning
                # below then reports it instead of silently absorbing it.
                rec[f"r_{name}"] = np.nan
            else:
                rec[f"r_{name}"] = stats.pearsonr(g.survey_mean[ok], p[ok])[0]
        rows.append(rec)

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "aggregate_benchmark.csv"), index=False)

    labels = [("model", "the model, as produced"),
              ("debias", "the model, per-item bias removed (UPPER BOUND, needs truth)"),
              ("region", "leave-one-out region mean (no model)"),
              ("grand", "grand mean, IN-SAMPLE (no model, cannot rank at all)"),
              ("grand_loo", "grand mean, leave-one-out (matched to the region arm)")]
    if gdp is not None:
        labels.insert(3, ("gdp", "log GDP per capita + region dummies (no model)"))

    print(f"\n  {'predictor':56s} {'median r':>9s} {'median nMAE':>12s}")
    print("  " + "-" * 79)
    for k, lab in labels:
        rr = d[f"r_{k}"].median()
        print(f"  {lab:56s} {rr:+9.4f} {d[f'mae_{k}'].median():12.4f}")

    n_nan = {k: int(d[f"r_{k}"].isna().sum()) for k, _ in labels}
    if any(n_nan.values()):
        print(f"\n  WARNING: undefined correlations by predictor: {n_nan}")
        print("  Head-to-head counts below exclude those items; investigate before")
        print("  quoting the counts.")
    print(f"\n  head-to-head, item by item (out of {len(d)}):")
    for k, lab in labels[2:]:
        # AUDIT FIX (B8): a comparator whose r is undefined for every item must
        # not be given a win count. NaN loses every ">" comparison silently, so
        # the count would print as 0 and read as "the model never wins".
        _rcol = d[f"r_{k}"]
        _rtxt = "  n/a" if _rcol.isna().all() else f"{int((d.r_model > _rcol).sum()):3d}"
        print(f"    model beats {lab.split(' (')[0]:44s} "
              f"on r: {_rtxt}   "
              f"on nMAE: {int((d.nmae_model < d[f'nmae_{k}']).sum()):2d}")
        print(f"    debiased model beats {' ' * 37} "
              f"on nMAE: {int((d.mae_debias < d[f'mae_{k}']).sum()):2d}")

    # AUDIT FIX (P6): the released 64.2% is a RATIO OF MEDIANS, the largest of
    # four defensible estimators. For a claim phrased as "almost all of the level
    # error is a single per-item offset", the median of the per-item reductions is
    # the natural quantity, and it is 59.9%. All four are emitted and named so a
    # quoted figure can be identified.
    _item_red = 1 - d.nmae_debias / d.nmae_model
    _est = [
        ("ratio of medians (as released)",
         1 - d.nmae_debias.median() / d.nmae_model.median()),
        ("ratio of means", 1 - d.nmae_debias.mean() / d.nmae_model.mean()),
        ("MEDIAN of the per-item reductions (recommended)", _item_red.median()),
        ("mean of the per-item reductions", _item_red.mean()),
    ]
    red = _est[2][1]
    print("\n  Removing a single per-item offset cuts the model's error by:")
    for _lab, _val in _est:
        print(f"    {100 * _val:5.1f}%   {_lab}")
    print(f"  Report {100 * red:.1f}% and name the estimator. Quoting the ratio of")
    print("  medians without naming it invites a reader who recomputes per item to")
    print("  get a figure five points lower.")
    pd.DataFrame([dict(estimator=_l, value=_v) for _l, _v in _est] + [
        dict(estimator="items where removing the offset helps",
             value=int((_item_red > 0).sum())),
        dict(estimator="items where removing the offset hurts",
             value=int((_item_red < 0).sum())),
    ]).to_csv(os.path.join(OUT, "offset_reduction_estimators.csv"), index=False)
    print("  That is the quantitative form of 'the levels are displaced but the")
    print("  country-to-country pattern is largely intact'. It also means the")
    print("  headline error is dominated by a constant, not by country-specific")
    print("  mistakes — which is what a group-level prior with a fixed offset")
    print("  would produce, and is consistent with the compression and bias")
    print("  results reported elsewhere.")
    print("\n  IMPORTANT FRAMING: the region and grand-mean comparators both")
    print("  require survey data from other countries. State that explicitly.")
    print("  The comparison answers 'should a researcher who already holds real")
    print("  data use the model to extend it', which is the realistic use case,")
    print("  not 'is the model better than nothing'.")

# ===========================================================================
print("\n" + "=" * 78)
print("2. LLAMA THIRD-PERSON COVERAGE SENSITIVITY")
print("=" * 78)
cov_p = os.path.join(OUT, "coverage_by_item_llama_3p.csv")
sc3 = scatter("llama_3p")
if not os.path.exists(cov_p) or sc3 is None:
    print(f"  needs {cov_p} (written by analyze_v13b.py) and the llama_3p")
    print("  scatter file — skipping.")
else:
    cov = pd.read_csv(cov_p)
    cov.columns = ["variable", "coverage"]
    sc3 = sc3.dropna(subset=["survey_mean", "silicon_mean"]).merge(cov, on="variable")
    print(f"  {'threshold':22s} {'items kept':>10s} {'median r':>9s} {'median nMAE':>12s}")
    print("  " + "-" * 56)
    rows = []
    for thr in (0.0, 0.60, 0.70, 0.80, 0.90, 0.95):
        s = sc3[sc3.coverage >= thr]
        per = {}
        for v, g in s.groupby("variable"):
            if len(g) >= 10 and g.survey_mean.std() > 0 and g.silicon_mean.std() > 0:
                sp = RANGE[v][1] - RANGE[v][0]
                per[v] = (stats.pearsonr(g.survey_mean, g.silicon_mean)[0],
                          ((g.survey_mean - g.silicon_mean).abs() / sp).mean())
            else:
                per[v] = (np.nan, np.nan)
        if not per:
            continue
        rs = np.array([x[0] for x in per.values()], float)
        ms = np.array([x[1] for x in per.values()], float)
        print(f"  coverage >= {thr:.2f}      {len(per):10d} "
              f"{np.nanmedian(rs):+9.4f} {np.nanmedian(ms):12.4f}")
        rows.append(dict(threshold=thr, n_items=len(per),
                         median_r=np.nanmedian(rs), median_nmae=np.nanmedian(ms)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "llama3p_coverage_sensitivity.csv"),
                              index=False)
    if len(rows) >= 3:
        t = pd.DataFrame(rows)
        loose = float(t.iloc[0].median_r)
        # N4a FIX: this averaged the three tightest thresholds while labelling
        # itself "at coverage >= 0.80". The 0.80 row alone is a different number,
        # and the mislabelled value would have been copied into the text as a
        # false statement. Report the single 0.80 row, and the mean over tight
        # thresholds separately and under its own name.
        row80 = t[t.threshold == 0.80]
        at80 = float(row80.median_r.iloc[0]) if len(row80) else np.nan
        tight = float(t[t.threshold >= 0.80].median_r.mean())
        print(f"\n  median r at no threshold:            {loose:+.4f}")
        print(f"  median r at the 0.80 threshold ONLY: {at80:+.4f}")
        print(f"  mean of the three tightest rows:     {tight:+.4f}")
        print("  Note the series is NOT monotone: it falls to about zero and then")
        print("  recovers slightly at the sparsest thresholds (n = 12 and n = 7).")
        print("  Neither figure nor text may describe it as monotone.")
        if tight > loose + 0.05:
            print("""
  The correlation RISES as the threshold tightens, so the condition's weak
  headline number is partly a parsing failure on specific items rather than a
  uniform failure to recover. Report the restricted estimate alongside the full
  one and name the dropped items.""")
        elif tight < loose - 0.05:
            print("""
  The correlation FALLS as the threshold tightens. This is the opposite of a
  parsing artefact: the items this condition parsed most reliably are the ones it
  recovered WORST, so the poorly covered items were carrying what little signal
  there was. The explanation is that the low-coverage items are the wide 0-10
  scales, and those are precisely the items on which cross-national recovery is
  otherwise strongest, so third-person framing fails to produce a parsable answer
  on exactly the items that would have recovered best.

  Two consequences for the write-up. First, better parsing would NOT rescue this
  condition, so do not present the coverage problem as an excuse for the weak
  result. Second, the usable subset is a BIASED subset, not a cleaner one, so the
  full-sample estimate is the one to report, with the selection stated
  explicitly.""")
        else:
            print("""
  The correlation is flat across thresholds, so the weak result is uniform across
  items rather than driven by the poorly covered ones. Report the full-sample
  estimate and note that restricting to well-covered items does not change it.""")

# ===========================================================================
print("\n" + "=" * 78)
print("3. THE euftf ANOMALY — which construct does the model appear to answer?")
print("=" * 78)
if sc is None:
    print("  qwen_1p scatter not available — skipping.")
else:
    piv_h = sc.pivot_table(index="cntry", columns="variable", values="survey_mean")
    piv_s = sc.pivot_table(index="cntry", columns="variable", values="silicon_mean")
    if "euftf" not in piv_s.columns:
        print("  euftf not present in this condition — skipping.")
    else:
        target = piv_s["euftf"]
        cors = {}
        for v in piv_h.columns:
            x, y = piv_h[v], target
            m = x.notna() & y.notna()
            if m.sum() >= 10 and x[m].std() > 0 and y[m].std() > 0:
                cors[v] = stats.pearsonr(x[m], y[m])[0]
        cs = pd.Series(cors).sort_values()
        own = cs.get("euftf", np.nan)
        print(f"  the model's euftf country means correlate with HUMAN euftf at "
              f"{own:+.4f}")
        print(f"\n  most positively correlated human items:")
        for v, r in cs.nlargest(5).items():
            print(f"    {v:10s} {r:+.4f}")
        print(f"  most negatively correlated human items:")
        for v, r in cs.nsmallest(5).items():
            print(f"    {v:10s} {r:+.4f}")
        cs.rename("r_with_silicon_euftf").to_csv(
            os.path.join(OUT, "euftf_diagnostic.csv"))
        print("\n  Read this as a hypothesis generator, not a test. euftf is a")
        print("  bipolar item whose numeric direction is a convention (0 = unification")
        print("  has gone too far, 10 = should go further), so an inverted reading is")
        print("  possible without the item being reverse-coded in the ESS sense. If the")
        print("  model's euftf tracks a substantively adjacent human item more strongly")
        print("  than it tracks human euftf, that is the substantive explanation the")
        print("  revision plan asks for, and it should be stated as one item's")
        print("  interpretation rather than generalised.")

print("\n" + "=" * 78)
print(f"Tables written to {os.path.abspath(OUT)}/")
print("=" * 78)
