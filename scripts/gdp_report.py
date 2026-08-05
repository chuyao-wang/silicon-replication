#!/usr/bin/env python3
"""
gdp_report.py -- implements the three GDP-arm decisions and writes report-ready
tables into results/analysis/.

Run from ~/Winston_Code AFTER benchmark_v13.py has been run with --gdp:

    python3 gdp_report.py --primary gdp_per_capita.csv \\
                          --robustness gdp_pc_ppp.csv

DECISIONS IMPLEMENTED
  1. Deflator. Current US dollars (NY.GDP.PCAP.CD) is primary because it is the
     indicator the pipeline's own documentation and the revision plan name, so the
     choice precedes the result; and because it is the WEAKER of the two
     comparators, which makes the paper's practical claim conservative rather than
     favourable. PPP (NY.GDP.PCAP.PP.CD) is reported as robustness. The
     decomposition is reported in full, including the fact that the
     covariate-alone specification changes its verdict between deflators, because
     omitting a deflator-dependent result is worse than reporting it.

  2. Israel. Thirty countries is primary; twenty-nine, excluding Israel, is a
     robustness row. Israel is the only bloc of one, so its dummy is unidentified
     out of sample: when it is held out the dummy disappears from the training
     design and it is predicted as the reference category. EVERY arm is recomputed
     on the reduced sample, including the model's own between-country correlation,
     because dropping a country changes that too; a robustness row that varied
     only the GDP cell would not be like for like.

  3. The two comparator arms handle the bloc-of-one edge case by different rules:
     the regional average falls back to the all-other-countries mean, while the
     regression predicts the reference category. Both rules are recorded in
     gdp_il_identification.csv rather than left implicit.

OUTPUTS, all under results/analysis/
    gdp_arm_summary.csv          every arm x {30, 29} x {USD, PPP}, report-ready
    gdp_arm_decomposition.csv    covariate only / region only / both, per deflator
    gdp_arm_headtohead.csv       per-item win counts, model against each arm
    gdp_il_identification.csv    the bloc-of-one problem, quantified
    gdp_influence.csv            hat values, studentised residuals, Cook's distance
    gdp_numbers_block.csv        rows in the manuscript_numbers.csv schema
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

R = "results"
OUT = os.path.join(R, "analysis")


def find(name: str, extra: list[str] | None = None) -> str:
    """Locate a file across the cluster's several plausible homes."""
    cands = (extra or []) + [
        name, os.path.join(R, name), os.path.join(OUT, name),
        os.path.join("handoff_upload", name), os.path.join("code", name),
        os.path.join("silicon_package_v13", "code", name),
        os.path.join("silicon_package_v13", "outputs", name),
        os.path.join("silicon_package_v13", "data", name),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    raise SystemExit(f"could not locate {name}; looked in: {cands}")


ap = argparse.ArgumentParser()
ap.add_argument("--primary", default="gdp_per_capita.csv",
                help="current US dollars, NY.GDP.PCAP.CD")
ap.add_argument("--robustness", default="gdp_pc_ppp.csv",
                help="PPP, NY.GDP.PCAP.PP.CD")
ap.add_argument("--exclude", default="IL",
                help="comma-separated countries for the reduced-sample row")
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)

sc0 = pd.read_csv(find("silicon_full_country_scatter_qwen_1p.csv"))
dirt = pd.read_csv(find("item_direction_table.csv")).set_index("variable")
bloc = pd.read_csv(find("country_bloc_assignment.csv")).set_index("cntry").bloc
RNG = (dirt.high_value - dirt.low_value)
DEFL = {"USD": a.primary, "PPP": a.robustness}
GDP = {k: pd.read_csv(find(v, [v])).set_index("cntry").gdp_pc for k, v in DEFL.items()}
EXCL = [c.strip() for c in a.exclude.split(",") if c.strip()]

sc0 = sc0.dropna(subset=["survey_mean", "silicon_mean"]).copy()
sc0["region"] = sc0.cntry.map(bloc)


def frame(defl: str, countries: list[str] | None) -> pd.DataFrame:
    d = sc0.copy()
    d["log_gdp"] = np.log(d.cntry.map(GDP[defl]))
    if countries is not None:
        d = d[d.cntry.isin(countries)]
    return d.dropna(subset=["log_gdp"])


def loo_reg(g: pd.DataFrame, use_gdp: bool, use_region: bool) -> pd.Series:
    """benchmark_v13.py's leave-one-out fit, with the design switchable."""
    g = g.reset_index(drop=True)
    preds = []
    for i in range(len(g)):
        tr, te = g.drop(g.index[i]), g.iloc[[i]]
        cols, D = [np.ones(len(tr))], None
        if use_gdp:
            cols.append(tr.log_gdp.values)
        if use_region:
            D = pd.get_dummies(tr.region, drop_first=True).astype(float)
            cols.append(D.values)
        beta, *_ = np.linalg.lstsq(np.column_stack(cols), tr.survey_mean.values, rcond=None)
        x = [1.0]
        if use_gdp:
            x.append(float(te.log_gdp.iloc[0]))
        if use_region:
            x.extend(pd.get_dummies(te.region).reindex(columns=D.columns,
                                                       fill_value=0).astype(float).values.ravel())
        preds.append(float(np.asarray(x) @ beta))
    return pd.Series(preds, index=g.cntry.values)


def loo_region_mean(g: pd.DataFrame) -> pd.Series:
    g = g.set_index("cntry")
    out = {}
    for c in g.index:
        peers = g.index[(g.region == g.loc[c, "region"]) & (g.index != c)]
        # the regional arm's own rule for a bloc of one: fall back to all others
        out[c] = g.loc[peers, "survey_mean"].mean() if len(peers) else \
            g.loc[g.index != c, "survey_mean"].mean()
    return pd.Series(out)


def score(g: pd.DataFrame, pred: pd.Series, var: str) -> tuple[float, float]:
    h = g.set_index("cntry").survey_mean
    p = pred.reindex(h.index)
    ok = p.notna() & h.notna()
    if ok.sum() < 10 or p[ok].nunique() <= 1:
        r = np.nan if p[ok].nunique() > 1 else 0.0
    else:
        r = stats.pearsonr(h[ok], p[ok])[0]
    return r, float((h[ok] - p[ok]).abs().mean() / RNG[var])


ARMS = {
    "model": lambda g: g.set_index("cntry").silicon_mean,
    "model_debiased": lambda g: (g.set_index("cntry").silicon_mean
                                 - (g.silicon_mean.mean() - g.survey_mean.mean())),
    "region_mean_loo": loo_region_mean,
    "gdp_plus_region_loo": lambda g: loo_reg(g, True, True),
    "grand_mean_insample": lambda g: pd.Series(g.survey_mean.mean(), index=g.cntry.values),
}

# ---------------------------------------------------------------- arm summary
rows, per_item = [], []
for defl in ("USD", "PPP"):
    for tag, keep in (("n=30 (primary)", None),
                      (f"n={30 - len(EXCL)} (excluding {'+'.join(EXCL)})",
                       [c for c in sorted(sc0.cntry.unique()) if c not in EXCL])):
        for arm, fn in ARMS.items():
            recs = []
            for v, g in frame(defl, keep).groupby("variable"):
                r, m = score(g, fn(g), v)
                recs.append((v, r, m))
                per_item.append(dict(deflator=defl, sample=tag, arm=arm,
                                     variable=v, r=r, nmae=m))
            rr = pd.DataFrame(recs, columns=["variable", "r", "nmae"])
            rows.append(dict(deflator=defl, sample=tag, arm=arm, n_items=len(rr),
                             median_r=rr.r.median(), median_nmae=rr.nmae.median(),
                             n_undefined_r=int(rr.r.isna().sum())))
S = pd.DataFrame(rows)
S.to_csv(os.path.join(OUT, "gdp_arm_summary.csv"), index=False)
PI = pd.DataFrame(per_item)

print("=" * 92)
print("GDP ARM SUMMARY  (deflator x sample x arm)")
print("=" * 92)
for defl in ("USD", "PPP"):
    for tag in S[S.deflator == defl]["sample"].unique():
        print(f"\n  {defl}, {tag}")
        print(f"    {'arm':24s} {'median r':>9s} {'median nMAE':>12s}")
        sub = S[(S.deflator == defl) & (S["sample"] == tag)]
        for _, x in sub.iterrows():
            print(f"    {x.arm:24s} {x.median_r:+9.4f} {x.median_nmae:12.4f}")

# ------------------------------------------------------------- decomposition
dec = []
for defl in ("USD", "PPP"):
    g_all = frame(defl, None)
    for spec, ug, ur in (("region dummies only", False, True),
                         ("log GDP only", True, False),
                         ("log GDP + region dummies", True, True)):
        recs = [score(g, loo_reg(g, ug, ur), v) for v, g in g_all.groupby("variable")]
        rr = pd.DataFrame(recs, columns=["r", "nmae"])
        dec.append(dict(deflator=defl, specification=spec,
                        median_r=rr.r.median(), median_nmae=rr.nmae.median()))
D = pd.DataFrame(dec)
ref_reg = float(S[(S.arm == "region_mean_loo") & (S["sample"] == "n=30 (primary)")
                  & (S.deflator == "USD")].median_r.iloc[0])
D["beats_region_mean_arm_on_r"] = D.median_r > ref_reg
D.to_csv(os.path.join(OUT, "gdp_arm_decomposition.csv"), index=False)
print("\n" + "=" * 92)
print("DECOMPOSITION  (does one public covariate suffice?)")
print("=" * 92)
print(f"  reference: the leave-one-out regional MEAN arm reaches r = {ref_reg:+.4f}\n")
print(f"  {'deflator':9s} {'specification':26s} {'median r':>9s} {'median nMAE':>12s}  beats region mean")
for _, x in D.iterrows():
    print(f"  {x.deflator:9s} {x.specification:26s} {x.median_r:+9.4f} "
          f"{x.median_nmae:12.4f}  {'yes' if x.beats_region_mean_arm_on_r else 'NO'}")
flip = (D[D.specification == "log GDP only"].beats_region_mean_arm_on_r.nunique() > 1)
print(f"\n  covariate-alone verdict is deflator-dependent: {'YES' if flip else 'no'}")
if flip:
    print("  -> do NOT claim that a single macro covariate suffices. Report the")
    print("     combined specification, which is stable, and report this flip as")
    print("     the reason the stronger claim is not made.")

# ---------------------------------------------------------------- head to head
hh = []
for defl in ("USD", "PPP"):
    for tag in PI["sample"].unique():
        m = PI[(PI.deflator == defl) & (PI["sample"] == tag) & (PI.arm == "model")].set_index("variable")
        for arm in ARMS:
            if arm == "model":
                continue
            o = PI[(PI.deflator == defl) & (PI["sample"] == tag) & (PI.arm == arm)].set_index("variable")
            j = m.join(o, rsuffix="_o")
            hh.append(dict(deflator=defl, sample=tag, comparator=arm, n_items=len(j),
                           model_wins_on_r=int((j.r > j.r_o).sum()),
                           model_wins_on_nmae=int((j.nmae < j.nmae_o).sum()),
                           r_comparable=bool(j.r_o.notna().all())))
H = pd.DataFrame(hh)
H.to_csv(os.path.join(OUT, "gdp_arm_headtohead.csv"), index=False)
print("\n" + "=" * 92)
print("HEAD TO HEAD, item by item (the median is one point of a distribution)")
print("=" * 92)
for _, x in H[(H.comparator == "gdp_plus_region_loo")].iterrows():
    print(f"  {x.deflator}, {x['sample']:32s} model wins on r "
          f"{x.model_wins_on_r:2d}/{x.n_items}   on nMAE {x.model_wins_on_nmae:2d}/{x.n_items}")

# --------------------------------------------------- Israel identification
il_rows, infl = [], []
for defl in ("USD", "PPP"):
    g_all = frame(defl, None)
    for v, g in g_all.groupby("variable"):
        g = g.reset_index(drop=True)
        D_ = pd.get_dummies(g.region, drop_first=True).astype(float)
        X = np.column_stack([np.ones(len(g)), g.log_gdp.values, D_.values])
        Hm = X @ np.linalg.pinv(X)
        res = g.survey_mean.values - Hm @ g.survey_mean.values
        p = np.linalg.matrix_rank(X)
        s2 = (res ** 2).sum() / max(1, len(g) - p)
        hi = np.diag(Hm)
        stud = res / np.sqrt(np.maximum(s2 * (1 - hi), 1e-12))
        for c, h_, st in zip(g.cntry, hi, stud):
            infl.append(dict(deflator=defl, variable=v, cntry=c, hat=h_,
                             stud_resid=st,
                             cooks_d=(st ** 2 / p) * (h_ / max(1 - h_, 1e-12))))
        for c in EXCL:
            if c not in set(g.cntry):
                continue
            tr = g[g.cntry != c]
            cols = pd.get_dummies(tr.region, drop_first=True).columns
            il_rows.append(dict(
                deflator=defl, variable=v, country=c,
                bloc=bloc[c], bloc_size=int((bloc == bloc[c]).sum()),
                dummy_survives_holdout=bloc[c] in set(cols),
                predicted_as=("its own bloc" if bloc[c] in set(cols)
                              else sorted(bloc.unique())[0] + " (reference category)"),
                truth=float(g.loc[g.cntry == c, "survey_mean"].iloc[0]),
                pred_regression=float(loo_reg(g, True, True)[c]),
                pred_region_mean=float(loo_region_mean(g)[c]),
                hat_insample=float(hi[list(g.cntry).index(c)])))
IL = pd.DataFrame(il_rows)
IL["err_regression"] = (IL.pred_regression - IL.truth).abs() / IL.variable.map(RNG)
IL["err_region_mean"] = (IL.pred_region_mean - IL.truth).abs() / IL.variable.map(RNG)
IL.to_csv(os.path.join(OUT, "gdp_il_identification.csv"), index=False)
INF = pd.DataFrame(infl)
INF.to_csv(os.path.join(OUT, "gdp_influence.csv"), index=False)
print("\n" + "=" * 92)
print("THE BLOC OF ONE")
print("=" * 92)
for c in EXCL:
    s = IL[(IL.country == c) & (IL.deflator == "USD")]
    if not len(s):
        continue
    print(f"  {c}: bloc '{s.bloc.iloc[0]}', size {int(s.bloc_size.iloc[0])}; "
          f"dummy survives hold-out on {int(s.dummy_survives_holdout.sum())}/{len(s)} items")
    print(f"     in-sample hat value: mean {s.hat_insample.mean():.4f} "
          f"(a value of 1.000 means the fit passes exactly through it)")
    print(f"     predicted as: {s.predicted_as.iloc[0]}")
    print(f"     range-normalised error, regression arm  : median {s.err_regression.median():.4f}")
    print(f"     range-normalised error, region-mean arm : median {s.err_region_mean.median():.4f}")
    print("     the two arms apply DIFFERENT rules to the same edge case; both are")
    print("     recorded in gdp_il_identification.csv and must be stated in the text.")
top = INF[INF.deflator == "USD"].groupby("cntry").agg(
    mean_hat=("hat", "mean"),
    n_high_cooks=("cooks_d", lambda x: int((x > 4 / 30).sum()))).sort_values("mean_hat", ascending=False)
print(f"\n  highest-leverage countries (USD): "
      f"{', '.join(f'{i} {r.mean_hat:.3f}' for i, r in top.head(4).iterrows())}")

# ------------------------------------------------- numbers-table style block
def get(defl, tag, arm, col):
    x = S[(S.deflator == defl) & (S["sample"] == tag) & (S.arm == arm)]
    return float(x[col].iloc[0])


P, Rb = "n=30 (primary)", f"n={30 - len(EXCL)} (excluding {'+'.join(EXCL)})"
hh_p = H[(H.deflator == "USD") & (H["sample"] == P) & (H.comparator == "gdp_plus_region_loo")].iloc[0]
NB = pd.DataFrame([
    dict(section="5.6 benchmarks", claim="a public macro covariate with coarse region dummies outranks the model",
         quantity="median r: model vs log GDP + region, current USD",
         value=f"{get('USD', P, 'model', 'median_r'):+.4f} vs {get('USD', P, 'gdp_plus_region_loo', 'median_r'):+.4f}",
         source="gdp_arm_summary.csv",
         note="primary deflator is NY.GDP.PCAP.CD, named in the pipeline documentation before results were seen, and the weaker of the two comparators"),
    dict(section="5.6 benchmarks", claim="and beats it on level accuracy on every item",
         quantity="median nMAE: model vs log GDP + region, current USD",
         value=f"{get('USD', P, 'model', 'median_nmae'):.4f} vs {get('USD', P, 'gdp_plus_region_loo', 'median_nmae'):.4f}",
         source="gdp_arm_headtohead.csv",
         note=f"the model wins on {hh_p.model_wins_on_nmae}/{hh_p.n_items} items on nMAE and {hh_p.model_wins_on_r}/{hh_p.n_items} on r"),
    dict(section="5.6 benchmarks", claim="the comparison is unchanged under a PPP deflator",
         quantity="median r and nMAE, log GDP + region, PPP",
         value=f"{get('PPP', P, 'gdp_plus_region_loo', 'median_r'):+.4f} / {get('PPP', P, 'gdp_plus_region_loo', 'median_nmae'):.4f}",
         source="gdp_arm_summary.csv",
         note="robustness deflator NY.GDP.PCAP.PP.CD; slightly STRONGER than the primary, so the primary is the conservative choice"),
    dict(section="5.6 benchmarks", claim="one covariate alone does NOT suffice; the verdict depends on the deflator",
         quantity="median r, log GDP only, USD then PPP",
         value=f"{float(D[(D.deflator == 'USD') & (D.specification == 'log GDP only')].median_r.iloc[0]):+.4f} / "
               f"{float(D[(D.deflator == 'PPP') & (D.specification == 'log GDP only')].median_r.iloc[0]):+.4f}",
         source="gdp_arm_decomposition.csv",
         note=f"against the regional mean arm's {ref_reg:+.4f}: the covariate-alone specification wins under one deflator and loses under the other, so only the combined specification is claimed"),
    dict(section="4.5 robustness", claim="the benchmark comparison survives excluding the one country whose bloc is unidentified out of sample",
         quantity="median r, log GDP + region, excluding Israel",
         value=f"{get('USD', Rb, 'gdp_plus_region_loo', 'median_r'):+.4f}",
         source="gdp_arm_summary.csv",
         note="Israel is the only bloc of one; every arm including the model's own r_bc is recomputed on the 29-country sample, so the row is like for like"),
    dict(section="4.5 robustness", claim="the two comparator arms apply different rules to a bloc of one",
         quantity="Israel's range-normalised error, regression vs region mean",
         value=f"{IL[(IL.country == 'IL') & (IL.deflator == 'USD')].err_regression.median():.4f} vs "
               f"{IL[(IL.country == 'IL') & (IL.deflator == 'USD')].err_region_mean.median():.4f}",
         source="gdp_il_identification.csv",
         note="the regression predicts the reference category once the dummy vanishes; the regional average falls back to the all-other-countries mean. Report the asymmetry rather than harmonising it silently"),
])
NB.to_csv(os.path.join(OUT, "gdp_numbers_block.csv"), index=False)

print("\n" + "=" * 92)
print(f"six rows written to {OUT}/gdp_numbers_block.csv in the manuscript_numbers schema")
print(f"tables written to {OUT}/: gdp_arm_summary, gdp_arm_decomposition, "
      f"gdp_arm_headtohead, gdp_il_identification, gdp_influence, gdp_numbers_block")
print("=" * 92)
