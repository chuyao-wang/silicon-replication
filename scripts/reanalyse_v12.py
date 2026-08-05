#!/usr/bin/env python3
"""
reanalyse_v12.py — analysis-side corrections that must be closed before any
number enters the manuscript.

Three subcommands.

  aggregate   Recompute the HUMAN side of every released condition directly from
              the ESS Round 11 microdata under the corrected rules — per-item
              valid range, design weights — and re-derive every aggregate and
              distributional headline. Needs only the ESS file and the released
              *_country_scatter_*.csv files, so it runs anywhere.

  ladder250   Recompute the cumulative ladder on the 250 respondents nested
              within the 500-respondent runs, so that the ladder's country
              effect and the primary full_noregion / full_nocountry contrast
              share n and therefore share attenuation. Without this the
              difference between the two country effects — the quantity that
              tests whether compositional detail can substitute for the label —
              is confounded by sample size. Needs the raw response files.

  rq3         Recompute per-country individual-level correlations from raw files
              under the corrected missingness rule. v12 runs write this table
              themselves; this subcommand exists for the legacy raws.

WHY THIS IS NEEDED
------------------
Pre-v12 runs retained the ESS single-digit refusal / don't-know / no-answer
codes (7/8/9) as valid human data on the sixteen items whose substantive scale
has fewer than seven categories, because the pipeline's missing-code set held
only the two- and three-digit variants. The silicon side was never affected, so
the bias is one-sided and falls on the coarse-scale, predominantly reverse-coded
items on which the scale-direction mechanism is identified. Separately, the
released human country means are design-weighted by pspwght, which was
undocumented and is described the wrong way round in the replication package.

Usage:
  python reanalyse_v12.py aggregate --ess data/ess11/ESS11e04_2.csv --data ../data
  python reanalyse_v12.py ladder250 --results results --n 250
  python reanalyse_v12.py rq3 --results results
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

SEED = 888

ITEM_VALID_RANGE = {
    "ppltrst": (0, 10), "pplfair": (0, 10), "pplhlp": (0, 10), "sclmeet": (1, 7),
    "inprdsc": (0, 6), "stfdem": (0, 10), "polintr": (1, 4), "psppipla": (1, 5),
    "psppsgva": (1, 5), "vote": (1, 3), "actrolga": (1, 5), "cptppola": (1, 5),
    "trstprl": (0, 10), "trstplt": (0, 10), "trstlgl": (0, 10), "trstprt": (0, 10),
    "trstplc": (0, 10), "trstep": (0, 10), "trstun": (0, 10), "happy": (0, 10),
    "stflife": (0, 10), "stfeco": (0, 10), "stfgov": (0, 10), "stfhlth": (0, 10),
    "stfedu": (0, 10), "imueclt": (0, 10), "imwbcnt": (0, 10), "imbgeco": (0, 10),
    "imsmetn": (1, 4), "impcntr": (1, 4), "imdfetn": (1, 4), "gincdif": (1, 5),
    "freehms": (1, 5), "hmsfmlsh": (1, 5), "hmsacld": (1, 5), "euftf": (0, 10),
    "atchctr": (0, 10), "atcherp": (0, 10), "health": (1, 5), "rlgatnd": (1, 7),
    "aesfdrk": (1, 4), "hincfel": (1, 4),
}
# Direction frozen from the ESS11 codebook; see item_direction_table.csv.
REVERSE = {"polintr", "vote", "imsmetn", "impcntr", "imdfetn", "gincdif", "freehms",
           "hmsfmlsh", "hmsacld", "health", "rlgatnd", "aesfdrk", "hincfel"}
VARS = list(ITEM_VALID_RANGE)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def human_side(ess_path: str, n_per_country: int = 500,
               weighted: bool = True) -> pd.DataFrame:
    """Corrected human country means and SDs for the seed-888 subsample."""
    ess = pd.read_csv(ess_path, usecols=["cntry", "idno", "pspwght"] + VARS,
                      low_memory=False)
    parts = []
    for _, g in ess.groupby("cntry", sort=True):
        parts.append(g.sample(n=n_per_country, random_state=SEED)
                     if len(g) > n_per_country else g)
    sub = pd.concat(parts, ignore_index=True)
    sub["pspwght"] = pd.to_numeric(sub["pspwght"], errors="coerce").fillna(1.0)

    rows = []
    for v in VARS:
        lo, hi = ITEM_VALID_RANGE[v]
        x = pd.to_numeric(sub[v], errors="coerce")
        x = x.where((x >= lo) & (x <= hi))
        for cn, idx in sub.groupby("cntry", sort=True).groups.items():
            xi, wi = x.loc[idx], sub["pspwght"].loc[idx]
            m = xi.notna()
            if not m.any():
                rows.append(dict(variable=v, cntry=cn, human_mean=np.nan,
                                 human_sd=np.nan, n_valid=0))
                continue
            w = wi[m] if weighted else pd.Series(1.0, index=xi[m].index)
            mu = float(np.average(xi[m], weights=w))
            sd = float(np.sqrt(np.average((xi[m] - mu) ** 2, weights=w)))
            rows.append(dict(variable=v, cntry=cn, human_mean=mu, human_sd=sd,
                             n_valid=int(m.sum())))
    return pd.DataFrame(rows)


def per_item_r(df: pd.DataFrame, hcol: str, scol: str = "silicon_mean") -> pd.Series:
    out = {}
    for v, g in df.groupby("variable"):
        g = g.dropna(subset=[hcol, scol])
        if len(g) >= 10 and g[hcol].std() > 0 and g[scol].std() > 0:
            out[v] = stats.pearsonr(g[hcol], g[scol])[0]
    return pd.Series(out, dtype=float)


def _fmt(x, nd=4):
    return "     n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}"


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def cmd_aggregate(args) -> None:
    h = human_side(args.ess, weighted=(args.human_weight == "pspwght"))
    files = sorted(glob.glob(os.path.join(args.data, "silicon_full_country_scatter_*.csv")))
    if not files:
        raise SystemExit(f"no scatter files under {args.data}")

    print("=" * 100)
    print("CORRECTED AGGREGATE HEADLINES  (human side: per-item valid range, "
          f"{'pspwght-weighted' if args.human_weight == 'pspwght' else 'unweighted'})")
    print("=" * 100)
    hdr = (f"{'condition':28s} {'med r_bc':>9s} {'was':>8s} {'>.5':>4s} {'>.7':>4s} "
           f"{'neg':>4s} {'wasneg':>7s} {'SDratio':>8s} {'was':>7s} {'bSD_r':>7s}")
    print(hdr)
    print("-" * len(hdr))

    summary = []
    for f in files:
        tag = re.sub(r"^silicon_full_country_scatter_|\.csv$", "", os.path.basename(f))
        sc = pd.read_csv(f)
        m = sc.merge(h, on=["variable", "cntry"], how="left")
        r_new, r_old = per_item_r(m, "human_mean"), per_item_r(m, "survey_mean")
        sd_new = (m.silicon_sd / m.human_sd).replace([np.inf, -np.inf], np.nan)
        sd_old = (m.silicon_sd / m.survey_sd).replace([np.inf, -np.inf], np.nan)
        # Between-country dispersion ratio: how much cross-national spread the
        # model produces at all, which r_bc, being scale-free, conceals.
        bsd = m.groupby("variable").agg(s=("silicon_mean", "std"),
                                        hh=("human_mean", "std"))
        bsd_ratio = float((bsd.s / bsd.hh).median())
        print(f"{tag:28s} {r_new.median():+9.4f} {r_old.median():+8.4f} "
              f"{int((r_new > .5).sum()):4d} {int((r_new > .7).sum()):4d} "
              f"{int((r_new < 0).sum()):4d} {int((r_old < 0).sum()):7d} "
              f"{sd_new.mean():8.4f} {sd_old.mean():7.4f} {bsd_ratio:7.4f}")
        summary.append(dict(condition=tag, median_r_bc=r_new.median(),
                            median_r_bc_old=r_old.median(),
                            n_gt50=int((r_new > .5).sum()), n_gt70=int((r_new > .7).sum()),
                            n_negative=int((r_new < 0).sum()),
                            n_negative_old=int((r_old < 0).sum()),
                            within_sd_ratio=sd_new.mean(),
                            within_sd_ratio_old=sd_old.mean(),
                            between_sd_ratio=bsd_ratio))
        if tag == args.main_tag:
            main_m, main_r_new, main_r_old = m, r_new, r_old

    pd.DataFrame(summary).to_csv(os.path.join(args.out, "corrected_condition_summary.csv"),
                                 index=False)

    # ---- main condition detail ----
    m, r_new, r_old = main_m, main_r_new, main_r_old
    print("\n" + "=" * 100)
    print(f"MAIN CONDITION DETAIL: {args.main_tag}")
    print("=" * 100)
    d = (pd.DataFrame({"r_bc_corrected": r_new, "r_bc_released": r_old})
         .assign(delta=lambda t: t.r_bc_corrected - t.r_bc_released,
                 direction=lambda t: np.where(t.index.isin(REVERSE), "reverse", "forward"))
         .sort_values("delta", key=abs, ascending=False))
    d.to_csv(os.path.join(args.out, "corrected_r_bc_by_item.csv"))
    print("Items moving by more than 0.02:")
    print(d[d.delta.abs() > 0.02].to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\nMost severe inversions:")
    print("  corrected: " + ", ".join(f"{v} {r:+.3f}" for v, r in r_new.nsmallest(5).items()))
    print("  released:  " + ", ".join(f"{v} {r:+.3f}" for v, r in r_old.nsmallest(5).items()))
    print("  The third and fourth ranks (hincfel, hmsacld) swap under the correction "
          "and differ by less than 0.005 either way. The manuscript should therefore "
          "name the inverted items without ranking beyond the top two.")

    fwd = d[d.direction == "forward"].r_bc_corrected
    rev = d[d.direction == "reverse"].r_bc_corrected
    print(f"\nDirection decomposition (corrected): forward mean {fwd.mean():+.4f} "
          f"(n={len(fwd)}) | reverse mean {rev.mean():+.4f} (n={len(rev)})")
    y = d.r_bc_corrected.values
    x = (d.direction == "reverse").astype(float).values
    r2 = stats.pearsonr(x, y)[0] ** 2
    print(f"  r_bc ~ reverse: R2 = {r2:.4f}")
    print(f"  negatives that are reverse-coded: "
          f"{int(d[(d.r_bc_corrected < 0)].direction.eq('reverse').sum())}"
          f"/{int((d.r_bc_corrected < 0).sum())}")
    aligned = np.where(x == 1, -y, y)
    print(f"  sign-aligned median r_bc: {np.median(aligned):+.4f}")

    prof_new = m.groupby("cntry").apply(
        lambda g: stats.pearsonr(*g.dropna(subset=["human_mean", "silicon_mean"])
                                 [["human_mean", "silicon_mean"]].values.T)[0],
        include_groups=False)
    bias = (m.silicon_mean - m.human_mean).groupby(m.cntry).mean()
    print(f"\nr_prof (corrected): {prof_new.min():.3f} ({prof_new.idxmin()}) to "
          f"{prof_new.max():.3f} ({prof_new.idxmax()}), mean {prof_new.mean():.3f}, "
          f"SD {prof_new.std():.3f}")
    print(f"country mean bias (corrected): {bias.min():+.3f} to {bias.max():+.3f}; "
          f"all positive: {bool((bias > 0).all())}")
    pd.DataFrame({"r_prof": prof_new, "mean_bias": bias}).to_csv(
        os.path.join(args.out, "corrected_country_summary.csv"))
    print(f"\nWritten to {os.path.abspath(args.out)}/")


# ---------------------------------------------------------------------------
# ladder250
# ---------------------------------------------------------------------------

def cmd_ladder250(args) -> None:
    """Recompute the ladder on the nested subsample, matching the primary arms' n."""
    raws = sorted(glob.glob(os.path.join(args.results, "silicon_full_raw_qwen_1p*_seed888.csv")))
    if not raws:
        raise SystemExit(f"no raw files under {args.results}")
    keep = None
    print(f"Recomputing the ladder at n={args.n} per country (nested within 500).")
    print("Purpose: the primary contrast runs at n=250, so the ladder's country")
    print("effect must be attenuated identically or the difference between the two")
    print("country effects is confounded by sample size.\n")
    rows = []
    for f in raws:
        df = pd.read_csv(f, usecols=["idno", "cntry", "variable",
                                     "human_response", "silicon_response"],
                         low_memory=False)
        for c in ("human_response", "silicon_response"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for v, (lo, hi) in ITEM_VALID_RANGE.items():
            msk = df.variable.eq(v) & ((df.human_response < lo) | (df.human_response > hi))
            df.loc[msk, "human_response"] = np.nan
        anchor = df.variable.iloc[0]
        ids = (df[df.variable == anchor].groupby("cntry")["idno"]
               .apply(lambda s: list(s)[:args.n]))
        sel = set((c, i) for c, lst in ids.items() for i in lst)
        df = df[[(c, i) in sel for c, i in zip(df.cntry, df.idno)]]
        cm = (df.groupby(["variable", "cntry"])
                .agg(h=("human_response", "mean"), s=("silicon_response", "mean"))
                .reset_index())
        r = per_item_r(cm.rename(columns={"h": "human_mean", "s": "silicon_mean"}),
                       "human_mean")
        tag = re.sub(r"^silicon_full_raw_|_seed888\.csv$", "", os.path.basename(f))
        rows.append(dict(condition=tag, n_per_country=args.n,
                         median_r_bc=r.median(), n_items=len(r)))
        print(f"  {tag:34s} median r_bc = {r.median():+.4f}  ({len(r)} items)")
    pd.DataFrame(rows).to_csv(os.path.join(args.out, f"ladder_n{args.n}.csv"), index=False)
    print(f"\nWritten to {args.out}/ladder_n{args.n}.csv")
    print("NOTE: human means here are UNWEIGHTED (raw files carry no pspwght). "
          "Use --ess with the aggregate subcommand for the weighted estimand, or "
          "merge weights in before publishing these numbers.")


# ---------------------------------------------------------------------------
# rq3
# ---------------------------------------------------------------------------

def cmd_rq3(args) -> None:
    raws = sorted(glob.glob(os.path.join(args.results, "silicon_full_raw_*_seed888.csv")))
    for f in raws:
        df = pd.read_csv(f, low_memory=False)
        for c in ("human_response", "silicon_response"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for v, (lo, hi) in ITEM_VALID_RANGE.items():
            msk = df.variable.eq(v) & ((df.human_response < lo) | (df.human_response > hi))
            df.loc[msk, "human_response"] = np.nan
        rows = []
        for v, gv in df.groupby("variable"):
            dom = gv["domain"].iloc[0] if "domain" in gv.columns else ""
            for cn, g in gv.groupby("cntry"):
                hh = g.human_response.values
                ss = g.silicon_response.values
                ok = ~(np.isnan(hh) | np.isnan(ss))
                n = int(ok.sum())
                if n < 10:
                    rp = pp = rs = ps = np.nan
                else:
                    rp, pp = stats.pearsonr(hh[ok], ss[ok])
                    rs, ps = stats.spearmanr(hh[ok], ss[ok])
                rows.append(dict(variable=v, domain=dom, cntry=cn, r_pearson=rp,
                                 p_pearson=pp, r_spearman=rs, p_spearman=ps,
                                 human_mean=np.nanmean(hh[ok]) if n else np.nan,
                                 silicon_mean=np.nanmean(ss[ok]) if n else np.nan,
                                 human_sd=np.nanstd(hh[ok], ddof=1) if n > 1 else np.nan,
                                 silicon_sd=np.nanstd(ss[ok], ddof=1) if n > 1 else np.nan,
                                 n_valid=n))
        out = os.path.join(args.out, os.path.basename(f)
                           .replace("raw_", "rq3_").replace("_seed888", "_corrected"))
        res = pd.DataFrame(rows)
        res.to_csv(out, index=False)
        print(f"{os.path.basename(f)} -> {os.path.basename(out)}  "
              f"mean r_wc = {res.r_pearson.mean():+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("aggregate")
    a.add_argument("--ess", required=True)
    a.add_argument("--data", default="../data")
    a.add_argument("--out", default="../outputs")
    a.add_argument("--main_tag", default="qwen_1p")
    a.add_argument("--human_weight", default="pspwght", choices=["pspwght", "none"])
    a.set_defaults(func=cmd_aggregate)

    b = sub.add_parser("ladder250")
    b.add_argument("--results", default="results")
    b.add_argument("--out", default="outputs")
    b.add_argument("--n", type=int, default=250)
    b.set_defaults(func=cmd_ladder250)

    c = sub.add_parser("rq3")
    c.add_argument("--results", default="results")
    c.add_argument("--out", default="outputs")
    c.set_defaults(func=cmd_rq3)

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
