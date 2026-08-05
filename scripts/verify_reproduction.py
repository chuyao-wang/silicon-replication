#!/usr/bin/env python3
"""
verify_reproduction.py (v12) — reproduction gate and repeated-run tolerance.

WHAT CHANGED RELATIVE TO THE v11 VERSION OF THIS FILE, AND WHY
--------------------------------------------------------------
The v11 script had three defects that would have made the gate uninformative.

(1) --repb was advertised by submit_v11.sh but never implemented, so the
    A-versus-B replicate comparison — the quantity that was supposed to replace
    the 630k-prompt stability arm — could not be computed at all. The command
    printed by the submit script exited with 'unrecognized arguments: --repb'.

(2) Criterion (b) was a fixed tolerance, |delta r_bc| < 0.03 for every item.
    That constant is smaller than Monte Carlo noise by an order of magnitude on
    the compressed items. Simulating two independent replicates at n = 500 from
    the observed per-country silicon standard deviations gives median
    |delta r_bc| of 0.092 for sclmeet, 0.081 for inprdsc and 0.063 for health,
    because their between-country silicon dispersion is only 0.04-0.06 scale
    points. The probability that at least one of the eight verify items exceeds
    0.03 under exact code equivalence is 0.999. The gate would have failed
    almost surely and the failure would have been uninterpretable.

(3) Criterion (a) required at least 95 per cent of cells within 2 Monte Carlo
    standard errors, but the expected pass rate under exact equivalence is
    P(|z| <= 2) = 0.9545. With 240 cells the failure probability under the null
    is about 0.30. Raising the multiplier to 2.5 gives an expected pass rate of
    0.9876 and a null failure probability below 0.001.

This version therefore replaces the fixed tolerance with a noise-scaled test. The per-item
standard deviation of r_bc is estimated parametrically by resampling country
means from the run's own per-country silicon standard deviations; the A-versus-B
replicate pair is then used to check that this parametric estimate is calibrated
rather than to supply a one-degree-of-freedom tolerance of its own. The decision
statistic is an omnibus chi-square over standardised per-item differences, which
has a known null distribution and is sensitive to a systematic shift while
tolerating the noise that defeated the constant tolerance.

A further defect was one of estimand rather than calibration: The v11 script computed r_bc
from unweighted human means and applied no valid-range rule, whereas the
manuscript's r_bc uses design-weighted human means and (from v12 onwards) the
per-item valid range. A gate decided on a different estimand does not transfer.
This version applies the valid-range rule to both files and, when the ESS microdata are
supplied, design-weights the human means as the manuscript does.

TESTS
-----
  test1   Sampling ESS11 with seed 888 reproduces, per country, the exact idno
          set of an existing raw file.
  test1b  sample_per_country = 250 is nested within 500. The nopolitical arm
          compares a 250-respondent run against a legacy 500-respondent run
          truncated to the same idnos, which is valid only if nesting holds.
          Nesting follows from RandomState.choice(replace=False) being
          implemented as permutation(pop)[:size]; it is a library-version
          property, not a guarantee, so it is asserted rather than assumed.
  test2   Reproduction gate plus repeated-run tolerance.
            (a) cell-level silicon means: |delta mean| <= 2.5 * MC-SE for at
                least 95 per cent of item x country cells.
            (b) item-level r_bc: omnibus chi-square on standardised differences
                against legacy, with per-item flags at 3 sigma.
            (c) calibration: median |r_A - r_B| / (sqrt(2) * sigma_hat) should be
                near 0.674, the median of the half-normal.

USAGE (on the HPC, from ~/Winston_Code)
---------------------------------------
  python verify_reproduction.py --test1 --test1b \
      --ess data/ESS11e04_2.csv \
      --legacy_raw results/silicon_full_raw_qwen_1p_seed888.csv

  python verify_reproduction.py --test2 \
      --new    results/silicon_full_raw_qwen_1p_verifyA_seed888.csv \
      --repb   results/silicon_full_raw_qwen_1p_verifyB_seed888.csv \
      --legacy_raw results/silicon_full_raw_qwen_1p_seed888.csv \
      --ess data/ESS11e04_2.csv
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

SEED = 888
N_PER_COUNTRY = 500

# Valid response range per item, i.e. the range shown to the model in the
# prompt. Values outside it are ESS refusal / don't-know / no-answer codes.
# Kept in this file so the gate does not depend on importing the pipeline.
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


# ---------------------------------------------------------------------------
# loading and harmonisation
# ---------------------------------------------------------------------------

def _load_raw(path: str, weights: pd.DataFrame | None) -> pd.DataFrame:
    """Load a raw response file and apply the manuscript's estimand.

    Both response columns are coerced to numeric, the per-item valid range is
    applied to the human side (a no-op for v12 files, effective for legacy
    files, which is precisely what makes the two comparable), and design
    weights are attached when available.
    """
    usecols = ["idno", "cntry", "variable", "human_response", "silicon_response"]
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    for c in ("human_response", "silicon_response"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    dropped = 0
    for var, (lo, hi) in ITEM_VALID_RANGE.items():
        m = df["variable"].eq(var)
        if not m.any():
            continue
        bad = m & ((df["human_response"] < lo) | (df["human_response"] > hi))
        dropped += int(bad.sum())
        df.loc[bad, "human_response"] = np.nan
    if dropped:
        print(f"  [{path}] valid-range rule set {dropped} human values missing")

    if weights is not None:
        df = df.merge(weights, on=["idno", "cntry"], how="left")
        df["pspwght"] = pd.to_numeric(df["pspwght"], errors="coerce").fillna(1.0)
    else:
        df["pspwght"] = 1.0
    return df


def _load_weights(ess_path: str | None) -> pd.DataFrame | None:
    if not ess_path:
        print("  NOTE: --ess not supplied; human means will be UNWEIGHTED, which "
              "is not the manuscript's estimand. Supply --ess for a transferable gate.")
        return None
    w = pd.read_csv(ess_path, usecols=["idno", "cntry", "pspwght"], low_memory=False)
    w["pspwght"] = pd.to_numeric(w["pspwght"], errors="coerce")
    return w


def _wmean(x: pd.Series, w: pd.Series) -> float:
    m = x.notna() & w.notna()
    return float(np.average(x[m], weights=w[m])) if m.any() else np.nan


def _cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per item x country: weighted human mean, unweighted silicon mean and SD."""
    rows = []
    for (var, cn), g in df.groupby(["variable", "cntry"], sort=True):
        s = g["silicon_response"]
        rows.append({
            "variable": var, "cntry": cn,
            "h": _wmean(g["human_response"], g["pspwght"]),
            "s": s.mean(), "s_sd": s.std(ddof=1), "n": int(s.notna().sum()),
        })
    return pd.DataFrame(rows)


def _rbc(cells: pd.DataFrame) -> pd.Series:
    out = {}
    for var, g in cells.groupby("variable"):
        g = g.dropna(subset=["h", "s"])
        if len(g) >= 10:
            out[var] = stats.pearsonr(g["h"], g["s"])[0]
    return pd.Series(out, dtype=float)


def _rbc_sigma(cells: pd.DataFrame, n_sim: int = 4000, seed: int = 20260728) -> pd.Series:
    """Parametric Monte Carlo SD of r_bc, per item.

    Country silicon means are resampled as mean + N(0, sd / sqrt(n)) using the
    run's own per-country dispersion, holding the human means fixed. This is the
    sampling noise that two independent replicates of the same code would show,
    and it is what the fixed 0.03 tolerance ignored.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for var, g in cells.groupby("variable"):
        g = g.dropna(subset=["h", "s", "s_sd", "n"])
        if len(g) < 10:
            continue
        h = g["h"].to_numpy()
        se = g["s_sd"].to_numpy() / np.sqrt(np.maximum(g["n"].to_numpy(), 1))
        draws = g["s"].to_numpy() + rng.normal(size=(n_sim, len(g))) * se
        dc = draws - draws.mean(axis=1, keepdims=True)
        hc = h - h.mean()
        denom = np.sqrt((dc ** 2).sum(axis=1) * (hc ** 2).sum())
        r = np.where(denom > 0, (dc @ hc) / np.where(denom > 0, denom, 1), np.nan)
        out[var] = float(np.nanstd(r, ddof=1))
    return pd.Series(out, dtype=float)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test1_idno(ess_path: str, legacy_raw_path: str) -> bool:
    raw = pd.read_csv(legacy_raw_path, usecols=["idno", "cntry", "variable"])
    anchor = raw["variable"].iloc[0]
    legacy_ids = raw[raw["variable"] == anchor].groupby("cntry")["idno"].apply(set)
    ess = pd.read_csv(ess_path, usecols=["idno", "cntry"], low_memory=False)
    bad = []
    for cn in sorted(legacy_ids.index):
        c = ess[ess["cntry"] == cn]
        n = min(N_PER_COUNTRY, len(c))
        if set(c.sample(n=n, random_state=SEED)["idno"]) != legacy_ids[cn]:
            bad.append(cn)
    if bad:
        print(f"TEST 1 FAIL: {len(bad)} countries mismatch: {bad[:6]}")
        return False
    print(f"TEST 1 PASS: exact idno match in {len(legacy_ids)}/{len(legacy_ids)} countries")
    return True


def test1b_nesting(ess_path: str, n_small: int = 250, n_large: int = N_PER_COUNTRY) -> bool:
    """Assert that the smaller subsample is nested within the larger one."""
    ess = pd.read_csv(ess_path, usecols=["idno", "cntry"], low_memory=False)
    bad, too_small = [], []
    for cn, g in ess.groupby("cntry"):
        if len(g) <= n_large:
            too_small.append(cn)
            continue
        big = set(g.sample(n=n_large, random_state=SEED)["idno"])
        small = set(g.sample(n=n_small, random_state=SEED)["idno"])
        if not small.issubset(big):
            bad.append((cn, len(small & big)))
    print(f"  pandas {pd.__version__} / numpy {np.__version__}")
    if too_small:
        print(f"  NOTE: {len(too_small)} countries have n <= {n_large}: {too_small}")
    if bad:
        print(f"TEST 1b FAIL: {n_small} not nested within {n_large} in {len(bad)} countries: {bad[:6]}")
        print("  -> do NOT truncate a legacy 500-respondent run to compare against a 250 arm.")
        return False
    print(f"TEST 1b PASS: {n_small} nested within {n_large} in all countries")
    return True


def test2(new_path: str, legacy_raw_path: str, repb_path: str | None,
          ess_path: str | None, mean_se_mult: float = 2.5,
          cell_pass_rate: float = 0.95, sigma_flag: float = 3.0,
          chi2_alpha: float = 0.01) -> bool:
    weights = _load_weights(ess_path)
    new = _load_raw(new_path, weights)
    old = _load_raw(legacy_raw_path, weights)
    rep = _load_raw(repb_path, weights) if repb_path else None

    common = sorted(set(new["variable"]) & set(old["variable"]))
    if not common:
        print("TEST 2 ERROR: no overlapping variables")
        return False
    new = new[new["variable"].isin(common)]
    old = old[old["variable"].isin(common)]
    if rep is not None:
        rep = rep[rep["variable"].isin(common)]
    print(f"Comparing {len(common)} items: {common}\n")

    cn_, co_ = _cell_table(new), _cell_table(old)
    cr_ = _cell_table(rep) if rep is not None else None

    # (a) cell-level silicon means -------------------------------------------
    a = cn_.merge(co_, on=["variable", "cntry"], suffixes=("_new", "_old"))
    se = np.sqrt(a["s_sd_new"] ** 2 / a["n_new"] + a["s_sd_old"] ** 2 / a["n_old"])
    a["pass"] = (a["s_new"] - a["s_old"]).abs() <= mean_se_mult * se
    rate = a["pass"].mean()
    expected = 2 * stats.norm.cdf(mean_se_mult) - 1
    ok_a = rate >= cell_pass_rate
    print(f"(a) cell silicon means within +/-{mean_se_mult} MC-SE: "
          f"{a['pass'].sum()}/{len(a)} = {rate:.2%} "
          f"(expected under equivalence {expected:.2%}; threshold {cell_pass_rate:.0%}) "
          f"-> {'PASS' if ok_a else 'FAIL'}")
    if not ok_a:
        print(a.loc[~a["pass"], ["variable", "cntry", "s_new", "s_old"]]
              .head(10).to_string(index=False))

    # (b) item-level r_bc, noise-scaled --------------------------------------
    r_new, r_old = _rbc(cn_), _rbc(co_)
    sig_new, sig_old = _rbc_sigma(cn_), _rbc_sigma(co_)
    items = [v for v in r_new.index if v in r_old.index and v in sig_new.index
             and v in sig_old.index]
    sig_pair = np.sqrt(sig_new[items] ** 2 + sig_old[items] ** 2)
    z = (r_new[items] - r_old[items]) / sig_pair.replace(0, np.nan)
    q = float(np.nansum(z ** 2))
    dof = int(z.notna().sum())
    crit = stats.chi2.ppf(1 - chi2_alpha, dof)
    ok_b = q <= crit
    print(f"\n(b) item-level r_bc, standardised by parametric MC sigma:")
    print(f"    {'item':10s} {'r_legacy':>9s} {'r_new':>8s} {'sigma_pair':>11s} {'z':>7s}")
    for v in sorted(items, key=lambda v: -abs(z[v]) if pd.notna(z[v]) else 0):
        flag = "  <-- flag" if pd.notna(z[v]) and abs(z[v]) > sigma_flag else ""
        print(f"    {v:10s} {r_old[v]:+9.3f} {r_new[v]:+8.3f} {sig_pair[v]:11.3f} "
              f"{z[v]:+7.2f}{flag}")
    print(f"    omnibus chi2 = {q:.2f} on {dof} df; {1 - chi2_alpha:.0%} critical "
          f"value {crit:.2f} -> {'PASS' if ok_b else 'FAIL'}")
    print(f"    NOTE: a fixed +/-0.03 tolerance would flag "
          f"{int((r_new[items] - r_old[items]).abs().gt(0.03).sum())}/{len(items)} items "
          f"here purely from Monte Carlo noise; that criterion is retired.")

    # (c) replicate calibration ---------------------------------------------
    ok_c = True
    if cr_ is not None:
        r_rep = _rbc(cr_)
        it2 = [v for v in items if v in r_rep.index]
        s2 = np.sqrt(2.0) * sig_new[it2]
        ratio = ((r_new[it2] - r_rep[it2]).abs() / s2.replace(0, np.nan)).dropna()
        med = float(ratio.median())
        ok_c = 0.30 <= med <= 1.50
        print(f"\n(c) replicate A vs B calibration: median |r_A - r_B| / (sqrt2 * sigma) "
              f"= {med:.3f} (half-normal median 0.674; accept 0.30-1.50) "
              f"-> {'PASS' if ok_c else 'FAIL'}")
        print(f"    empirical repeated-run tolerance, max |r_A - r_B| over items "
              f"= {float((r_new[it2] - r_rep[it2]).abs().max()):.3f}")
        print("    Report this as the repeated-run stability result; it also closes the "
              "manuscript limitation that run-to-run stability was untested.")
        if not ok_c:
            print("    A median far above 1.5 means the parametric sigma understates real "
                  "run-to-run variation (e.g. non-determinism beyond sampling noise); "
                  "rescale sigma by the observed ratio before judging (b).")
    else:
        print("\n(c) SKIPPED: pass --repb to compute the repeated-run tolerance.")

    ok = ok_a and ok_b and ok_c
    print("\nTEST 2 " + ("PASS: legacy files usable as comparators"
                         if ok else "FAIL: run paired arms; do not compare against legacy"))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test1", action="store_true")
    ap.add_argument("--test1b", action="store_true")
    ap.add_argument("--test2", action="store_true")
    ap.add_argument("--ess", default="data/ESS11e04_2.csv")
    ap.add_argument("--legacy_raw", required=True)
    ap.add_argument("--new")
    ap.add_argument("--repb", help="second unseeded replicate (verifyB)")
    ap.add_argument("--n_small", type=int, default=250)
    args = ap.parse_args()

    ok = True
    if args.test1:
        ok &= test1_idno(args.ess, args.legacy_raw)
    if args.test1b:
        ok &= test1b_nesting(args.ess, n_small=args.n_small)
    if args.test2:
        if not args.new:
            ap.error("--test2 requires --new")
        ok &= test2(args.new, args.legacy_raw, args.repb, args.ess)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
