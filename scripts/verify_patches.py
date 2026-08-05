#!/usr/bin/env python3
"""verify_patches.py -- run after patch_A.py and patch_B.py, and after rerunning
analyze_v13.py and benchmark_v13.py.

    python3 verify_patches.py                       # default results/analysis
    python3 verify_patches.py --out results/analysis --ref handoff_upload

Two things are checked. First, that every quantity the patches were supposed to
add now exists and holds the expected value. Second, that no column present
before the patches has moved: the patches were designed to be additive, so any
drift in a shared column is a defect in the patch rather than a finding.
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/analysis")
ap.add_argument("--ref", default="handoff_upload",
                help="pre-patch bundle to regress against; skipped if absent")
a = ap.parse_args()

fails = []


def chk(name, got, want, tol=5e-4):
    ok = got is not None and want is not None and abs(float(got) - float(want)) < tol
    print(f"  {'pass' if ok else 'FAIL'}  {name:58s} {float(got):+12.6f}  "
          f"(expect {float(want):+.6f})")
    if not ok:
        fails.append(name)


def chk_true(name, cond, detail=""):
    print(f"  {'pass' if cond else 'FAIL'}  {name:58s} {detail}")
    if not cond:
        fails.append(name)


print("=" * 88)
print("PATCH A -- analyze_v13.py")
print("=" * 88)

cs = pd.read_csv(os.path.join(a.out, "condition_summary.csv")).set_index("condition")
need = ["within_sd_ratio_mean", "within_sd_ratio_median", "within_sd_ratio_of_means",
        "between_sd_ratio_median", "between_sd_ratio_mean"]
chk_true("P2 new dispersion columns present", all(c in cs.columns for c in need),
         f"missing: {[c for c in need if c not in cs.columns]}")
if all(c in cs.columns for c in need):
    q = cs.loc["qwen_1p"]
    chk("P2 within, mean of ratios (the released estimator)", q.within_sd_ratio_mean, 0.605241)
    chk("P2 within, median of ratios", q.within_sd_ratio_median, 0.572690)
    chk("P2 within, ratio of means", q.within_sd_ratio_of_means, 0.585903)
    chk("P2 between, median of ratios (the released estimator)",
        q.between_sd_ratio_median, 0.282081)
    chk("P2 between, mean of ratios", q.between_sd_ratio_mean, 0.295173)
    chk_true("P2 legacy columns alias the same estimators",
             bool((cs.within_sd_ratio == cs.within_sd_ratio_mean).all()
                  and (cs.between_sd_ratio == cs.between_sd_ratio_median).all()))

relp = os.path.join(a.out, "reliability_attenuation_n685.csv")
chk_true("E2 reliability table written", os.path.exists(relp), relp)
if os.path.exists(relp):
    rel = pd.read_csv(relp).set_index("variable")
    chk_true("E2 table covers 42 items", len(rel) == 42, f"n={len(rel)}")
    chk("E2 median human-side country-mean reliability", rel.human_reliability.median(), 0.9897, 1e-3)
    chk("E2 median silicon-side country-mean reliability", rel.silicon_reliability.median(), 0.9416, 1e-3)
    chk("E2 median attenuation factor on r_bc", rel.attenuation_factor.median(), 0.9667, 1e-3)
    low = sorted(rel.index[rel.silicon_reliability < 0.5])
    chk_true("E2 the two sub-0.50 items are exactly the pre-declared exclusions",
             low == ["hmsfmlsh", "rlgatnd"], f"{low}")
    chk_true("E2 inprdsc is above the 0.50 floor (the screen is defensible)",
             float(rel.loc["inprdsc", "silicon_reliability"]) > 0.5,
             f"{rel.loc['inprdsc', 'silicon_reliability']:.4f}")

sump = os.path.join(a.out, "reliability_attenuation_summary.csv")
chk_true("E2 summary table written", os.path.exists(sump), sump)
if os.path.exists(sump):
    su = pd.read_csv(sump).set_index("quantity").value
    det_w = float(su.get("detectable r, 80% power, within one country"))
    det_p = float(su.get("detectable r, 80% power, pooled within-country"))
    r_pw = float(su.get("observed median r_pw"))
    r_wc = float(su.get("observed mean r_wc per cell"))
    chk("E2 detectable r, within one country", det_w, 0.1077, 2e-3)
    chk("E2 detectable r, pooled within-country", det_p, 0.0197, 2e-3)
    chk("E2 observed median r_pw", r_pw, 0.0429, 1e-3)
    chk_true("E2 the within-country null IS underpowered (r_wc below its threshold)",
             r_wc < det_w, f"{r_wc:.4f} < {det_w:.4f}")
    chk_true("E2 the pooled estimate is NOT underpowered (r_pw above its threshold)",
             r_pw > det_p, f"{r_pw:.4f} > {det_p:.4f}")

print()
print("=" * 88)
print("PATCH B -- benchmark_v13.py")
print("=" * 88)

ab = pd.read_csv(os.path.join(a.out, "aggregate_benchmark.csv")).set_index("variable")
chk_true("P4 nmae_* aliases present",
         all(f"nmae_{k}" in ab.columns for k in ("model", "debias", "region", "grand")))
chk_true("P4 nmae_* equals mae_* exactly",
         all(bool((ab[f"nmae_{k}"] == ab[f"mae_{k}"]).all())
             for k in ("model", "debias", "region", "grand")))
chk_true("P3 leave-one-out grand mean present", "nmae_grand_loo" in ab.columns)
if "nmae_grand_loo" in ab.columns:
    chk("P3 median nMAE, grand mean in-sample (as released)", ab.nmae_grand.median(), 0.071521)
    chk("P3 median nMAE, grand mean leave-one-out (matched)", ab.nmae_grand_loo.median(), 0.073987)
    chk("P3 items where the model beats the in-sample grand mean on nMAE",
        int((ab.nmae_model < ab.nmae_grand).sum()), 2, 0.5)
    chk("P3 items where the model beats the LOO grand mean on nMAE",
        int((ab.nmae_model < ab.nmae_grand_loo).sum()), 3, 0.5)
    chk_true("P3 corollary: r_grand_loo is NaN for every item, not -1",
             bool(ab.r_grand_loo.isna().all()),
             "a leave-one-out constant predictor correlates -1 with the truth by "
             "construction; reporting that as a finding would be an artefact")
chk_true("P5' r_grand stays 0.0 for every item (exactly constant, deliberate)",
         bool((ab.r_grand == 0.0).all()))

offp = os.path.join(a.out, "offset_reduction_estimators.csv")
chk_true("P6 offset-reduction estimators written", os.path.exists(offp), offp)
if os.path.exists(offp):
    off = pd.read_csv(offp).set_index("estimator").value
    chk("P6 ratio of medians (the released 64.2%)",
        off.get("ratio of medians (as released)"), 0.641548)
    chk("P6 ratio of means", off.get("ratio of means"), 0.629032)
    chk("P6 MEDIAN of per-item reductions (recommended)",
        off.get("MEDIAN of the per-item reductions (recommended)"), 0.598961)
    chk("P6 mean of per-item reductions",
        off.get("mean of the per-item reductions"), 0.537296)

print()
print("=" * 88)
print("REGRESSION -- no pre-existing column may move")
print("=" * 88)
if not os.path.isdir(a.ref):
    print(f"  skipped: reference bundle {a.ref} not found")
else:
    moved, checked, files = 0, 0, 0
    for f in sorted(glob.glob(os.path.join(a.ref, "*.csv"))):
        b = os.path.basename(f)
        n = os.path.join(a.out, b)
        if not os.path.exists(n):
            continue
        files += 1
        x, y = pd.read_csv(f), pd.read_csv(n)
        if len(x) != len(y):
            print(f"  FAIL  {b}: row count {len(x)} -> {len(y)}")
            moved += 1
            continue
        for col in [c for c in x.columns if c in y.columns]:
            checked += 1
            u, v = x[col], y[col]
            if pd.api.types.is_bool_dtype(u) or pd.api.types.is_bool_dtype(v):
                if not u.astype(str).equals(v.astype(str)):
                    print(f"  FAIL  {b}.{col}: boolean mismatch")
                    moved += 1
            elif pd.api.types.is_numeric_dtype(u) and pd.api.types.is_numeric_dtype(v):
                d = (u.astype(float) - v.astype(float)).abs().max()
                if pd.notna(d) and d > 1e-12:
                    print(f"  FAIL  {b}.{col}: max|d| = {d:.3e}")
                    moved += 1
            elif not u.equals(v):
                print(f"  FAIL  {b}.{col}: mismatch")
                moved += 1
    print(f"  {files} shared tables, {checked} shared columns, {moved} moved")
    if moved:
        fails.append("regression")

print()
print("=" * 88)
if fails:
    print(f"{len(fails)} FAILURE(S): " + "; ".join(fails))
    raise SystemExit(1)
print("ALL CHECKS PASSED")
print("=" * 88)
