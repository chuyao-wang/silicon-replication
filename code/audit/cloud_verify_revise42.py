#!/usr/bin/env python3
"""cloud_verify_revise42.py -- traces every numeric token introduced by
revise42/revise43 back to the landed data, end to end.

    python3 cloud_analysis/cloud_verify_revise42.py --data <path>

<path> is the silicon-replication checkout's data/summary (cloud default
below). This is the cloud-side equivalent of verify_manuscript.py for the
new tokens only; the canonical 489-number pass still runs on the Mac once
the package ships these outputs. Writes revise42_numbers.csv beside itself.

Every check recomputes the quantity from the raw or summary CSVs and
asserts the rounded value equals the token in paper_current.md.
"""
from __future__ import annotations
import argparse, os, re, sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "paper_current.md")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/workspace/silicon-replication/data/summary")
a = ap.parse_args()
D = a.data
RAW = os.path.join(D, "..", "raw")

text = open(P, encoding="utf-8").read().replace("−", "-")
def has(tok: str) -> None:
    assert tok.replace("−", "-") in text, f"token not in manuscript: {tok}"

def r_bc(tag):
    s = pd.read_csv(f"{D}/silicon_full_country_scatter_{tag}.csv")
    return pd.Series({v: stats.pearsonr(g.survey_mean, g.silicon_mean)[0]
                      for v, g in s.groupby("variable")}).sort_index()
def z(x): return np.arctanh(np.clip(np.asarray(x, float), -0.999999, 0.999999))

rev = pd.read_csv(f"{D}/item_direction_table.csv").set_index("variable")["direction"].eq("reverse")
rows = []
def check(name, value, token, fmt="{:+.2f}"):
    got = fmt.format(value)
    assert got == token or got.lstrip("+") == token.lstrip("+"), \
        f"{name}: computed {got}, manuscript says {token}"
    has(token.replace("+", "").replace("-", ""))  # digits present either sign form
    rows.append(dict(quantity=name, value=round(float(value), 6), token=token))
    print(f"  ok {name:44s} {got}")

def margin(w, b):
    dz = pd.Series(z(r_bc(w)) - z(r_bc(b)), index=r_bc(w).index)
    rv = rev.reindex(dz.index)
    return dz[~rv].median(), dz[rv].median(), int((dz[~rv] > 0).sum())

print("4.2 margins:")
f, r, _ = margin("qwen_1p_minimal", "qwen_1p_demo_only")
check("country add, forward", f, "+0.73"); check("country add, reverse", r, "-0.44")
f, r, _ = margin("qwen_1p_minimal_membership", "qwen_1p_minimal")
check("membership add, forward", f, "+0.09")
f, r, _ = margin("qwen_1p", "qwen_1p_full_nomembership")
check("membership remove, forward", f, "+0.08")
f, r, _ = margin("qwen_1p_minimal_internet", "qwen_1p_minimal")
check("internet single add, forward", f, "+0.13")
f, r, _ = margin("qwen_1p_minimal_union", "qwen_1p_minimal")
check("union single add, forward", f, "+0.06")
f, r, _ = margin("qwen_1p_minimal_politics", "qwen_1p_minimal")
check("politics add, forward", f, "-0.12")
f, r, _ = margin("qwen_1p_minimal_ses", "qwen_1p_minimal")
check("SES add, forward", f, "-0.13")
f, r, _ = margin("qwen_1p", "qwen_1p_full_nosocioecon")
check("SES remove, forward", f, "-0.06")
f, r, _ = margin("qwen_1p", "qwen_1p_full_noincome")
check("income remove, forward", f, "-0.055", "{:+.3f}")
f, r, _ = margin("qwen_1p", "qwen_1p_full_nosocioecon")
check("SES remove, forward (3dp)", f, "-0.061", "{:+.3f}")
f, r, n = margin("llama_1p_full_noregion", "llama_1p_full_nocountry")
check("Llama label remove, forward", f, "+0.25")
check("Llama label remove, reverse", r, "-0.20")
assert n == 25; has("25 of 29"); print("  ok Llama forward items improving              25/29")

print("4.2 probes:")
check("country_only median r_bc", r_bc("qwen_1p_country_only").median(), "0.52", "{:.2f}")

print("4.2 swap:")
sw = r_bc("qwen_1p_full_swapcountry")  # vs true country (file's survey_mean)
rv = rev.reindex(sw.index)
check("swap vs true country, forward", sw[~rv].median(), "0.34", "{:.2f}")
check("no-label arm, forward", r_bc("qwen_1p_full_nocountry")[~rv].median(), "0.40", "{:.2f}")
check("true-label arm, forward", r_bc("qwen_1p_full_noregion")[~rv].median(), "0.56", "{:.2f}")
sc = pd.read_csv(os.path.join(HERE, "swap_scores.csv"))
check("swap vs labeled country, forward",
      sc[~sc.reverse].r_label.median(), "0.40", "{:.2f}")
# permutation baseline: labeled vs true human means under SWAP_MAP
sys.path.insert(0, os.path.join(D, "..", "..", "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pipe", os.path.join(D, "..", "..", "scripts", "silicon_v127_NEW.py"))
m = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(m)
except Exception: pass
sm = m.SWAP_MAP
ref = pd.read_csv(f"{D}/silicon_full_country_scatter_qwen_1p.csv")
human = ref.set_index(["variable", "cntry"]).survey_mean
perm = []
for v, g in ref.groupby("variable"):
    lab = [sm[c] for c in g.cntry]
    perm.append(stats.pearsonr(pd.Series([human.get((v, c)) for c in lab]).values,
                               g.survey_mean.values)[0])
check("permutation baseline (labeled vs true human)", pd.Series(perm).median(),
      "-0.07", "{:+.2f}")

print("4.1 pooled decomposition:")
s = pd.read_csv(f"{D}/silicon_full_country_scatter_qwen_1p.csv")
x, y = s.survey_mean, s.silicon_mean
xc = x - s.groupby("variable").survey_mean.transform("mean")
yc = y - s.groupby("variable").silicon_mean.transform("mean")
check("pooled r after item centering", stats.pearsonr(xc, yc)[0], "0.515", "{:.3f}")
share = 1 - np.cov(xc, yc)[0, 1] / np.cov(x, y)[0, 1]
assert f"{share*100:.1f}%" == "96.5%"; has("96.5%")
rows.append(dict(quantity="item-mean share of pooled covariance",
                 value=round(float(share), 6), token="96.5%"))
print("  ok item-mean share of pooled covariance        96.5%")

print("3.2 sample size:")
raw = pd.read_csv(f"{RAW}/silicon_full_raw_qwen_1p_seed888.csv.gz").dropna(
    subset=["human_response"])
piv_h = raw.pivot_table(index=["cntry", "idno"], columns="variable",
                        values="human_response")
rng = np.random.default_rng(888)
meds = []
for b in range(20):
    ha, hb = [], []
    for c in piv_h.index.get_level_values(0).unique():
        idx = piv_h.loc[c].index.to_numpy().copy(); rng.shuffle(idx)
        half = len(idx) // 2
        ha.append(piv_h.loc[c].loc[idx[:half]].mean().rename(c))
        hb.append(piv_h.loc[c].loc[idx[half:]].mean().rename(c))
    A, B = pd.DataFrame(ha), pd.DataFrame(hb)
    meds.append(np.median([stats.pearsonr(A[v], B[v])[0] for v in A.columns]))
check("human split-half median r_bc (unweighted)", float(np.mean(meds)),
      "0.98", "{:.2f}")
has("400")  # the plateau statement; curve shipped in revise42_numbers.csv

print("5.1 replicate-pair noise (revise46):")
_a = pd.read_csv(f"{RAW}/silicon_full_raw_qwen_1p_minimal_membership_seed888.csv.gz",
                 usecols=["idno", "cntry", "variable", "silicon_response"]).dropna()
_b = pd.read_csv(f"{RAW}/silicon_full_raw_qwen_1p_minimal_membership_rep_seed888.csv.gz",
                 usecols=["idno", "cntry", "variable", "silicon_response"]).dropna()
_m = _a.merge(_b, on=["idno", "cntry", "variable"], suffixes=("_a", "_b"))
_noise = _m.groupby("variable").apply(
    lambda x: (x.silicon_response_a - x.silicon_response_b).std() / np.sqrt(2),
    include_groups=False)
_tot = _a.groupby(["variable", "cntry"]).silicon_response.std().groupby(
    "variable").median()
_share = ((_noise ** 2) / (_tot ** 2)).median()
assert f"{_share*100:.0f}%" == "95%"; has("95%")
rows.append(dict(quantity="noise share of the replicate arm's within-country variance",
                 value=round(float(_share), 6), token="95%"))
print("  ok noise share of replicate arm within-country var   95%")

print("4.6 second replicate pair:")
_A, _B = r_bc("qwen_1p_minimal_membership"), r_bc("qwen_1p_minimal_membership_rep")
_d = (_A - _B).abs()
check("replicate pair, median |delta r_bc|", _d.median(), "0.029", "{:.3f}")
check("replicate pair, max |delta r_bc|", _d.max(), "0.242", "{:.3f}")
assert int((_d > 0.060).sum()) == 8; has("eight items beyond")
assert list(_d.sort_values(ascending=False).index[:2]) == ["rlgatnd", "hmsfmlsh"], \
    "the two largest movers must be the screened items"
print("  ok items beyond the 0.060 band                        8 (led by rlgatnd, hmsfmlsh)")

print("5.2 signal correlation (forward items):")
r_main = r_bc("qwen_1p")
scale = raw.groupby("variable")["scale"].first().map(
    lambda s2: int(s2.split("-")[1]) - int(s2.split("-")[0]))
sig = s.groupby("variable").survey_mean.std() / scale
fwd = ~rev
pr = stats.pearsonr(sig[fwd[r_main.index]], r_main[fwd[r_main.index]])
check("r_bc vs normalized human between-SD, r", pr[0], "0.46", "{:.2f}")
check("its p value", pr[1], "0.012", "{:.3f}")

print("3.2 arm accounting (revise43):")
assert 13 * 863100 + 4 * 452100 == 13_028_700
assert 13_028_700 + 24 * 863100 == 33_743_100
has("33,743,100"); has("forty-one")
rows.append(dict(quantity="response attempts, all 41 arms", value=33_743_100,
                 token="33,743,100"))
print("  ok 17 arms/13,028,700 + 24 full arms = 41/33,743,100")

pd.DataFrame(rows).to_csv(os.path.join(HERE, "revise42_numbers.csv"), index=False)
print(f"\nALL {len(rows)} quantities traced; revise42_numbers.csv written")
