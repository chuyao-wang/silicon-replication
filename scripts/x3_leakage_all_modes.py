#!/usr/bin/env python3
"""
X3 -- is it the country label, or is it country identifiability?

Run on Winston from ~/Winston_Code. Writes out/leakage_all_modes.csv.

WHY THIS IS THE HIGHEST-VALUE ANALYSIS LEFT
The leakage check returned 28.99 per cent for full_nocountry against a 3.33 per
cent baseline, and across the three conditions measured so far the between-country
recovery tracked how identifiable the country was rather than whether a label was
present:

    demo_only        7.48 per cent identifiable    median r_bc  -0.0286
    full_nocountry  28.99 per cent                              +0.3441
    full_noregion  100.00 per cent                              +0.4157

Monotone, and the Pearson correlation of log-identifiability on r_bc is 0.94 -- on
three points, which is not enough to claim anything. This script measures the same
quantity for every backstory mode the chapter uses, which turns three points into
eight and makes the relation testable rather than suggestive.

WHAT TURNS ON IT
If the relation holds, the chapter's central claim becomes: the aggregate ordering
is produced by country IDENTIFIABILITY, and the explicit label is merely the
cheapest route to it. That statement survives the leakage objection instead of being
wounded by it, because leakage stops being a confound and becomes the mechanism. If
the relation does not hold -- if, say, ses is highly identifiable but recovers
poorly -- then identifiability is not sufficient and the label is doing something
extra, which is also a result and a sharper one than the current draft has.

READ THE CONTROLS FIRST. full_noregion and full_clean contain the country sentence
and must classify at or near 100 per cent. demo_only is NOT a zero-information
control: it carries age and year of birth, which are country-patterned in the ESS
samples, and it returned 7.48 per cent. It is the composition floor.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
import silicon_sampling_extended_v12 as M

ESS = "data/ESS Data/ESS11e04_1.csv"
N, FOLDS, OUT = 685, 3, "out"
# Median between-country correlation for each mode, from the frozen bundle, so the
# two columns can be read side by side. NaN where the mode was not run as an arm.
RBC = {"demo_only": -0.0286, "minimal": 0.5232,
       "minimal_politics": 0.3161, "minimal_politics_econ": 0.4228,   # v13.2
       "ses": 0.3967, "political": 0.2870,
       "full_clean": 0.4433, "full_nocountry": 0.3441, "full_noregion": 0.4157,
       "full_nopolitical": np.nan}
MODES = list(RBC)

os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(ESS, low_memory=False)
if "essround" not in df.columns:
    df["essround"] = 11
assert hasattr(M, "harmonize_ess_variables"), "wrong pipeline version"
df = M.harmonize_ess_variables(df)
df["survey_year"] = df["essround"].map(M.ESS_ROUND_YEAR)
rng = np.random.default_rng(888)
s = pd.concat([g.iloc[rng.choice(len(g), min(N, len(g)), replace=False)]
               for _, g in df.groupby("cntry")], ignore_index=True)
y = s.cntry.to_numpy()
base = float(pd.Series(y).value_counts(normalize=True).max())
print(f"{len(s):,} respondents, {s.cntry.nunique()} countries, baseline {100*base:.2f}%",
      flush=True)
cv = StratifiedKFold(FOLDS, shuffle=True, random_state=888)

rows = []
for mode in MODES:
    t0 = time.time()
    try:
        txt = s.apply(lambda r: M.generate_backstory(r, mode=mode), axis=1)
    except Exception as e:
        print(f"{mode:18s} SKIPPED: {e}", flush=True); continue
    p = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True),
                      LinearSVC(C=1.0, max_iter=4000))
    acc = float(cross_val_score(p, txt, y, cv=cv, scoring="accuracy").mean())
    p.fit(txt, y)
    v, c = p.steps[0][1], p.steps[1][1]
    names = np.array(v.get_feature_names_out())
    top = ", ".join(names[np.argsort(-np.abs(c.coef_).max(0))[:8]])
    rows.append(dict(mode=mode, n_chars_median=int(txt.str.len().median()),
                     country_recoverable=round(100 * acc, 2),
                     times_baseline=round(acc / base, 2),
                     median_r_bc=RBC[mode], top_ngrams=top,
                     seconds=round(time.time() - t0, 1)))
    print(f"{mode:18s} {100*acc:7.2f}%  {acc/base:5.1f}x  r_bc={RBC[mode]!s:>8s}  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    print(f"                   {top}", flush=True)

out = pd.DataFrame(rows)
out.insert(0, "baseline_pct", round(100 * base, 2))
out.to_csv(os.path.join(OUT, "leakage_all_modes.csv"), index=False)
print(f"\nwrote {OUT}/leakage_all_modes.csv", flush=True)
d = out.dropna(subset=["median_r_bc"])
if len(d) >= 4:
    r = float(np.corrcoef(np.log(d.country_recoverable), d.median_r_bc)[0, 1])
    from scipy.stats import spearmanr
    rho = float(spearmanr(d.country_recoverable, d.median_r_bc).statistic)
    print(f"log-identifiability against r_bc: Pearson {r:+.4f}, Spearman {rho:+.4f}, "
          f"n={len(d)}", flush=True)
print("DONE", flush=True)
