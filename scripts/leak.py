import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
import silicon_sampling_extended_v12 as M
ESS = sys.argv[1] if len(sys.argv) > 1 else "data/ESS Data/ESS11e04_1.csv"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 685
df = pd.read_csv(ESS, low_memory=False)
if "essround" not in df.columns: df["essround"] = 11
df = M.harmonize_ess_variables(df)
df["survey_year"] = df["essround"].map(M.ESS_ROUND_YEAR)
rng = np.random.default_rng(888)
s = pd.concat([g.iloc[rng.choice(len(g), min(N, len(g)), replace=False)]
               for _, g in df.groupby("cntry")], ignore_index=True)
y = s.cntry.to_numpy(); base = pd.Series(y).value_counts(normalize=True).max()
print(f"{len(s):,} respondents, {s.cntry.nunique()} countries, baseline {100*base:.2f}%")
cv = StratifiedKFold(5, shuffle=True, random_state=888)
for mode, role in [("full_noregion", "POSITIVE CONTROL, expect ~100%"),
                   ("demo_only", "NEGATIVE CONTROL, expect ~baseline"),
                   ("full_nocountry", "*** THE NUMBER TO REPORT ***")]:
    txt = s.apply(lambda r: M.generate_backstory(r, mode=mode), axis=1)
    p = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True),
                      LogisticRegression(max_iter=2000, C=4.0))
    a = cross_val_score(p, txt, y, cv=cv, scoring="accuracy").mean()
    print(f"{mode:16s} {100*a:7.2f}%  {a/base:5.1f}x baseline   {role}")
    if mode == "full_nocountry":
        p.fit(txt, y); v, c = p.steps[0][1], p.steps[1][1]
        n = np.array(v.get_feature_names_out())
        print("  strongest n-grams:", ", ".join(n[np.argsort(-np.abs(c.coef_).max(0))[:12]]))
