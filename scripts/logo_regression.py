#!/usr/bin/env python3
"""v12.4 regression: all 15 pre-existing modes render byte-identically under
the new module; each of the six LOGO modes differs from full_clean by exactly
its own group's sentences."""
import importlib.util, sys, re
import pandas as pd

OLD, NEW, ESS = sys.argv[1], sys.argv[2], sys.argv[3]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

old = load("old_pipe", OLD)
new = load("new_pipe", NEW)

df = pd.read_csv(ESS, low_memory=False)
# deterministic spread: every country, up to 25 respondents each
sub = df.groupby("cntry", group_keys=False).head(25).copy()
sub["survey_year"] = 2024
print(f"{len(sub)} respondents across {sub.cntry.nunique()} countries")

OLD_MODES = ['v4','v5','v5_clean','demo_only','minimal','ses','political',
             'full_clean','full_nocountry','full_noregion','full_nopolitical',
             'minimal_politics','minimal_politics_ses','minimal_politics_ses_geo',
             'minimal_politics_econ']
bad = 0
for mode in OLD_MODES:
    for _, row in sub.iterrows():
        a = old.generate_backstory(row, mode)
        b = new.generate_backstory(row, mode)
        if a != b:
            print(f"MISMATCH mode={mode} idresp={row.get('idno')}")
            print("  old:", a[:200]); print("  new:", b[:200]); bad += 1; break
print(f"{len(OLD_MODES)} pre-existing modes: {'ALL BYTE-IDENTICAL' if not bad else str(bad)+' MISMATCHED'}")

# each LOGO mode: full_clean minus exactly its own sentences
GROUPS = {
 'full_noascriptive': [r"I am (fe)?male", r"years old", r"born in \d{4}"],
 'full_nosocioecon':  [r"level of education", r"years of full-time education",
                       r"main activity", r"income is in"],
 'full_nohousehold':  [r"I am (married|separated|legally|widowed|never|divorced|in a legally)",
                       r"household of \d+"],
 'full_nocivic':      [r"born in this country", r"citizen of this country"],
 'full_nomembership': [r"trade union", r"use the internet"],
 'full_nominority':   [r"discriminat"],
 'full_nodomicil':    [r"I live in (a big city|the suburbs|a town|a country village|a farm|the outskirts)"],
}
ALL_PATS = sorted({p for ps in GROUPS.values() for p in ps})
fails = 0
for mode, pats in GROUPS.items():
    others = [p for p in ALL_PATS if p not in pats]
    for _, row in sub.iterrows():
        full = new.generate_backstory(row, 'full_clean')
        abl  = new.generate_backstory(row, mode)
        # 1) ablated sentences absent
        for p in pats:
            if re.search(p, abl):
                print(f"FAIL {mode}: pattern {p!r} present:\n  {abl[:250]}"); fails += 1; break
        # 2) every full_clean sentence NOT matching an ablated pattern survives
        keep = [t for t in full.split(". ") if not any(re.search(p, t) for p in pats)]
        for t in keep:
            frag = t.rstrip(".")
            if frag and frag not in abl and "region" not in frag:
                print(f"FAIL {mode}: lost non-target sentence {frag[:90]!r}"); fails += 1; break
        # 3) nodomicil keeps the region code
        if mode == 'full_nodomicil':
            r = row.get('region')
            if isinstance(r, str) and r and f"the {r} region" in full and f"the {r} region" not in abl:
                print(f"FAIL nodomicil: region {r} lost"); fails += 1
print("LOGO checks:", "ALL PASS" if not fails else f"{fails} FAILURES")
# show one example of each
row = sub.iloc[3]
print("\n--- full_clean:", new.generate_backstory(row, 'full_clean'))
for m in GROUPS: print(f"--- {m}:", new.generate_backstory(row, m))
sys.exit(1 if (bad or fails) else 0)
