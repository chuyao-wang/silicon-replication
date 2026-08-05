#!/usr/bin/env python3
"""v12.6 regression: all 22 pre-existing modes byte-identical; add-one arms
render exactly minimal + their block; twins/probes exact; swap arm equals
full_noregion with only the country name substituted."""
import importlib.util, sys, re
import pandas as pd

OLD, NEW, ESS = sys.argv[1], sys.argv[2], sys.argv[3]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
old = load("old_pipe", OLD); new = load("new_pipe", NEW)

df = pd.read_csv(ESS, low_memory=False)
sub = df.groupby("cntry", group_keys=False).head(25).copy()
sub["survey_year"] = 2024
print(f"{len(sub)} respondents, {sub.cntry.nunique()} countries")

OLDM = ['v4','v5','v5_clean','demo_only','minimal','ses','political','full_clean',
        'full_nocountry','full_noregion','full_nopolitical','minimal_politics',
        'minimal_politics_ses','minimal_politics_ses_geo','minimal_politics_econ',
        'full_noascriptive','full_nosocioecon','full_nohousehold','full_nocivic',
        'full_nomembership','full_nominority','full_nodomicil']
bad=0
for m in OLDM:
    for _,row in sub.iterrows():
        if old.generate_backstory(row,m)!=new.generate_backstory(row,m):
            print("MISMATCH",m); bad+=1; break
print(f"{len(OLDM)} pre-existing modes:", "ALL BYTE-IDENTICAL" if not bad else f"{bad} BAD")

PATS={'ses3':[r"level of education",r"years of full-time education",r"main activity",r"income is in"],
      'member':[r"trade union",r"use the internet"],
      'house':[r"I am (married|separated|legally|widowed|never|divorced|in a legally)",r"household of \d+"],
      'civic':[r"born in this country",r"citizen of this country"],
      'minor':[r"discriminat"],
      'domic':[r"I live in (a big city|the suburbs|a town|a country village|a farm|the outskirts)"],
      'region':[r"I live in the [A-Z]{2}\S* region"],
      'pol':[r"left \(0\) to right",r"political party"]}
ALL=[p for ps in PATS.values() for p in ps]
fails=0
ADD={'minimal_ses':'ses3','minimal_membership':'member','minimal_household':'house',
     'minimal_civic':'civic','minimal_minority':'minor','minimal_domicil':'domic',
     'minimal_region':'region'}
for mode,g in ADD.items():
    for _,row in sub.iterrows():
        mini=new.generate_backstory(row,'minimal'); out=new.generate_backstory(row,mode)
        if not out.startswith(mini):
            print(f"FAIL {mode}: minimal not a prefix"); fails+=1; break
        extra=out[len(mini):]
        for p in [q for q in ALL if q not in PATS[g]]:
            if re.search(p,extra): print(f"FAIL {mode}: alien pattern {p!r} in {extra[:120]!r}"); fails+=1; break
# twins
for _,row in sub.iterrows():
    ao=new.generate_backstory(row,'minimal_ageonly'); yo=new.generate_backstory(row,'minimal_yrbrnonly')
    co=new.generate_backstory(row,'country_only')
    if "born in 1" in ao or "born in 2" in ao: print("FAIL ageonly has birth year"); fails+=1; break
    if "years old" in yo: print("FAIL yrbrnonly has age"); fails+=1; break
    if re.search(r"years old|born in \d|I am (fe)?male|education|income|political",co):
        print("FAIL country_only extra content:",co[:150]); fails+=1; break
# income probe
for _,row in sub.iterrows():
    fc=new.generate_backstory(row,'full_clean'); ni=new.generate_backstory(row,'full_noincome')
    want=re.sub(r" My household income is in [^.]+ decile for this country\.","",fc)
    if ni!=want: print("FAIL noincome mismatch"); fails+=1; print(repr(fc[:100]),"\n",repr(ni[:100])); break
# swap: equals full_noregion with only the country sentence changed
cn=new.COUNTRY_NAMES; sm=new.SWAP_MAP
for _,row in sub.iterrows():
    nr=new.generate_backstory(row,'full_noregion'); sw=new.generate_backstory(row,'full_swapcountry')
    t=cn.get(row['cntry']); f=cn.get(sm[row['cntry']])
    if nr.replace(f"I live in {t}.",f"I live in {f}.")!=sw:
        print("FAIL swap:",row['cntry']); print(repr(nr[:160])); print(repr(sw[:160])); fails+=1; break
print("NEW-MODE checks:", "ALL PASS" if not fails else f"{fails} FAILURES")
r=sub.iloc[3]
print("\nswap example:", new.generate_backstory(r,'full_swapcountry')[:200])
print("country_only:", new.generate_backstory(r,'country_only'))
print("ageonly:", new.generate_backstory(r,'minimal_ageonly'))
print("yrbrnonly:", new.generate_backstory(r,'minimal_yrbrnonly'))
print("add ses:", new.generate_backstory(r,'minimal_ses'))
sys.exit(1 if (bad or fails) else 0)
