#!/usr/bin/env python3
"""W4 -- verify every reference against Crossref.

Checks three things per entry, which are three different ways to be wrong:
  resolves   the DOI exists
  surname    Crossref's first author surname matches the one I cite
  year       Crossref's issued year matches the year I cite
An entry with no DOI is searched by title; a confident hit is reported as a
DOI that could be added, not as an error.
"""
import argparse, json, re, sys, time, urllib.parse, urllib.request

UA = "silicon-sampling-replication/1.0 (mailto:C.Wang85@lse.ac.uk)"

def get(url):
    for attempt in range(3):
        try:
            rq = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(rq, timeout=30) as f:
                return json.load(f)["message"]
        except Exception as e:
            if getattr(e, "code", None) == 404:
                return None
            if attempt == 2:
                return {"_error": str(e)}
            time.sleep(2)
    return None

def get_csl(doi):
    """doi.org content negotiation. Crossref returns 404 for DOIs it does not
    register -- arXiv and the DataCite data DOIs among them -- and doi.org
    resolves both registries, so a Crossref miss is re-checked here before it
    is reported as NOT FOUND."""
    url = "https://doi.org/" + urllib.parse.quote(doi)
    for attempt in range(3):
        try:
            rq = urllib.request.Request(url, headers={
                "User-Agent": UA,
                "Accept": "application/vnd.citationstyles.csl+json"})
            with urllib.request.urlopen(rq, timeout=30) as f:
                return json.load(f)
        except Exception as e:
            if getattr(e, "code", None) == 404:
                return None
            if attempt == 2:
                return {"_error": str(e)}
            time.sleep(2)
    return None

ap = argparse.ArgumentParser()
ap.add_argument("--manuscript", default="paper_final.md",
                help="manuscript whose References section is checked")
args = ap.parse_args()
lines = open(args.manuscript).read().split("\n")
i = max(k for k, l in enumerate(lines) if l.strip() == "References")
j = min(k for k, l in enumerate(lines) if l.strip() == "Appendix" and k > i)
ents = [l.strip() for l in lines[i+1:j] if len(l.strip()) > 40]

def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return " ".join(s.split())

rows = []
for e in ents:
    m = re.search(r"doi\.org/(10\.[^\s,)]+)", e)
    doi = m.group(1).rstrip(".") if m else ""
    yr = re.search(r"\((\d{4})[a-z]?\)", e)
    year = int(yr.group(1)) if yr else None
    surname = e.split(",")[0].strip()
    # title: text between the year-paren and the following period
    t = re.split(r"\(\d{4}[a-z]?\)\.\s*", e, maxsplit=1)
    title = t[1].split(". ")[0].strip() if len(t) > 1 else ""
    title = re.sub(r"\s*\[[^\]]*\]", "", title)   # drop [Data set] and kin
    r = dict(surname=surname, year=year, title=title, doi=doi,
             status="", cr_year=None, cr_surname="", cr_title="", note="")
    if doi:
        m = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
        if m is None:
            c = get_csl(doi)
            if c and "_error" not in c:
                r["status"] = "resolves"
                r["note"] = "not in Crossref; resolved via doi.org content negotiation"
                ct = c.get("title") or ""
                r["cr_title"] = ct if isinstance(ct, str) else (ct or [""])[0]
                au = c.get("author") or []
                r["cr_surname"] = au[0].get("family", "") if au else ""
                dp = (c.get("issued") or {}).get("date-parts") or []
                if dp and dp[0] and dp[0][0]:
                    r["cr_year"] = int(dp[0][0])
            else:
                r["status"] = "NOT FOUND"
        elif "_error" in m:
            r["status"] = "ERROR"; r["note"] = m["_error"]
        else:
            r["status"] = "resolves"
            r["cr_title"] = (m.get("title") or [""])[0]
            au = m.get("author") or []
            r["cr_surname"] = au[0].get("family", "") if au else ""
            for k in ("issued", "published-print", "published-online", "created"):
                dp = (m.get(k) or {}).get("date-parts") or []
                if dp and dp[0] and dp[0][0]:
                    r["cr_year"] = dp[0][0]; break
    else:
        q = urllib.parse.urlencode({"query.bibliographic": f"{surname} {title}",
                                    "rows": 3})
        m = get("https://api.crossref.org/works?" + q)
        r["status"] = "no DOI cited"
        if m and "_error" not in m and m.get("items"):
            for it in m["items"]:
                ct = (it.get("title") or [""])[0]
                if norm(ct)[:44] == norm(title)[:44] and norm(title):
                    r["note"] = f"Crossref has one: {it['DOI']}"
                    r["cr_title"] = ct
                    break
            else:
                r["note"] = "no confident Crossref match (preprint or report)"
    rows.append(r)
    time.sleep(0.3)

bad = 0
print(f"{'':3} {'status':12} {'yr':>5} {'cr':>5}  cited surname / Crossref surname")
print("-" * 96)
for n, r in enumerate(rows, 1):
    ymis = r["cr_year"] is not None and r["year"] is not None and \
           abs(r["cr_year"] - r["year"]) > 0
    smis = r["cr_surname"] and norm(r["cr_surname"]) != norm(r["surname"])
    # a registry title that is a prefix of the cited one (or the reverse) is a
    # subtitle truncation in the registry record, not a wrong work
    _a, _b = norm(r["cr_title"]), norm(r["title"])
    tmis = r["cr_title"] and r["doi"] and _a[:40] != _b[:40] \
        and not (_a.startswith(_b) or _b.startswith(_a))
    flag = ""
    if r["status"] in ("NOT FOUND", "ERROR"): flag = "  <== DOI"
    elif ymis: flag = "  <== YEAR"
    elif smis: flag = "  <== AUTHOR"
    elif tmis: flag = "  <== TITLE"
    if flag: bad += 1
    print(f"{n:3} {r['status']:12} {r['year'] or '?':>5} {r['cr_year'] or '-':>5}  "
          f"{r['surname'][:22]:22} {r['cr_surname'][:20]:20}{flag}")
    if tmis:
        print(f"      cited : {r['title'][:78]}")
        print(f"      actual: {r['cr_title'][:78]}")
    if r["note"]:
        print(f"      {r['note']}")

print("-" * 96)
print(f"{len(rows)} references, {sum(1 for r in rows if r['status']=='resolves')} DOIs resolve, "
      f"{sum(1 for r in rows if not r['doi'])} cite no DOI, {bad} flagged")
json.dump(rows, open("w4_results.json", "w"), indent=1)
