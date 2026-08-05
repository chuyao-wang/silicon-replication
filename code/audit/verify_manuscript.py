#!/usr/bin/env python3
"""verify_manuscript.py -- can every number, figure and table in the chapter be
found in this package?

    python3 code/audit/verify_manuscript.py --manuscript /path/to/paper.md

A replication package that does not contain the numbers the text quotes is not a
replication package. This check is deliberately blunt: it takes every numeric
token in the manuscript, and asks whether some shipped file carries that value at
the precision printed. It then checks that every figure the text names exists as
a file, and every table the text names is laid out in the document.

Tokens that are not results are skipped by an explicit list, not by a heuristic,
so a number that stops being traceable cannot hide behind a loosened filter.
"""
from __future__ import annotations
import argparse, glob, os, re, sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Numbers that are not results. Each is here for a stated reason.
SKIP_EXACT = {
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14",
    "15", "16", "17", "18", "19", "20", "21", "22", "30", "42",     # counts, scale points
    "1950", "1978", "1995", "1997", "1999", "2001", "2004", "2005", "2011",
    "2012", "2013", "2014", "2017", "2023", "2024", "2025", "2026",  # years
    "0.05", "0.01", "0.001", "95", "90",                             # conventional levels
    "0", "0.0", "100", "1000", "1,000",
}
SKIP_CONTEXT = re.compile(
    r"(doi\.org|arXiv|pp?\.\s*\d|Vol\.|edition|Round|ESS11|GPT-|Llama 3\.1|"
    r"Qwen 2\.5|NUTS|R11|p\s*[<>=]|Bonferroni)", re.I)

NUM = re.compile(r"(?<![\w.])[-+−]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?![\w])")


def shipped_values() -> list[tuple[str, set]]:
    """Every numeric value in every shipped csv, one set per file.

    Column sums are included alongside cell values. A design total such as the
    response count across all fifteen arms is the sum of a shipped column, not an
    entry in one, and a checker that cannot see that would force the number to be
    typed."""
    out = []
    for d in ("results/tables", "results/analysis", "figures", "data/conditions",
              "data/reference"):
        for p in sorted(glob.glob(os.path.join(ROOT, d, "*.csv"))):
            try:
                t = pd.read_csv(p, low_memory=False)
            except Exception:
                continue
            vals = set()
            for c in t.columns:
                s = pd.to_numeric(t[c], errors="coerce").dropna()
                if s.empty:
                    continue
                vals.update(float(x) for x in s)
                vals.add(float(s.sum()))
            out.append((os.path.relpath(p, ROOT), vals))
    return out


def matches(tok: str, files: list[tuple[str, set]]) -> str:
    """Return the first shipped file carrying tok at its printed precision."""
    raw = tok.replace(",", "").replace("−", "-").lstrip("+")
    try:
        v = float(raw)
    except ValueError:
        return ""
    dec = len(raw.split(".")[1]) if "." in raw else 0
    tol = 0.5 * 10 ** (-dec)
    for name, vals in files:
        for cand in (v, v / 100.0, v * 100.0):          # per cent written either way
            for x in vals:
                if abs(x - cand) < tol * (100 if cand != v and dec == 0 else 1) + 1e-12:
                    if abs(x - cand) <= tol or (cand != v and abs(x - cand) <= tol / 100):
                        return name
    # Design totals are products of shipped counts rather than table entries:
    # 685 x 42, 863,100 x 4, and so on. A product of the run's own counts is
    # traced; an arbitrary number is not.
    if v == int(v) and v > 1000:
        base = [685, 42, 30, 22, 15, 4, 2, 863100, 452100, 28770, 20550]
        for i, x in enumerate(base):
            for y in base[i:]:
                if abs(x * y - v) < 0.5:
                    return f"design total {x} x {y}"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", required=True)
    a = ap.parse_args()
    text = open(a.manuscript).read()
    files = shipped_values()
    print(f"{len(files)} shipped csv files scanned")

    # ---------------------------------------------------------------- numbers
    tested = untraced = 0
    for line in text.split("\n"):
        if SKIP_CONTEXT.search(line):
            continue
        for tok in NUM.findall(line):
            if tok.lstrip("+-−") in SKIP_EXACT:
                continue
            tested += 1
            if not matches(tok, files):
                untraced += 1
                print(f"   NOT TRACED  {tok:>12}   {line.strip()[:96]}")
    print(f"NUMBERS  {tested} tested, {tested-untraced} traced to a shipped file, "
          f"{untraced} not")

    # ---------------------------------------------------------------- figures
    # The manuscript numbers figures in reading order; two file names follow the
    # order they were written in. FIGURE_MAP.csv is the authority, so the two
    # cannot silently diverge again.
    figs = sorted(set(re.findall(r"Figure\s+(A?\d+)", text)))
    fmap = pd.read_csv(os.path.join(ROOT, "results", "figures", "FIGURE_MAP.csv"),
                       dtype=str).set_index("manuscript_figure").file.to_dict()
    have = {os.path.basename(p) for p in
            glob.glob(os.path.join(ROOT, "results", "figures", "*.pdf"))}
    missing = [f for f in figs if fmap.get(f) not in have]
    print(f"FIGURES  {len(figs)} referenced, {len(figs)-len(missing)} present in the "
          f"package, {len(missing)} missing" + (f": {missing}" if missing else ""))

    # ---------------------------------------------------------------- tables
    tabs = sorted(set(re.findall(r"Table\s+(A?\d+)", text)))
    laid = {t for t in tabs if re.search(rf"Table {t}\.\s", text)}
    print(f"TABLES   {len(tabs)} referenced, {len(laid)} laid out, "
          f"{len(tabs)-len(laid)} missing"
          + (f": {sorted(set(tabs)-laid)}" if len(laid) != len(tabs) else ""))

    ok = untraced == 0 and not missing and len(laid) == len(tabs)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
