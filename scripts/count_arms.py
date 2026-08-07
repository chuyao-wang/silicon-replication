#!/usr/bin/env python3
"""count_arms.py -- derive the arm count and the response total from manifests.

    python3 scripts/count_arms.py                       # from data/summary/
    python3 scripts/count_arms.py --data results        # cluster layout
    python3 scripts/count_arms.py --out results/analysis/arm_census.csv

WHY THIS EXISTS
---------------
Section 3 of the manuscript says "33,743,100 response attempts across all 41
arms". Both tokens are hand-maintained, and every new arm makes them stale
without anything failing. Two Llama anchored arms were added on 7 August 2026
and the tokens did not move. This script recomputes them from the shipped
manifests so the manuscript can cite a produced file instead of arithmetic.

WHAT COUNTS AS AN ARM
---------------------
A manifest counts when it has a shipped country-scatter file and more than two
respondents per country. That excludes:

  qwen_1p_anchored_pilotanch, qwen_1p_pilotnum   two-respondent pilots
  qwen_1p_minimal_politics_ses                   a manifest with no results

The rule is not invented here, it is reverse-engineered from the manuscript:
applied to the 44 manifests that existed before the Llama anchored run it
returns exactly 41 arms and 33,743,100 attempts, which is what the chapter
says. Adding the two new arms gives 43 and 34,647,300.

Response attempts per arm are n_respondents x variables_n, the same product
that gives the 863,100 per model-prompt condition already in Section 3.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

# what the manuscript said before the Llama anchored arms landed
BASELINE = (41, 33_743_100)


def census(data_dir: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(data_dir, "manifest_*.json"))):
        m = json.load(open(f, encoding="utf-8"))
        tag = m["tag"]
        scatter = os.path.join(data_dir, f"silicon_full_country_scatter_{tag}.csv")
        has_results = os.path.exists(scatter)
        per_country = m["sample_per_country"]
        counted = has_results and per_country > 2
        out.append({
            "tag": tag,
            "model": m["model"],
            "prompt": m["prompt"],
            "items": m["variables_n"],
            "respondents": m["n_respondents"],
            "per_country": per_country,
            "attempts": m["n_respondents"] * m["variables_n"],
            "has_results": int(has_results),
            "counted_as_arm": int(counted),
            "excluded_because": ""
            if counted
            else ("no shipped country-scatter file" if not has_results
                  else f"pilot, {per_country} respondents per country"),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/summary",
                    help="directory holding manifest_*.json and the scatter files")
    ap.add_argument("--out", default="", help="write the per-arm census here")
    a = ap.parse_args()

    rows = census(a.data)
    if not rows:
        raise SystemExit(f"no manifests under {a.data}")

    arms = [r for r in rows if r["counted_as_arm"]]
    total = sum(r["attempts"] for r in arms)
    skipped = [r for r in rows if not r["counted_as_arm"]]

    print(f"{len(rows)} manifests under {a.data}")
    if skipped:
        print("\nnot counted as arms:")
        for r in skipped:
            print(f"   {r['tag']:<40} {r['excluded_because']}")

    print(f"\n  ARMS              {len(arms)}")
    print(f"  RESPONSE ATTEMPTS {total:,}")

    print(f"\n  manuscript before the Llama anchored run: "
          f"{BASELINE[0]} arms, {BASELINE[1]:,}")
    delta_arms = len(arms) - BASELINE[0]
    delta_att = total - BASELINE[1]
    if (delta_arms, delta_att) == (0, 0):
        print("  the two tokens in Section 3 are current.")
    else:
        print(f"  Section 3 is stale by {delta_arms} arm(s) and "
              f"{delta_att:,} attempts. Write {len(arms)} and {total:,}.")

    by_model = {}
    for r in arms:
        by_model[r["model"]] = by_model.get(r["model"], 0) + r["attempts"]
    print("\n  by model:")
    for k in sorted(by_model):
        print(f"    {k:<8} {by_model[k]:>12,}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
            w.writerow({"tag": "TOTAL", "model": "", "prompt": "",
                        "items": "", "respondents": "", "per_country": "",
                        "attempts": total, "has_results": "",
                        "counted_as_arm": len(arms), "excluded_because": ""})
        print(f"\nwrote {a.out} ({len(rows)} rows + TOTAL)")


if __name__ == "__main__":
    main()
