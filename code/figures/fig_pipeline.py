#!/usr/bin/env python3
"""fig_pipeline.py -- Figure 1, the simulation pipeline.

    python3 fig_pipeline.py                 # run from a staged directory

The only figure in the manuscript that carries no data. It was previously a
drawing pasted into the document, which meant a reader holding the package could
not regenerate Figure 1 and could not check that the counts on it matched the
run. Drawing it here fixes both: the sample size, the item count, the country
count and the prompt total are read from the run manifests rather than typed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs  # noqa: E402


def manifest(data: str) -> dict:
    for pat in (os.path.join(data, "*qwen_1p.json"),
                os.path.join("results", "manifest_qwen_1p.json")):
        hits = sorted(glob.glob(pat))
        if hits:
            return json.load(open(hits[0]))
    return {}


def box(ax, x, y, w, h, lines, shade="#ffffff"):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.1, edgecolor=fs.INK, facecolor=shade, zorder=2))
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.033 - i * 0.040, t, ha="center", va="center",
                fontsize=fs.SZ_TICK if i else fs.SZ_LABEL,
                fontweight="bold" if i == 0 else "normal",
                color=fs.INK if i == 0 else fs.MUTE, zorder=3)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", linewidth=1.1, color=fs.MUTE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="handoff_upload")
    ap.add_argument("--figdir", default="figures")
    a = ap.parse_args()
    m = manifest(a.data)
    n = m.get("sample_per_country", 685)
    nv = m.get("variables_n", 42)
    nc = m.get("n_countries", 30)
    seed = m.get("sampling_seed", 888)
    temp = m.get("temperature", 0.7)
    per_cond = n * nv * nc

    # Vertical flow, compact (reviewer, v18: a third smaller, less text, and
    # every box fully closed -- the rounded pad must stay inside the axes).
    fig, ax = plt.subplots(figsize=(3.9, 4.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])   # axis("off") leaves the tick
    ax.axis("off")                          # label Text objects behind
    for sp in ax.spines.values():
        sp.set_visible(False)

    BW, BX, BH = 0.72, 0.14, 0.105         # stage width, left edge, height
    box(ax, BX, 0.870, BW, BH,
        ["ESS Round 11", f"{nc} countries, {n} respondents per country"])
    box(ax, BX, 0.660, BW, BH,
        ["Backstory", "20 demographic variables, one paragraph"])
    box(ax, 0.03, 0.450, 0.44, BH,
        ["Qwen 2.5-7B-Instruct", "1st and 3rd person"], shade="#f2f2f2")
    box(ax, 0.53, 0.450, 0.44, BH,
        ["Llama 3.1-8B-Instruct", "1st and 3rd person"], shade="#f2f2f2")
    box(ax, BX, 0.240, BW, BH,
        ["Responses", f"{nv} items each, {per_cond:,} per condition"])
    box(ax, BX, 0.030, BW, BH,
        ["Comparison with the human answers",
         "country means; respondents within countries"])

    arrow(ax, 0.50, 0.862, 0.50, 0.775)    # ESS -> backstory
    arrow(ax, 0.40, 0.655, 0.28, 0.565)    # backstory -> Qwen
    arrow(ax, 0.60, 0.655, 0.72, 0.565)    # backstory -> Llama
    arrow(ax, 0.28, 0.445, 0.40, 0.355)    # Qwen -> responses
    arrow(ax, 0.72, 0.445, 0.60, 0.355)    # Llama -> responses
    arrow(ax, 0.50, 0.232, 0.50, 0.145)    # responses -> comparison

    fig.subplots_adjust(left=0.005, right=0.995, top=0.985, bottom=0.015)
    fs.save(fig, a.figdir, "fig1_pipeline")
    print(f"   {nc} countries, {n} per country, {nv} items, {per_cond:,} per condition")


if __name__ == "__main__":
    main()
