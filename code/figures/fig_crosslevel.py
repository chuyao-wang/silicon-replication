#!/usr/bin/env python3
"""
fig_crosslevel.py -- Figure 7: country rank under aggregate profile recovery
against country rank under individual recovery, Qwen 1P.

    python3 fig_crosslevel.py --data handoff_upload --figdir figures

WHY THIS FIGURE EXISTS, AND WHY IT IS A RANK DISPLAY
Section 4.5 reports the aggregate-individual dissociation across countries as
a bounded correlation (-0.21, 90% CI [-0.49, 0.10]) and as a pairwise count:
232 of the 435 country pairs are ordered one way by aggregate profile
recovery and the other way by individual recovery. The reviewer (v29) asked
for a display. A scatter of the two quantities would draw the noise the
interval describes, so the display is the decision-relevant object instead:
the two rank orderings joined per country. The crossing lines ARE the
discordant pairs, and the two countries the text names (Switzerland and
Israel) are emphasized.

The count drawn in the annotation is computed from the same table the text
uses and asserted equal to the manuscript's 232, so figure and text cannot
drift apart.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

import figstyle as fs
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="handoff_upload")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--name", default="fig7_crosslevel_ranks")
    a = ap.parse_args()

    d = pd.read_csv(os.path.join(a.data,
                                 "aggregate_vs_individual_by_country.csv"))
    assert len(d) == 30, f"expected 30 countries, found {len(d)}"
    d["name"] = d.cntry.map(fs.CNAME).fillna(d.cntry)
    d["rk_prof"] = d.profile_r.rank(ascending=False, method="first").astype(int)
    d["rk_wc"] = d.mean_within_r.rank(ascending=False, method="first").astype(int)

    disc = sum(1
               for i in range(len(d))
               for j in range(i + 1, len(d))
               if (d.profile_r.iat[i] - d.profile_r.iat[j])
               * (d.mean_within_r.iat[i] - d.mean_within_r.iat[j]) < 0)
    assert disc == 232, f"manuscript says 232 discordant pairs, computed {disc}"

    n = len(d)
    fig, ax = plt.subplots(figsize=(6.3, 6.9))
    fig.subplots_adjust(left=0.24, right=0.76, top=0.90, bottom=0.055)
    for _, r in d.iterrows():
        hot = r.cntry in ("CH", "IL")
        y0, y1 = n - r.rk_prof, n - r.rk_wc
        ax.plot([0, 1], [y0, y1],
                color=fs.INK if hot else fs.REV_SHADE,
                linewidth=1.7 if hot else 0.8,
                zorder=3 if hot else 2)
        kw = dict(fontsize=8, va="center",
                  color=fs.INK if hot else fs.MUTE,
                  fontweight="bold" if hot else "normal")
        ax.text(-0.035, y0, f"{r['name']}  {r.profile_r:.2f}",
                ha="right", **kw)
        ax.text(1.035, y1, f"{r.mean_within_r:+.3f}  {r['name']}",
                ha="left", **kw)
    ax.text(0.5, n + 0.7,
            "crossing lines are rank reversals:\n"
            "232 of the 435 country pairs reverse",
            ha="center", va="bottom", fontsize=fs.SZ_DENSE, color=fs.MUTE)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-1.0, n + 2.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ranked by aggregate\nprofile recovery  $r_{prof}$",
                        "ranked by individual\nrecovery  $r_{wc}$"])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(False)
    fs.save(fig, a.figdir, a.name)


if __name__ == "__main__":
    main()
