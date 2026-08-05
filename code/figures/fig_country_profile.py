#!/usr/bin/env python3
"""
fig_country_profile.py -- the country-level figure the letter looked for and did
not find.

    python3 fig_country_profile.py --data handoff_upload --figdir figures

THE CRITICISM THIS ANSWERS
"After Figure 3, country-level profile correlations are discussed without being
shown graphically. I found myself looking for a figure that was not there."

The figure does more than supply the missing display, because simply plotting the
profile correlation would invite a reader to rank countries on a single metric. The
substantive point, however, is NOT that the two metrics disagree wildly. Measured,
they largely agree: the Spearman correlation between profile recovery and
normalised error is -0.598, and 125 of 435 country pairs are ordered differently.
Twenty-nine per cent is enough to make country selection metric-dependent and not
enough to call the rankings unrelated, and the caption says so in those terms.

The stronger objection to the profile correlation is not rank disagreement but lack
of discrimination. Every one of the thirty countries falls between 0.685 and 0.911.
A metric on which the worst country in the sample scores 0.69 cannot separate
countries that should be trusted from countries that should not, whatever its rank
agreement with a metric that can.

  (a) Profile correlation by country, descending. This is the quantity the text
      already discussed: within a country, the correlation across the 42 item
      means between the human and the silicon profile. Every country is high,
      0.685 to 0.911, which is why a reader who sees only this panel concludes
      that the model reproduces national profiles well.

  (b) Range-normalised mean absolute error, same country order. The spread here is
      wide where panel (a) is narrow. The profile correlation is invariant to an
      additive offset and the model applies a large positive one in every country,
      so panel (a) cannot see the largest single component of the error.

  (c) The rank crossings, drawn explicitly, with the count stated rather than
      characterised. Each line joins a country's rank on the two metrics.

WHAT THE FIGURE IS FOR ARGUMENTATIVELY
It supports one sentence: a metric that is invariant to the largest single component
of the error, and that compresses all thirty countries into a band of width 0.23,
cannot be used to select countries for use. The 29 per cent of discordant pairs is a
secondary observation and must be reported as such; the primary one is the narrowness
of panel (a) against the spread of panel (b).

It is a weaker instrument than the cross-level counterexample argument in section
4.4, where 232 of 435 pairs invert between the aggregate and the individual level.
That comparison should be drawn explicitly in the text so that the two counts, 125
and 232, are not read as measuring the same thing.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import figstyle as fs
import matplotlib.pyplot as plt

CONDITION = "qwen_1p"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="handoff_upload")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--name", default="fig5_country_profile_vs_error")
    ap.add_argument("--condition", default=CONDITION)
    a = ap.parse_args()

    d = pd.read_csv(os.path.join(a.data, "country_profile_and_mae.csv"))
    d = d[d.condition == a.condition].copy()
    assert len(d) == 30, f"expected 30 countries, found {len(d)}"
    d["name"] = d.cntry.map(fs.CNAME).fillna(d.cntry)

    d = d.sort_values("profile_r", ascending=False).reset_index(drop=True)
    d["rank_profile"] = np.arange(1, len(d) + 1)
    d["rank_error"] = d.norm_mae.rank(method="min").astype(int)

    rho = stats.spearmanr(d.profile_r, d.norm_mae).statistic
    pairs = 0
    n = len(d)
    for i in range(n):
        for j in range(i + 1, n):
            a_better_profile = d.profile_r[i] > d.profile_r[j]
            a_better_error = d.norm_mae[i] < d.norm_mae[j]
            if a_better_profile != a_better_error:
                pairs += 1

    ys = np.arange(n)[::-1]

    # Compact strip design (reviewer, v18: only the extremes and the median
    # need naming). Each panel is a one-dimensional strip of all thirty
    # countries; panel (a) runs over the full 0-1 scale, so the narrowness of
    # the band IS the display.
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(6.3, 5.6),
                                   gridspec_kw=dict(hspace=0.60))

    def beeswarm(vals, min_dx, dy=0.24):
        """Deterministic non-overlapping offsets: place each point at the
        lowest free row whose occupied x-positions are all at least min_dx
        away."""
        order = np.argsort(vals)
        offs = np.zeros(len(vals))
        rows = {}
        for i in order:
            k = 0
            while True:
                for cand in ([0] if k == 0 else [k, -k]):
                    taken = rows.get(cand, [])
                    if all(abs(vals[i] - t) >= min_dx for t in taken):
                        offs[i] = cand * dy
                        rows.setdefault(cand, []).append(vals[i])
                        cand = None
                        break
                if cand is None:
                    break
                k += 1
        return offs
    fig.subplots_adjust(left=0.045, right=0.985, top=0.905, bottom=0.155)

    def strip(ax, vals, codes, xlabel, title, xlim, fmt):
        fs.grid(ax, axis="x")
        v = vals.to_numpy(dtype=float)
        jit = beeswarm(v, min_dx=(xlim[1] - xlim[0]) / 55.0)
        ax.scatter(vals, jit, s=34, marker="o", facecolors="none",
                   edgecolors=fs.INK, linewidths=0.9, zorder=3)
        # full country names, extremes only (reviewer, v20)
        aorder = np.argsort(v)
        for which, idx, ytxt in (("min", aorder[0], 1.02),
                                 ("max", aorder[-1], 1.02)):
            vv = float(vals.iloc[idx])
            name = fs.CNAME.get(str(codes.iloc[idx]), str(codes.iloc[idx]))
            ax.annotate(f"{name}  {fmt % vv}",
                        xy=(float(vals.iloc[idx]), jit[idx]), xytext=(float(vals.iloc[idx]), ytxt),
                        fontsize=fs.SZ_ANNOT, ha="center", va="center",
                        bbox=fs.BOX,
                        arrowprops=dict(arrowstyle="-", color=fs.INK,
                                        linewidth=0.8))
        ax.set_ylim(-1.25, 1.25)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")

    strip(axA, d.profile_r, d.cntry,
          "country profile correlation, 42 item means",
          "(a)  all 30 countries sit in a band of width 0.23",
          (0.0, 1.0), "%.2f")
    strip(axB, d.norm_mae, d.cntry,
          "range-normalized mean absolute error",
          "(b)  level accuracy spreads the same countries out",
          (0.0, 1.06 * float(d.norm_mae.max())), "%.3f")
    axB.annotate(f"Spearman between the two metrics {rho:+.2f}\n"
                 f"{pairs} of {n * (n - 1) // 2} pairs ordered differently",
                 xy=(0.02, 0.10), xycoords="axes fraction", ha="left", va="bottom",
                 fontsize=fs.SZ_DENSE, color=fs.MUTE, bbox=fs.BOX)

    fs.save(fig, a.figdir, a.name)

    p = os.path.join(a.figdir, f"{a.name}_data.csv")
    d[["cntry", "name", "profile_r", "norm_mae", "rank_profile", "rank_error"]] \
        .to_csv(p, index=False)
    print(f"per-country values written to {p}")
    print(f"  Spearman between the two metrics: {rho:+.4f}")
    print(f"  discordant country pairs: {pairs} of {n * (n - 1) // 2} "
          f"({100 * pairs / (n * (n - 1) // 2):.1f} per cent)")


if __name__ == "__main__":
    main()
