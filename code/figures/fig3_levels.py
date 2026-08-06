#!/usr/bin/env python3
"""RETIRED 6 Aug 2026. Its two panels are now the second panel of
manuscript Figure 2 and of Figure 6, both produced by fig2_recovery.py, so
that each level has one figure. Kept for provenance; not in FIGURE_MAP.


fig3_levels.py -- Figure 3, rebuilt on a shared scale.

    python3 fig3_levels.py --data handoff_upload --figdir figures

THE CRITICISM THIS ANSWERS
"I found Figure 3 difficult to interpret because the two panels use different
y-axis scales, making visual comparisons misleading. Several variables show
strongly negative correlations at both the country and individual levels, and
these deserve much more discussion."

Both points are structural, and both are answered by construction rather than by
prose.

  1. ONE SCALE. The two panels share the vertical axis, so the individual-level
     series is flat. That flatness is the finding and not a defect of the figure:
     the aggregate series spans 1.64 correlation units and the individual series
     0.30, a ratio of 5.5 to one, and the earlier figure's independent axes
     concealed exactly that. The magnitude ratio is stated in
     the panel title so that a reader cannot mistake the flat series for a
     rendering failure.

  2. THE NEGATIVE ITEMS ARE MARKED, NOT DISCUSSED AFTER THE FACT. Reverse-coded
     items use the open square and the lighter shade throughout. Ten of the
     eleven items with a negative between-country correlation are reverse-coded,
     which is visible here before it is argued anywhere. This is not a coding
     error: the direction classification is frozen in item_direction_table.csv,
     built from the ESS11 codebook independently of any result, and the
     supervisor's hypothesis that the negatives reflect scale interpretation
     rather than a bug is what the figure supports.

     The one exception is marked separately. euftf is forward-coded and still
     recovers negatively, at -0.231. It is a bipolar policy item whose numeric
     direction is a convention rather than a substantive ordering, and it is the
     counterexample the discussion has to treat rather than leave for a reader
     to find.

WHY THE ITEMS ARE ORDERED BY THE AGGREGATE SERIES
So that the individual panel is read as a function of the aggregate one. The
item-level correlation between the two series is +0.937 as coded, and this figure
is where that number becomes visible; section 4.4 then reports that roughly a
third of the magnitude is carried by the same direction failure operating at both
levels, since the correlation falls to +0.775 on forward items alone.

WHICH INDIVIDUAL-LEVEL MEASURE IS PLOTTED
The pooled within-country correlation, r_pw, which pools respondents across
countries after centring within country. It is preferred here to the per-cell mean
because it is the quantity for which the design is adequately powered: the
detectable correlation at eighty per cent power is 0.0197 pooled against 0.1077
within a single country. The per-cell series is shown as a faint reference so that
the choice is visible rather than silent.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import figstyle as fs
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="handoff_upload")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--name", default="fig3_levels_shared_scale")
    a = ap.parse_args()

    d = pd.read_csv(os.path.join(a.data, "cross_level_agreement.csv"))
    d = d.rename(columns={d.columns[0]: "variable"}).set_index("variable")
    dirt = pd.read_csv(os.path.join(a.data, "item_direction_table.csv")).set_index("variable")
    d["reverse"] = dirt["direction"].reindex(d.index).eq("reverse")
    d = d.sort_values("r_bc", ascending=False)

    span_bc = float(d.r_bc.max() - d.r_bc.min())
    span_pw = float(d.r_pooled_within.max() - d.r_pooled_within.min())
    ratio = span_bc / span_pw

    lim = 1.02 * max(d.r_bc.abs().max(), d.r_pooled_within.abs().max())
    xs = np.arange(len(d))

    # Compact, full text width (reviewer, v18): two stacked panels on one
    # shared scale, items along the horizontal axis. Per-item labels live in
    # Appendix Figures A1 and A3; here only the extremes and the euftf
    # counterexample are named.
    # v31 review: the separate country-profile figure is retired and its two
    # strips are absorbed here as panels (c) and (d), so that 4.1 carries two
    # figures instead of three and the backstory contrast moves up a slot.
    dc = pd.read_csv(os.path.join(a.data, "country_profile_and_mae.csv"))
    dc = dc[dc.condition == "qwen_1p"].copy()
    assert len(dc) == 30, f"expected 30 countries, found {len(dc)}"

    # v32 review: panel (d), the error strip, is cut with the rest of the
    # 4.1 slimming; the metric-dependence numbers stay in the prose.
    # v40 review: the country strip that was panel (c) is cut. It carried a
    # different unit (countries) on a different scale, which is the fault the
    # shared item scale here is meant to fix, and the 30 countries are ranked
    # in Figure 7. Its two numbers stay in the 4.1 prose.
    fig = plt.figure(figsize=(6.3, 5.6))
    gsT = fig.add_gridspec(2, 1, hspace=0.22)
    axA = fig.add_subplot(gsT[0])
    axB = fig.add_subplot(gsT[1], sharex=axA, sharey=axA)
    fig.subplots_adjust(bottom=0.085, top=0.945, left=0.095, right=0.985)

    for ax, col, title in (
            (axA, "r_bc", f"(a)  aggregate $r_{{bc}}$, range {span_bc:.2f}"),
            (axB, "r_pooled_within",
             f"(b)  individual $r_{{pw}}$, same scale, range {span_pw:.2f}")):
        fs.grid(ax)
        ax.axhline(0, color=fs.INK, linewidth=0.9, zorder=2)
        for mask, shade, mark, lab in (
                (~d.reverse.values, fs.FWD_SHADE, fs.FWD_MARK, "forward (n=29)"),
                (d.reverse.values, fs.REV_SHADE, fs.REV_MARK, "reverse (n=13)")):
            ax.scatter(xs[mask], d.loc[mask, col], s=26, marker=mark,
                       facecolors="none", edgecolors=shade, linewidths=1.0,
                       zorder=4, label=lab)
            ax.vlines(xs[mask], 0, d.loc[mask, col], color=shade, linewidth=0.8,
                      zorder=3)
        ax.set_ylim(-1.06, 0.92)
        ax.set_ylabel("correlation")
        ax.set_title(title, loc="left")

    # name only the extremes and the counterexample
    axA.annotate(fs.VLABEL.get(d.index[0], d.index[0]) + f"  {d.r_bc.iloc[0]:+.2f}",
                 xy=(0, d.r_bc.iloc[0]), xytext=(1.5, -0.32), fontsize=fs.SZ_ANNOT,
                 ha="left", va="center", bbox=fs.BOX,
                 arrowprops=dict(arrowstyle="-", color=fs.INK, linewidth=0.8))
    axA.annotate(fs.VLABEL.get(d.index[-1], d.index[-1]) + f"  {d.r_bc.iloc[-1]:+.2f}",
                 xy=(len(d) - 1, d.r_bc.iloc[-1]), xytext=(len(d) - 1, -0.97),
                 fontsize=fs.SZ_ANNOT, ha="right", va="center", bbox=fs.BOX)
    # the one forward-coded inversion (European unification) is visible as
    # the lone circle in the negative tail; the marker shape carries it, and
    # Section 4.1 discusses it, so no extra emphasis element (reviewer, v23)

    axA.legend(loc="lower left", fontsize=fs.SZ_LEG, handlelength=1.4)
    axB.legend(loc="lower left", fontsize=fs.SZ_LEG, handlelength=1.6, ncol=2)

    axB.set_xticks([])
    axB.set_xlim(-0.8, len(d) - 0.2)
    axB.set_xlabel("42 items, ordered by the aggregate series "
                   "(per-item labels: Appendix Figures A1 and A3)")

    # ------- panels (c) and (d): the country strips, absorbed from the
    # retired fig5_country_profile_vs_error (v31 review)
    def beeswarm(vals, min_dx, dy=0.24):
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

    def strip(ax, vals, codes, xlabel, title, xlim, fmt):
        fs.grid(ax, axis="x")
        v = vals.to_numpy(dtype=float)
        jit = beeswarm(v, min_dx=(xlim[1] - xlim[0]) / 55.0)
        ax.scatter(vals, jit, s=30, marker="o", facecolors="none",
                   edgecolors=fs.INK, linewidths=0.9, zorder=3)
        aorder = np.argsort(v)
        for idx in (aorder[0], aorder[-1]):
            vv = float(vals.iloc[idx])
            name = fs.CNAME.get(str(codes.iloc[idx]), str(codes.iloc[idx]))
            ax.annotate(f"{name}  {fmt % vv}",
                        xy=(vv, jit[idx]), xytext=(vv, 1.45),
                        fontsize=fs.SZ_ANNOT, ha="center", va="center",
                        bbox=fs.BOX,
                        arrowprops=dict(arrowstyle="-", color=fs.INK,
                                        linewidth=0.8))
        ax.set_ylim(-1.25, 1.95)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel, fontsize=fs.SZ_DENSE)
        ax.set_title(title, loc="left")

    fs.save(fig, a.figdir, a.name)

    out = d.reset_index()[["variable", "reverse", "r_bc", "r_pooled_within", "r_ind"]]
    p = os.path.join(a.figdir, f"{a.name}_data.csv")
    out.to_csv(p, index=False)
    print(f"per-item values written to {p}")
    print(f"  aggregate range {span_bc:.4f}, individual range {span_pw:.4f}, "
          f"ratio {ratio:.2f}")
    neg = d.r_bc < 0
    print(f"  items negative at the aggregate level: {int(neg.sum())}, "
          f"of which reverse-coded: {int((neg & d.reverse).sum())}")


if __name__ == "__main__":
    main()
