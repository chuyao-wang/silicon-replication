#!/usr/bin/env python3
"""
fig2_recovery.py -- Figure 2, restructured into three panels, absorbing Figure 5.

    python3 fig2_recovery.py --data handoff_upload --figdir figures

WHY THE FIGURE IS BUILT THIS WAY
The supervisor's letter made three criticisms of the original Figure 2 and one of
Figure 5. This figure answers all four in one place.

  1. "The pooled correlations are extremely high, around 0.8, while the typical
     item-level correlations are much lower. Simply describing the overall
     recovery as moderate is confusing." The pooled correlation is not a
     competing estimate of the same quantity; it is a different quantity, and it
     is inflated because pooling item x country cells lets between-ITEM variance
     in the means masquerade as cross-national signal. It therefore appears here
     exactly once, as a labelled annotation on panel (a) marked as a
     counterexample, and nowhere else in the main text. The per-item
     between-country correlation is the sole primary aggregate metric.

  2. "One of the most interesting questions raised by the results is why some
     variables recover well while others perform so poorly." Panel (a) used to
     mark forward and reverse items separately. The reviewer (v29) had Figure 3
     carry that distinction alone; here every item draws with one marker and the
     panel's only job is the item spread against the pooled counterexample.

  3. "The paper should say more explicitly what constitutes good performance."
     Panel (b) counts items against pre-declared thresholds rather than
     describing the distribution as moderate. Panel (c) plots the individual
     level against the demographic-explainable ceiling, so the reader sees the
     achieved value as a fraction of what is attainable rather than as a bare
     correlation.

  4. "I was also unconvinced by Figure 5. It largely repeats information already
     presented in Figure 2." Correct, as a separate figure. The
     aggregate-individual dissociation is retained as panel (c) of this one and
     the figure number is retired, which is the merge recorded as planned
     deviation two in the response memo.

WHAT PANEL (c) SHOWS THAT THE OLD FIGURE 5 DID NOT
Intervals. The old figure gave point estimates, which invited the reading that a
near-zero individual correlation might be an underpowered null. The interval on
each condition is a percentile bootstrap over item x country cells, and the
dashed reference line is the demographic-explainable ceiling. The anchored arms
are included because they triple individual-level recovery, which is not a result
the chapter had anywhere: it belongs here because it is the same manipulation that
carries mechanism two at the aggregate level.

A CAVEAT THE PANEL MUST CARRY. llama_3p parses only 77.3 per cent of scheduled
responses and the loss is concentrated on the eleven-point scales, so its bar is
drawn in the excluded shade and labelled. It is shown rather than dropped because
a reader comparing this figure with the earlier draft will look for it.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import figstyle as fs
import matplotlib.pyplot as plt

# tag, short label, whether the arm is trustworthy for substantive reading
CONDITIONS = [
    ("qwen_1p", "Qwen\n1P", True),
    ("qwen_3p", "Qwen\n3P", True),
    ("llama_1p", "Llama\n1P", True),
    ("llama_3p", "Llama\n3P", False),
    ("qwen_1p_anchored", "Qwen 1P\nanchored", True),
]

# Pre-declared thresholds. Stated before the counts, so the counts are not a
# description of the distribution chosen after seeing it.
THRESHOLDS = [(0.70, "r > 0.70"), (0.50, "r > 0.50"), (0.00, "r < 0")]

CEILING = 0.2103   # median demographic-explainable within-country multiple R


def r_bc(tag: str, d: str) -> pd.Series:
    s = pd.read_csv(os.path.join(d, f"silicon_full_country_scatter_{tag}.csv"))
    return pd.Series({v: stats.pearsonr(g.survey_mean, g.silicon_mean)[0]
                      for v, g in s.groupby("variable")})


def r_wc_cells(tag: str, d: str) -> np.ndarray:
    s = pd.read_csv(os.path.join(d, f"silicon_full_rq3_{tag}.csv"))
    return s.r_pearson.to_numpy(dtype=float)


def boot_median(x: np.ndarray, rng, B: int = 4000) -> tuple[float, float, float]:
    x = x[~np.isnan(x)]
    med = float(np.median(x))
    idx = rng.integers(0, len(x), size=(B, len(x)))
    draws = np.median(x[idx], axis=1)
    return med, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="handoff_upload")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--name", default="fig2_recovery")
    a = ap.parse_args()

    dirt = pd.read_csv(os.path.join(a.data, "item_direction_table.csv")).set_index("variable")
    rev_all = dirt["direction"].eq("reverse")
    rng = np.random.default_rng(888)

    R = {t: r_bc(t, a.data) for t, _, _ in CONDITIONS}
    W = {t: r_wc_cells(t, a.data) for t, _, _ in CONDITIONS}

    pooled = pd.read_csv(os.path.join(a.data, "pooled_vs_item.csv")).set_index("condition")

    # (a) across the top, (b) the individual level below (reviewer, v32:
    # the threshold-count panel is cut; its counts live in the text and in
    # this figure's data csv, and 4.1 was carrying too many panels).
    fig = plt.figure(figsize=(6.3, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.10, 0.90], hspace=0.28)
    axA = fig.add_subplot(gs[0, 0])
    axC = fig.add_subplot(gs[1, 0])
    fig.subplots_adjust(bottom=0.060, top=0.960, left=0.105, right=0.985)

    xs = np.arange(len(CONDITIONS))
    labels = [lab for _, lab, _ in CONDITIONS]

    # ----------------------------------------------------------- panel (a)
    axA.axhline(0, color=fs.INK, linewidth=0.9, zorder=1)
    fs.grid(axA)
    for i, (tag, _, ok) in enumerate(CONDITIONS):
        v = R[tag]
        # one marker for every item (reviewer, v29): the direction split is
        # Figure 3's job, and carrying it here twice cluttered the swarm.
        y = v.to_numpy(dtype=float)
        jitter = rng.uniform(-0.17, 0.17, size=len(y))
        axA.scatter(i + jitter, y, s=17, marker=fs.FWD_MARK,
                    facecolors="none",
                    edgecolors=fs.SERIES if ok else fs.EXCL_SHADE,
                    linewidths=0.8, zorder=3)
        med = float(np.median(v))
        axA.plot([i - 0.30, i + 0.30], [med, med],
                 color=fs.INK if ok else fs.EXCL_SHADE, linewidth=2.4, zorder=5,
                 solid_capstyle="butt")
        # PATCH U: nine points above the median bar is the densest part of the
        # swarm, so the label needs an opaque background or the markers are
        # drawn through the digits and "+0.443" reads as a smear.
        axA.annotate(f"{med:+.3f}", (i, med), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=fs.SZ_ANNOT,
                     fontweight="bold", zorder=6, bbox=fs.BOX,
                     color=fs.INK if ok else fs.MUTE)
    axA.set_xticks(xs)
    axA.set_xticklabels(labels, fontsize=fs.SZ_DENSE)
    # Headroom for the two-line pooled-r counterexample label. The ticks stay
    # inside [-1, 1] so the axis cannot be read as allowing r > 1.
    axA.set_ylim(-1.0, 1.18)
    axA.set_yticks(np.arange(-1.0, 1.001, 0.25))
    axA.set_ylabel(r"between-country correlation  $r_{bc}$")
    axA.set_title("(a)  every item, every condition", loc="left")

    # The pooled correlation, once, as a labelled counterexample.
    pr = float(pooled.loc["qwen_1p", "r_pooled"])
    axA.axhline(pr, color=fs.MUTE, linewidth=1.0, linestyle=":", zorder=2)
    axA.annotate(
        f"pooled $r$ = {pr:.3f}, a counterexample (Section 4.1)",
        xy=(0.02, pr), xytext=(0.02, 0.985), textcoords="axes fraction",
        fontsize=fs.SZ_DENSE, color=fs.MUTE, va="top", ha="left",
        bbox=fs.BOX)   # opaque: it sits over the top of the qwen_1p swarm

    hitem = plt.Line2D([], [], linestyle="none", marker=fs.FWD_MARK,
                       markerfacecolor="none", markeredgecolor=fs.SERIES,
                       markersize=5.5, label="one item")
    hmed = plt.Line2D([], [], color=fs.INK, linewidth=2.4, label="median across items")
    axA.legend(handles=[hmed, hitem], loc="lower left",
               fontsize=fs.SZ_DENSE, handlelength=1.6, frameon=True,
               facecolor=fs.PAPER, edgecolor="none", framealpha=1.0)

    # ----------------------------------------------------------- panel (c)
    fs.grid(axC)
    axC.axhline(0, color=fs.INK, linewidth=0.9, zorder=1)
    axC.axhline(CEILING, color=fs.INK, linewidth=1.1, linestyle="--", zorder=2)
    axC.annotate(f"demographic-explainable ceiling {CEILING:.3f}",
                 xy=(-0.5, CEILING), textcoords="offset points",
                 xytext=(2, 5), ha="left", fontsize=fs.SZ_DENSE)
    for i, (tag, _, ok) in enumerate(CONDITIONS):
        med, lo, hi = boot_median(W[tag], rng)
        col = fs.INK if ok else fs.EXCL_SHADE
        axC.plot([i, i], [lo, hi], color=col, linewidth=1.2, zorder=3)
        axC.plot([i - 0.09, i + 0.09], [lo, lo], color=col, linewidth=1.2, zorder=3)
        axC.plot([i - 0.09, i + 0.09], [hi, hi], color=col, linewidth=1.2, zorder=3)
        axC.scatter([i], [med], s=52, marker="D", color=col, zorder=4)
        # Value only: at half width the ceiling percentages collide, and the
        # text reports them (13.5% and 38% of the ceiling in Section 4.4).
        axC.annotate(f"{med:.3f}",
                     (i, hi), textcoords="offset points", xytext=(0, 5),
                     ha="center", va="bottom", fontsize=fs.SZ_DENSE, bbox=fs.BOX,
                     color=fs.INK if ok else fs.MUTE)
    axC.set_xticks(xs)
    axC.set_xticklabels(labels, fontsize=fs.SZ_DENSE)
    axC.set_xlim(-0.60, len(CONDITIONS) - 0.40)
    axC.set_ylim(-0.025, 0.255)
    axC.set_ylabel(r"within-country correlation  $r_{wc}$")
    axC.set_title("(b)  the same conditions, individual level", loc="left")

    fs.save(fig, a.figdir, a.name)

    out = []
    for tag, lab, ok in CONDITIONS:
        v = R[tag]
        med, lo, hi = boot_median(W[tag], rng)
        out.append(dict(condition=tag, trustworthy=ok, n_items=len(v),
                        median_r_bc=float(np.median(v)),
                        n_gt70=int((v > 0.70).sum()), n_gt50=int((v > 0.50).sum()),
                        n_negative=int((v < 0).sum()),
                        median_r_wc=med, r_wc_ci_lo=lo, r_wc_ci_hi=hi,
                        pct_of_ceiling=100 * med / CEILING))
    p = os.path.join(a.figdir, f"{a.name}_data.csv")
    pd.DataFrame(out).to_csv(p, index=False)
    print(f"per-condition values written to {p}")
    print(pd.DataFrame(out).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
