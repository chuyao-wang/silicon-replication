#!/usr/bin/env python3
"""
fig_2x2.py -- the figure that carries the chapter's central mechanism claim.

    python3 fig_2x2.py --data handoff_upload --figdir figures

The claim is a SINGLE INTERACTING mechanism, not two parallel ones: the country
label supplies cross-national ordering information, and the response-scale
direction determines the sign with which that information is expressed. The
design that identifies it is a 2 x 2, country label present or absent crossed
with numeric or verbally anchored scales, on the 22-item batch.

Three panels, in the order a reader needs them.

  (a) Every item, both scale regimes. The country-label effect is plotted on the
      Fisher-z scale, once under numeric scales and once under anchored ones,
      with a line joining the two estimates for the same item. Under numeric
      scales the reverse-coded items sit below zero: the country label makes
      recovery WORSE on them. Under anchored scales they move above zero. Forward
      items barely move. This is the interaction, item by item, before any
      averaging.

  (b) The four cells with bootstrap intervals. The two lines are forward and
      reverse items; the horizontal axis is the scale regime. Two non-parallel
      lines are the interaction; the vertical distance between them is the
      forward-minus-reverse gap that the triple difference tests.

  (c) The triple difference under all three pre-specified reverse-item sets, with
      the 95 per cent bootstrap intervals as released. All three exclude zero,
      which is the robust claim. The gap figures are reported separately in the
      text because elimination, unlike the interaction, does not survive the most
      conservative set.

STYLE. Carried over from the March round's plots.py: a single grayscale scheme,
categories distinguished by shade, marker and line style rather than colour, and
Liberation Serif at sizes close to the 12-pt body text. Nothing in the figure
relies on colour.

ESTIMATORS. Cell means are means over items, matching the pre-specified triple
difference so that the four cells reconstruct it exactly. Bootstrap intervals for
the cells are resampled over items with seed 888, the project's convention. The
triple-difference intervals are READ from twoxtwo_triple_difference.csv and are
not recomputed.
"""
from __future__ import annotations

import argparse
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------- style
INK, GRID, MUTE = "#000000", "#d9d9d9", "#595959"
FWD_SHADE, REV_SHADE = "#1a1a1a", "#8c8c8c"
FWD_MARK, REV_MARK = "o", "s"
FWD_LS, REV_LS = "-", "--"
EXCL_SHADE = "#c8c8c8"
SZ_TITLE, SZ_LABEL, SZ_TICK, SZ_LEG, SZ_ANNOT, SZ_DENSE = 11, 10, 10, 10, 10, 10

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10, "axes.titlesize": SZ_TITLE, "axes.titleweight": "bold",
    "axes.labelsize": SZ_LABEL, "axes.edgecolor": INK, "axes.linewidth": 0.8,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": SZ_TICK, "ytick.labelsize": SZ_TICK,
    "legend.frameon": False, "legend.fontsize": SZ_LEG,
})

VLABEL = {
    "actrolga": "Active role in political group", "aesfdrk": "Safety after dark",
    "cptppola": "Confident participating in politics", "freehms": "Gays and lesbians free to live",
    "gincdif": "Govt should reduce income differences", "happy": "How happy",
    "health": "Subjective general health", "hincfel": "Feeling about household income",
    "hmsacld": "Same-sex couples' right to adopt", "hmsfmlsh": "Ashamed if close family gay",
    "imdfetn": "Allow immigrants of different race", "impcntr": "Allow immigrants, poorer countries",
    "imsmetn": "Allow immigrants of same race", "inprdsc": "People to discuss intimate matters",
    "polintr": "Interest in politics", "psppipla": "Political system allows a say",
    "psppsgva": "Political system allows influence", "rlgatnd": "Religious attendance",
    "sclmeet": "Frequency of social meetings", "stfdem": "Satisfaction with democracy",
    "stflife": "Life satisfaction", "trstplc": "Trust in the police",
}

ARMS = {("numeric", "with"): "qwen_1p_full_noregion",
        ("numeric", "without"): "qwen_1p_full_nocountry",
        ("anchored", "with"): "qwen_1p_full_noregion_anchored",
        ("anchored", "without"): "qwen_1p_full_nocountry_anchored"}
PREDECLARED_EXCL = ("hmsfmlsh", "rlgatnd")
FURTHER_EXCL = ("freehms", "hmsacld")


def z(x):
    return np.arctanh(np.clip(np.asarray(x, dtype=float), -0.999999, 0.999999))


def r_bc(tag, d):
    s = pd.read_csv(os.path.join(d, f"silicon_full_country_scatter_{tag}.csv"))
    out = {}
    for v, g in s.groupby("variable"):
        out[v] = stats.pearsonr(g.survey_mean, g.silicon_mean)[0]
    return pd.Series(out)


def boot_ci(vals, n=10000, seed=888):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    draws = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return np.percentile(draws, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="handoff_upload")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--name", default="fig_2x2_country_by_scale")
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)

    dirt = pd.read_csv(os.path.join(a.data, "item_direction_table.csv")).set_index("variable")
    rev_flag = dirt["direction"].eq("reverse")
    items = sorted(r_bc("qwen_1p_anchored", a.data).index)
    ce = {}
    for regime in ("numeric", "anchored"):
        w = r_bc(ARMS[(regime, "with")], a.data).reindex(items)
        wo = r_bc(ARMS[(regime, "without")], a.data).reindex(items)
        ce[regime] = pd.Series(z(w) - z(wo), index=items)
    E = pd.DataFrame(ce)
    E["reverse"] = rev_flag.reindex(items)
    E["excluded"] = E.index.isin(PREDECLARED_EXCL)
    E["label"] = [VLABEL.get(v, v) for v in E.index]

    fwd = E.index[~E.reverse].tolist()
    rev12 = E.index[E.reverse].tolist()
    rev10 = [v for v in rev12 if v not in PREDECLARED_EXCL]

    td = pd.read_csv(os.path.join(a.data, "twoxtwo_triple_difference.csv"))

    # Two full-width panels (reviewer, v24): the four-cell interaction leads
    # with a plain-language annotation; the per-item movement is the evidence
    # beneath it. The triple-difference estimates live in the text and Table
    # A5.
    fig = plt.figure(figsize=(6.75, 8.6))
    # Panel (a) taller (reviewer, v29), and repositioned after drawing so it
    # spans the full page width instead of inheriting the left margin that
    # panel (b)'s long item labels force on the shared gridspec column.
    gs = fig.add_gridspec(2, 1, height_ratios=[1.02, 2.10], hspace=0.38)
    axB = fig.add_subplot(gs[0, 0])
    axA = fig.add_subplot(gs[1, 0])

    # ------------------------------------------------------- panel (a)
    order = (sorted(rev12, key=lambda v: E.loc[v, "numeric"])
             + sorted(fwd, key=lambda v: E.loc[v, "numeric"]))
    ypos = {v: i for i, v in enumerate(order)}
    axA.axvline(0, color=INK, linewidth=0.9, zorder=1)
    for v in order:
        y = ypos[v]
        isrev, isex = bool(E.loc[v, "reverse"]), bool(E.loc[v, "excluded"])
        shade = EXCL_SHADE if isex else (REV_SHADE if isrev else FWD_SHADE)
        x0, x1 = E.loc[v, "numeric"], E.loc[v, "anchored"]
        axA.annotate("", xy=(x1, y), xytext=(x0, y),
                     arrowprops=dict(arrowstyle="-|>", linewidth=0.9, color=shade,
                                     shrinkA=2.5, shrinkB=0.5,
                                     linestyle=(0, (2, 1.6)) if isex else "-"))
        axA.plot([x0], [y], marker=REV_MARK if isrev else FWD_MARK, markersize=4.6,
                 markerfacecolor="#ffffff", markeredgecolor=shade,
                 markeredgewidth=0.9, zorder=3)
        axA.plot([x1], [y], marker=REV_MARK if isrev else FWD_MARK, markersize=4.6,
                 markerfacecolor=shade, markeredgecolor=shade, zorder=3)
    axA.set_yticks(range(len(order)))
    axA.set_yticklabels([E.loc[v, "label"] + ("  \u2020" if E.loc[v, "excluded"] else "")
                         for v in order], fontsize=SZ_DENSE)
    axA.set_ylim(-0.9, len(order) - 0.1)
    axA.axhline(len(rev12) - 0.5, color=MUTE, linewidth=0.7, linestyle=(0, (4, 3)))
    axA.set_xlabel("country-label effect, Fisher $z$  ($z(r_{\\rm with}) - z(r_{\\rm without})$)")
    axA.set_title("(b)  the same movement, item by item", loc="left", pad=22)
    xl = max(abs(E[["numeric", "anchored"]].to_numpy()).max() * 1.14, 0.95)
    axA.set_xlim(-xl, xl)
    axA.text(-xl * 0.96, len(rev12) - 0.5 + 0.45, "forward-coded items",
             fontsize=SZ_DENSE, color=MUTE, va="bottom", style="italic")
    axA.text(-xl * 0.96, len(rev12) - 0.5 - 0.45, "reverse-coded items",
             fontsize=SZ_DENSE, color=MUTE, va="top", style="italic")
    axA.grid(axis="x", color=GRID, linewidth=0.7)
    axA.set_axisbelow(True)
    from matplotlib.lines import Line2D
    axA.legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor="#ffffff",
               markeredgecolor=INK, markersize=5.4, label="numbers only"),
        Line2D([], [], marker="o", color="none", markerfacecolor=INK,
               markeredgecolor=INK, markersize=5.4, label="numbers with verbal endpoints"),
    ], loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=2,
       fontsize=SZ_LEG, handletextpad=0.5, frameon=False)

    # ------------------------------------------------------- panel (b)
    xs = [0, 1]
    cells = {}
    for grp, name, shade, mark, ls in ((fwd, "forward (n=10)", FWD_SHADE, FWD_MARK, FWD_LS),
                                       (rev10, "reverse (n=10)", REV_SHADE, REV_MARK, REV_LS)):
        means = [E.loc[grp, r].mean() for r in ("numeric", "anchored")]
        cis = [boot_ci(E.loc[grp, r].to_numpy()) for r in ("numeric", "anchored")]
        cells[name] = (means, cis)
        axB.plot(xs, means, color=shade, linestyle=ls, linewidth=1.5, marker=mark,
                 markersize=7, markerfacecolor=shade, markeredgecolor=INK,
                 markeredgewidth=0.7, label=name, zorder=3)
        for x, m, (lo, hi) in zip(xs, means, cis):
            axB.plot([x, x], [lo, hi], color=shade, linewidth=1.1, zorder=2)
            axB.plot([x - 0.028, x + 0.028], [lo, lo], color=shade, linewidth=1.1, zorder=2)
            axB.plot([x - 0.028, x + 0.028], [hi, hi], color=shade, linewidth=1.1, zorder=2)
    axB.axhline(0, color=INK, linewidth=0.9)
    for x, tag in zip(xs, ("numeric", "anchored")):
        gap = cells["forward (n=10)"][0][x] - cells["reverse (n=10)"][0][x]
        lo = min(cells["forward (n=10)"][0][x], cells["reverse (n=10)"][0][x])
        hi = max(cells["forward (n=10)"][0][x], cells["reverse (n=10)"][0][x])
        axB.annotate("", xy=(x + 0.16, hi), xytext=(x + 0.16, lo),
                     arrowprops=dict(arrowstyle="<|-|>", linewidth=0.9, color=MUTE,
                                     shrinkA=0, shrinkB=0))
        axB.text(x + 0.20, (lo + hi) / 2, f"gap\n{gap:+.3f}", fontsize=SZ_ANNOT,
                 color=MUTE, va="center", ha="left",
                 bbox=dict(facecolor="#ffffff", edgecolor="none", pad=1.4))
    # v42 review: the leader line ran diagonally across the panel and read
    # as a plotted series. The sentence now sits in empty space and names
    # the column it is about.
    axB.text(0.62, -0.30, "under the anchored scale the label\nhelps reverse-worded items too",
             fontsize=SZ_DENSE, color=INK, ha="left", va="center",
             bbox=dict(facecolor="#ffffff", edgecolor="none", pad=1.6))
    axB.set_xticks(xs)
    axB.set_xticklabels(["numbers only", "numbers with\nverbal endpoints"])
    axB.set_xlim(-0.35, 1.75)
    axB.set_ylabel("mean country-label effect, Fisher $z$")
    axB.set_title("(a)  the country-label effect, by scale format", loc="left")
    axB.grid(axis="y", color=GRID, linewidth=0.7)
    axB.set_axisbelow(True)
    axB.legend(loc="lower left", fontsize=SZ_LEG, frameon=True,
               facecolor="#ffffff", edgecolor="none", framealpha=1.0)

    # Reposition panel (a): its left edge (including its own y-axis label)
    # aligns with the leftmost extent of panel (b)'s item labels, so the two
    # panels share the same printed width and (a) no longer wastes the label
    # gutter (reviewer, v29: "figure 6a bigger, ylabel from the page's left").
    fig.canvas.draw()
    R = fig.canvas.get_renderer()
    M = 0.015                      # printed left edge, in figure coordinates

    def tb(ax):
        return ax.get_tightbbox(R).transformed(fig.transFigure.inverted())

    # 1. pull the item panel in until its labels sit inside the canvas, so the
    #    tight bounding box stops expanding to the left and the two panels can
    #    be aligned in the saved figure and not only on the canvas.
    pa = axA.get_position()
    d = M - tb(axA).x0
    axA.set_position([pa.x0 + d, pa.y0, pa.width - d, pa.height])
    fig.canvas.draw()
    # 2. the four-cell panel starts at the same printed column, its y-axis
    #    label flush with the item labels above it (v42 review).
    pa, pb = axA.get_position(), axB.get_position()
    d = M - tb(axB).x0
    axB.set_position([pb.x0 + d, pb.y0, pa.x1 - (pb.x0 + d), pb.height])

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.figdir, f"{a.name}.{ext}"))
    print(f"wrote {a.name}.pdf / .png to {a.figdir}/")

    print("\nnumbers drawn, for checking against manuscript_numbers.csv")
    print(f"  {'cell':34s} {'mean z':>8s}   95% bootstrap CI over items")
    for name, (means, cis) in cells.items():
        for r, m, ci in zip(("numeric", "anchored"), means, cis):
            print(f"  {name + ', ' + r:34s} {m:+8.4f}   [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    for x, tag in zip(xs, ("numeric", "anchored")):
        print(f"  gap, {tag:28s} "
              f"{cells['forward (n=10)'][0][x] - cells['reverse (n=10)'][0][x]:+8.4f}")
    E.round(6).to_csv(os.path.join(a.figdir, f"{a.name}_data.csv"))
    print(f"\nper-item values written to {a.figdir}/{a.name}_data.csv")


if __name__ == "__main__":
    main()
