#!/usr/bin/env python3
"""
fig_two_margin.py -- Figure 4 after the two-margin integration (revise42).

    python3 fig_two_margin.py                       (inside .stage, as rebuild.sh calls it)
    python3 fig_two_margin.py --data <scatter dir> --lib <figstyle dir> --figdir <out>

Successor to the Figure 4 half of code/figures/fig_ladder.py; the ladder half
is retired with Appendix Figure A4 (user decision, 5 Aug 2026). Panel (a) is
unchanged: the identifying country-label contrast on all 42 items, computed
from clean_ablations.csv exactly as before. Panels (b) and (c) replace the
three-block summary with the two margins for every block of the 20-variable
partition: added alone to the sparse base (b), removed alone from the full
profile (c). The ascriptive base has no add margin, because it is the base.

Defaults match fig_ladder.py, the script this one replaces, so rebuild.sh
can call it with no arguments: data in handoff_upload/, figures out to
figures/, figstyle imported from the working directory. The block margins
are computed here from the arm files, so the figure does not depend on
another script having run first; --margins cross-checks against a shipped
two_margin_by_block.csv when one is given.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# figstyle ships in the package at code/lib; resolve it from --lib.
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="handoff_upload")
ap.add_argument("--margins", default=None,
                help="optional shipped two_margin_by_block.csv to cross-check against")
ap.add_argument("--lib", default=None,
                help="directory holding figstyle.py; omit when it is importable already")
ap.add_argument("--figdir", default="figures")
ap.add_argument("--name", default="fig4_backstory_ladder")
a = ap.parse_args()
if a.lib:
    sys.path.insert(0, a.lib)

import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

BLOCK_LABEL = {
    "country label": "country\nlabel",
    "NUTS region code": "NUTS\nregion code",
    "political identity": "political\nidentity",
    "ascriptive": "ascriptive\nbase",
    "education, activity, income": "education,\nactivity, income",
    "union, internet use": "union,\ninternet use",
    "household": "household",
    "migration, citizenship": "migration,\ncitizenship",
    "minority status": "minority\nstatus",
    "domicile": "domicile",
}
ORDER = list(BLOCK_LABEL)


def r_bc(tag, d):
    s = pd.read_csv(os.path.join(d, f"silicon_full_country_scatter_{tag}.csv"))
    return pd.Series({v: stats.pearsonr(g.survey_mean, g.silicon_mean)[0]
                      for v, g in s.groupby("variable")})


def z(x):
    return np.arctanh(np.clip(np.asarray(x, dtype=float), -0.999999, 0.999999))


dirt = pd.read_csv(os.path.join(a.data, "item_direction_table.csv")).set_index("variable")
rev = dirt["direction"].eq("reverse")

# ---- panel (a) data: the identifying contrast, from clean_ablations as before
abl = pd.read_csv(os.path.join(a.data, "clean_ablations.csv"))
row = abl[abl.block == "country label"].iloc[0]
w, wo = (r_bc(row.with_arm, a.data).sort_index(),
         r_bc(row.without_arm, a.data).sort_index())
country = pd.Series(z(w) - z(wo), index=w.index)
rev42 = rev.reindex(country.index).fillna(False)
fwd_items = country[~rev42].sort_values(ascending=False)
rev_items = country[rev42].sort_values(ascending=False)
n_fwd_up, n_rev_up = int((fwd_items > 0).sum()), int((rev_items > 0).sum())
assert (n_fwd_up, len(fwd_items)) == (27, 29), (n_fwd_up, len(fwd_items))
assert (n_rev_up, len(rev_items)) == (1, 13), (n_rev_up, len(rev_items))

# ---- panels (b, c) data: both margins per block, computed here
CONTRASTS = {
    "country label": (("qwen_1p_minimal", "qwen_1p_demo_only"),
                      (row.with_arm, row.without_arm)),
    "NUTS region code": (("qwen_1p_minimal_region", "qwen_1p_minimal"),
                         ("qwen_1p", "qwen_1p_full_noregion")),
    "political identity": (("qwen_1p_minimal_politics", "qwen_1p_minimal"),
                           ("qwen_1p", "qwen_1p_full_nopolitical")),
    "ascriptive": (None, ("qwen_1p", "qwen_1p_full_noascriptive")),
    "education, activity, income": (("qwen_1p_minimal_ses", "qwen_1p_minimal"),
                                    ("qwen_1p", "qwen_1p_full_nosocioecon")),
    "union, internet use": (("qwen_1p_minimal_membership", "qwen_1p_minimal"),
                            ("qwen_1p", "qwen_1p_full_nomembership")),
    "household": (("qwen_1p_minimal_household", "qwen_1p_minimal"),
                  ("qwen_1p", "qwen_1p_full_nohousehold")),
    "migration, citizenship": (("qwen_1p_minimal_civic", "qwen_1p_minimal"),
                               ("qwen_1p", "qwen_1p_full_nocivic")),
    "minority status": (("qwen_1p_minimal_minority", "qwen_1p_minimal"),
                        ("qwen_1p", "qwen_1p_full_nominority")),
    "domicile": (("qwen_1p_minimal_domicil", "qwen_1p_minimal"),
                 ("qwen_1p", "qwen_1p_full_nodomicil")),
}
recs = []
for block, (add, rem) in CONTRASTS.items():
    for margin, pair in (("add", add), ("remove", rem)):
        if pair is None:
            continue
        w, b = pair
        dz = pd.Series(z(r_bc(w, a.data).sort_index()) - z(r_bc(b, a.data).sort_index()),
                       index=r_bc(w, a.data).sort_index().index)
        rv = rev.reindex(dz.index).fillna(False)
        recs.append(dict(block=block, margin=margin, with_arm=w, base_arm=b,
                         fwd_median_dz=dz[~rv].median(), rev_median_dz=dz[rv].median(),
                         fwd_improved=int((dz[~rv] > 0).sum()), n_fwd=int((~rv).sum()),
                         rev_improved=int((dz[rv] > 0).sum()), n_rev=int(rv.sum())))
tm = pd.DataFrame(recs)
piv = {(r.block, r.margin): r for r in tm.itertuples()}
# the country remove margin is the identifying contrast in panel (a)
assert abs(piv[("country label", "remove")].fwd_median_dz - fwd_items.median()) < 1e-9
if a.margins:
    ship = pd.read_csv(a.margins).set_index(["block", "margin"])
    for k, r in piv.items():
        assert abs(ship.loc[k, "fwd_median_dz"] - r.fwd_median_dz) < 1e-5, k

fig = plt.figure(figsize=(6.3, 9.6))
gs = fig.add_gridspec(2, 2, height_ratios=[2.05, 1.0], hspace=0.15, wspace=0.16)
axI = fig.add_subplot(gs[0, :])
axAdd = fig.add_subplot(gs[1, 0])
axRem = fig.add_subplot(gs[1, 1])
fig.subplots_adjust(bottom=0.118, top=0.972, left=0.315, right=0.975)

# ------------------------------------------------ panel (a): item by item
ordered = pd.concat([fwd_items, rev_items])
ys = np.arange(len(ordered))[::-1]
axI.axvspan(-0.060, 0.060, color=fs.GRID, alpha=0.55, zorder=0)
axI.axvline(0, color=fs.INK, linewidth=0.9, zorder=2)
for y, (v, dzv) in zip(ys, ordered.items()):
    isrev = bool(rev42[v])
    shade = fs.REV_SHADE if isrev else fs.FWD_SHADE
    axI.plot([0, dzv], [y, y], color=shade, linewidth=1.0, zorder=3)
    axI.plot([dzv], [y], marker=fs.REV_MARK if isrev else fs.FWD_MARK,
             markersize=4.6, markerfacecolor=shade, markeredgecolor=shade,
             zorder=4)
div = ys[len(fwd_items) - 1] - 0.5
axI.axhline(div, color=fs.MUTE, linewidth=0.7, linestyle=(0, (4, 3)))
axI.set_yticks(ys)
axI.set_yticklabels([fs.VLABEL.get(v, v) for v in ordered.index], fontsize=8)
axI.set_ylim(-0.8, len(ordered) - 0.2)
axI.annotate(f"forward-coded items:\n{n_fwd_up} of {len(fwd_items)} improve",
             xy=(0.025, 0.990), xycoords="axes fraction", ha="left",
             va="top", fontsize=fs.SZ_DENSE, color=fs.MUTE, bbox=fs.BOX)
axI.annotate(f"reverse-coded items:\n{n_rev_up} of {len(rev_items)} improves",
             xy=(0.980, (div + 0.3) / len(ordered)),
             xycoords="axes fraction", ha="right", va="top",
             fontsize=fs.SZ_DENSE, color=fs.MUTE, bbox=fs.BOX)
axI.set_xlabel("country-label effect per item ($r_{bc}$, Fisher $z$): "
               "with the label minus without", fontsize=fs.SZ_DENSE)
axI.set_title("(a)  the country-label contrast, item by item", loc="left")
fs.grid(axI, axis="x")

# -------------------------------------- panels (b, c): the two margins
bw = 0.34
for ax, margin, ttl, xlim in ((axAdd, "add", "(b)  added to sparse base", (-0.62, 1.02)),
                              (axRem, "remove", "(c)  removed from full profile", (-0.30, 0.34))):
    yb = np.arange(len(ORDER))[::-1]
    ax.axvspan(-0.060, 0.060, color=fs.GRID, alpha=0.55, zorder=0)
    ax.axvline(0, color=fs.INK, linewidth=0.9, zorder=2)
    for y, block in zip(yb, ORDER):
        r = piv.get((block, margin))
        if r is None:
            ax.annotate("(base)", (0.03, y), ha="left", va="center",
                        fontsize=fs.SZ_DENSE, color=fs.MUTE, bbox=fs.BOX)
            continue
        ax.barh(y + bw / 2, r.fwd_median_dz, height=bw, color=fs.FWD_SHADE,
                edgecolor=fs.INK, linewidth=0.7, zorder=3,
                label="forward-worded items (29)" if (y == yb[0] and margin == "add") else None)
        ax.barh(y - bw / 2, r.rev_median_dz, height=bw, color=fs.NEG_FILL,
                edgecolor=fs.NEG_EDGE, linewidth=0.9, hatch=fs.NEG_HATCH,
                zorder=3,
                label="reverse-worded items (13)" if (y == yb[0] and margin == "add") else None)
        # label only values beyond the noise band; the in-band clutter is
        # in the shipped figure data, not on the page
        for yy, val in ((y + bw / 2, r.fwd_median_dz), (y - bw / 2, r.rev_median_dz)):
            if abs(val) <= 0.060:
                continue
            ax.annotate(f"{val:+.2f}", (val, yy), textcoords="offset points",
                        xytext=(4 if val >= 0 else -4, 0),
                        ha="left" if val >= 0 else "right", va="center",
                        fontsize=8, bbox=fs.BOX)
    ax.set_yticks(yb)
    ax.set_yticklabels([BLOCK_LABEL[b] for b in ORDER] if margin == "add" else [],
                       fontsize=8.5)
    ax.set_ylim(-0.7, len(ORDER) - 0.3)
    ax.set_title(ttl, loc="left", fontsize=fs.SZ_LABEL)
    ax.set_xlim(*xlim)
    fs.grid(ax, axis="x")
fig.text(0.645, 0.083, "median change per block ($r_{bc}$, Fisher $z$)",
         ha="center", fontsize=fs.SZ_DENSE)
axAdd.legend(loc="upper center", bbox_to_anchor=(1.08, -0.175), ncol=2,
             fontsize=fs.SZ_DENSE, handlelength=1.2, frameon=False)

fs.save(fig, a.figdir, a.name)
ordered.rename("country_label_dz").to_csv(
    os.path.join(a.figdir, f"{a.name}_country_item_dz.csv"))
tm.to_csv(os.path.join(a.figdir, f"{a.name}_data.csv"), index=False)
print(tm[["block", "margin", "fwd_median_dz", "rev_median_dz"]].round(3).to_string(index=False))
