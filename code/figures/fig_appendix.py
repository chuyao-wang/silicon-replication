#!/usr/bin/env python3
"""fig_appendix.py -- the three appendix figures, and the cell counts the text quotes.

    python3 fig_appendix.py                 # run from a staged directory

WHY THIS EXISTS
The manuscript cites Appendix Figures A1, A2 and A3, and quotes two counts from
the per-cell individual-level file. None of the four was produced by anything in
this package: the figures existed only as images in the document, and the counts
only in the sentence that reported them. A reader holding the package could not
regenerate them, which is the one thing a replication package has to allow.

  A1  per-variable country-level recovery, all 42 items, ranked
  A2  the country scatter for the best- and worst-recovered items, side by side
  A3  per-variable individual-level recovery, the counterpart of A1

Each writes the values behind it to a _data.csv, so a caption can be checked
without rerunning anything. rq3_cell_summary.csv carries the counts.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs  # noqa: E402

TAG = "qwen_1p"


def load(kind: str, data: str, tag: str = TAG) -> pd.DataFrame:
    for p in (os.path.join(data, f"silicon_full_{kind}_{tag}.csv"),
              os.path.join("results", f"silicon_full_{kind}_{tag}.csv")):
        if os.path.exists(p):
            return pd.read_csv(p)
    sys.exit(f"silicon_full_{kind}_{tag}.csv not found under {data} or results/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="handoff_upload")
    ap.add_argument("--figdir", default="figures")
    a = ap.parse_args()

    sc = load("country_scatter", a.data)
    rq3 = load("rq3", a.data)
    direction = None
    for p in (os.path.join(a.data, "item_direction_table.csv"),
              "item_direction_table.csv",
              os.path.join("results", "item_direction_table.csv")):
        if os.path.exists(p):
            direction = pd.read_csv(p).set_index("variable").direction
            break
    rev = (direction == "reverse") if direction is not None else None

    # ---------------------------------------------------------------- A1
    rows = []
    for v, g in sc.groupby("variable"):
        g = g.dropna(subset=["survey_mean", "silicon_mean"])
        if len(g) >= 10 and g.survey_mean.std() > 0 and g.silicon_mean.std() > 0:
            rows.append(dict(variable=v, r_bc=stats.pearsonr(g.survey_mean,
                                                             g.silicon_mean)[0]))
    A1 = pd.DataFrame(rows).sort_values("r_bc", ascending=False).reset_index(drop=True)
    A1["reverse"] = A1.variable.map(rev).fillna(False) if rev is not None else False

    fig, ax = fs.new(figsize=(6.3, 7.3)) if hasattr(fs, "new") else _fallback(6.3, 7.3)
    y = np.arange(len(A1))[::-1]
    for r, yy in zip(A1.itertuples(), y):
        ax.plot([0, r.r_bc], [yy, yy], color=fs.GRID, linewidth=0.9, zorder=1)
        ax.scatter(r.r_bc, yy, s=26, zorder=3,
                   marker=fs.REV_MARK if r.reverse else fs.FWD_MARK,
                   facecolors="none",
                   edgecolors=fs.REV_SHADE if r.reverse else fs.FWD_SHADE,
                   linewidths=0.9)
    ax.axvline(0, color=fs.INK, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([fs.VLABEL.get(v, v) for v in A1.variable], fontsize=fs.SZ_DENSE)
    ax.set_xlabel(r"between-country correlation  $r_{bc}$")
    ax.set_title("(A1)  per-variable country-level recovery, Qwen 1P", loc="left")
    ax.set_xlim(-1.0, 1.0)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.96, bottom=0.06)
    fs.save(fig, a.figdir, "figA1_item_country_recovery")
    A1.to_csv(os.path.join(a.figdir, "figA1_item_country_recovery_data.csv"), index=False)

    # ---------------------------------------------------------------- A3
    rc = [c for c in rq3.columns if "pearson" in c.lower()][0]
    pc = [c for c in rq3.columns if c.lower().startswith("p_")][0]
    rq3["_r"] = pd.to_numeric(rq3[rc], errors="coerce")
    rq3["_p"] = pd.to_numeric(rq3[pc], errors="coerce")
    A3 = (rq3.groupby("variable")._r.mean().rename("r_wc").reset_index()
          .sort_values("r_wc", ascending=False).reset_index(drop=True))
    A3["reverse"] = A3.variable.map(rev).fillna(False) if rev is not None else False

    fig, ax = _fallback(6.3, 7.3)
    y = np.arange(len(A3))[::-1]
    for r, yy in zip(A3.itertuples(), y):
        ax.plot([0, r.r_wc], [yy, yy], color=fs.GRID, linewidth=0.9, zorder=1)
        ax.scatter(r.r_wc, yy, s=26, zorder=3,
                   marker=fs.REV_MARK if r.reverse else fs.FWD_MARK,
                   facecolors="none",
                   edgecolors=fs.REV_SHADE if r.reverse else fs.FWD_SHADE,
                   linewidths=0.9)
    ax.axvline(0, color=fs.INK, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([fs.VLABEL.get(v, v) for v in A3.variable], fontsize=fs.SZ_DENSE)
    ax.set_xlabel(r"mean within-country correlation  $r_{wc}$")
    ax.set_title("(A3)  per-variable individual-level recovery, Qwen 1P", loc="left")
    fig.subplots_adjust(left=0.30, right=0.98, top=0.96, bottom=0.06)
    fs.save(fig, a.figdir, "figA3_item_individual_recovery")
    A3.to_csv(os.path.join(a.figdir, "figA3_item_individual_recovery_data.csv"), index=False)

    # ---------------------------------------------------------------- A2
    # side by side, compact (reviewer, v18)
    best, worst = A1.variable.iloc[0], A1.variable.iloc[-1]
    fig, axes = _fallback(6.3, 3.5, ncols=2)
    for ax, v in zip(axes, (best, worst)):
        g = sc[sc.variable == v].dropna(subset=["survey_mean", "silicon_mean"])
        r = stats.pearsonr(g.survey_mean, g.silicon_mean)[0]
        ax.scatter(g.survey_mean, g.silicon_mean, s=30, facecolors="none",
                   edgecolors=fs.INK, linewidths=0.9, zorder=3)
        lo = float(min(g.survey_mean.min(), g.silicon_mean.min()))
        hi = float(max(g.survey_mean.max(), g.silicon_mean.max()))
        pad = 0.06 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle=":",
                color=fs.MUTE, linewidth=1.0, zorder=2)
        b, c = np.polyfit(g.survey_mean, g.silicon_mean, 1)
        xs = np.array([g.survey_mean.min(), g.survey_mean.max()])
        ax.plot(xs, b * xs + c, linestyle="--", color=fs.INK, linewidth=1.2, zorder=2)
        ax.set_xlabel("human country mean")
        ax.set_ylabel("model country mean")
        ax.set_title(f"{fs.VLABEL.get(v, v)}  ($r_{{bc}}$ = {r:+.2f})", loc="left")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.15, wspace=0.28)
    fs.save(fig, a.figdir, "figA2_best_worst_scatter")
    sc[sc.variable.isin([best, worst])].to_csv(
        os.path.join(a.figdir, "figA2_best_worst_scatter_data.csv"), index=False)

    # -------------------------------------------- the counts the text quotes
    out = os.path.join("results", "analysis")
    os.makedirs(out, exist_ok=True)
    summ = pd.DataFrame([dict(
        condition=TAG,
        n_cells=int(rq3._r.notna().sum()),
        mean_r_wc=float(rq3._r.mean()),
        sd_r_wc=float(rq3._r.std()),
        median_r_wc=float(rq3._r.median()),
        n_positive=int((rq3._r > 0).sum()),
        n_p_lt_05=int((rq3._p < 0.05).sum()),
        n_items_negative_mean=int((A3.r_wc < 0).sum()),
        best_item=A3.variable.iloc[0], best_r=float(A3.r_wc.iloc[0]),
        worst_item=A3.variable.iloc[-1], worst_r=float(A3.r_wc.iloc[-1]))])
    summ.to_csv(os.path.join(out, "rq3_cell_summary.csv"), index=False)
    print(summ.to_string(index=False))
    print(f"\nfigures A1, A2, A3 written to {a.figdir}/; counts to {out}/rq3_cell_summary.csv")


def _fallback(w, h, ncols=1, nrows=1):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, h))
    n = ncols * nrows
    return (fig, ax) if n == 1 else (fig, list(np.atleast_1d(ax).ravel()))


if __name__ == "__main__":
    main()
