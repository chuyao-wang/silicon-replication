#!/usr/bin/env python3
"""
figstyle.py -- the single grayscale scheme every figure in the chapter uses.

Carried over from the March round's plots.py so that the restructured figures
remain visually continuous with the ones the supervisor has already seen.
Categories are distinguished by shade, marker and line style rather than colour,
and the sign of a bar by a solid dark fill against a white hatched one, so
nothing depends on colour reproduction. Type sizes are held between 9 and 12
points so that on-page sizes match across figures and the 12-point serif body
text.
"""
from __future__ import annotations

import os

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, GRID, MUTE = "#000000", "#d9d9d9", "#595959"
# PATCH T. The canvas colour, and the keyword dict for an opaque text
# background. Any label that sits inside the data must carry bbox=BOX, or the
# markers are drawn through the digits: see patch_S.py for the four figures
# where they were.
PAPER = "#ffffff"
BOX = dict(facecolor=PAPER, edgecolor="none", pad=1.4, alpha=1.0)
FWD_SHADE, REV_SHADE, EXCL_SHADE = "#1a1a1a", "#8c8c8c", "#c8c8c8"
FWD_MARK, REV_MARK = "o", "s"
FWD_LS, REV_LS = "-", "--"
POS_FILL, POS_EDGE = "#4d4d4d", "#000000"
NEG_FILL, NEG_EDGE, NEG_HATCH = "#ffffff", "#000000", "////"
SERIES, HIST_FILL = "#333333", "#a6a6a6"

# Around 10 pt everywhere (reviewer, v19), with an 11 pt bold panel title;
# figures embed at natural size, so these are true on-page point sizes.
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
    "atchctr": "Attachment to country", "atcherp": "Attachment to Europe",
    "cptppola": "Confident participating in politics",
    "euftf": "European unification: further vs too far",
    "freehms": "Gays and lesbians free to live", "gincdif": "Govt should reduce income differences",
    "happy": "How happy", "health": "Subjective general health",
    "hincfel": "Feeling about household income", "hmsacld": "Same-sex couples' right to adopt",
    "hmsfmlsh": "Ashamed if close family gay", "imbgeco": "Immigration good for economy",
    "imdfetn": "Allow immigrants of different race",
    "impcntr": "Allow immigrants from poorer countries",
    "imsmetn": "Allow immigrants of same race", "imueclt": "Immigrants enrich culture",
    "imwbcnt": "Immigrants make country better",
    "inprdsc": "People to discuss intimate matters", "polintr": "Interest in politics",
    "pplfair": "Most people try to be fair", "pplhlp": "Most people are helpful",
    "ppltrst": "Most people can be trusted", "psppipla": "Political system allows a say",
    "psppsgva": "Political system allows influence", "rlgatnd": "Religious attendance",
    "sclmeet": "Frequency of social meetings", "stfdem": "Satisfaction with democracy",
    "stfeco": "Satisfaction with the economy", "stfedu": "Satisfaction with education",
    "stfgov": "Satisfaction with government", "stfhlth": "Satisfaction with health services",
    "stflife": "Life satisfaction", "trstep": "Trust in the European Parliament",
    "trstlgl": "Trust in the legal system", "trstplc": "Trust in the police",
    "trstplt": "Trust in politicians", "trstprl": "Trust in parliament",
    "trstprt": "Trust in political parties", "trstun": "Trust in the United Nations",
    "vote": "Voted in last national election",
}

CNAME = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CH": "Switzerland",
    "CY": "Cyprus", "DE": "Germany", "EE": "Estonia", "ES": "Spain",
    "FI": "Finland", "FR": "France", "GB": "United Kingdom", "GR": "Greece",
    "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IL": "Israel",
    "IS": "Iceland", "IT": "Italy", "LT": "Lithuania", "LV": "Latvia",
    "ME": "Montenegro", "NL": "Netherlands", "NO": "Norway", "PL": "Poland",
    "PT": "Portugal", "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia",
    "SK": "Slovakia", "UA": "Ukraine",
}


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def sign_bars(ax, values, horizontal=True, width=0.72, offset=0.0):
    """Bars whose sign is encoded by fill, so they read in grayscale and print."""
    for i, v in enumerate(values):
        pos = v >= 0
        kw = dict(color=(POS_FILL if pos else NEG_FILL),
                  edgecolor=(POS_EDGE if pos else NEG_EDGE),
                  hatch=(None if pos else NEG_HATCH), linewidth=0.8, zorder=2)
        if horizontal:
            ax.barh(i + offset, v, height=width, **kw)
        else:
            ax.bar(i + offset, v, width=width, **kw)


def _in_view(ax, t, which):
    """True if a tick label's tick lies inside the current view limits."""
    try:
        lo, hi = (ax.get_xlim() if which == "x" else ax.get_ylim())
        lo, hi = min(lo, hi), max(lo, hi)
        pos = t.get_position()[0 if which == "x" else 1]
        return lo <= pos <= hi
    except Exception:
        return True


def check_layout(fig, min_overlap_pt2=90.0, verbose=True, min_pt=8.0):
    """Report overlapping text, text outside the canvas, and two further faults.

    A PARTIAL substitute for eyeballing the figure, and it must not be mistaken for
    the whole of one: it cannot judge whether a panel makes its argument, whether a
    shade ordering reads in print, or whether an annotation points at the right
    thing. Open the PDF as well.

    Five changes, each prompted by a fault this function had passed:

      * text is compared across groups, not only within them. A caption added
        with fig.text lives in the figure group and tick labels live in an axes
        group, so before patch S the two were never compared and a caption drawn
        over forty-two tick labels was reported as no overlap at all.
      * text drawn over plotted points is reported, and so is text that HIDES
        them behind an opaque background. A label with no opaque background in
        a dense swarm is unreadable; one with an opaque background is legible
        but conceals the data underneath it. Both are reported, separately, so
        the choice between them is made rather than discovered. Legend text is
        exempt: a legend covers the data on purpose.

      * tick labels are compared only when their tick lies inside the axes view.
        A label outside the view limits is never drawn, so reporting it as
        out-of-canvas is a false positive, and false positives are what make a
        checker get ignored.
      * type smaller than `min_pt` points is reported. It survives on screen and
        fails on the page.
      * plotted points outside the view limits are reported. That is silent
        clipping, and nothing else here would catch it.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    problems = []
    fw, fh = (fig.get_size_inches() * fig.dpi)

    def extent(t):
        try:
            if not t.get_visible() or not t.get_text().strip():
                return None
            return t.get_window_extent(renderer=rend)
        except Exception:
            return None

    groups = [("figure", [t for t in fig.texts])]
    for i, ax in enumerate(fig.axes):
        items = list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        items += [t for t in ax.get_xticklabels() if _in_view(ax, t, "x")]
        items += [t for t in ax.get_yticklabels() if _in_view(ax, t, "y")]
        if ax.get_legend() is not None:
            items += ax.get_legend().get_texts()
        groups.append((f"axes[{i}]", items))

        # Silent clipping: points drawn outside the view limits.
        x0, x1 = sorted(ax.get_xlim())
        y0, y1 = sorted(ax.get_ylim())
        for coll in list(ax.collections):
            try:
                o = np.asarray(coll.get_offsets(), dtype=float)
            except Exception:
                continue
            if o.size == 0:
                continue
            o = o[~np.isnan(o).any(axis=1)]
            if len(o) and ((o[:, 0] < x0).any() or (o[:, 0] > x1).any()
                           or (o[:, 1] < y0).any() or (o[:, 1] > y1).any()):
                problems.append(f"axes[{i}]: plotted points fall outside the view "
                                "limits and are silently clipped")
                break

    # PATCH S. Text over plotted points. Checked per axes, because that is the
    # only place data coordinates mean anything. Legend text is exempt -- a
    # legend is drawn over the data deliberately -- and so is text carrying an
    # opaque bbox, since that is the remedy this check exists to prompt.
    for i, ax in enumerate(fig.axes):
        pts = []
        for coll in list(ax.collections):
            try:
                o = np.asarray(coll.get_offsets(), dtype=float)
            except Exception:
                continue
            o = o[~np.isnan(o).any(axis=1)] if o.size else o
            if len(o):
                pts.append(np.asarray(coll.get_offset_transform().transform(o)))
        for ln in list(ax.lines):
            try:
                if str(ln.get_marker()) in ("", "None", "none"):
                    continue
                xy = np.column_stack([np.asarray(ln.get_xdata(), dtype=float),
                                      np.asarray(ln.get_ydata(), dtype=float)])
                xy = xy[~np.isnan(xy).any(axis=1)]
                if len(xy):
                    pts.append(np.asarray(ln.get_transform().transform(xy)))
            except Exception:
                continue
        if not pts:
            continue
        P = np.vstack(pts)
        exempt = set()
        if ax.get_legend() is not None:
            exempt = {id(t) for t in ax.get_legend().get_texts()}
        for t in list(ax.texts):
            if id(t) in exempt:
                continue
            # PATCH V. An opaque background is no longer an exemption. It
            # changes the fault from "the markers are drawn through the digits"
            # to "the markers are hidden", and the second is not obviously
            # better: it must be reported so the trade-off is on the record.
            bb = t.get_bbox_patch()
            opaque = (bb is not None and bb.get_alpha() not in (0,)
                      and bb.get_facecolor()[3] > 0.85)
            b = extent(t)
            if b is None or b.width <= 0:
                continue
            n = int(((P[:, 0] >= b.x0) & (P[:, 0] <= b.x1)
                     & (P[:, 1] >= b.y0) & (P[:, 1] <= b.y1)).sum())
            if n and opaque:
                problems.append(f"axes[{i}]: {n} plotted point(s) hidden behind the "
                                f"opaque background of {t.get_text()[:30]!r}")
            elif n:
                problems.append(f"axes[{i}]: {n} plotted point(s) fall inside the "
                                f"text {t.get_text()[:30]!r}, which has no opaque "
                                "background")

    # PATCH S. Compare every piece of text against every other piece, not only
    # against text in the same group. The group name is kept for the message.
    all_boxes = []
    for gname, items in groups:
        for t in items:
            b = extent(t)
            if b is not None and b.width > 0:
                all_boxes.append((gname, t, b))

    for gname, items in [("__all__", None)]:
        boxes = [(t, b) for _, t, b in all_boxes]
        names = [g for g, _, _ in all_boxes]
        for a in range(len(boxes)):
            ta, ba = boxes[a]
            gname = names[a]
            if ba.x0 < -2 or ba.y0 < -2 or ba.x1 > fw + 2 or ba.y1 > fh + 2:
                problems.append(f"{gname}: text outside canvas: "
                                f"{ta.get_text()[:44]!r}")
            try:
                if float(ta.get_fontsize()) < min_pt:
                    problems.append(f"{gname}: type below {min_pt:g} pt: "
                                    f"{ta.get_text()[:30]!r}")
            except Exception:
                pass
            for b in range(a + 1, len(boxes)):
                tb, bb = boxes[b]
                ov = (min(ba.x1, bb.x1) - max(ba.x0, bb.x0)) * \
                     (min(ba.y1, bb.y1) - max(ba.y0, bb.y0))
                if (min(ba.x1, bb.x1) > max(ba.x0, bb.x0)
                        and min(ba.y1, bb.y1) > max(ba.y0, bb.y0)
                        and ov > min_overlap_pt2):
                    where = (gname if gname == names[b]
                             else f"{gname} against {names[b]}")
                    problems.append(
                        f"{where}: overlap {ov:.0f} pt^2 between "
                        f"{ta.get_text()[:26]!r} and {tb.get_text()[:26]!r}")
    if verbose:
        if problems:
            print(f"  LAYOUT: {len(problems)} issue(s)")
            for x in problems[:14]:
                print(f"    {x}")
        else:
            print(f"  LAYOUT: no overlap between any two pieces of text, none over "
                  f"a plotted point, no clipping, no type below {min_pt:g} pt. "
                  "This is not a substitute for opening the PDF.")
    return problems


def place_caption(fig, text, x=0.052, fontsize=None, color=None,
                  gap_pt=12.0, margin_pt=10.0, va="top"):
    """Place a caption below everything already drawn, growing the figure to fit.

    PATCH T. The figures placed their captions at hard-coded figure fractions,
    which is a guess about how much room the tick labels will take. fig3 rotates
    forty-two item labels through ninety degrees, the longest of them thirty
    characters, and the guess was wrong by the height of the caption: all
    forty-two labels and all five caption lines were drawn on top of one another
    and check_layout could not see it, because a fig.text and a tick label lived
    in different comparison groups. fig2's caption ran off the canvas.

    This measures instead. It finds the lowest text already drawn -- tick labels
    included -- puts the caption gap_pt below it, and if the caption would fall
    off the bottom it makes the figure taller by the shortfall and shifts the
    axes up by the same number of PIXELS, so their size and aspect do not change
    and only the margin grows.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    dpi = fig.dpi
    fh_px = fig.get_size_inches()[1] * dpi

    lows = []
    for ax in fig.axes:
        cand = list(ax.get_xticklabels()) + [ax.xaxis.label]
        for t in cand:
            try:
                if t.get_visible() and t.get_text().strip():
                    lows.append(t.get_window_extent(renderer=rend).y0)
            except Exception:
                pass
    top_px = (min(lows) if lows else 0.0) - gap_pt * dpi / 72.0

    kw = dict(fontsize=fontsize if fontsize is not None else SZ_DENSE,
              color=color if color is not None else MUTE,
              va=va, linespacing=1.5)
    txt = fig.text(x, top_px / fh_px, text, **kw)

    fig.canvas.draw()
    need = margin_pt * dpi / 72.0 - txt.get_window_extent(renderer=rend).y0
    if need > 0:
        sp = fig.subplotpars
        new_h_px = fh_px + need
        fig.set_size_inches(fig.get_size_inches()[0], new_h_px / dpi)
        # Keep every axes the same size in pixels; the new space is all margin.
        fig.subplots_adjust(bottom=(sp.bottom * fh_px + need) / new_h_px,
                            top=(sp.top * fh_px + need) / new_h_px)
        txt.set_y((top_px + need) / new_h_px)
        fig.canvas.draw()
    return txt


def save(fig, figdir, name, check=True):
    os.makedirs(figdir, exist_ok=True)
    if check:
        check_layout(fig)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"{name}.{ext}"))
    print(f"wrote {name}.pdf / .png to {figdir}/")
