import os, sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, sys.argv[1])
import figstyle as fs
import matplotlib.pyplot as plt
D = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "summary")
rev=pd.read_csv(f"{D}/item_direction_table.csv").set_index("variable")["direction"].eq("reverse")
def med(tag):
    s=pd.read_csv(f"{D}/silicon_full_country_scatter_{tag}.csv")
    r=pd.Series({v:stats.pearsonr(g.survey_mean,g.silicon_mean)[0] for v,g in s.groupby("variable")})
    rv=rev.reindex(r.index).fillna(False); return r[~rv].median(), r[rv].median()
_H = os.path.dirname(os.path.abspath(__file__))
_SC = next(q for q in (os.path.join(_H, "swap_scores.csv"),
                       os.path.join(_H, "..", "..", "results", "analysis", "swap_scores.csv"))
           if os.path.exists(q))
sc=pd.read_csv(_SC)
true_f,true_r = med("qwen_1p_full_noregion")
none_f,none_r = med("qwen_1p_full_nocountry")
lab_f,lab_r   = sc[~sc.reverse].r_label.median(), sc[sc.reverse].r_label.median()
tru_f,tru_r   = sc[~sc.reverse].r_true.median(), sc[sc.reverse].r_true.median()

ROWS = [("the true name",                              true_f, true_r),
        ("a wrong name, scored\nagainst the named country",  lab_f, lab_r),
        ("a wrong name, scored\nagainst the true country",   tru_f, tru_r)]
fig, ax = plt.subplots(figsize=(6.3, 3.35))
fig.subplots_adjust(left=0.30, right=0.955, top=0.855, bottom=0.30)
ys = np.arange(len(ROWS))[::-1]
ax.axvline(0, color=fs.INK, linewidth=0.8, zorder=1)
ax.axvline(none_f, color=fs.MUTE, linewidth=1.0, linestyle=(0,(5,3)), zorder=1)
ax.annotate("no country name\nin the prompt at all", xy=(none_f, -0.52), ha="center",
            va="bottom", fontsize=9, color=fs.MUTE, bbox=fs.BOX)
for y,(lbl,f_,r_) in zip(ys, ROWS):
    ax.plot([0,f_],[y+0.17,y+0.17], color=fs.FWD_SHADE, linewidth=1.8, zorder=2)
    ax.plot([f_],[y+0.17], marker=fs.FWD_MARK, markersize=5.6, color=fs.FWD_SHADE, zorder=3)
    ax.annotate(f"{f_:+.2f}", (f_,y+0.17), textcoords="offset points", xytext=(8,0),
                ha="left", va="center", fontsize=9.5, bbox=fs.BOX)
    ax.plot([0,r_],[y-0.17,y-0.17], color=fs.REV_SHADE, linewidth=1.8, zorder=2)
    ax.plot([r_],[y-0.17], marker=fs.REV_MARK, markersize=4.8, color=fs.REV_SHADE, zorder=3)
ax.set_yticks(ys); ax.set_yticklabels([l for l,_,_ in ROWS], fontsize=9)
ax.set_ylim(-0.75, len(ROWS)-0.38)
ax.set_xlim(-0.45, 0.70)
ax.set_xlabel("between-country correlation $r_{bc}$ (median item)", fontsize=fs.SZ_DENSE)
ax.set_title("a wrong name costs the country the respondents came from", loc="left")
fs.grid(ax, axis="x")
h_f = plt.Line2D([0],[0], color=fs.FWD_SHADE, marker=fs.FWD_MARK, markersize=5.6, linewidth=1.8,
                 label="forward-worded items (29)")
h_r = plt.Line2D([0],[0], color=fs.REV_SHADE, marker=fs.REV_MARK, markersize=4.8, linewidth=1.8,
                 label="reverse-worded items (13)")
ax.legend(handles=[h_f,h_r], loc="upper center", bbox_to_anchor=(0.5,-0.36), ncol=2,
          fontsize=fs.SZ_DENSE, handlelength=1.8, frameon=False)
fs.check_layout(fig)
fig.savefig(sys.argv[2], dpi=300); print("wrote", sys.argv[2])
