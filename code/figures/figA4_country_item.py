import os, sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, sys.argv[1])
import figstyle as fs
import matplotlib.pyplot as plt
D = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "summary")
def r_bc(tag):
    s=pd.read_csv(f"{D}/silicon_full_country_scatter_{tag}.csv")
    return pd.Series({v:stats.pearsonr(g.survey_mean,g.silicon_mean)[0] for v,g in s.groupby("variable")}).sort_index()
def z(x): return np.arctanh(np.clip(np.asarray(x,float),-0.999999,0.999999))
rev=pd.read_csv(f"{D}/item_direction_table.csv").set_index("variable")["direction"].eq("reverse")
abl=pd.read_csv(f"{D}/clean_ablations.csv"); row=abl[abl.block=="country label"].iloc[0]
w,wo=r_bc(row.with_arm),r_bc(row.without_arm)
country=pd.Series(z(w)-z(wo),index=w.index)
rv=rev.reindex(country.index).fillna(False)
fwd=country[~rv].sort_values(ascending=False); rvs=country[rv].sort_values(ascending=False)
assert (int((fwd>0).sum()),len(fwd))==(27,29) and (int((rvs>0).sum()),len(rvs))==(1,13)
ordered=pd.concat([fwd,rvs]); ys=np.arange(len(ordered))[::-1]
fig,ax=plt.subplots(figsize=(6.3,7.6))
fig.subplots_adjust(left=0.335,right=0.975,top=0.955,bottom=0.085)
ax.axvspan(-0.060,0.060,color=fs.GRID,alpha=0.55,zorder=0)
ax.axvline(0,color=fs.INK,linewidth=0.9,zorder=2)
for y,(v,dzv) in zip(ys,ordered.items()):
    isrev=bool(rv[v]); shade=fs.REV_SHADE if isrev else fs.FWD_SHADE
    ax.plot([0,dzv],[y,y],color=shade,linewidth=1.0,zorder=3)
    ax.plot([dzv],[y],marker=fs.REV_MARK if isrev else fs.FWD_MARK,markersize=4.6,
            markerfacecolor=shade,markeredgecolor=shade,zorder=4)
div=ys[len(fwd)-1]-0.5
ax.axhline(div,color=fs.MUTE,linewidth=0.7,linestyle=(0,(4,3)))
ax.set_yticks(ys); ax.set_yticklabels([fs.VLABEL.get(v,v) for v in ordered.index],fontsize=8)
ax.set_ylim(-0.8,len(ordered)-0.2)
ax.annotate(f"forward-coded items:\n{int((fwd>0).sum())} of {len(fwd)} improve",
            xy=(0.025,0.990),xycoords="axes fraction",ha="left",va="top",
            fontsize=fs.SZ_DENSE,color=fs.MUTE,bbox=fs.BOX)
ax.annotate(f"reverse-coded items:\n{int((rvs>0).sum())} of {len(rvs)} improves",
            xy=(0.980,(div+0.3)/len(ordered)),xycoords="axes fraction",ha="right",va="top",
            fontsize=fs.SZ_DENSE,color=fs.MUTE,bbox=fs.BOX)
ax.set_xlabel("country-label effect per item ($r_{bc}$, Fisher $z$): with the label minus without",
              fontsize=fs.SZ_DENSE)
ax.set_title("(A4)  the country-label contrast, item by item",loc="left")
fs.grid(ax,axis="x")
fs.check_layout(fig)
fig.savefig(sys.argv[2],dpi=300); print("wrote",sys.argv[2])
