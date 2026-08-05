import sys, numpy as np, pandas as pd
from collections import Counter
C=["variable","cntry","scale","silicon_response","raw_response"]
for t in sys.argv[1:]:
    n=0; k=Counter(); w=Counter(); Lf=[]
    for d in pd.read_csv(f"results/silicon_full_raw_{t}_seed888.csv",usecols=lambda c:c in C,chunksize=200000,low_memory=False):
        r=d.raw_response.astype(str); f=pd.to_numeric(d.silicon_response,errors="coerce").isna()
        v=pd.to_numeric(r.str.extract(r"(-?\d+\.?\d*)",expand=False),errors="coerce")
        lo=pd.to_numeric(d.scale.astype(str).str.split("-").str[0],errors="coerce")
        hi=pd.to_numeric(d.scale.astype(str).str.split("-").str[-1],errors="coerce")
        e=d.raw_response.isna()|r.str.strip().isin(["","nan","NaN","None","<NA>"])
        c=np.where(~f,"ok",np.where(e,"empty",np.where(v.isna(),"no digit (refusal/prose)",
          np.where((v<lo)|(v>hi),"OUT OF SCALE RANGE","other"))))
        n+=len(d); k.update(Counter(c))
        w.update(Counter(zip(np.where(hi-lo+1>=11,"11pt","short"),c)))
        Lf.append(r.str.len().to_numpy()[f.to_numpy()].astype(np.int32))
        print(f"  {t} {n:,}",flush=True)
    print(f"\n=== {t}: {n:,} ===",flush=True)
    for a,b in k.most_common(): print(f"  {a:26s} {b:8,d} {100*b/n:6.2f}%",flush=True)
    for a,b in sorted(w.items()): print(f"  {a[0]:6s} {a[1]:26s} {100*b/n:6.2f}%",flush=True)
    L=np.concatenate(Lf); m=int(L.max())
    print(f"  failed len: med {np.median(L):.0f} max {m} | within 2% of max {100*(L>=.98*m).mean():.1f}% (high=truncation)",flush=True)
print("DONE",flush=True)
