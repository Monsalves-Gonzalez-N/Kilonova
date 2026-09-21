"""La constante por plantilla, ¿es el maximo de la plantilla en alguna banda?

Si lo es, la fase se calcula: phase0 = (mjd0 - peak_mjd)/(1+z) + C(plantilla), sin ajuste.
"""
import sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import sncosmo
from kilonova.simulation import intermediate_z_contaminants as izc
from kilonova.simulation import openuniverse_parents as oup

S = "/tmp/claude-1000/-home-nicolas-nico-git-Kilonova/0be13fce-0731-4085-bd8f-d900fc6438b0/scratchpad"
t = pd.read_csv(f"{S}/phase_truth.csv")
t["res"] = t.delta - t.groupby("source").delta.transform("median")
C = t.groupby("source").agg(n=("delta", "size"), C=("delta", "median"))
C = C[C.n >= 8]

izc.register_sources()
bands = {"B": "bessellb", "V": "bessellv", "R062": "f062", "Y106": "f106"}
rows = []
for source_name, row in C.iterrows():
    src = sncosmo.get_source(source_name)
    ph = np.arange(src.minphase(), src.maxphase(), 0.25)
    out = {"source": source_name, "n": int(row.n), "C": row.C}
    for label, bandname in bands.items():
        try:
            model = sncosmo.Model(source=src)
            mag = model.bandmag(bandname, "ab", ph)
            ok = np.isfinite(mag)
            out[f"peak_{label}"] = float(ph[ok][np.argmin(mag[ok])]) if ok.any() else np.nan
        except Exception:
            out[f"peak_{label}"] = np.nan
    out["minphase"] = float(src.minphase())
    rows.append(out)
d = pd.DataFrame(rows)
for label in bands:
    col = d[f"peak_{label}"]
    print(f"C vs -peak_{label}:  corr {np.corrcoef(d.C, -col)[0,1]:+.3f}   "
          f"resid(C + peak) mediana {np.median(d.C + col):+.3f}  MAD {1.4826*np.median(np.abs(d.C + col - np.median(d.C + col))):.3f} d")
print(f"C vs minphase: corr {np.corrcoef(d.C, d.minphase)[0,1]:+.3f}")
print()
print(d.round(2).to_string(index=False))
d.to_csv(f"{S}/template_C.csv", index=False)
