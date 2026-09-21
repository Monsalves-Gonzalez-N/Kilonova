"""C sin muestra de calibracion: el primer MJD valido del modelo de OU ES la minphase del template.

    C = minphase(plantilla) - (mjd_inicio_modelo - peak_mjd)/(1+z)

Si eso reproduce el C que medimos ajustando colores, la fase sale de dos numeros del catalogo y uno
de la plantilla. Cero ajuste, cero calibracion.
"""
import sys, warnings
import numpy as np, pandas as pd, h5py
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import sncosmo
from kilonova.simulation import openuniverse_parents as oup
from kilonova.simulation import intermediate_z_contaminants as izc
from kilonova.simulation.early_windows import model_from_hdf5_group

S = "/tmp/claude-1000/-home-nicolas-nico-git-Kilonova/0be13fce-0731-4085-bd8f-d900fc6438b0/scratchpad"
BANDS = ["F184", "H158", "J129", "Y106", "Z087"]
izc.register_sources()
cat = oup.read_parent_catalog("/home/nicolas/Dropbox/Kilonova/openuniverse2025")
cat = cat[cat.healpix == 10050].set_index("id")
by_index = izc.core_collapse_source_by_template_index(cat.reset_index())
span = {}
for ti, (name, _) in by_index.items():
    s = sncosmo.get_source(name)
    span[ti] = (float(s.minphase()), float(s.maxphase()))

fits = pd.read_csv(f"{S}/batch_z05.csv").set_index("object_id")
rows = []
with h5py.File("/home/nicolas/Dropbox/Kilonova/openuniverse2025/snana_10050.hdf5", "r") as h5:
    for object_id in fits.index:
        sid = object_id.split("_")[-1]
        if sid not in h5:
            continue
        model = model_from_hdf5_group(h5[sid], {"bands": BANDS})
        if not model:
            continue
        lo = min(m.min() for m, _ in model.values())
        hi = max(m.max() for m, _ in model.values())
        row = cat.loc[int(sid)]
        ti = int(row.template_index)
        if ti not in span:
            continue
        z = float(fits.loc[object_id, "z"])
        rows.append(dict(object_id=object_id, ti=ti, z=z,
                         C_lo=span[ti][0] - (lo - float(row.peak_mjd)) / (1 + z),
                         C_hi=span[ti][1] - (hi - float(row.peak_mjd)) / (1 + z)))
d = pd.DataFrame(rows)
t = pd.read_csv(f"{S}/phase_truth.csv")
t["ti"] = t.object_id.map(fits.template_index)
C_fit = t.groupby("ti").delta.median()
d["C_fit"] = d.ti.map(C_fit)
print(f"objetos: {len(d)}\n")
for col in ("C_lo", "C_hi"):
    e = d.dropna(subset=["C_fit"])
    r = e[col] - e.C_fit
    print(f"{col} - C(ajuste):  mediana {r.median():+.3f}  MAD {1.4826*np.median(np.abs(r-r.median())):.3f}  "
          f"|>1d| {100*(r.abs()>1).mean():.0f}%")
print("\npor plantilla (C del borde inferior vs C del ajuste):")
g = d.groupby("ti").agg(n=("z","size"), C_lo=("C_lo","median"),
                        lo_scatter=("C_lo", lambda s: 1.4826*np.median(np.abs(s-s.median()))),
                        C_fit=("C_fit","first"))
g["diff"] = g.C_lo - g.C_fit
print(g.round(2).to_string())
