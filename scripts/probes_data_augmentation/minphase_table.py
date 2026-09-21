"""Cobertura de fase por plantilla: donde empieza cada SED y cuantos objetos caen antes."""
import sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import sncosmo
from kilonova.simulation import openuniverse_parents as oup
from kilonova.simulation import intermediate_z_contaminants as izc

S = "/tmp/claude-1000/-home-nicolas-nico-git-Kilonova/0be13fce-0731-4085-bd8f-d900fc6438b0/scratchpad"
izc.register_sources()
cat = oup.read_parent_catalog("/home/nicolas/Dropbox/Kilonova/openuniverse2025")
cat = cat[cat.healpix == 10050].set_index("id")
by_index = izc.core_collapse_source_by_template_index(cat.reset_index())

fits = pd.read_csv(f"{S}/batch_z05.csv")
fa = pd.read_csv(f"{S}/fast_anchor.csv")
t = pd.read_csv(f"{S}/phase_truth.csv")
t["ti"] = t.object_id.map(fits.set_index("object_id").template_index)
C = t.groupby("ti").delta.median()
fa["ph0"] = fa.truth + fa.template_index.map(C)
fa["covered"] = np.isfinite(fa.cal_spread)

rows = []
for ti, (name, subtype) in sorted(by_index.items()):
    s = sncosmo.get_source(name)
    sub = fa[fa.template_index == ti]
    o = fits[fits.template_index == ti]
    rows.append(dict(ti=ti, subtype=subtype, source=name.replace("ou-", ""),
                     minphase=float(s.minphase()), maxphase=float(s.maxphase()),
                     span=float(s.maxphase() - s.minphase()), C=C.get(ti, np.nan),
                     n_obj=len(o), n_test=len(sub),
                     cubierto=100 * sub.covered.mean() if len(sub) else np.nan,
                     ph0_p5=sub.ph0.quantile(.05) if len(sub) else np.nan))
d = pd.DataFrame(rows).sort_values(["subtype", "minphase"])
pd.set_option("display.width", 220)
print(d.round(2).to_string(index=False))
print()
print(f"plantillas: {len(d)}   minphase: mediana {d.minphase.median():.1f}  rango {d.minphase.min():.0f} a {d.minphase.max():.0f} d")
print(f"span: mediana {d.span.median():.0f} d  rango {d.span.min():.0f} a {d.span.max():.0f} d")
print("\nminphase por subtipo:")
print(d.groupby("subtype").minphase.agg(["size", "median", "min", "max"]).round(1).to_string())
print("\ncorrelacion minphase vs %cubierto:",
      round(float(np.corrcoef(d.dropna(subset=["cubierto"]).minphase, d.dropna(subset=["cubierto"]).cubierto)[0, 1]), 3))
d.to_csv(f"{S}/minphase_table.csv", index=False)
