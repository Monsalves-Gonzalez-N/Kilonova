"""Cuantos objetos recupera ampliar REST_FRAME_PHASES, con el peak_phase real de cada plantilla."""
import sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from kilonova.simulation import openuniverse_parents as oup
from kilonova.simulation import intermediate_z_contaminants as izc
import sncosmo

S = "/tmp/claude-1000/-home-nicolas-nico-git-Kilonova/0be13fce-0731-4085-bd8f-d900fc6438b0/scratchpad"
izc.register_sources()
cat = oup.read_parent_catalog("/home/nicolas/Dropbox/Kilonova/openuniverse2025")
cat = cat[cat.healpix == 10050].set_index("id")
by_index = izc.core_collapse_source_by_template_index(cat.reset_index())

# peak_phase: la fase de maximo que usa el render, medida sobre la misma ventana de brillo
info = {}
for ti, (name, _) in by_index.items():
    s = sncosmo.get_source(name)
    ph = np.arange(s.minphase(), s.maxphase(), 0.25)
    lo, hi = izc.BRIGHTNESS_REST_WAVELENGTH_LIMITS
    wl = np.arange(lo, hi, 20.0)
    flux = np.array([np.trapezoid(s.flux(p, wl), wl) for p in ph])
    info[ti] = (float(ph[np.argmax(flux)]), float(s.minphase()), float(s.maxphase()), name)

fa = pd.read_csv(f"{S}/fast_anchor.csv")
b = pd.read_csv(f"{S}/batch_z05.csv").set_index("object_id")
t = pd.read_csv(f"{S}/phase_truth.csv"); t["ti"] = t.object_id.map(b.template_index)
C = t.groupby("ti").delta.median()
fa["ph0"] = fa.truth + fa.template_index.map(C)
fa["phN"] = fa.ph0 + 15.0 / (1.0 + fa.z)
fa = fa[fa.ph0.notna() & fa.template_index.isin(info)]
fa["peak"] = fa.template_index.map(lambda i: info[i][0])
fa["minph"] = fa.template_index.map(lambda i: info[i][1])
fa["maxph"] = fa.template_index.map(lambda i: info[i][2])

print(f"objetos: {len(fa)}   (peak_phase medido sobre {izc.BRIGHTNESS_REST_WAVELENGTH_LIMITS} A)")
print("\nventana del render      cubiertos    limitado por la plantilla")
for X in (20, 25, 30, 40, 60, 100, 200):
    start = np.maximum(fa.minph, fa.peak - X)
    end = np.minimum(fa.maxph, fa.peak + 70.0)
    ok = (fa.ph0 >= start) & (fa.phN <= end)
    tpl = (fa.ph0 >= fa.peak - X) & (fa.ph0 < fa.minph)
    print(f"  peak-{X:<4d} a peak+70   {100*ok.mean():5.1f}%      {100*tpl.mean():5.1f}%")
full = (fa.ph0 >= fa.minph) & (fa.phN <= fa.maxph)
print(f"\ntoda la plantilla        {100*full.mean():5.1f}%   <- techo real: lo que la SED tiene")
print("\npor plantilla, con peak-20 (hoy) vs toda la plantilla:")
g = fa.assign(hoy=(fa.ph0 >= np.maximum(fa.minph, fa.peak - 20)) & (fa.phN <= np.minimum(fa.maxph, fa.peak + 70)),
              todo=full).groupby("template_index").agg(
    n=("ph0", "size"), peak=("peak", "first"), minph=("minph", "first"),
    hoy=("hoy", lambda s: 100 * s.mean()), todo=("todo", lambda s: 100 * s.mean()))
g["source"] = [info[i][3].replace("ou-", "") for i in g.index]
g["gana"] = g.todo - g.hoy
print(g.sort_values("gana", ascending=False).head(16).round(1).to_string())
