"""Cuanta fase se pierde por el np.ceil del borde de entrada, leyendo los .SED crudos."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, importlib.util
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("b", "scripts/build_openuniverse_cc_templates.py")
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

MODELS = Path("/home/nicolas/nico/openuniverse_models/MODELS-1_TRANSIENT_SED/NON1ASED.V19_CC+HostXT_WAVEEXT")
S = "/tmp/claude-1000/-home-nicolas-nico-git-Kilonova/0be13fce-0731-4085-bd8f-d900fc6438b0/scratchpad"
a = np.load("data/openuniverse/cc_templates.npz", allow_pickle=True)
stored = {int(ti): (str(n), a[f"phase_{k}"]) for k, (ti, n) in enumerate(zip(a["template_indices"], a["template_names"]))}
file_by_index, _ = b.read_template_types(MODELS)

rows = []
for ti, (name, ph) in sorted(stored.items()):
    path = MODELS / file_by_index[ti]
    if not path.exists():
        path = Path(str(path) + ".gz")
    phase, wl, flux = b.read_sed_file(path)
    first, last = b.complete_phase_run(phase, wl, flux)
    rows.append(dict(ti=ti, source=name, file_first=float(phase[first]), stored_first=float(ph[0]),
                     perdido_inicio=float(ph[0] - phase[first]),
                     file_last=float(phase[last]), stored_last=float(ph[-1]),
                     perdido_final=float(phase[last] - ph[-1]),
                     file_min=float(phase[0]), recortado_por_trim=float(phase[first] - phase[0])))
d = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(d.round(3).to_string(index=False))
print()
print(f"perdido en el borde de entrada por el ceil: mediana {d.perdido_inicio.median():.3f} d, "
      f"max {d.perdido_inicio.max():.3f} d, en {(d.perdido_inicio > 0.01).sum()} de {len(d)} plantillas")
print(f"perdido en el borde de salida por el floor: mediana {d.perdido_final.median():.3f} d, max {d.perdido_final.max():.3f} d")
print(f"recortado ADEMAS por complete_phase_run:    {(d.recortado_por_trim > 0.01).sum()} plantillas, "
      f"mediana {d[d.recortado_por_trim > 0.01].recortado_por_trim.median():.1f} d")
d.to_csv(f"{S}/ceil_loss.csv", index=False)

# contra el exceso medido objeto a objeto
fa = pd.read_csv(f"{S}/fast_anchor.csv"); bt = pd.read_csv(f"{S}/batch_z05.csv").set_index("object_id")
t = pd.read_csv(f"{S}/phase_truth.csv"); t["ti"] = t.object_id.map(bt.template_index)
C = t.groupby("ti").delta.median()
fa["ph0"] = fa.truth + fa.template_index.map(C)
fa["minph"] = fa.template_index.map(d.set_index("ti").stored_first)
e = fa[fa.ph0 < fa.minph].groupby("template_index").apply(lambda g: (g.minph - g.ph0).median())
cmp = pd.DataFrame({"exceso_medido": e, "perdido_por_ceil": d.set_index("ti").perdido_inicio}).dropna()
print(f"\ncorrelacion exceso medido vs perdida por ceil, {len(cmp)} plantillas: "
      f"{np.corrcoef(cmp.exceso_medido, cmp.perdido_por_ceil)[0,1]:+.3f}")
print(cmp.round(3).to_string())
