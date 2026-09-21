"""Primer test: ¿los colores de un ejemplo del training set fijan la fase de su plantilla?

Toma un objeto de early_windows_deep, su z y su plantilla (la que dice su template_index).
Renderiza la plantilla a ESE z con una normalizacion arbitraria, y busca la fase en que los
colores del modelo coinciden con los del objeto. La normalizacion no entra: los colores no
dependen de ella. Lo que sobra despues, magnitud observada - magnitud del modelo, es el zero point.
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq, sys

from kilonova.simulation import openuniverse_parents as oup
from kilonova.simulation import intermediate_z_contaminants as izc

HEALPIX = 10050
CATALOG_DIR = "/home/nicolas/Dropbox/Kilonova/openuniverse2025"
WINDOWS = "data/openuniverse/early_windows_deep.parquet"
MAG = sys.argv[1] if len(sys.argv) > 1 else "mag_true"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# --- 1. un objeto del training set -------------------------------------------------------------
f = pq.ParquetFile(WINDOWS)
cols = ["object_id","gentype","label","z_CMB","epoch","days_since_detection","band","observed",MAG]
wanted = None
for g in range(f.num_row_groups):
    d = f.read_row_group(g, columns=cols).to_pandas()
    d = d[d.object_id.str.startswith(f"snana_{HEALPIX}_") & d.label.isin(["SN II","SN Ib","SN Ic"])]
    if len(d):
        rng = np.random.default_rng(SEED)
        oid = rng.choice(sorted(d.object_id.unique()))
        wanted = d[d.object_id == oid].sort_values(["epoch"])
        break
assert wanted is not None
obj = wanted.iloc[0]
z = float(obj.z_CMB)
print(f"objeto {obj.object_id}  label={obj.label}  z={z:.4f}  usando {MAG}")

obs = wanted[wanted.observed].copy()
print(f"{len(obs)} medidas = {obs.epoch.nunique()} epocas x 3 bandas; "
      f"bandas {sorted(obs.band.unique())}")

# --- 2. su plantilla ---------------------------------------------------------------------------
catalog = oup.read_parent_catalog(CATALOG_DIR)
catalog = catalog[catalog.healpix == HEALPIX]
row = catalog[catalog.id == int(obj.object_id.split("_")[-1])]
assert len(row) == 1, f"el objeto no esta en el catalogo padre ({len(row)} filas)"
row = row.iloc[0]
by_index = izc.core_collapse_source_by_template_index(catalog)
source_name, sub = by_index[int(row.template_index)]
print(f"plantilla: template_index={int(row.template_index)} -> {source_name} ({sub})")

# --- 3. el modelo a ESE z, normalizacion arbitraria ---------------------------------------------
realization = izc.realization_from_parent(row, 0, z, by_index, np.random.default_rng(0))
realization = izc.apply_brightness_offset(realization, 0.0, float("nan"), 0)  # M_ref, arbitrario
curves = izc.rendered_band_curves(realization, z)
print(f"modelo renderizado en {len(curves)} bandas, "
      f"{len(next(iter(curves.values()))[0])} fases en reposo")

# --- 4. buscar la fase donde coinciden los COLORES ----------------------------------------------
# Las 4 epocas estan separadas por days_since_detection (dias de observador). Una sola incognita:
# la fase en reposo de la primera epoca. Los colores se toman contra el ancla de cada epoca.
phases_model = next(iter(curves.values()))[0]
interp = {b: (lambda ph, p=curves[b][0], m=curves[b][1]: np.interp(ph, p, m, np.nan, np.nan))
          for b in curves}
epochs = obs.groupby("epoch")
day0 = obs.days_since_detection.min()

def residual(phase0):
    """rms de los colores observados menos los del modelo, sobre las 4 epocas."""
    diffs = []
    for _, e in epochs:
        ph = phase0 + (e.days_since_detection.iloc[0] - day0) / (1.0 + z)
        bands = list(e.band); mags = list(e[MAG])
        if any(b not in interp for b in bands): return np.nan
        model = [interp[b](ph) for b in bands]
        if not np.all(np.isfinite(model)): return np.nan
        # colores: todo contra la primera banda de la epoca
        for i in range(1, len(bands)):
            diffs.append((mags[i] - mags[0]) - (model[i] - model[0]))
    return np.sqrt(np.mean(np.square(diffs))) if diffs else np.nan

grid = np.arange(phases_model.min(), phases_model.max(), 0.25)
scores = np.array([residual(p) for p in grid])
ok = np.isfinite(scores)
print(f"fases probadas: {ok.sum()} de {len(grid)}")
best = grid[ok][np.argmin(scores[ok])]
print(f"\n==> mejor fase: {best:+.2f} d en reposo   residuo de color = {np.nanmin(scores):.4f} mag")

# cuantos minimos hay (degeneracion subida/bajada)
s = scores[ok]; g = grid[ok]
mins = [g[i] for i in range(1,len(s)-1) if s[i]<s[i-1] and s[i]<s[i+1] and s[i]<np.nanmin(s)*3]
print(f"    minimos locales competitivos (<3x el mejor): {len(mins)} -> {np.round(mins,1)}")

# --- 5. el zero point que sobra -----------------------------------------------------------------
offsets = []
for _, e in epochs:
    ph = best + (e.days_since_detection.iloc[0] - day0) / (1.0 + z)
    for b, m in zip(e.band, e[MAG]):
        offsets.append((int(e.epoch.iloc[0]), b, m - interp[b](ph)))
off = pd.DataFrame(offsets, columns=["epoch","band","offset"])
print(f"\nzero point (observado - modelo), sobre {len(off)} medidas:")
print(f"    mediana = {off.offset.median():+.4f} mag   dispersion = {off.offset.std():.4f} mag "
      f"   rango = {off.offset.max()-off.offset.min():.4f} mag")
print(off.pivot_table(index="epoch", columns="band", values="offset").round(3).to_string())
print(f"\nM_abs implicada = {izc.REFERENCE_ABSOLUTE_MAGNITUDE + off.offset.median():.3f} "
      f"(referencia {izc.REFERENCE_ABSOLUTE_MAGNITUDE})")
