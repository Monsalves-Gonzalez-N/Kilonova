"""Muestra aleatoria: recuperamos la plantilla de cada objeto, separado POR plantilla.

Por objeto: fija la fase con los colores de sus 4 epocas x 3 bandas (mag_true, sin ruido), y mide
el zero point que sobra. Si el metodo es sano, la dispersion del zero point entre las 12 medidas
tiene que quedar en el piso de cuantizacion de mag_true (~1 dia de grilla), y tiene que quedar ahi
para TODAS las plantillas, no en promedio.
"""
import sys, warnings
import numpy as np, pandas as pd, pyarrow.parquet as pq
from multiprocessing import Pool

warnings.filterwarnings("ignore")
HEALPIX = 10050
CATALOG_DIR = "/home/nicolas/Dropbox/Kilonova/openuniverse2025"
WINDOWS = "data/openuniverse/early_windows_deep.parquet"
N_OBJECTS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8

_state = {}

def init():
    from kilonova.simulation import openuniverse_parents as oup
    from kilonova.simulation import intermediate_z_contaminants as izc
    catalog = oup.read_parent_catalog(CATALOG_DIR)
    catalog = catalog[catalog.healpix == HEALPIX].set_index("id")
    _state["catalog"] = catalog
    _state["by_index"] = izc.core_collapse_source_by_template_index(catalog.reset_index())
    _state["izc"] = izc

def one(payload):
    object_id, z, rows = payload
    izc, catalog, by_index = _state["izc"], _state["catalog"], _state["by_index"]
    try:
        row = catalog.loc[int(object_id.split("_")[-1])]
    except KeyError:
        return None
    row = row.copy(); row["id"] = int(object_id.split("_")[-1])
    template_index = int(row.template_index)
    if template_index not in by_index:
        return None
    source_name, subtype = by_index[template_index]
    realization = izc.realization_from_parent(row, 0, z, by_index, np.random.default_rng(0))
    realization = izc.apply_brightness_offset(realization, 0.0, float("nan"), 0)
    curves = izc.rendered_band_curves(realization, z)
    if not curves:
        return None
    interp = {b: (curves[b][0], curves[b][1]) for b in curves}
    def mag(b, ph):
        p, m = interp[b]
        return np.interp(ph, p, m, np.nan, np.nan)
    epochs = [(d, list(g["band"]), list(g["mag_true"])) for d, g in rows.groupby("days_since_detection")]
    if len(epochs) < 2:
        return None
    day0 = min(d for d, _, _ in epochs)
    phases = next(iter(interp.values()))[0]
    def residual(phase0):
        diffs = []
        for d, bands, mags in epochs:
            ph = phase0 + (d - day0) / (1.0 + z)
            if any(b not in interp for b in bands): return np.nan
            model = [mag(b, ph) for b in bands]
            if not np.all(np.isfinite(model)): return np.nan
            for i in range(1, len(bands)):
                diffs.append((mags[i] - mags[0]) - (model[i] - model[0]))
        return np.sqrt(np.mean(np.square(diffs))) if diffs else np.nan
    grid = np.arange(phases.min(), phases.max(), 0.25)
    scores = np.array([residual(p) for p in grid])
    ok = np.isfinite(scores)
    if ok.sum() < 10:
        return None
    best = grid[ok][np.argmin(scores[ok])]
    colour_rms = float(np.nanmin(scores))
    s, g = scores[ok], grid[ok]
    n_minima = sum(1 for i in range(1, len(s)-1)
                   if s[i] < s[i-1] and s[i] < s[i+1] and s[i] < colour_rms*3)
    offsets, per_epoch = [], []
    for d, bands, mags in epochs:
        ph = best + (d - day0) / (1.0 + z)
        this = [m - mag(b, ph) for b, m in zip(bands, mags)]
        per_epoch.append(this)
        offsets.extend(this)
    offsets = np.array(offsets)
    # DENTRO de una epoca solo varian las bandas; ENTRE epocas solo varia la fase.
    band_spread = float(np.mean([max(e) - min(e) for e in per_epoch]))
    epoch_medians = [float(np.median(e)) for e in per_epoch]
    epoch_spread = float(max(epoch_medians) - min(epoch_medians))
    return dict(object_id=object_id, z=z, template_index=template_index, source=source_name,
                subtype=subtype, phase=best, colour_rms=colour_rms, n_minima=n_minima,
                n_meas=len(offsets), offset=float(np.median(offsets)),
                offset_spread=float(offsets.max()-offsets.min()),
                band_spread=band_spread, epoch_spread=epoch_spread, grid_min=float(phases.min()))

def main():
    f = pq.ParquetFile(WINDOWS)
    cols = ["object_id","label","z_CMB","days_since_detection","band","observed","mag_true"]
    frames = []
    for gi in range(f.num_row_groups):
        d = f.read_row_group(gi, columns=cols).to_pandas()
        d = d[d.object_id.str.startswith(f"snana_{HEALPIX}_") & d.observed
              & d.label.isin(["SN II","SN Ib","SN Ic"]) & (d.z_CMB > 0.5)]
        if len(d): frames.append(d)
        if sum(x.object_id.nunique() for x in frames) > N_OBJECTS*3: break
    d = pd.concat(frames)
    rng = np.random.default_rng(12345)
    ids = rng.choice(sorted(d.object_id.unique()), size=min(N_OBJECTS, d.object_id.nunique()),
                     replace=False)
    d = d[d.object_id.isin(ids)]
    payloads = [(oid, float(g.z_CMB.iloc[0]), g[["days_since_detection","band","mag_true"]])
                for oid, g in d.groupby("object_id")]
    print(f"objetos: {len(payloads)}  workers: {WORKERS}", flush=True)
    with Pool(WORKERS, initializer=init) as pool:
        out = [r for r in pool.imap_unordered(one, payloads, chunksize=4) if r]
    r = pd.DataFrame(out)
    r.to_csv("/tmp/claude-1000/-home-nicolas-nico-git-Kilonova/0be13fce-0731-4085-bd8f-d900fc6438b0/scratchpad/batch_z05.csv", index=False)
    print(f"\najustados {len(r)} de {len(payloads)}  |  z de {r.z.min():.2f} a {r.z.max():.2f}\n")
    print("=== GLOBAL ===")
    print(f"  residuo de color   mediana {r.colour_rms.median():.4f}  p95 {r.colour_rms.quantile(.95):.4f}  max {r.colour_rms.max():.4f} mag")
    print(f"  spread zero point  mediana {r.offset_spread.median():.4f}  p95 {r.offset_spread.quantile(.95):.4f}  max {r.offset_spread.max():.4f} mag")
    print(f"  objetos con >1 minimo competitivo: {(r.n_minima>1).sum()} de {len(r)}")
    print("\n=== POR PLANTILLA ===")
    g = r.groupby(["subtype","source"]).agg(
        n=("object_id","size"), z_med=("z","median"),
        col_med=("colour_rms","median"), col_max=("colour_rms","max"),
        spr_med=("offset_spread","median"), spr_max=("offset_spread","max"),
        multi=("n_minima", lambda s: int((s>1).sum())),
    ).sort_values(["subtype","spr_max"], ascending=[True,False])
    pd.set_option("display.width", 200)
    print(g.round(4).to_string())

if __name__ == "__main__":
    main()
