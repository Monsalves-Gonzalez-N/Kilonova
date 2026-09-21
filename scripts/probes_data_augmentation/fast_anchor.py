"""Fase analitica en vez de busqueda: phase0 = (mjd0 - peak_mjd)/(1+z) + C(plantilla).

C se calibra en la MITAD de los objetos y se mide en la otra mitad, sobre TODOS -- tambien los que
el ajuste por colores rielaba. Sin grilla, sin minimizacion.
"""
import sys, time, warnings
import numpy as np, pandas as pd, pyarrow.parquet as pq, h5py
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from multiprocessing import Pool

S = "/tmp/claude-1000/-home-nicolas-nico-git-Kilonova/0be13fce-0731-4085-bd8f-d900fc6438b0/scratchpad"
HDF5 = "/home/nicolas/Dropbox/Kilonova/openuniverse2025/snana_10050.hdf5"
CAT = "/home/nicolas/Dropbox/Kilonova/openuniverse2025"
BANDS = ["F184", "H158", "J129", "Y106", "Z087"]

_s = {}

def init(C):
    _s["C"] = C
    from kilonova.simulation import openuniverse_parents as oup
    from kilonova.simulation import intermediate_z_contaminants as izc
    catalog = oup.read_parent_catalog(CAT)
    catalog = catalog[catalog.healpix == 10050].set_index("id")
    _s["cat"] = catalog
    _s["by_index"] = izc.core_collapse_source_by_template_index(catalog.reset_index())
    _s["izc"] = izc

def curves(object_id, z, template_index):
    """Modelo renderizado a ese z: {banda: (fases, mags)}. Es lo unico caro y se cachea por (tpl,z)."""
    izc, cat, by_index = _s["izc"], _s["cat"], _s["by_index"]
    row = cat.loc[int(object_id.split("_")[-1])].copy()
    row["id"] = int(object_id.split("_")[-1])
    r = izc.realization_from_parent(row, 0, z, by_index, np.random.default_rng(0))
    r = izc.apply_brightness_offset(r, 0.0, float("nan"), 0)
    c = izc.rendered_band_curves(r, z)
    return {b: (c[b][0], c[b][1]) for b in c}

def one(payload):
    object_id, z, mjd0, peak_mjd, template_index, epochs = payload
    try:
        interp = curves(object_id, z, template_index)
    except Exception:
        return None
    if not interp:
        return None
    truth = (mjd0 - peak_mjd) / (1.0 + z)
    day0 = min(d for d, _, _ in epochs)
    out = {"object_id": object_id, "z": z, "truth": truth, "template_index": template_index}
    for tag, C in (("raw", 0.0), ("cal", _s["C"].get(template_index, np.nan))):
        if not np.isfinite(C):
            continue
        offs, per_epoch = [], []
        ok = True
        for d, bands, mags in epochs:
            ph = truth + C + (d - day0) / (1.0 + z)
            this = []
            for b, m in zip(bands, mags):
                if b not in interp:
                    ok = False; break
                p, mm = interp[b]
                model = np.interp(ph, p, mm, np.nan, np.nan)
                if not np.isfinite(model):
                    ok = False; break
                this.append(m - model)
            if not ok: break
            per_epoch.append(this); offs.extend(this)
        if not ok or len(offs) < 4:
            out[f"{tag}_spread"] = np.nan
            continue
        offs = np.array(offs)
        out[f"{tag}_offset"] = float(np.median(offs))
        out[f"{tag}_spread"] = float(offs.max() - offs.min())
        out[f"{tag}_band"] = float(np.mean([max(e) - min(e) for e in per_epoch]))
        med = [np.median(e) for e in per_epoch]
        out[f"{tag}_epoch"] = float(max(med) - min(med))
    return out

def load():
    from kilonova.simulation import openuniverse_parents as oup
    fits = pd.read_csv(f"{S}/batch_z05.csv").set_index("object_id")
    cat = oup.read_parent_catalog(CAT); cat = cat[cat.healpix == 10050].set_index("id")
    f = pq.ParquetFile("data/openuniverse/early_windows_deep.parquet")
    cols = ["object_id", "days_since_detection", "band", "observed", "mag_true"]
    frames = []
    for gi in range(f.num_row_groups):
        d = f.read_row_group(gi, columns=cols).to_pandas()
        d = d[d.object_id.isin(fits.index) & d.observed]
        if len(d): frames.append(d)
    win = pd.concat(frames)
    from kilonova.simulation.early_windows import model_from_hdf5_group, nearest_model_magnitude
    payloads = []
    with h5py.File(HDF5, "r") as h5:
        for object_id, g in win.groupby("object_id"):
            sid = object_id.split("_")[-1]
            if sid not in h5: continue
            model = model_from_hdf5_group(h5[sid], {"bands": BANDS})
            if not model: continue
            epochs = [(d, list(gg["band"]), list(gg["mag_true"])) for d, gg in g.groupby("days_since_detection")]
            day0 = min(d for d, _, _ in epochs)
            grid = next(iter(model.values()))[0]
            best, berr = None, np.inf
            for cand in grid:
                err = n = 0
                for d, bands, mags in epochs:
                    t = np.array([cand + (d - day0)])
                    for b, m in zip(bands, mags):
                        if b in model and np.isfinite(m):
                            err += (nearest_model_magnitude(model[b][0], model[b][1], t)[0] - m) ** 2; n += 1
                if n and err / n < berr: berr, best = err / n, float(cand)
            if best is None or berr > 1e-4: continue
            r = cat.loc[int(sid)]
            payloads.append((object_id, float(fits.loc[object_id, "z"]), best, float(r.peak_mjd),
                             int(r.template_index), epochs))
    return payloads, fits

def main():
    payloads, fits = load()
    print(f"objetos con mjd0 recuperado: {len(payloads)} de {len(fits)}")
    tr = pd.read_csv(f"{S}/phase_truth.csv")
    idx = pd.read_csv(f"{S}/batch_z05.csv")[["object_id", "template_index"]].drop_duplicates()
    tr = tr.merge(idx, on="object_id")
    # CALIBRACION: mitad de los objetos buenos, elegida por hash del id
    tr["half"] = [hash(o) % 2 for o in tr.object_id]
    calib = tr[tr.half == 0]
    C = calib.groupby("template_index").delta.median().to_dict()
    n_cal = calib.groupby("template_index").size()
    print(f"C calibrado en {len(calib)} objetos, {len(C)} plantillas (min {n_cal.min()} por plantilla)")
    test_ids = set(tr[tr.half == 1].object_id) | (set(fits.index) - set(tr.object_id))
    payloads = [p for p in payloads if p[0] in test_ids]
    print(f"evaluando en {len(payloads)} objetos de test (los buenos de la otra mitad + los que rielaban)\n")
    t0 = time.time()
    with Pool(8, initializer=init, initargs=(C,)) as pool:
        out = [r for r in pool.imap_unordered(one, payloads, chunksize=4) if r]
    dt = time.time() - t0
    r = pd.DataFrame(out)
    r = r.merge(fits.reset_index()[["object_id", "phase", "grid_min", "colour_rms", "offset_spread"]], on="object_id")
    r["railed"] = (r.phase - r.grid_min) <= 1.0
    r.to_csv(f"{S}/fast_anchor.csv", index=False)
    print(f"tiempo: {dt:.1f} s para {len(r)} objetos = {1000*dt/len(r):.0f} ms/objeto (8 workers, incl. render)\n")
    for tag, name in (("raw", "sin C (phase0 = (mjd0-peak)/(1+z))"), ("cal", "con C por plantilla")):
        c = r[np.isfinite(r[f"{tag}_spread"])]
        if not len(c): continue
        print(f"{name}:  n={len(c)}  spread zp mediana {c[f'{tag}_spread'].median():.4f}  p90 {c[f'{tag}_spread'].quantile(.9):.4f}")
    print()
    print("=== comparacion contra el ajuste por colores, sobre el mismo objeto ===")
    for sel, name in ((~r.railed, "los que el ajuste resolvia"), (r.railed, "los que el ajuste RIELABA")):
        c = r[sel & np.isfinite(r.cal_spread)]
        if not len(c): continue
        print(f"  {name} (n={len(c)}):")
        print(f"     ajuste  spread mediana {c.offset_spread.median():.4f}  p90 {c.offset_spread.quantile(.9):.4f}")
        print(f"     analit. spread mediana {c.cal_spread.median():.4f}  p90 {c.cal_spread.quantile(.9):.4f}"
              f"   (banda {c.cal_band.median():.4f}, epoca {c.cal_epoch.median():.4f})")

if __name__ == "__main__":
    main()
