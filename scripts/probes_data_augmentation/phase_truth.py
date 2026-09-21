"""Que error tiene la fase que recuperamos, contra la fase VERDADERA de OpenUniverse.

La fase verdadera sale del hdf5: recuperamos el MJD de la epoca 0 haciendo casar los mag_true de la
ventana con la curva del modelo, y la comparamos con peak_mjd del catalogo padre.
"""
import sys, warnings
import numpy as np, pandas as pd, pyarrow.parquet as pq, h5py
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

SCRATCH = "/tmp/claude-1000/-home-nicolas-nico-git-Kilonova/0be13fce-0731-4085-bd8f-d900fc6438b0/scratchpad"
HDF5 = "/home/nicolas/Dropbox/Kilonova/openuniverse2025/snana_10050.hdf5"
from kilonova.simulation import openuniverse_parents as oup
from kilonova.simulation.early_windows import model_from_hdf5_group, nearest_model_magnitude
from kilonova.photometry.roman_noise import roman_bandpasses

fits = pd.read_csv(f"{SCRATCH}/batch_z05.csv")
good = fits[(fits.phase - fits.grid_min > 1.0) & (fits.colour_rms < 0.05)].set_index("object_id")
print(f"objetos buenos: {len(good)}")

catalog = oup.read_parent_catalog("/home/nicolas/Dropbox/Kilonova/openuniverse2025")
catalog = catalog[catalog.healpix == 10050].set_index("id")

f = pq.ParquetFile("data/openuniverse/early_windows_deep.parquet")
cols = ["object_id", "days_since_detection", "band", "observed", "mag_true"]
frames = []
for gi in range(f.num_row_groups):
    d = f.read_row_group(gi, columns=cols).to_pandas()
    d = d[d.object_id.isin(good.index) & d.observed]
    if len(d):
        frames.append(d)
win = pd.concat(frames)
constants = {"bands": sorted(win.band.unique())}
print("bandas:", constants["bands"])

rows, steps = [], []
with h5py.File(HDF5, "r") as h5:
    for object_id, g in win.groupby("object_id"):
        snana_id = object_id.split("_")[-1]
        if snana_id not in h5:
            continue
        model = model_from_hdf5_group(h5[snana_id], constants)
        epochs = [(d, list(gg["band"]), np.array(gg["mag_true"])) for d, gg in g.groupby("days_since_detection")]
        day0 = min(d for d, _, _ in epochs)
        any_mjd = next(iter(model.values()))[0]
        steps.append(float(np.median(np.diff(any_mjd))))
        # el MJD de la epoca 0 es el punto de la grilla del modelo que reproduce los mag_true
        best_mjd, best_err = None, np.inf
        for cand in any_mjd:
            err, n = 0.0, 0
            for d, bands, mags in epochs:
                t = np.array([cand + (d - day0)])
                for b, m in zip(bands, mags):
                    if b not in model or not np.isfinite(m):
                        continue
                    bm, bg = model[b]
                    err += (nearest_model_magnitude(bm, bg, t)[0] - m) ** 2
                    n += 1
            if n and err / n < best_err:
                best_err, best_mjd = err / n, float(cand)
        if best_mjd is None or best_err > 1e-4:
            continue
        row = catalog.loc[int(snana_id)]
        z = float(good.loc[object_id, "z"])
        rows.append(dict(object_id=object_id, source=good.loc[object_id, "source"], z=z,
                         fitted=float(good.loc[object_id, "phase"]),
                         truth=(best_mjd - float(row.peak_mjd)) / (1.0 + z),
                         match=np.sqrt(best_err)))
r = pd.DataFrame(rows)
r["delta"] = r.fitted - r.truth
r.to_csv(f"{SCRATCH}/phase_truth.csv", index=False)
print(f"paso de la grilla del modelo OU: mediana {np.median(steps):.3f} d observer-frame")
print(f"casaron {len(r)} objetos, residuo del match mediana {r.match.median():.2e} mag\n")
print("=== fitted - truth (d, rest-frame) ===")
print(f"  global: mediana {r.delta.median():+.3f}  MAD {1.4826*np.median(np.abs(r.delta-r.delta.median())):.3f}  p5 {r.delta.quantile(.05):+.2f}  p95 {r.delta.quantile(.95):+.2f}")
g = r.groupby("source").agg(n=("delta","size"), offset=("delta","median"),
                            scatter=("delta", lambda s: 1.4826*np.median(np.abs(s-s.median()))),
                            p95=("delta", lambda s: s.abs().quantile(.95)))
g = g[g.n >= 8].sort_values("scatter")
print("\n=== por plantilla (n>=8) ===")
print(g.round(3).to_string())
