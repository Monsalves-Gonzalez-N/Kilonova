"""Un objeto del training set a alto z, regenerado a bajo z desde SU ventana.

Demo de punta a punta de `kilonova.simulation.window_anchor`, que es donde vive el metodo:

    phase0     = (mjd - peak_mjd)/(1+z) + C[plantilla]        <- la fase NO se busca
    zero_point = mediana(mag_true - modelo(banda, phase0))     <- el brillo del objeto
    -> se vuelve a renderizar la MISMA plantilla a z_nuevo con ese mismo brillo intrinseco

Lo unico que entra del training set es la ventana (mjd, band, mag_true); peak_mjd y template_index
salen del catalogo padre y C de `data/openuniverse/template_phase_anchor.csv`. El hdf5 de
OpenUniverse NO se toca para regenerar -- solo aqui, para medir el brillo por el camino del izc
(pico de la curva completa) y comprobar que da lo mismo.
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import h5py

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from kilonova.simulation import intermediate_z_contaminants as izc
from kilonova.simulation import openuniverse_parents as oup
from kilonova.simulation import window_anchor
from kilonova.photometry.spectra import ALL_ROMAN_BANDS

SOURCE_DIR = "/home/nicolas/Dropbox/Kilonova/openuniverse2025"
BAND_COLOUR = {"R062": "#8c564b", "Z087": "#1f77b4", "Y106": "#2ca02c",
               "J129": "#ff7f0e", "H158": "#d62728", "F184": "#9467bd"}


def draw(plots, z_target, path):
    """El mismo objeto arriba como lo vio el survey a alto z, abajo regenerado a bajo z."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, len(plots), figsize=(4.2 * len(plots), 6.4), squeeze=False,
                                sharex="col")
    for column, (parent_key, label, z_parent, window, result) in enumerate(plots):
        for row, (frame, z, title) in enumerate((
                (window, z_parent, "OpenUniverse"),
                (result[result.observed], z_target, "regenerado"))):
            axis = axes[row][column]
            for band, block in frame.groupby("band"):
                block = block.sort_values("mjd")
                days = block.mjd - frame.mjd.min()
                axis.errorbar(days, block.mag_observed, yerr=block.mag_err, marker="o", ms=4,
                              lw=1, color=BAND_COLOUR.get(band, "k"), label=band)
                axis.plot(days, block.mag_true, ls="--", lw=1, color=BAND_COLOUR.get(band, "k"))
            axis.invert_yaxis()
            axis.set_title(f"{title}  z={z:.3f}", fontsize=9)
            axis.set_ylabel("AB mag")
            if row:
                axis.set_xlabel("dias desde la 1a epoca")
        axes[0][column].text(0.02, 0.04, f"{parent_key}\n{label}", fontsize=7, va="bottom",
                             transform=axes[0][column].transAxes)
    axes[0][0].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=140)


def pick_window(parquet_path, keys, z_min, z_max, wanted):
    """Las ventanas observadas de los primeros objetos de `keys` que aparezcan en el parquet."""
    columns = ["object_id", "z_CMB", "epoch", "mjd", "band", "observed",
               "mag_true", "mag_observed", "mag_err", "detected"]
    keep = set(keys)
    file = pq.ParquetFile(parquet_path)
    found = {}
    for group in range(file.num_row_groups):
        frame = file.read_row_group(group, columns=columns).to_pandas()
        frame = frame[frame.object_id.isin(keep) & frame.observed & frame.z_CMB.between(z_min, z_max)]
        for object_id, block in frame.groupby("object_id"):
            found.setdefault(object_id, block)
            if len(found) >= wanted:
                return found
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--healpix", type=int, default=10050)
    parser.add_argument("--tier", default="deep")
    parser.add_argument("--z-min", type=float, default=1.0)
    parser.add_argument("--z-max", type=float, default=1.6)
    parser.add_argument("--z-target", type=float, default=0.15)
    parser.add_argument("--objects", type=int, default=1)
    parser.add_argument("--classes", nargs="*", default=None,
                        help="clases a regenerar (default: las core-collapse)")
    parser.add_argument("--seed", type=int, default=0, help="muestreo de parents en el catalogo")
    parser.add_argument("--plot", default=None, help="PNG con alto z vs bajo z")
    parser.add_argument("--out", default=None, help="CSV con las ventanas regeneradas")
    args = parser.parse_args()

    izc.register_sources()
    catalog = oup.read_parent_catalog(SOURCE_DIR)
    catalog = catalog[catalog.healpix == args.healpix]
    source_by_index = izc.core_collapse_source_by_template_index(catalog)
    anchors = window_anchor.phase_anchors()
    catalog = catalog.set_index("parent_key")
    hdf5_path = f"{SOURCE_DIR}/snana_{args.healpix}.hdf5"

    inside = catalog.redshift.between(args.z_min, args.z_max)
    block = catalog[inside & (catalog.label.isin(args.classes) if args.classes
                              else catalog.gentype.isin(oup.CORE_COLLAPSE_GENTYPES))]
    sample = block.index.to_series().sample(min(len(block), 50 * args.objects + 50),
                                            random_state=args.seed)
    windows = pick_window(f"data/openuniverse/early_windows_{args.tier}.parquet",
                          sample, args.z_min, args.z_max, args.objects)
    print(f"objetos candidatos en el catalogo: {len(block)};  con ventana: {len(windows)}\n")

    regenerated, plots = [], []
    for parent_key, window in windows.items():
        parent = catalog.loc[parent_key].copy()
        parent["parent_key"] = parent_key
        object_id = int(parent.id)
        z_parent = float(window.z_CMB.iloc[0])
        family = window_anchor.family_of(parent)
        label = str(parent.label) if not isinstance(family, int) else source_by_index[family][1]
        source_name = str(anchors[family].source) if family in anchors else "?"
        anchor = anchors[family]
        phases = window_anchor.window_phases(window.mjd, parent.peak_mjd, z_parent, anchor)
        print(f"=== {parent_key}  {label}  z={z_parent:.4f}  familia {family} "
              f"({source_name}) ===")
        print(f"C = {anchor.C:+.3f} d (tabla congelada, residuo medido {anchor.residual_mag:.4f} "
              f"mag);  plantilla cubre [{anchor.minphase:.1f}, {anchor.maxphase:.1f}] d")
        print("fases de las epocas: " + ", ".join(f"{p:+.2f}" for p in np.unique(phases)))

        # --- el anclaje, que es una sola llamada --------------------------------------------
        low = window_anchor.anchor_window(window, parent, args.z_target, source_by_index,
                                          np.random.default_rng(object_id), anchors=anchors)
        if low is None:
            outside = window_anchor.uncovered_phases(np.unique(phases), anchor)
            print(f"   DESCARTADO: {len(outside)} fases fuera de la plantilla "
                  f"({', '.join(f'{p:+.2f}' for p in outside)})\n")
            continue
        print(f"zero point = {low['brightness_offset']:+.4f} mag sobre "
              f"{low['brightness_bands']} medidas;  spread {low['brightness_residual']:.4f}")

        # --- contraste independiente: el brillo por el camino del izc ------------------------
        with h5py.File(hdf5_path, "r") as h5:
            peaks = oup.parent_peak_magnitudes(
                h5[str(object_id)], z_parent, float(parent.peak_mjd), ALL_ROMAN_BANDS)
        by_peak, spread_izc, n_izc = izc.measure_brightness_offset(
            izc.realization_from_parent(parent, 0, z_parent, source_by_index,
                                        np.random.default_rng(object_id)), peaks)
        print(f"por el camino del izc (pico de la curva completa del hdf5): {by_peak:+.4f} mag "
              f"({n_izc} bandas, spread {spread_izc:.4f})   ->  diferencia "
              f"{low['brightness_offset']-by_peak:+.4f} mag")

        # --- el mismo objeto, a z bajo ------------------------------------------------------
        result, rejected = izc.build_izc_windows([low], args.tier)
        if result.empty:
            print(f"a z={args.z_target}: sin ventana ({rejected})\n")
            continue
        seen = result[result.observed]
        print(f"a z={args.z_target:.3f}: {seen.epoch.nunique()} epocas, {len(seen)} observaciones, "
              f"{int(seen.detected.sum())} detectadas;  mag {seen.mag_observed.min():.2f}-"
              f"{seen.mag_observed.max():.2f} (a z={z_parent:.3f} eran "
              f"{window.mag_observed.min():.2f}-{window.mag_observed.max():.2f})")
        print(f"   id nuevo: {result.object_id.iloc[0]}\n")
        if args.plot:
            plots.append((parent_key, label, z_parent, window, result))
        result["parent_object_id"] = parent_key
        result["parent_z"] = z_parent
        regenerated.append(result)

    if plots:
        draw(plots, args.z_target, args.plot)
        print(f"-> {args.plot}")
    if regenerated and args.out:
        pd.concat(regenerated, ignore_index=True).to_csv(args.out, index=False)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
