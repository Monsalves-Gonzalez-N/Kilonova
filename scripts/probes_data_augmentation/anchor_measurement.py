"""Medicion limpia del anclaje analitico, con `cc_templates.npz` regenerado y la ventana entera.

Rehace lo que el plan pide en su punto 1, y de paso produce la tabla del punto 2. Lo que cambia
respecto de las sondas del 2026-09-18: hay columna `mjd` (no hay que recuperar el MJD casando
`mag_true` contra el hdf5) y la verdad contra la que se mide ya no es un ajuste por colores sino el
brillo que el izc lee del PICO de la curva completa del padre -- una medida que no depende de la
fase en absoluto.

Por cada objeto se barre un desplazamiento `delta` alrededor de C y se mira que le pasa a dos cosas:

    zp(delta)     = mediana(mag_true - modelo(banda, phase0 + delta))
    spread(delta) = max - min de esos residuos

`zp` se compara contra el brillo del izc: si la fase esta bien, coinciden. `spread` no necesita
nada externo. Que los dos criterios elijan el mismo `delta` es lo que permite congelar C sin
arrastrar el hdf5.
"""

import argparse
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import h5py

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from kilonova.simulation import intermediate_z_contaminants as izc
from kilonova.simulation import openuniverse_parents as oup
from kilonova.simulation.early_windows import model_from_hdf5_group
from kilonova.photometry.spectra import ALL_ROMAN_BANDS, spectrum_to_roman_magnitudes

SOURCE_DIR = "/home/nicolas/Dropbox/Kilonova/openuniverse2025"
# Con que clave se agrupa un padre para darle su C. Las core-collapse van por plantilla: son 44
# SED distintas y cada una pone su fase cero donde quiere. Las otras cuatro clases van por CLASE,
# porque su modelo es uno solo -- o, en SN Iax, un banco de 919 que comparte el cero de fase del
# SED base -- asi que una sola C las cubre a todas. Que eso sea cierto lo dice la dispersion
# medida, no este comentario.
CORE_COLLAPSE_LABELS = ("SN Ib", "SN Ic", "SN IIP", "SN IIL")
ARCHIVE_NEW = "data/openuniverse/cc_templates.npz"
ARCHIVE_OLD = "data/openuniverse/_old_izc/cc_templates.npz.ceil-phase-edge"
HDF5_BANDS = ["F184", "H158", "J129", "Y106", "Z087", "R062"]
DELTAS = np.round(np.arange(-1.5, 1.501, 0.05), 3)  # por defecto; --delta-max lo ensancha
PHASE_STEP = 0.05

_state = {}


def archive_phase_edges(path):
    """{nombre de plantilla: (minphase, maxphase, paso)} tal como esta escrito el archivo."""
    edges = {}
    with np.load(path) as archive:
        names = [str(one) for one in archive["template_names"]]
        for index, name in enumerate(names):
            phase = archive[f"phase_{index}"]
            edges[name] = (float(phase.min()), float(phase.max()), float(np.median(np.diff(phase))))
    return edges


def init(catalog, source_by_index, C_by_template, deltas):
    izc.register_sources()
    _state.update(catalog=catalog, source_by_index=source_by_index, C=C_by_template,
                  deltas=deltas, hdf5={})


def _hdf5(healpix):
    if healpix not in _state["hdf5"]:
        _state["hdf5"][healpix] = h5py.File(f"{SOURCE_DIR}/snana_{healpix}.hdf5", "r")
    return _state["hdf5"][healpix]


def model_magnitudes(realization, redshift, phases):
    """{banda: mags} en fases NATIVAS de la plantilla. `rendered_band_curves` sin su ventana."""
    model = izc.build_model(dict(realization, redshift=float(redshift)))
    blue = max(model.minwave(), izc.OBSERVED_WAVELENGTH_LIMITS[0])
    red = min(model.maxwave(), izc.OBSERVED_WAVELENGTH_LIMITS[1])
    if red <= blue:
        return {}
    wavelength = izc._sampling_grid(blue, red, izc.OBSERVED_WAVELENGTH_STEP)
    rows = []
    for phase in phases:
        flux = np.clip(model.flux(phase * (1.0 + redshift), wavelength), 0.0, None)
        rows.append(dict.fromkeys(ALL_ROMAN_BANDS, np.nan) if flux.max() < izc.PHOTOMETRY_FLOOR
                    else spectrum_to_roman_magnitudes(wavelength, flux))
    return {band: np.array([row[band] for row in rows]) for band in ALL_ROMAN_BANDS}


def one(payload):
    parent_key, healpix, object_id, z, family, epochs, minphase, maxphase = payload
    catalog, by_index, C = _state["catalog"], _state["source_by_index"], _state["C"]
    parent = catalog.loc[parent_key].copy()
    parent["id"] = object_id
    parent["parent_key"] = parent_key  # es el indice de la tabla, y la realizacion lo pide
    deltas = _state["deltas"]
    base = np.array([(mjd - float(parent.peak_mjd)) / (1.0 + z) + C[family]
                     for mjd, _, _ in epochs])
    result = {"parent_key": parent_key, "z": z, "family": family,
              "phase_first": float(base.min()), "phase_last": float(base.max())}
    # La grilla que hace falta para barrer delta, recortada a lo que la plantilla tiene.
    low, high = base.min() + deltas.min(), base.max() + deltas.max()
    if high < minphase or low > maxphase:
        return result
    grid = np.arange(max(low, minphase), min(high, maxphase) + 1e-9, PHASE_STEP)
    if len(grid) < 2:
        return result
    try:
        realization = izc.realization_from_parent(
            parent, 0, z, by_index, np.random.default_rng(object_id))
        at_reference = izc.apply_brightness_offset(dict(realization), 0.0, float("nan"), 0)
        rendered = model_magnitudes(at_reference, z, grid)
        peaks = oup.parent_peak_magnitudes(
            _hdf5(healpix)[str(object_id)], z, float(parent.peak_mjd), ALL_ROMAN_BANDS)
        offset_izc, spread_izc, n_izc = izc.measure_brightness_offset(realization, peaks)
    except Exception as error:  # una plantilla que no cubre una banda, un grupo ausente
        result["error"] = repr(error)[:80]
        return result
    result.update(offset_izc=offset_izc, spread_izc=spread_izc, bands_izc=n_izc)
    for delta in deltas:
        wanted = base + delta
        if wanted.min() < grid[0] or wanted.max() > grid[-1]:
            continue
        residuals = []
        for phase, (_, bands, magnitudes) in zip(wanted, epochs):
            for band, magnitude in zip(bands, magnitudes):
                if band in rendered and np.isfinite(magnitude):
                    model = np.interp(phase, grid, rendered[band], np.nan, np.nan)
                    residuals.append(magnitude - model)
        residuals = np.array([one for one in residuals if np.isfinite(one)])
        if len(residuals) < 4:
            continue
        result[f"zp_{delta:+.2f}"] = float(np.median(residuals))
        result[f"sp_{delta:+.2f}"] = float(residuals.max() - residuals.min())
    return result


def collect(parquet_path, catalog, per_template, z_limits, family_of):
    """Ventanas observadas de hasta `per_template` objetos por familia, con su fila del catalogo."""
    columns = ["object_id", "z_CMB", "mjd", "band", "observed", "mag_true"]
    wanted = catalog.index.to_series()
    file = pq.ParquetFile(parquet_path)
    by_template = {}
    windows = {}
    for group in range(file.num_row_groups):
        frame = file.read_row_group(group, columns=columns).to_pandas()
        frame = frame[frame.object_id.isin(wanted) & frame.observed
                      & frame.z_CMB.between(*z_limits) & np.isfinite(frame.mag_true)]
        if frame.empty:
            continue
        for parent_key, block in frame.groupby("object_id"):
            family = family_of(catalog.loc[parent_key])
            seen = by_template.setdefault(family, 0)
            if seen >= per_template:
                continue
            by_template[family] = seen + 1
            windows[parent_key] = block
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--healpix", type=int, nargs="+", default=[10050])
    parser.add_argument("--tier", default="deep")
    parser.add_argument("--per-template", type=int, default=30)
    parser.add_argument("--z-min", type=float, default=0.3)
    parser.add_argument("--z-max", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--templates", type=int, nargs="*", default=None,
                        help="solo estos template_index (core-collapse)")
    parser.add_argument("--classes", nargs="*", default=list(CORE_COLLAPSE_LABELS),
                        help="clases a medir; las core-collapse van por plantilla y el resto por clase")
    parser.add_argument("--delta-center", type=float, default=0.0,
                        help="centro del barrido; SN1987A vive en +45 d, no cerca de cero")
    parser.add_argument("--delta-max", type=float, default=1.5)
    parser.add_argument("--delta-step", type=float, default=0.05)
    parser.add_argument("--out", default="scripts/probes_data_augmentation/anchor_measurement.csv")
    parser.add_argument("--table", default="scripts/probes_data_augmentation/template_C_2026-09-21.csv")
    args = parser.parse_args()

    deltas = np.round(args.delta_center
                      + np.arange(-args.delta_max, args.delta_max + 1e-9, args.delta_step), 3)
    izc.register_sources()
    catalog = oup.read_parent_catalog(SOURCE_DIR)
    catalog = catalog[catalog.healpix.isin(args.healpix)]
    source_by_index = izc.core_collapse_source_by_template_index(catalog)
    wanted = set(args.classes)
    core_collapse = wanted & set(CORE_COLLAPSE_LABELS)
    label_of = {index: label for index, (_, label) in source_by_index.items()}
    catalog = catalog[
        catalog.template_index.map(label_of).isin(core_collapse) | catalog.label.isin(wanted)
    ].set_index("parent_key")
    if catalog.empty:
        raise SystemExit(f"el catalogo no tiene objetos de {sorted(wanted)}")

    def family_of(row):
        """La clave con la que un padre busca su C. Ver CORE_COLLAPSE_LABELS."""
        index = row.template_index
        return int(index) if label_of.get(int(index)) in CORE_COLLAPSE_LABELS else str(row.label)

    new_edges, old_edges = archive_phase_edges(ARCHIVE_NEW), archive_phase_edges(ARCHIVE_OLD)
    # El archivo guarda "pycoco_SN2013ab" y la fuente registrada es "ou-SN2013ab": se casa por la
    # misma regla que usa `core_collapse_source_by_template_index`.
    by_source = {izc.OPENUNIVERSE_SOURCE_PREFIX + name.replace("pycoco_", ""): name
                 for name in new_edges}
    edges = {}
    for template_index, (source_name, label) in source_by_index.items():
        if label not in core_collapse:
            continue
        name = by_source[source_name]
        edges[template_index] = (name, label, *new_edges[name], old_edges[name][0])
    # Las clases que no son core-collapse no salen del archivo: su modelo lo arma el izc, asi que
    # los bordes se le preguntan a la fuente misma sobre un padre cualquiera de esa clase.
    for label in sorted(wanted - set(CORE_COLLAPSE_LABELS)):
        block = catalog[catalog.label == label]
        if block.empty:
            continue
        example = block.iloc[0].copy()
        example["parent_key"] = block.index[0]
        source = izc.source_of(izc.realization_from_parent(
            example, 0, 0.5, source_by_index, np.random.default_rng(0)))
        edges[label] = (source.name or label, label, float(source.minphase()),
                        float(source.maxphase()), 1.0, float(source.minphase()))

    print(f"familias: {len(edges)}   objetos: {len(catalog)}   clases: {sorted(wanted)}")
    ceil_moved = [v[5] - v[2] for key, v in edges.items() if isinstance(key, int)]
    if ceil_moved:
        print(f"ceil: {sum(1 for one in ceil_moved if abs(one) > 1e-6)} plantillas movieron su "
              f"minphase, mediana {np.median(ceil_moved):+.3f} d\n")

    # --- C desde el borde del modelo, objeto a objeto ------------------------------------------
    rows = []
    for healpix in args.healpix:
        with h5py.File(f"{SOURCE_DIR}/snana_{healpix}.hdf5", "r") as h5:
            block = catalog[catalog.healpix == healpix]
            for parent_key, row in block.iterrows():
                object_id = int(row.id)
                if str(object_id) not in h5:
                    continue
                model = model_from_hdf5_group(h5[str(object_id)], {"bands": HDF5_BANDS})
                if not model:
                    continue
                start = min(mjd.min() for mjd, _ in model.values())
                family = family_of(row)
                if family not in edges:
                    continue
                rows.append(dict(parent_key=parent_key, family=family,
                                 C_edge=edges[family][2]
                                 - (start - float(row.peak_mjd)) / (1.0 + float(row.redshift))))
    edge = pd.DataFrame(rows)
    C_by_template = edge.groupby("family").C_edge.median().to_dict()
    scatter = edge.groupby("family").C_edge.apply(
        lambda one: 1.4826 * np.median(np.abs(one - one.median())))
    print(f"C del borde: {len(edge)} objetos, {len(C_by_template)} familias "
          f"(dispersion mediana intra-familia {scatter.median():.3f} d, maxima {scatter.max():.3f} "
          f"en {scatter.idxmax()})\n")

    # --- barrido de delta sobre una muestra por plantilla ---------------------------------------
    if args.templates:
        catalog = catalog[catalog.template_index.isin(args.templates)]
    windows = collect(f"data/openuniverse/early_windows_{args.tier}.parquet",
                      catalog, args.per_template, (args.z_min, args.z_max), family_of)
    payloads = []
    for parent_key, block in windows.items():
        row = catalog.loc[parent_key]
        family = family_of(row)
        if family not in C_by_template:
            continue
        if args.templates and family not in args.templates:
            continue
        epochs = [(mjd, list(one.band), list(one.mag_true)) for mjd, one in block.groupby("mjd")]
        payloads.append((parent_key, int(row.healpix), int(row.id), float(block.z_CMB.iloc[0]),
                         family, epochs, edges[family][2], edges[family][3]))
    print(f"muestra: {len(payloads)} objetos, {len({p[4] for p in payloads})} familias")
    with Pool(args.workers, initializer=init,
              initargs=(catalog, source_by_index, C_by_template, deltas)) as pool:
        measured = pd.DataFrame([one for one in pool.imap_unordered(one_task, payloads, chunksize=2)])
    measured["C_edge"] = measured.family.map(C_by_template)
    measured.to_csv(args.out, index=False)
    print(f"-> {args.out}\n")
    report(measured, edges, C_by_template, args.table)


one_task = one


def report(measured, edges, C_by_template, table_path):
    if "offset_izc" not in measured:
        raise SystemExit("ningun objeto llego al brillo del izc; revisa la columna `error`")
    usable = measured[np.isfinite(measured.offset_izc)]
    print(f"objetos con brillo del izc: {len(usable)} de {len(measured)}")
    zp_columns = {float(c[3:]): c for c in measured.columns if c.startswith("zp_")}
    sp_columns = {float(c[3:]): c for c in measured.columns if c.startswith("sp_")}

    def best(block, columns, score):
        # Un `delta` que empuja la fase fuera de la plantilla deja de medir en casi todos los
        # objetos, y con tres que queden el criterio encuentra un minimo espurio: hay que exigir
        # que la mayoria de la muestra siga entrando.
        floor = max(4, int(0.5 * len(block)))
        values = {}
        for delta, column in columns.items():
            if column in block and block[column].notna().sum() >= floor:
                value = score(block, column)
                if np.isfinite(value):
                    values[delta] = value
        return min(values, key=values.get) if values else np.nan

    bias = lambda b, c: np.abs(np.nanmedian(b[c] - b.offset_izc)) if c in b else np.nan
    scatter = lambda b, c: np.nanmedian(b[c]) if c in b else np.nan

    delta_zp = best(usable, zp_columns, bias)
    delta_sp = best(usable, sp_columns, scatter)
    print(f"\nglobal:  delta que anula el sesgo contra el izc = {delta_zp:+.2f} d;  "
          f"delta que minimiza el spread = {delta_sp:+.2f} d")
    for delta in sorted({0.0, round(delta_zp, 2), round(delta_sp, 2)}):
        zp, sp = zp_columns.get(delta), sp_columns.get(delta)
        if zp is None:
            continue
        residual = usable[zp] - usable.offset_izc
        print(f"  delta {delta:+.2f}:  sesgo {np.nanmedian(residual):+.4f} mag  "
              f"|residuo| p50 {np.nanmedian(np.abs(residual)):.4f} p90 "
              f"{np.nanquantile(np.abs(residual), 0.9):.4f}   spread p50 "
              f"{np.nanmedian(usable[sp]):.4f}")

    rows = []
    for family, block in usable.groupby("family"):
        name, label, minphase, maxphase, step, minphase_old = edges[family]
        one_zp = best(block, zp_columns, bias)
        one_sp = best(block, sp_columns, scatter)
        column = zp_columns.get(round(one_zp, 2))
        rows.append(dict(
            family=family, source=name, label=label, n=len(block),
            C_edge=C_by_template[family], delta_zp=one_zp, delta_spread=one_sp,
            C=C_by_template[family] + (one_zp if np.isfinite(one_zp) else 0.0),
            minphase=minphase, maxphase=maxphase, ceil_loss=minphase_old - minphase,
            residual=float(np.nanmedian(np.abs(block[column] - block.offset_izc)))
            if column else np.nan))
    table = pd.DataFrame(rows).sort_values("family", key=lambda c: c.astype(str))
    table.to_csv(table_path, index=False)
    print(f"\npor plantilla:\n{table.round(3).to_string(index=False)}")
    print(f"\n-> {table_path}")
    agree = table.delta_zp - table.delta_spread
    print(f"\nlos dos criterios coinciden: mediana {agree.median():+.2f} d, "
          f"|dif|<=0.2 d en {100*(agree.abs() <= 0.2).mean():.0f}% de las plantillas")


if __name__ == "__main__":
    main()
