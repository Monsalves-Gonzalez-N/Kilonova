import os
import time
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Receta de error fotometrico Roman (fuente unica de verdad). Constantes y primitivas se importan de
# aqui en vez de redefinirlas, para que pipeline y notebooks no se desincronicen.
from kilonova.photometry.roman_noise import (
    SNR_DETECTION, ZP_JITTER_SIGMA, FIELD_SEED,
    EXPOSURE_TIME_BY_TIER, BASE_CADENCE_DAYS, TIER_ANCHOR_BAND, ALL_BANDS_BY_WAVELENGTH,
    HLTDS_FIELD_CENTER, HLTDS_FIELDS_BY_TIER, PSF_NEA_PIX,
    collecting_area_cm2,
    read_noise_electrons, field_center_for_tier, build_tier_constants,
    source_flux_electrons, flux_error_electrons, limiting_magnitude_5sigma,
    epochs_from_first_detection, cadence_schedule,
)
# Receta de fotometria sintetica de kilonovas (SED LANL -> mag AB Roman), validada en el dataloader.
from kilonova.photometry.spectra import ALL_ROMAN_BANDS, magnitudes_for_bands

# Todo el pipeline vive en esta carpeta: hdf5 de entrada (OpenUniverse), catalogo y HDF5 de salida.
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
# Un hdf5 por tier (se corren por separado). Por defecto ambos apuntan al mismo snana_10307_full.hdf5.
HDF5_PATH_BY_TIER = {
    "deep": os.path.join(DATA_DIR, "snana_10307_full.hdf5"),
    "wide": os.path.join(DATA_DIR, "snana_10307_full.hdf5"),
}
CATALOG_PATH = os.path.join(DATA_DIR, "snana_10307.parquet")
OUTPUT_DIR = DATA_DIR

# Grid de kilonovas LANL (espectros rest-frame ya cacheados; un row group por archivo .dat).
LANL_SPECTRA_PATH = "/Users/bhianca/Kilonova/data/dust_generation/lanl_spectra.parquet"
LANL_METADATA_COLUMNS = ["simulation_id", "time_index", "time_days", "angle_index"]
N_ANGLE_BINS = 54

# Inyeccion de kilonovas: una KN por transiente con z < KN_REDSHIFT_MAX, a ese mismo z.
KN_REDSHIFT_MAX = 0.5
KN_REALIZATION_SEED = 0       # reproducibilidad del sorteo simulacion/angulo/offset
N_KN_VISITS = 8               # visitas sinteticas (margen para hallar 1a deteccion + 4 epocas)
EXPLOSION_OFFSET_MAX_DAYS = 5.0  # t0 ~ U[0, max]: dias observador entre el merger y la 1a visita
KN_GENTYPE = 50
# Offset para la semilla de ruido de las KN: las KN ya no usan object_id (ahora string) para sembrar,
# asi que el ruido se siembra con KN_SEED_OFFSET + id_fuente, disjunto de los ids OU (~8e7).
KN_SEED_OFFSET = 2_000_000_000

# Limites opcionales para smoke-tests (0 = sin limite). No afectan la corrida completa.
OU_LIMIT = int(os.environ.get("EARLY_WINDOWS_OU_LIMIT", "0")) or None
KN_LIMIT = int(os.environ.get("EARLY_WINDOWS_KN_LIMIT", "0")) or None

NUMBER_OF_EPOCHS = 4

GENTYPE_LABEL = {10: "SN Ia", 12: "SN Iax", 21: "SN Ib", 26: "SN Ic", 32: "SN II",
                 40: "SLSN-I", 42: "TDE", 50: "KN", 57: "PISN-H", 58: "PISN-He", 99: "FIXMAG"}


def visit_schedule(window_start, window_end, constants):
    """base_epochs = visitas (cada 5 d) sobre [window_start, window_end] + su cadencia (anchor en todas,
    las otras 4 en 2 pares intercalados). La cadencia sale de roman_photometry.cadence_schedule."""
    base_epochs = np.arange(window_start, window_end + 1e-9, BASE_CADENCE_DAYS)
    return base_epochs, cadence_schedule(base_epochs, constants["bands"], constants["anchor_band"])


def nearest_model_magnitude(model_mjd, model_magnitude, target_mjds):
    nearest = np.abs(model_mjd[None, :] - target_mjds[:, None]).argmin(axis=1)
    return model_magnitude[nearest]


def model_from_hdf5_group(group, constants):
    """{band: (mjd, mag)} de una light curve OpenUniverse, descartando mag invalidas (NaN / >=99)."""
    mjd_grid = group["mjd"][:]
    model = {}
    for band in constants["bands"]:
        magnitude = group["mag_" + band[0]][:]
        valid = np.isfinite(magnitude) & (magnitude < 99)
        if valid.any():
            model[band] = (mjd_grid[valid], magnitude[valid])
    return model


def build_window_from_model(object_id, model, constants, redshift, gentype,
                            base_epochs=None, noise_seed=None):
    """Grilla fija NUMBER_OF_EPOCHS épocas × bandas del tier desde la primera detección S/N>=5.
    Magnitud observada = realización ruidosa en espacio de FLUJO -> mag. Filas no observadas por la
    cadencia llevan observed=False y mag_observed/mag_err = NaN (pero conservan mag_true del modelo).
    Si base_epochs es None se deriva del rango MJD del modelo (camino OpenUniverse); si se pasa, se usa
    tal cual (camino KN: offset de explosion + cadencia). object_id se guarda tal cual (los OU pasan su
    id entero como string; las KN pasan "simulation_id_angle_index_explosion_offset_days"); el ruido se
    siembra con noise_seed (default int(object_id), valido para los OU)."""
    if len(model) == 0:
        return None
    if noise_seed is None:
        noise_seed = int(object_id)

    if base_epochs is None:
        window_start = min(band_mjd.min() for band_mjd, _ in model.values())
        window_end = max(band_mjd.max() for band_mjd, _ in model.values())
        base_epochs, schedule = visit_schedule(window_start, window_end, constants)
    else:
        base_epochs = np.asarray(base_epochs, dtype=float)
        schedule = cadence_schedule(base_epochs, constants["bands"], constants["anchor_band"])

    exposure_by_band = constants["exposure_time"]
    base_zeropoint_by_band = constants["zeropoint"]
    noise_floor_by_band = constants["noise_floor_variance"]

    # Dos streams independientes por objeto (deterministas): jitter del zeropoint (posición FOV) y la
    # realización de ruido del flujo. spawn evita correlación entre ambos.
    jitter_rng, noise_rng = (np.random.default_rng(seed)
                             for seed in np.random.SeedSequence(int(noise_seed)).spawn(2))

    # Zeropoint con jitter FOV por (banda, visita): una realización por observación, fija por objeto.
    # Orden de bandas determinista para que las extracciones del rng sean reproducibles.
    jittered_zeropoint = {}
    for band in constants["bands"]:
        if band not in model:
            continue
        band_mjd, _ = model[band]
        observed_mjds = schedule.get(band, np.empty(0))
        epochs = observed_mjds[(observed_mjds >= band_mjd.min()) & (observed_mjds <= band_mjd.max())]
        jitters = jitter_rng.normal(0.0, ZP_JITTER_SIGMA, size=len(epochs))
        for epoch_mjd, jitter in zip(epochs, jitters):
            jittered_zeropoint[(band, round(float(epoch_mjd), 6))] = base_zeropoint_by_band[band] + jitter

    def observation_quantities(band, epoch_mjd, mag_true):
        zeropoint = jittered_zeropoint.get((band, round(float(epoch_mjd), 6)), base_zeropoint_by_band[band])
        exposure = exposure_by_band[band]
        flux_true = source_flux_electrons(mag_true, exposure, zeropoint)
        flux_error = flux_error_electrons(flux_true, noise_floor_by_band[band])
        return zeropoint, exposure, flux_true, flux_error

    # Paso 1: SNR/detección en las (visita, banda) observadas (con jitter), para hallar la 1ª detección.
    detection_mjds = []
    for band in constants["bands"]:
        if band not in model:
            continue
        band_mjd, band_mag = model[band]
        observed_mjds = schedule.get(band, np.empty(0))
        epochs = observed_mjds[(observed_mjds >= band_mjd.min()) & (observed_mjds <= band_mjd.max())]
        if len(epochs) == 0:
            continue
        mag_true_epochs = nearest_model_magnitude(band_mjd, band_mag, epochs)
        for epoch_mjd, mag_true in zip(epochs, mag_true_epochs):
            _, _, flux_true, flux_error = observation_quantities(band, epoch_mjd, mag_true)
            if flux_true / flux_error >= SNR_DETECTION:
                detection_mjds.append(epoch_mjd)
    if len(detection_mjds) == 0:
        return None

    first_detection_mjd = min(detection_mjds)
    epoch_mjds = epochs_from_first_detection(base_epochs, first_detection_mjd, NUMBER_OF_EPOCHS)

    # Paso 2: grilla fija de épocas × bandas con observed/detected/mag_observed(ruidosa)/mag_err.
    # El eje de tiempo es days_since_detection = epoch_mjd - first_detection_mjd: cero en la 1ª detección,
    # misma convención para OpenUniverse y KN (el MJD absoluto no se usa). object_id: los OU pasan su id
    # entero como string; las KN codifican "simulation_id_angle_index_explosion_offset_days".
    rows = []
    for epoch_index, epoch_mjd in enumerate(epoch_mjds, start=1):
        days_since_detection = float(epoch_mjd - first_detection_mjd)
        for band in constants["bands"]:
            observed = band in schedule and np.isclose(schedule[band], epoch_mjd).any()
            mag_true = np.nan
            if band in model:
                band_mjd, band_mag = model[band]
                if band_mjd.min() <= epoch_mjd <= band_mjd.max():
                    mag_true = float(nearest_model_magnitude(band_mjd, band_mag,
                                                             np.array([epoch_mjd]))[0])
            row = {"object_id": object_id, "gentype": int(gentype),
                   "label": GENTYPE_LABEL.get(gentype, "UNKNOWN"), "z_CMB": redshift,
                   "epoch": epoch_index, "days_since_detection": days_since_detection, "band": band,
                   "observed": bool(observed and np.isfinite(mag_true)),
                   "mag_true": mag_true, "mag_observed": np.nan, "mag_err": np.nan,
                   "snr": np.nan, "detected": False, "mag_limit_5sigma": np.nan}
            if row["observed"]:
                zeropoint, exposure, flux_true, flux_error = observation_quantities(band, epoch_mjd, mag_true)
                snr = flux_true / flux_error
                flux_observed = flux_true + noise_rng.normal(0.0, flux_error)
                row["snr"] = snr
                row["mag_err"] = 1.0857 / snr
                row["detected"] = bool(snr >= SNR_DETECTION)
                row["mag_limit_5sigma"] = limiting_magnitude_5sigma(flux_error, exposure, zeropoint)
                if flux_observed > 0:
                    row["mag_observed"] = zeropoint - 2.5 * np.log10(flux_observed / exposure
                                                                     / collecting_area_cm2())
                else:
                    row["mag_observed"] = row["mag_limit_5sigma"]
            rows.append(row)
    return pd.DataFrame(rows)


def build_early_window(object_id, group, constants, redshift, gentype):
    model = model_from_hdf5_group(group, constants)
    # object_id como string (la columna se comparte con las KN, que llevan un id no entero); noise_seed
    # = int(object_id) por defecto -> el ruido OU queda idéntico al early_windows_{tier}.parquet previo.
    return build_window_from_model(str(int(object_id)), model, constants, redshift, gentype)


# ---------------------------------------------------------------------------
# Kilonovas LANL: misma receta de ruido, mag_true desde la SED redshifteada.
# ---------------------------------------------------------------------------

_LANL_PARQUET_FILE = None


def _lanl_parquet_file():
    global _LANL_PARQUET_FILE
    if _LANL_PARQUET_FILE is None:
        _LANL_PARQUET_FILE = pq.ParquetFile(LANL_SPECTRA_PATH)
    return _LANL_PARQUET_FILE


def load_lanl_wavelength_grid():
    """Grilla de longitud de onda rest-frame (Å), guardada en la metadata del schema parquet."""
    metadata = _lanl_parquet_file().schema_arrow.metadata or {}
    wavelength_bytes = metadata.get(b"wavelength_rest_aa")
    if wavelength_bytes is None:
        raise RuntimeError(f"{LANL_SPECTRA_PATH}: falta la metadata wavelength_rest_aa")
    return np.frombuffer(wavelength_bytes, dtype=np.float32).astype(float)


def load_lanl_catalog_metadata():
    catalog = _lanl_parquet_file().read(columns=LANL_METADATA_COLUMNS).to_pandas()
    catalog.index.name = "spectrum_id"
    return catalog


def load_simulation_spectra(simulation_id):
    """{(angle_index, time_index): flux_rest} de una simulacion LANL, leyendo su(s) row group(s) una
    sola vez. Pensado para iterar todos los espectros de una simulacion sin reescanear el parquet."""
    simulation_id = int(simulation_id)
    parquet_file = _lanl_parquet_file()
    field_index = parquet_file.schema_arrow.get_field_index("simulation_id")
    spectra = {}
    for group_index in range(parquet_file.num_row_groups):
        statistics = parquet_file.metadata.row_group(group_index).column(field_index).statistics
        if statistics is None or not (statistics.min <= simulation_id <= statistics.max):
            continue
        row_group = parquet_file.read_row_group(
            group_index, columns=["simulation_id", "time_index", "angle_index", "flux_rest"]
        )
        simulation_array = row_group.column("simulation_id").to_numpy()
        mask = simulation_array == simulation_id
        if not mask.any():
            continue
        time_array = row_group.column("time_index").to_numpy()
        angle_array = row_group.column("angle_index").to_numpy()
        flux_column = row_group.column("flux_rest")
        for local_index in np.flatnonzero(mask):
            local_index = int(local_index)
            key = (int(angle_array[local_index]), int(time_array[local_index]))
            spectra[key] = np.asarray(flux_column[local_index].as_py(), dtype=float)
    if not spectra:
        raise KeyError(f"simulation_id={simulation_id} no encontrada en {LANL_SPECTRA_PATH}")
    return spectra


def build_simulation_time_grids(lanl_catalog):
    """Por simulacion: (time_index, time_days) ordenados por fase, para el nearest-rest-phase lookup."""
    unique_times = lanl_catalog.drop_duplicates(["simulation_id", "time_index"])[
        ["simulation_id", "time_index", "time_days"]
    ]
    simulation_time_grids = {}
    for simulation_id, group in unique_times.groupby("simulation_id"):
        ordered = group.sort_values("time_days")
        simulation_time_grids[int(simulation_id)] = (
            ordered["time_index"].to_numpy(), ordered["time_days"].to_numpy(),
        )
    return simulation_time_grids


def nearest_time_index(simulation_time_grids, simulation_id, rest_phase_days):
    time_index_array, time_days_array = simulation_time_grids[int(simulation_id)]
    nearest_position = int(np.argmin(np.abs(time_days_array - rest_phase_days)))
    return int(time_index_array[nearest_position])


def sample_kn_realizations(transients, simulation_pool, rng, redshift_max=KN_REDSHIFT_MAX):
    """Una KN aleatoria por transiente con 0 < z_CMB < redshift_max: simulacion + angulo + offset de
    explosion, al z del transiente. kn_object_id = -id_fuente (distingue de OU y siembra el ruido).
    El sorteo es fijo por transiente y se comparte entre tiers (la misma KN observada en deep y wide)."""
    realizations = {}
    for row in transients.itertuples():
        redshift = float(row.z_CMB)
        if not (0.0 < redshift < redshift_max):
            continue
        kn_object_id = -int(row.id)
        realizations[kn_object_id] = {
            "source_object_id": int(row.id),
            "redshift": redshift,
            "simulation_id": int(rng.choice(simulation_pool)),
            "angle_index": int(rng.integers(N_ANGLE_BINS)),
            "explosion_offset_days": float(rng.uniform(0.0, EXPLOSION_OFFSET_MAX_DAYS)),
        }
    return realizations


def build_kn_models(realizations, simulation_time_grids, wavelength_rest_aa, bands=ALL_ROMAN_BANDS):
    """Modelo verdadero de cada KN (mag AB por banda en cada visita base) en UNA pasada, agrupando por
    simulacion para leer cada row group LANL una sola vez. base_epochs = offset + 5*arange(N_KN_VISITS);
    rest_phase = base_epoch/(1+z). Tier-independiente: depende solo de (sim, angulo, offset, z), por lo
    que el modelo de 6 bandas y base_epochs se reusan en deep y wide.
    Devuelve {kn_object_id: (model_6band, base_epochs, realization)}."""
    by_simulation = defaultdict(list)
    for kn_object_id, realization in realizations.items():
        by_simulation[realization["simulation_id"]].append(kn_object_id)

    kn_models = {}
    start = time.time()
    for counter, (simulation_id, kn_object_ids) in enumerate(by_simulation.items()):
        simulation_spectra = load_simulation_spectra(simulation_id)
        for kn_object_id in kn_object_ids:
            realization = realizations[kn_object_id]
            redshift = realization["redshift"]
            angle_index = realization["angle_index"]
            base_epochs = realization["explosion_offset_days"] + BASE_CADENCE_DAYS * np.arange(N_KN_VISITS)

            magnitudes_by_band = {band: np.full(len(base_epochs), np.nan) for band in bands}
            for epoch_position, base_epoch in enumerate(base_epochs):
                rest_phase_days = base_epoch / (1.0 + redshift)
                time_index = nearest_time_index(simulation_time_grids, simulation_id, rest_phase_days)
                flux_rest = simulation_spectra.get((angle_index, time_index))
                if flux_rest is None:
                    continue
                epoch_magnitudes = magnitudes_for_bands(wavelength_rest_aa, flux_rest, redshift, bands)
                for band in bands:
                    magnitudes_by_band[band][epoch_position] = epoch_magnitudes[band]

            model = {}
            for band in bands:
                band_magnitudes = magnitudes_by_band[band]
                finite = np.isfinite(band_magnitudes)
                if finite.any():
                    model[band] = (base_epochs[finite], band_magnitudes[finite])
            kn_models[kn_object_id] = (model, base_epochs, realization)

        if (counter + 1) % 50 == 0:
            print(f"[kn-models] {counter + 1}/{len(by_simulation)} simulations "
                  f"({len(kn_models)} kilonovas, {time.time() - start:.0f}s)", flush=True)
    return kn_models


def kn_windows_for_tier(kn_models, constants):
    """Ventanas KN del tier: subset de bandas del modelo de 6 bandas + misma receta/ventana que OU."""
    windows = []
    n_detected = 0
    n_skipped = 0
    tier_bands = set(constants["bands"])
    start = time.time()
    for counter, (kn_object_id, (model_6band, base_epochs, realization)) in enumerate(kn_models.items()):
        model_tier = {band: series for band, series in model_6band.items() if band in tier_bands}
        # object_id de la KN = simulation_id_angle_index_explosion_offset_days (toda la realizacion).
        kn_id = (f"{realization['simulation_id']}_{realization['angle_index']}_"
                 f"{realization['explosion_offset_days']:.4f}")
        noise_seed = KN_SEED_OFFSET + realization["source_object_id"]
        window = build_window_from_model(
            kn_id, model_tier, constants, realization["redshift"], KN_GENTYPE,
            base_epochs=base_epochs, noise_seed=noise_seed,
        )
        if window is None:
            n_skipped += 1
            continue
        n_detected += 1
        windows.append(window)
        if (counter + 1) % 5000 == 0:
            print(f"[{constants['tier']}] kilonovas {counter + 1}/{len(kn_models)} "
                  f"detected={n_detected} skipped={n_skipped} ({time.time() - start:.0f}s)", flush=True)
    return windows, n_detected, n_skipped


def run_tier(tier, object_records):
    constants = build_tier_constants(tier)
    print(f"[{tier}] field={constants['field_name']}  noise_floor_variance: "
          + ", ".join(f"{b}={v:.0f}" for b, v in constants["noise_floor_variance"].items()), flush=True)

    windows = []
    n_ou_detected = 0
    n_ou_full = 0
    start = time.time()
    with h5py.File(HDF5_PATH_BY_TIER[tier], "r") as hdf5:
        for counter, (object_id, redshift, gentype) in enumerate(object_records):
            if str(object_id) not in hdf5:
                continue
            window = build_early_window(object_id, hdf5[str(object_id)], constants, redshift, gentype)
            if window is None:
                continue
            n_ou_detected += 1
            if window["epoch"].nunique() == NUMBER_OF_EPOCHS:
                n_ou_full += 1
            windows.append(window)
            if (counter + 1) % 10000 == 0:
                print(f"[{tier}] openuniverse {counter + 1}/{len(object_records)} "
                      f"detected={n_ou_detected} full{NUMBER_OF_EPOCHS}={n_ou_full} "
                      f"({time.time() - start:.0f}s)", flush=True)

    result = pd.concat(windows, ignore_index=True)
    output_path = os.path.join(OUTPUT_DIR, f"early_windows_{tier}.parquet")
    result.to_parquet(output_path, index=False)
    print(f"[{tier}] DONE openuniverse_detected={n_ou_detected} (full{NUMBER_OF_EPOCHS}={n_ou_full}) "
          f"rows={len(result)} -> {output_path}", flush=True)
    return n_ou_detected


def main():
    catalog = pd.read_parquet(CATALOG_PATH)
    transients = catalog[catalog["gentype"] != 99]
    object_records = [(int(row.id), float(row.z_CMB), int(row.gentype))
                      for row in transients.itertuples()]
    if OU_LIMIT is not None:
        object_records = object_records[:OU_LIMIT]
    print(f"transients in catalog (gentype!=FIXMAG): {len(object_records)}", flush=True)

    summary = {}
    for tier in ("deep", "wide"):
        summary[tier] = run_tier(tier, object_records)

    print("\n=== SUMMARY ===")
    print(f"input transients (contaminants only): {len(object_records)}")
    for tier, n_ou_detected in summary.items():
        print(f"{tier}: openuniverse with>=1 detection={n_ou_detected}")


if __name__ == "__main__":
    main()
