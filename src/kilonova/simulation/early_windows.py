import logging
import multiprocessing
import os
import time
from collections import defaultdict
from functools import lru_cache

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Receta de error fotometrico Roman (fuente unica de verdad). Constantes y primitivas se importan de
# aqui en vez de redefinirlas, para que pipeline y notebooks no se desincronicen.
from kilonova.photometry.roman_noise import (
    BASE_CADENCE_DAYS,
    SNR_DETECTION,
    ZP_JITTER_SIGMA,
    build_tier_constants,
    cadence_schedule,
    collecting_area_cm2,
    epochs_from_first_detection,
    flux_error_electrons,
    limiting_magnitude_5sigma,
    source_flux_electrons,
)

# Receta de fotometria sintetica de kilonovas (SED LANL -> mag AB Roman), fijada por tests/test_spectra.py.
from kilonova.photometry.spectra import ALL_ROMAN_BANDS, magnitudes_for_bands

logger = logging.getLogger(__name__)

# Grid de kilonovas LANL (espectros rest-frame ya cacheados; un row group por archivo .dat).
# El path del parquet viene de configs/paths.yaml via kilonova.config (nunca hardcodeado aqui).
LANL_METADATA_COLUMNS = ["simulation_id", "time_index", "time_days", "angle_index"]
N_ANGLE_BINS = 54

# Inyeccion de kilonovas: grilla LANL simulada sobre una grilla de redshifts (las KN LANL son
# demasiado debiles para tener analogas en el catalogo OU, asi que no se emparejan por transiente).
KN_REALIZATION_SEED = 0  # reproducibilidad del sorteo simulacion/angulo/offset
N_KN_VISITS = 8  # visitas sinteticas (margen para hallar 1a deteccion + 4 epocas)
EXPLOSION_OFFSET_MAX_DAYS = 5.0  # t0 ~ U[0, max]: dias observador entre el merger y la 1a visita
# La cadencia se repite cada 2 visitas (par: ancla + las dos no-ancla azules; impar: ancla + las dos
# rojas), asi que la fase del merger dentro del ciclo tiene DOS grados de libertad: el retardo hasta
# la 1a visita (explosion_offset_days) y la PARIDAD de esa visita. La grilla KN se construye como
# offset + 5*arange(N), de modo que su indice 0 es siempre la 1a visita post-merger; sin sortear la
# paridad aparte, esa visita seria siempre par. Como la KN es rapida y casi siempre se detecta ahi,
# eso imprimia la fase de la cadencia en la clase KN (92% par vs 29% en los contaminantes: la
# mascara sola clasificaba al 80.7%). En un survey real la grilla es fija en tiempo absoluto y el
# merger cae uniforme en el ciclo de 10 d, o sea paridad 50/50 e independiente del retardo.
CADENCE_PARITY_PERIOD = 2  # visitas por ciclo del patron de bandas
KN_GENTYPE = 50
# Offset para la semilla de ruido de las KN: se siembra con KN_SEED_OFFSET + noise_id de la
# realizacion, disjunto de los ids OU (~8e7).
KN_SEED_OFFSET = 2_000_000_000

NUMBER_OF_EPOCHS = 4

GENTYPE_LABEL = {
    10: "SN Ia",
    12: "SN Iax",
    21: "SN Ib",
    26: "SN Ic",
    32: "SN II",
    40: "SLSN-I",
    42: "TDE",
    50: "KN",
    57: "PISN-H",
    58: "PISN-He",
    99: "FIXMAG",
}


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


def build_window_from_model(
    object_id,
    model,
    constants,
    redshift,
    gentype,
    base_epochs=None,
    noise_seed=None,
    visit_index_offset=0,
):
    """Grilla fija NUMBER_OF_EPOCHS épocas × bandas del tier desde la primera detección S/N>=5.
    Magnitud observada = realización ruidosa en espacio de FLUJO -> mag. Filas no observadas por la
    cadencia llevan observed=False y mag_observed/mag_err = NaN (pero conservan mag_true del modelo).
    Si base_epochs es None se deriva del rango MJD del modelo (camino OpenUniverse); si se pasa, se usa
    tal cual (camino KN: offset de explosion + cadencia). object_id se guarda tal cual (los OU pasan su
    id entero como string; las KN el de kn_object_id); el ruido se siembra con noise_seed (default
    int(object_id), valido para los OU). `visit_index_offset` fija la paridad de la primera visita de
    la grilla: 0 para los OU, cuyas visitas SON las del survey, y la paridad sorteada para las KN,
    cuya grilla se reconstruye desde el merger (ver CADENCE_PARITY_PERIOD)."""
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
        schedule = cadence_schedule(
            base_epochs, constants["bands"], constants["anchor_band"], visit_index_offset
        )

    exposure_by_band = constants["exposure_time"]
    base_zeropoint_by_band = constants["zeropoint"]
    noise_floor_by_band = constants["noise_floor_variance"]

    # Dos streams independientes por objeto (deterministas): jitter del zeropoint (posición FOV) y la
    # realización de ruido del flujo. spawn evita correlación entre ambos.
    jitter_rng, noise_rng = (
        np.random.default_rng(seed) for seed in np.random.SeedSequence(int(noise_seed)).spawn(2)
    )

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
        for epoch_mjd, jitter in zip(epochs, jitters, strict=True):
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
        for epoch_mjd, mag_true in zip(epochs, mag_true_epochs, strict=True):
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
                    mag_true = float(nearest_model_magnitude(band_mjd, band_mag, np.array([epoch_mjd]))[0])
            row = {
                "object_id": object_id,
                "gentype": int(gentype),
                "label": GENTYPE_LABEL.get(gentype, "UNKNOWN"),
                "z_CMB": redshift,
                "epoch": epoch_index,
                "days_since_detection": days_since_detection,
                "band": band,
                # mag_true infinita (banda sin flujo) sigue siendo observada: da una realizacion de
                # solo ruido. Solo NaN -- que el modelo no cubra esa epoca -- deja la fila sin observar.
                "observed": bool(observed and not np.isnan(mag_true)),
                "mag_true": mag_true,
                "mag_observed": np.nan,
                "mag_err": np.nan,
                "snr": np.nan,
                "detected": False,
                "mag_limit_5sigma": np.nan,
            }
            if row["observed"]:
                zeropoint, exposure, flux_true, flux_error = observation_quantities(band, epoch_mjd, mag_true)
                snr = flux_true / flux_error
                flux_observed = flux_true + noise_rng.normal(0.0, flux_error)
                row["snr"] = snr
                # Sin flujo no hay error de magnitud que reportar (1.0857/0). Los OU nunca llegan
                # aqui: con mag_true finita el flujo siempre es > 0.
                row["mag_err"] = 1.0857 / snr if snr > 0 else np.nan
                row["detected"] = bool(snr >= SNR_DETECTION)
                row["mag_limit_5sigma"] = limiting_magnitude_5sigma(flux_error, exposure, zeropoint)
                if flux_observed > 0:
                    row["mag_observed"] = zeropoint - 2.5 * np.log10(
                        flux_observed / exposure / collecting_area_cm2()
                    )
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


@lru_cache(maxsize=2)
def _lanl_parquet_file(lanl_spectra_path):
    return pq.ParquetFile(lanl_spectra_path)


def load_lanl_wavelength_grid(lanl_spectra_path):
    """Grilla de longitud de onda rest-frame (Å), guardada en la metadata del schema parquet."""
    metadata = _lanl_parquet_file(str(lanl_spectra_path)).schema_arrow.metadata or {}
    wavelength_bytes = metadata.get(b"wavelength_rest_aa")
    if wavelength_bytes is None:
        raise RuntimeError(f"{lanl_spectra_path}: falta la metadata wavelength_rest_aa")
    return np.frombuffer(wavelength_bytes, dtype=np.float32).astype(float)


def load_lanl_catalog_metadata(lanl_spectra_path):
    catalog = _lanl_parquet_file(str(lanl_spectra_path)).read(columns=LANL_METADATA_COLUMNS).to_pandas()
    catalog.index.name = "spectrum_id"
    return catalog


def load_simulation_spectra(simulation_id, lanl_spectra_path):
    """{(angle_index, time_index): flux_rest} de una simulacion LANL, leyendo su(s) row group(s) una
    sola vez. Pensado para iterar todos los espectros de una simulacion sin reescanear el parquet."""
    simulation_id = int(simulation_id)
    parquet_file = _lanl_parquet_file(str(lanl_spectra_path))
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
        raise KeyError(f"simulation_id={simulation_id} no encontrada en {lanl_spectra_path}")
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
            ordered["time_index"].to_numpy(),
            ordered["time_days"].to_numpy(),
        )
    return simulation_time_grids


def nearest_time_index(simulation_time_grids, simulation_id, rest_phase_days):
    time_index_array, time_days_array = simulation_time_grids[int(simulation_id)]
    nearest_position = int(np.argmin(np.abs(time_days_array - rest_phase_days)))
    return int(time_index_array[nearest_position])


def sample_kn_realizations_on_grid(redshift_grid, realizations_per_redshift, simulation_pool, rng):
    """realizations_per_redshift sorteos (simulacion, angulo, offset de explosion, paridad de la
    cadencia) por cada z de la grilla. noise_id secuencial siembra el ruido; el mismo set de
    realizaciones se comparte entre tiers (la misma KN observada en deep y wide).

    `cadence_parity` se sortea aparte del offset y con la misma probabilidad: juntos reproducen la
    fase uniforme del merger dentro del ciclo de 10 d de la cadencia (ver CADENCE_PARITY_PERIOD)."""
    realizations = {}
    noise_id = 0
    for redshift in np.asarray(redshift_grid, dtype=float):
        for _ in range(realizations_per_redshift):
            realizations[noise_id] = {
                "noise_id": noise_id,
                "redshift": float(redshift),
                "simulation_id": int(rng.choice(simulation_pool)),
                "angle_index": int(rng.integers(N_ANGLE_BINS)),
                "explosion_offset_days": float(rng.uniform(0.0, EXPLOSION_OFFSET_MAX_DAYS)),
                "cadence_parity": int(rng.integers(CADENCE_PARITY_PERIOD)),
            }
            noise_id += 1
    return realizations


def kn_object_id(realization):
    """object_id de la KN = simulation_id_angle_index_offset_redshift_parity_noiseid. El
    simulation_id va PRIMERO porque el split anti-fuga del training lo lee de ahi
    (training/openuniverse_data.py). El noise_id va al final y es lo que hace el id unico por
    construccion: sim/angulo/z/offset-a-4-decimales colisionaban ~2 veces por millon."""
    return (
        f"{realization['simulation_id']}_{realization['angle_index']}_"
        f"{realization['explosion_offset_days']:.4f}_{realization['redshift']:.4f}_"
        f"{realization['cadence_parity']}_{realization['noise_id']}"
    )


def kn_model_from_spectra(realization, simulation_spectra, simulation_time_grids, wavelength_rest_aa, bands):
    """Modelo verdadero de UNA realizacion (mag AB por banda en cada visita base) a partir de los
    espectros ya cargados de su simulacion. base_epochs = offset + 5*arange(N_KN_VISITS);
    rest_phase = base_epoch/(1+z). Tier-independiente: depende solo de (sim, angulo, offset, z)."""
    redshift = realization["redshift"]
    angle_index = realization["angle_index"]
    simulation_id = realization["simulation_id"]
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

    # Se descarta NaN (no hay espectro para esa fase/angulo) pero se conserva +inf, que significa
    # banda cubierta y sin flujo: es una medicion real, la de una no-deteccion.
    model = {}
    for band in bands:
        band_magnitudes = magnitudes_by_band[band]
        usable = ~np.isnan(band_magnitudes)
        if usable.any():
            model[band] = (base_epochs[usable], band_magnitudes[usable])
    return model, base_epochs


def realizations_by_simulation(realizations):
    """{simulation_id: [realizacion, ...]} ordenado por simulation_id. Agrupar por simulacion es lo
    que permite leer cada row group LANL una sola vez, tanto en serie como en paralelo."""
    by_simulation = defaultdict(list)
    for realization in realizations.values():
        by_simulation[realization["simulation_id"]].append(realization)
    return dict(sorted(by_simulation.items()))


def build_kn_models(
    realizations, simulation_time_grids, wavelength_rest_aa, lanl_spectra_path, bands=ALL_ROMAN_BANDS
):
    """Modelo verdadero de cada KN en UNA pasada, agrupando por simulacion para leer cada row group
    LANL una sola vez. Devuelve {kn_object_id: (model_6band, base_epochs, realization)}.
    Ruta secuencial; para grillas grandes ver run_kn_tiers_parallel, que ademas evita materializar
    todos los modelos a la vez."""
    by_simulation = realizations_by_simulation(realizations)

    kn_models = {}
    start = time.time()
    for counter, (simulation_id, realization_list) in enumerate(by_simulation.items()):
        simulation_spectra = load_simulation_spectra(simulation_id, lanl_spectra_path)
        for realization in realization_list:
            model, base_epochs = kn_model_from_spectra(
                realization, simulation_spectra, simulation_time_grids, wavelength_rest_aa, bands
            )
            kn_models[realization["noise_id"]] = (model, base_epochs, realization)

        if (counter + 1) % 50 == 0:
            logger.info(
                "[kn-models] %d/%d simulations (%d kilonovas, %.0fs)",
                counter + 1,
                len(by_simulation),
                len(kn_models),
                time.time() - start,
            )
    return kn_models


def kn_windows_for_tier(kn_models, constants):
    """Ventanas KN del tier: subset de bandas del modelo de 6 bandas + misma receta/ventana que OU."""
    windows = []
    n_detected = 0
    n_skipped = 0
    tier_bands = set(constants["bands"])
    start = time.time()
    for counter, (_kn_object_id, (model_6band, base_epochs, realization)) in enumerate(kn_models.items()):
        model_tier = {band: series for band, series in model_6band.items() if band in tier_bands}
        noise_seed = KN_SEED_OFFSET + realization["noise_id"]
        window = build_window_from_model(
            kn_object_id(realization),
            model_tier,
            constants,
            realization["redshift"],
            KN_GENTYPE,
            base_epochs=base_epochs,
            noise_seed=noise_seed,
            visit_index_offset=realization["cadence_parity"],
        )
        if window is None:
            n_skipped += 1
            continue
        n_detected += 1
        windows.append(window)
        if (counter + 1) % 5000 == 0:
            logger.info(
                "[%s] kilonovas %d/%d detected=%d skipped=%d (%.0fs)",
                constants["tier"],
                counter + 1,
                len(kn_models),
                n_detected,
                n_skipped,
                time.time() - start,
            )
    return windows, n_detected, n_skipped


def run_kn_tier(tier, kn_models, output_path):
    constants = build_tier_constants(tier)
    logger.info(
        "[%s] field=%s  noise_floor_variance: %s",
        tier,
        constants["field_name"],
        ", ".join(f"{b}={v:.0f}" for b, v in constants["noise_floor_variance"].items()),
    )
    windows, n_detected, n_skipped = kn_windows_for_tier(kn_models, constants)
    if not windows:
        logger.warning("[%s] no kilonova reached a detection; nothing written", tier)
        return 0
    result = pd.concat(windows, ignore_index=True)
    result.to_parquet(output_path, index=False)
    logger.info(
        "[%s] DONE kilonovas detected=%d skipped=%d rows=%d -> %s",
        tier,
        n_detected,
        n_skipped,
        len(result),
        output_path,
    )
    return n_detected


# ---------------------------------------------------------------------------
# Ruta paralela: una simulacion LANL por tarea. El worker construye el modelo Y las ventanas de
# todos los tiers en el mismo proceso, de modo que los modelos -- que no se necesitan despues --
# nunca cruzan el limite de proceso ni se materializan todos a la vez. Solo vuelven las ventanas,
# que son el producto final. El ruido va sembrado por realizacion (KN_SEED_OFFSET + noise_id), asi
# que el resultado es identico al secuencial: el paralelismo no cambia ni un fotometro.
# ---------------------------------------------------------------------------

_KN_WORKER_STATE = {}


def _kn_worker_initializer(simulation_time_grids, wavelength_rest_aa, lanl_spectra_path, tiers, bands):
    _KN_WORKER_STATE["simulation_time_grids"] = simulation_time_grids
    _KN_WORKER_STATE["wavelength_rest_aa"] = wavelength_rest_aa
    _KN_WORKER_STATE["lanl_spectra_path"] = lanl_spectra_path
    _KN_WORKER_STATE["tier_constants"] = {tier: build_tier_constants(tier) for tier in tiers}
    _KN_WORKER_STATE["bands"] = bands


def _kn_simulation_task(work_item):
    """Una simulacion LANL -> {tier: DataFrame de ventanas} + detectadas por tier."""
    simulation_id, realization_list = work_item
    state = _KN_WORKER_STATE
    simulation_spectra = load_simulation_spectra(simulation_id, state["lanl_spectra_path"])
    tier_bands = {tier: set(constants["bands"]) for tier, constants in state["tier_constants"].items()}

    windows = {tier: [] for tier in state["tier_constants"]}
    for realization in realization_list:
        model_6band, base_epochs = kn_model_from_spectra(
            realization,
            simulation_spectra,
            state["simulation_time_grids"],
            state["wavelength_rest_aa"],
            state["bands"],
        )
        kn_id = kn_object_id(realization)
        noise_seed = KN_SEED_OFFSET + realization["noise_id"]
        for tier, constants in state["tier_constants"].items():
            model_tier = {b: s for b, s in model_6band.items() if b in tier_bands[tier]}
            window = build_window_from_model(
                kn_id,
                model_tier,
                constants,
                realization["redshift"],
                KN_GENTYPE,
                base_epochs=base_epochs,
                noise_seed=noise_seed,
                visit_index_offset=realization["cadence_parity"],
            )
            if window is not None:
                windows[tier].append(window)

    detected = {tier: len(tier_windows) for tier, tier_windows in windows.items()}
    frames = {
        tier: (pd.concat(tier_windows, ignore_index=True) if tier_windows else None)
        for tier, tier_windows in windows.items()
    }
    return frames, detected, len(realization_list)


def run_kn_tiers_parallel(
    realizations,
    simulation_time_grids,
    wavelength_rest_aa,
    lanl_spectra_path,
    tiers,
    output_paths,
    workers,
    bands=ALL_ROMAN_BANDS,
):
    """Genera las ventanas KN de todos los tiers repartiendo las simulaciones entre `workers`
    procesos y escribe un parquet por tier. Devuelve {tier: kilonovas detectadas}.

    Se usa imap ordenado sobre las simulaciones ordenadas (no imap_unordered) para que el orden de
    filas del parquet sea reproducible entre corridas."""
    by_simulation = realizations_by_simulation(realizations)
    work_items = list(by_simulation.items())
    # Cada worker es monohilo: 25 procesos x N hilos BLAS satura la maquina y frena todo. Con
    # spawn los hijos heredan este environ e importan numpy ya con el limite puesto.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    logger.info(
        "[kn] %d realizaciones sobre %d simulaciones, %d workers, tiers=%s",
        len(realizations),
        len(work_items),
        workers,
        ",".join(tiers),
    )

    frames_by_tier = {tier: [] for tier in tiers}
    detected_by_tier = {tier: 0 for tier in tiers}
    processed = 0
    start = time.time()
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        workers,
        initializer=_kn_worker_initializer,
        initargs=(simulation_time_grids, wavelength_rest_aa, str(lanl_spectra_path), tuple(tiers), bands),
    ) as pool:
        for counter, (frames, detected, n_realizations) in enumerate(
            pool.imap(_kn_simulation_task, work_items, chunksize=1), start=1
        ):
            processed += n_realizations
            for tier in tiers:
                detected_by_tier[tier] += detected[tier]
                if frames[tier] is not None:
                    frames_by_tier[tier].append(frames[tier])
            if counter % 25 == 0 or counter == len(work_items):
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed else 0.0
                remaining = (len(realizations) - processed) / rate if rate else 0.0
                logger.info(
                    "[kn] %d/%d simulaciones  %d/%d KNe  detectadas=%s  %.0f KN/s  ETA %.0f min",
                    counter,
                    len(work_items),
                    processed,
                    len(realizations),
                    " ".join(f"{tier}={detected_by_tier[tier]}" for tier in tiers),
                    rate,
                    remaining / 60.0,
                )

    for tier in tiers:
        if not frames_by_tier[tier]:
            logger.warning("[%s] ninguna kilonova alcanzo deteccion; no se escribe nada", tier)
            continue
        result = pd.concat(frames_by_tier[tier], ignore_index=True)
        result.to_parquet(output_paths[tier], index=False)
        logger.info(
            "[%s] DONE kilonovas detectadas=%d skipped=%d rows=%d -> %s",
            tier,
            detected_by_tier[tier],
            len(realizations) - detected_by_tier[tier],
            len(result),
            output_paths[tier],
        )
    return detected_by_tier


def collect_object_records(catalog_path, limit=None):
    """(id, z_CMB, gentype) de los transientes del catalogo de un field (gentype 99 = FIXMAG, fuera)."""
    catalog = pd.read_parquet(catalog_path)
    transients = catalog[catalog["gentype"] != 99]
    object_records = [(int(row.id), float(row.z_CMB), int(row.gentype)) for row in transients.itertuples()]
    if limit is not None:
        object_records = object_records[:limit]
    return object_records


def run_tier(tier, object_records, hdf5_path, output_path):
    constants = build_tier_constants(tier)
    logger.info(
        "[%s] field=%s  noise_floor_variance: %s",
        tier,
        constants["field_name"],
        ", ".join(f"{b}={v:.0f}" for b, v in constants["noise_floor_variance"].items()),
    )

    windows = []
    n_ou_detected = 0
    n_ou_full = 0
    start = time.time()
    with h5py.File(hdf5_path, "r") as hdf5:
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
                logger.info(
                    "[%s] openuniverse %d/%d detected=%d full%d=%d (%.0fs)",
                    tier,
                    counter + 1,
                    len(object_records),
                    n_ou_detected,
                    NUMBER_OF_EPOCHS,
                    n_ou_full,
                    time.time() - start,
                )

    result = pd.concat(windows, ignore_index=True)
    result.to_parquet(output_path, index=False)
    logger.info(
        "[%s] DONE openuniverse_detected=%d (full%d=%d) rows=%d -> %s",
        tier,
        n_ou_detected,
        NUMBER_OF_EPOCHS,
        n_ou_full,
        len(result),
        output_path,
    )
    return n_ou_detected
