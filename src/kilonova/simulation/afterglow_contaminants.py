"""GRB afterglow contaminants: one continuous population over the afterglowpy parameter space.

Forward-shock synchrotron of a Gaussian structured jet (afterglowpy, Ryan et al. 2020), reddened
by host dust and folded through the galsim Roman bandpasses at the visits of the 5 d cadence. The
parameter limits and their provenance live in ``configs/afterglow.yaml``.
"""

import logging
import multiprocessing
import os
import time
from pathlib import Path

import afterglowpy
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from astropy import units as u
from astropy.cosmology import Planck18
from dust_extinction.averages import G03_SMCBar

from kilonova.config import REPOSITORY_ROOT
from kilonova.photometry.roman_noise import (
    BASE_CADENCE_DAYS,
    build_tier_constants,
    flux_error_electrons,
    limiting_magnitude_5sigma,
)
from kilonova.photometry.spectra import ALL_ROMAN_BANDS, spectrum_to_roman_magnitudes
from kilonova.simulation.early_windows import (
    CADENCE_PARITY_PERIOD,
    N_KN_VISITS,
    build_window_from_model,
)

AFTERGLOW_CONFIG_FILE = REPOSITORY_ROOT / "configs" / "afterglow.yaml"
# Not an OpenUniverse gentype; the window label is overwritten with AFTERGLOW_LABEL.
AFTERGLOW_GENTYPE = 80
AFTERGLOW_LABEL = "GRB afterglow"
# Noise seeds disjoint from the OpenUniverse ids (~8e7) and from the kilonovae (KN_SEED_OFFSET).
AFTERGLOW_SEED_OFFSET = 3_000_000_000
# Runs with different population seeds get disjoint afterglow ids, so a held-out sample never
# shares an object_id or a noise seed with the training sample.
AFTERGLOW_IDS_PER_SEED = 10_000_000
# afterglowpy dominates the cost and scales with the number of (time, frequency) points. The
# spectrum is a smooth broken power law: 20 points reproduce the band magnitudes of a 60 point
# grid to 0.002 mag.
WAVELENGTH_GRID_AA = np.geomspace(4000.0, 25000.0, 20)
# Cheap pre-check at the two ends of the Roman coverage. An afterglow fainter than the deepest
# 5 sigma limit by more than the margin at both wavelengths and every visit cannot reach
# S/N >= 5 in any band: the margin covers the zeropoint jitter (0.15 mag rms), the difference
# between a band and a monochromatic magnitude, and a spectral peak between the two ends.
FAINT_CHECK_WAVELENGTHS_AA = np.array([6200.0, 18400.0])
FAINT_CHECK_MARGIN_MAG = 2.0
SPEED_OF_LIGHT_AA_PER_S = 2.99792458e18
MILLIJANSKY_CGS = 1e-26
SECONDS_PER_DAY = 86400.0
PARAMETER_GROUPS = ("physical", "geometry")
logger = logging.getLogger(__name__)


def load_afterglow_config(config_file=AFTERGLOW_CONFIG_FILE):
    with open(config_file) as handle:
        return yaml.safe_load(handle)


def sample_range(parameter_range, number_of_samples, random_generator):
    # PyYAML reads exponent literals without a decimal point or sign (1e-6) as strings.
    minimum = float(parameter_range["minimum"])
    maximum = float(parameter_range["maximum"])
    sampling = parameter_range["sampling"]
    if sampling == "uniform":
        return random_generator.uniform(minimum, maximum, number_of_samples)
    if sampling == "log_uniform":
        return np.exp(random_generator.uniform(np.log(minimum), np.log(maximum), number_of_samples))
    if sampling == "isotropic":
        cosine = random_generator.uniform(np.cos(maximum), np.cos(minimum), number_of_samples)
        return np.arccos(cosine)
    raise ValueError(f"unknown sampling: {sampling!r}")


def sample_afterglow_parameters(number_of_afterglows, random_generator, config=None, first_afterglow_id=0):
    """One row per afterglow. Every parameter is drawn independently of the others."""
    if config is None:
        config = load_afterglow_config()
    columns = {}
    for group in PARAMETER_GROUPS:
        for name, parameter_range in config[group].items():
            if isinstance(parameter_range, dict):
                columns[name] = sample_range(parameter_range, number_of_afterglows, random_generator)
    columns["wing_angle_rad"] = np.minimum(
        float(config["geometry"]["wing_over_core"]) * columns["core_angle_rad"], 0.5 * np.pi
    )
    columns["redshift"] = sample_range(config["redshift"], number_of_afterglows, random_generator)
    columns["first_visit_time_days"] = sample_range(
        config["first_visit_time_days"], number_of_afterglows, random_generator
    )

    host_extinction = config["host_extinction"]
    extinction_scale = float(host_extinction["median_av"]) / np.log(2.0)
    host_extinction_av = random_generator.exponential(extinction_scale, number_of_afterglows)
    is_dust_free = random_generator.random(number_of_afterglows) < float(
        host_extinction["dust_free_fraction"]
    )
    host_extinction_av[is_dust_free] = 0.0
    columns["host_extinction_av"] = host_extinction_av
    columns["cadence_parity"] = random_generator.integers(0, CADENCE_PARITY_PERIOD, number_of_afterglows)

    population = pd.DataFrame(columns)
    population.insert(0, "afterglow_id", first_afterglow_id + np.arange(number_of_afterglows))
    return population


def jet_parameters(parameters):
    redshift = float(parameters["redshift"])
    return {
        "jetType": afterglowpy.jet.Gaussian,
        "specType": afterglowpy.jet.SimpleSpec,
        "thetaObs": float(parameters["viewing_angle_rad"]),
        "E0": float(parameters["isotropic_energy_erg"]),
        "thetaCore": float(parameters["core_angle_rad"]),
        "thetaWing": float(parameters["wing_angle_rad"]),
        "n0": float(parameters["density_cm3"]),
        "p": float(parameters["electron_index"]),
        "epsilon_e": float(parameters["epsilon_e"]),
        "epsilon_B": float(parameters["epsilon_b"]),
        "xi_N": 1.0,
        "d_L": Planck18.luminosity_distance(redshift).to(u.cm).value,
        "z": redshift,
    }


def host_transmission(wavelength_observed_aa, redshift, host_extinction_av):
    if host_extinction_av == 0.0:
        return np.ones_like(wavelength_observed_aa)
    wavelength_rest = wavelength_observed_aa / (1.0 + redshift) * u.AA
    return G03_SMCBar().extinguish(wavelength_rest, Av=host_extinction_av)


def afterglow_spectra(parameters, times_days, wavelength_observed_aa=WAVELENGTH_GRID_AA, with_host_dust=True):
    """Observer-frame f_lambda [erg / (s cm2 AA)] of shape (times, wavelengths), or None when
    afterglowpy fails on this parameter set."""
    times_days = np.atleast_1d(np.asarray(times_days, dtype=float))
    frequencies_hz = SPEED_OF_LIGHT_AA_PER_S / wavelength_observed_aa
    time_grid, frequency_grid = np.meshgrid(times_days * SECONDS_PER_DAY, frequencies_hz, indexing="ij")
    # afterglowpy's shell integrator raises "bad dFnu" on a few parameter sets; those afterglows
    # are reported and left out rather than patched.
    try:
        flux_millijansky = afterglowpy.fluxDensity(
            time_grid.ravel(), frequency_grid.ravel(), **jet_parameters(parameters)
        )
    except RuntimeError:
        return None
    flux_nu = flux_millijansky.reshape(time_grid.shape) * MILLIJANSKY_CGS
    flux_lambda = flux_nu * SPEED_OF_LIGHT_AA_PER_S / wavelength_observed_aa**2
    if with_host_dust:
        flux_lambda = flux_lambda * host_transmission(
            wavelength_observed_aa, float(parameters["redshift"]), float(parameters["host_extinction_av"])
        )
    return flux_lambda


def visit_times_days(parameters):
    return float(parameters["first_visit_time_days"]) + BASE_CADENCE_DAYS * np.arange(N_KN_VISITS)


def afterglow_roman_magnitudes(parameters, bands=ALL_ROMAN_BANDS):
    """{band: true AB magnitude at each of the N_KN_VISITS survey visits}, or None when
    afterglowpy fails."""
    flux_lambda = afterglow_spectra(parameters, visit_times_days(parameters))
    if flux_lambda is None:
        return None
    magnitudes = {}
    for band in bands:
        magnitudes[band] = np.empty(N_KN_VISITS)
    for visit_index in range(N_KN_VISITS):
        visit_magnitudes = spectrum_to_roman_magnitudes(WAVELENGTH_GRID_AA, flux_lambda[visit_index], bands)
        for band in bands:
            magnitudes[band][visit_index] = visit_magnitudes[band]
    return magnitudes


def exceeds_bright_limit(magnitudes, config=None):
    """True when any band is brighter than the saturation limit at any of the visits."""
    if config is None:
        config = load_afterglow_config()
    bright_limit = float(config["saturation"]["bright_limit_magnitude"])
    for band_magnitudes in magnitudes.values():
        if np.nanmin(band_magnitudes) < bright_limit:
            return True
    return False


def faintest_detection_limit(constants_by_tier):
    """Deepest 5 sigma limiting magnitude over every band of every tier."""
    limits = []
    for constants in constants_by_tier.values():
        for band in constants["bands"]:
            background_error = flux_error_electrons(0.0, constants["noise_floor_variance"][band])
            limits.append(
                limiting_magnitude_5sigma(
                    background_error, constants["exposure_time"][band], constants["zeropoint"][band]
                )
            )
    return max(limits)


def is_too_faint(parameters, detection_limit):
    """True when the afterglow stays fainter than detection_limit + FAINT_CHECK_MARGIN_MAG at
    every visit. False also when afterglowpy fails, so the failure is reported by the full
    evaluation."""
    flux_lambda = afterglow_spectra(parameters, visit_times_days(parameters), FAINT_CHECK_WAVELENGTHS_AA)
    if flux_lambda is None:
        return False
    flux_nu = flux_lambda * FAINT_CHECK_WAVELENGTHS_AA**2 / SPEED_OF_LIGHT_AA_PER_S
    brightest_flux_nu = np.nanmax(flux_nu)
    if not brightest_flux_nu > 0:
        return True
    brightest_magnitude = -2.5 * np.log10(brightest_flux_nu) - 48.6
    return bool(brightest_magnitude > detection_limit + FAINT_CHECK_MARGIN_MAG)


def evaluate_afterglow(parameters):
    """Picklable unit of work for a process pool: (afterglow_id, Roman magnitudes or None)."""
    return int(parameters["afterglow_id"]), afterglow_roman_magnitudes(parameters)


def build_afterglow_window(parameters, magnitudes, constants):
    """Noisy survey window of one afterglow in one tier, or None when no visit reaches S/N >= 5.
    Same grid convention as the kilonovae: index 0 is the first visit after the burst and its
    cadence parity is drawn separately."""
    base_epochs = BASE_CADENCE_DAYS * np.arange(N_KN_VISITS)
    model = {}
    for band in constants["bands"]:
        model[band] = (base_epochs, magnitudes[band])
    afterglow_id = int(parameters["afterglow_id"])
    window = build_window_from_model(
        f"afterglow_{afterglow_id:07d}",
        model,
        constants,
        float(parameters["redshift"]),
        AFTERGLOW_GENTYPE,
        base_epochs=base_epochs,
        noise_seed=AFTERGLOW_SEED_OFFSET + afterglow_id,
        visit_index_offset=int(parameters["cadence_parity"]),
    )
    if window is None:
        return None
    window["label"] = AFTERGLOW_LABEL
    window["tier"] = constants["tier"]
    return window


_WORKER_STATE = {}


def _worker_state(config_file, tiers):
    key = (str(config_file), tuple(tiers))
    if key not in _WORKER_STATE:
        constants = {}
        for tier in tiers:
            constants[tier] = build_tier_constants(tier)
        _WORKER_STATE[key] = (
            load_afterglow_config(config_file),
            constants,
            faintest_detection_limit(constants),
        )
    return _WORKER_STATE[key]


def _write_parquet_atomically(frame, path):
    temporary_path = path.with_suffix(".tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def parameters_part_path(parts_directory, chunk_index):
    return Path(parts_directory) / f"parameters_{chunk_index:06d}.parquet"


def windows_part_path(parts_directory, tier, chunk_index):
    return Path(parts_directory) / f"windows_{tier}_{chunk_index:06d}.parquet"


def process_afterglow(parameters, tiers, config, constants, detection_limit):
    """(status, {tier: window}) of one afterglow. Only the afterglows that pass the cheap faint
    check pay for the full spectrum."""
    status = {
        "afterglow_id": int(parameters["afterglow_id"]),
        "failed": False,
        "too_faint": False,
        "discarded_bright": False,
    }
    for tier in tiers:
        status[f"window_{tier}"] = False
    windows = {}
    if is_too_faint(parameters, detection_limit):
        status["too_faint"] = True
        return status, windows
    magnitudes = afterglow_roman_magnitudes(parameters)
    if magnitudes is None:
        status["failed"] = True
    elif exceeds_bright_limit(magnitudes, config):
        status["discarded_bright"] = True
    else:
        for tier in tiers:
            window = build_afterglow_window(parameters, magnitudes, constants[tier])
            if window is not None:
                status[f"window_{tier}"] = True
                windows[tier] = window
    return status, windows


def generate_chunk(task):
    """Evaluate one chunk of the population and write its part files. The parameters part is
    written last: its presence marks the chunk as complete, which is what a resumed run checks."""
    chunk_index, chunk, tiers, parts_directory, config_file = task
    config, constants, detection_limit = _worker_state(config_file, tiers)
    windows = {}
    for tier in tiers:
        windows[tier] = []
    status_rows = []
    for _, parameters in chunk.iterrows():
        status, afterglow_windows = process_afterglow(parameters, tiers, config, constants, detection_limit)
        for tier, window in afterglow_windows.items():
            windows[tier].append(window)
        status_rows.append(status)

    summary = {"afterglows": len(chunk)}
    for tier in tiers:
        if windows[tier]:
            _write_parquet_atomically(
                pd.concat(windows[tier], ignore_index=True),
                windows_part_path(parts_directory, tier, chunk_index),
            )
        summary[tier] = len(windows[tier])
    status_frame = pd.DataFrame(status_rows)
    for column in ("failed", "too_faint", "discarded_bright"):
        summary[column] = int(status_frame[column].sum())
    _write_parquet_atomically(
        chunk.merge(status_frame, on="afterglow_id"), parameters_part_path(parts_directory, chunk_index)
    )
    return summary


def merge_parts(part_paths, output_path):
    """Stream the part files into one parquet, so the merge never holds the whole sample."""
    writer = None
    number_of_rows = 0
    for part_path in part_paths:
        table = pq.read_table(part_path)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table.cast(writer.schema))
        number_of_rows += table.num_rows
    if writer is not None:
        writer.close()
    return number_of_rows


def run_afterglow_generation(
    population, tiers, output_dir, name, workers, chunk_size, config_file=AFTERGLOW_CONFIG_FILE
):
    """Generate the windows of `population` over `workers` processes and write
    {name}_windows_{tier}.parquet and {name}_parameters.parquet in `output_dir`.

    Each chunk writes its own part files under {name}_parts/, so an interrupted run resumes
    where it stopped: rerunning with the same seed, size and chunk size skips the chunks whose
    parameters part already exists. Returns {tier: number of afterglows with a window}."""
    output_dir = Path(output_dir)
    parts_directory = output_dir / f"{name}_parts"
    parts_directory.mkdir(parents=True, exist_ok=True)
    number_of_chunks = int(np.ceil(len(population) / chunk_size))
    tasks = []
    for chunk_index in range(number_of_chunks):
        if parameters_part_path(parts_directory, chunk_index).exists():
            continue
        chunk = population.iloc[chunk_index * chunk_size : (chunk_index + 1) * chunk_size]
        tasks.append((chunk_index, chunk, tuple(tiers), str(parts_directory), str(config_file)))
    logger.info(
        "[afterglow] %d afterglows in %d chunks, %d already done, %d workers, tiers=%s",
        len(population),
        number_of_chunks,
        number_of_chunks - len(tasks),
        workers,
        ",".join(tiers),
    )

    # Single-threaded workers: one process per core with multi-threaded BLAS in each would
    # oversubscribe the machine. Spawned children inherit this environment before importing numpy.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    totals = {"afterglows": 0, "failed": 0, "too_faint": 0, "discarded_bright": 0}
    for tier in tiers:
        totals[tier] = 0
    start = time.time()
    if tasks:
        context = multiprocessing.get_context("spawn")
        with context.Pool(workers) as pool:
            for counter, summary in enumerate(
                pool.imap_unordered(generate_chunk, tasks, chunksize=1), start=1
            ):
                for key, value in summary.items():
                    totals[key] += value
                if counter % 10 == 0 or counter == len(tasks):
                    elapsed = time.time() - start
                    rate = totals["afterglows"] / elapsed
                    remaining_afterglows = (len(tasks) - counter) * chunk_size
                    window_counts = []
                    for tier in tiers:
                        window_counts.append(f"{tier}={totals[tier]}")
                    logger.info(
                        "[afterglow] %d/%d chunks  windows %s  too faint=%d  too bright=%d  failed=%d"
                        "  %.0f afterglows/s  ETA %.0f min",
                        counter,
                        len(tasks),
                        " ".join(window_counts),
                        totals["too_faint"],
                        totals["discarded_bright"],
                        totals["failed"],
                        rate,
                        remaining_afterglows / rate / 60.0,
                    )

    parameter_parts = []
    for chunk_index in range(number_of_chunks):
        parameter_parts.append(parameters_part_path(parts_directory, chunk_index))
    parameters_path = output_dir / f"{name}_parameters.parquet"
    merge_parts(parameter_parts, parameters_path)
    logger.info("[afterglow] parameters of %d afterglows -> %s", len(population), parameters_path)

    parameters = pd.read_parquet(parameters_path)
    windows_by_tier = {}
    for tier in tiers:
        window_parts = []
        for chunk_index in range(number_of_chunks):
            part_path = windows_part_path(parts_directory, tier, chunk_index)
            if part_path.exists():
                window_parts.append(part_path)
        windows_path = output_dir / f"{name}_windows_{tier}.parquet"
        number_of_rows = merge_parts(window_parts, windows_path)
        windows_by_tier[tier] = int(parameters[f"window_{tier}"].sum())
        logger.info(
            "[%s] DONE afterglows with a window=%d rows=%d -> %s",
            tier,
            windows_by_tier[tier],
            number_of_rows,
            windows_path,
        )
    return windows_by_tier
