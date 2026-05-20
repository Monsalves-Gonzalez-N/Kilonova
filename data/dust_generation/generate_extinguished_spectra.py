"""Generate a parquet of Roman AB photometry for LANL kilonova spectra
attenuated by host + MW extinction, parallelized with Ray.

Physics functions are taken verbatim from Extinction.ipynb (already validated).
For each (cached LANL spectrum) x (redshift in the grid) we draw one realization
of (Av_host, Rv_host, EBV_MW) and apply the full observational pipeline:

    rest spectrum --> F99 host extinction (Av_host, Rv_host)
                  --> redshift (specutils)
                  --> cosmological distance dimming (Planck18 luminosity distance)
                  --> F99 Milky Way extinction (Av_MW = Rv_MW * EBV_MW)
                  --> synthetic Roman AB magnitudes via pyphot

STOP rules (validated in test_stop_rules.ipynb):
    STOP-A (z-loop, patience=z_patience): track consecutive redshifts with no
        detection across any (angle, time). Break the z-loop when the streak
        reaches `z_patience`; a detection at any z resets the counter.
    STOP-LC (per-band, within each (z, angle)): each Roman band keeps a counter
        of consecutive non-detected decaying steps. The time axis truncates
        when ALL bands reach `lc_patience`.
Dust is sampled once per redshift and reused across all angles at that z
(host galaxy fixed per redshift).

Input: lanl_spectra.parquet built by cache_lanl_spectra.py (one row group per
LANL .dat file, rest-frame flux as FixedSizeList<float32, 1024>).
Output: one parquet row per evaluated (simulation, time, angle, redshift)
with scalar AB magnitudes per Roman filter and the sampled dust parameters.
"""

import argparse
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyphot
import ray
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import fits
from astropy.table import Table
from dust_extinction.parameter_averages import F99
from dustmaps.sfd import SFDQuery
from specutils import Spectrum


MILKY_WAY_RV = 3.1
F99_VALID_RANGE = (1000.0, 33333.0)
FLUX_UNIT = u.Unit('erg / (s cm2 AA)')
CM_TO_ANG = (1 * u.cm).to(u.AA).value
N_ANGLE_BINS = 54
DEFAULT_KCOR_PATH = Path(__file__).resolve().parent.parent.parent / 'kcor' / 'kcor_ROMAN.fits'

_SPEC_PATTERN = re.compile(
    r'Run_(T[PS])_dyn_all_lanth_(wind\d)_all_'
    r'md([\d.]+)_vd([\d.]+)_mw([\d.]+)_vw([\d.]+)_spec_'
)


# ---------------------------------------------------------------------------
# Tested functions from Extinction.ipynb (verbatim, plot args removed)
# ---------------------------------------------------------------------------

def wood_vasey_pdf(
    extinction_av,
    exponential_amplitude=1.0,
    gaussian_amplitude=0.5,
    exponential_scale=1.7,
    gaussian_sigma=0.6,
):
    if np.isscalar(extinction_av):
        if extinction_av < 0:
            return 0.0
    else:
        extinction_av = np.asarray(extinction_av)
    exponential_term = (exponential_amplitude / exponential_scale) * np.exp(-extinction_av / exponential_scale)
    gaussian_term = (2 * gaussian_amplitude / (np.sqrt(2 * np.pi) * gaussian_sigma)) * np.exp(-extinction_av**2 / (2 * gaussian_sigma**2))
    probability = exponential_term + gaussian_term
    if not np.isscalar(extinction_av):
        probability[extinction_av < 0] = 0.0
    return probability


def sample_extinction_av(number_of_samples, exponential_scale=1.7, gaussian_sigma=0.6, av_max=10.0, grid_resolution=10000):
    av_grid = np.linspace(0, av_max, grid_resolution)
    probability_grid = wood_vasey_pdf(av_grid, exponential_scale=exponential_scale, gaussian_sigma=gaussian_sigma)
    probability_grid = probability_grid / probability_grid.sum()
    return np.random.choice(av_grid, size=number_of_samples, p=probability_grid)


def sample_extinction_rv(number_of_samples, left=3.0, mode=3.1, right=3.2):
    return np.random.triangular(left=left, mode=mode, right=right, size=number_of_samples)


def sample_hourglass_ebv(parquet_path, number_of_samples=10000, random_seed=123):
    hourglass_table = pd.read_parquet(parquet_path, columns=['ra', 'dec'])
    unique_table = hourglass_table.drop_duplicates(subset=['ra', 'dec']).reset_index(drop=True)
    random_generator = np.random.default_rng(random_seed)
    sample_size = min(number_of_samples, len(unique_table))
    sample_indices = random_generator.choice(len(unique_table), size=sample_size, replace=False)
    ra_deg = unique_table['ra'].to_numpy()[sample_indices]
    dec_deg = unique_table['dec'].to_numpy()[sample_indices]
    sky_coordinates = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    ebv_samples = SFDQuery()(sky_coordinates)
    return {
        'ebv_samples': ebv_samples,
        'ra_deg': ra_deg,
        'dec_deg': dec_deg,
    }


def shift_spectrum_to_redshift(rest_wavelength, rest_flux_lambda, redshift):
    spectrum_rest = Spectrum(
        spectral_axis=rest_wavelength * u.AA,
        flux=rest_flux_lambda * u.Unit('erg / (s cm2 AA)'),
    )
    spectrum_redshifted = Spectrum(
        spectral_axis=spectrum_rest.spectral_axis,
        flux=spectrum_rest.flux,
    )
    spectrum_redshifted.shift_spectrum_to(redshift=redshift)
    observed_wavelength = spectrum_redshifted.spectral_axis.to(u.AA).value
    observed_flux_lambda = spectrum_redshifted.flux.value
    return observed_wavelength, observed_flux_lambda


def ab_magnitude_to_flux_lambda(wavelength_angstrom, magnitude_ab):
    speed_of_light_angstrom_per_second = const.c.to(u.AA / u.s).value
    flux_nu = 10.0 ** (-(magnitude_ab + 48.6) / 2.5)
    return flux_nu * speed_of_light_angstrom_per_second / wavelength_angstrom ** 2


def apply_dust_extinction_within_valid_range(
    wavelength_angstrom,
    flux_lambda,
    extinction_av,
    extinction_rv,
    valid_range=F99_VALID_RANGE,
):
    extinction_model = F99(Rv=float(extinction_rv))
    transmission = np.ones_like(wavelength_angstrom, dtype=float)
    valid_mask = (
        (wavelength_angstrom >= valid_range[0])
        & (wavelength_angstrom <= valid_range[1])
    )
    if valid_mask.any():
        transmission[valid_mask] = extinction_model.extinguish(
            wavelength_angstrom[valid_mask] * u.AA, Av=float(extinction_av),
        )
    return flux_lambda * transmission


def generate_observed_kilonova_spectrum(
    wavelength_rest,
    spectrum_input,
    extinction_av_host,
    extinction_rv_host,
    redshift,
    ebv_milky_way,
    rv_milky_way=MILKY_WAY_RV,
    intrinsic_distance_parsec=10.0,
    luminosity_distance_parsec=None,
    cosmology=Planck18,
    input_mode='flux',
    output_mode='flux',
):
    wavelength_rest = np.asarray(wavelength_rest, dtype=float)
    if input_mode == 'flux':
        rest_flux_lambda = np.asarray(spectrum_input, dtype=float)
    elif input_mode == 'magnitude':
        rest_flux_lambda = ab_magnitude_to_flux_lambda(wavelength_rest, np.asarray(spectrum_input, dtype=float))
    else:
        raise ValueError(f'unknown input_mode: {input_mode!r}')

    flux_after_host_extinction = apply_dust_extinction_within_valid_range(
        wavelength_rest,
        rest_flux_lambda,
        extinction_av=extinction_av_host,
        extinction_rv=extinction_rv_host,
    )

    wavelength_observed, flux_after_redshift_at_intrinsic_distance = shift_spectrum_to_redshift(
        wavelength_rest,
        flux_after_host_extinction,
        redshift,
    )

    if luminosity_distance_parsec is None:
        if redshift > 0:
            luminosity_distance_parsec = cosmology.luminosity_distance(redshift).to(u.pc).value
        else:
            luminosity_distance_parsec = intrinsic_distance_parsec

    distance_dimming_factor = (intrinsic_distance_parsec / luminosity_distance_parsec) ** 2
    flux_after_distance_dimming = flux_after_redshift_at_intrinsic_distance * distance_dimming_factor

    extinction_av_milky_way = rv_milky_way * ebv_milky_way
    flux_after_milky_way_extinction = apply_dust_extinction_within_valid_range(
        np.asarray(wavelength_observed, dtype=float),
        flux_after_distance_dimming,
        extinction_av=extinction_av_milky_way,
        extinction_rv=rv_milky_way,
    )

    return {
        'wavelength_observed': wavelength_observed,
        'flux_observed': flux_after_milky_way_extinction,
        'parameters': {
            'extinction_av_host': extinction_av_host,
            'extinction_rv_host': extinction_rv_host,
            'redshift': redshift,
            'ebv_milky_way': ebv_milky_way,
            'rv_milky_way': rv_milky_way,
            'extinction_av_milky_way': extinction_av_milky_way,
            'intrinsic_distance_parsec': intrinsic_distance_parsec,
            'luminosity_distance_parsec': luminosity_distance_parsec,
        },
    }


def _read_header_times(filepath):
    result = subprocess.run(
        ['grep', '^#', filepath],
        capture_output=True, text=True, check=True,
    )
    return [float(line.split('time[d]=')[1]) for line in result.stdout.splitlines()]


def parse_spec(filepath):
    times, spectra, current = [], [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                if current:
                    spectra.append(np.array(current))
                    current = []
                times.append(float(line.split("time[d]=")[1]))
            elif line:
                current.append([float(x) for x in line.split()])
    if current:
        spectra.append(np.array(current))
    spectra = np.array(spectra)
    lam_lo = spectra[0, :, 0] * CM_TO_ANG
    lam_hi = spectra[0, :, 1] * CM_TO_ANG
    lam_AA = 0.5 * (lam_lo + lam_hi)
    flux = spectra[:, :, 2:]
    return np.array(times), lam_AA, flux


def build_lanl_catalog(grid_directory, save_path):
    file_rows = []
    for simulation_id, filepath in enumerate(sorted(Path(grid_directory).glob('*_spec_*.dat'))):
        match = _SPEC_PATTERN.search(filepath.name)
        if match is None:
            continue
        run_type, wind, md, vd, mw, vw = match.groups()
        times = _read_header_times(str(filepath))
        file_rows.append({
            'simulation_id': simulation_id,
            'run_type': run_type,
            'wind': wind,
            'mass_dynamical': float(md),
            'velocity_dynamical': float(vd),
            'mass_wind': float(mw),
            'velocity_wind': float(vw),
            'times': times,
            'filepath': str(filepath),
        })
    file_df = pd.DataFrame(file_rows)
    file_df['time_data'] = [list(enumerate(times)) for times in file_df['times']]
    time_df = file_df.drop(columns='times').explode('time_data').reset_index(drop=True)
    time_df['time_index'] = time_df['time_data'].apply(lambda entry: entry[0])
    time_df['time_days'] = time_df['time_data'].apply(lambda entry: entry[1])
    time_df = time_df.drop(columns='time_data')

    n_time_rows = len(time_df)
    catalog = time_df.loc[time_df.index.repeat(N_ANGLE_BINS)].reset_index(drop=True)
    catalog['angle_index'] = np.tile(np.arange(N_ANGLE_BINS), n_time_rows)
    catalog.index.name = 'spectrum_id'

    if save_path is not None:
        catalog.to_parquet(save_path, index=True)
    return catalog


def load_lanl_catalog(catalog_path, grid_directory):
    if Path(catalog_path).exists():
        return pd.read_parquet(catalog_path)
    return build_lanl_catalog(grid_directory=grid_directory, save_path=catalog_path)


# ---------------------------------------------------------------------------
# Roman synthetic photometry (matches Extinction.ipynb)
# ---------------------------------------------------------------------------

def load_roman_filters(kcor_path):
    with fits.open(kcor_path) as hdul:
        filter_table = Table(hdul['FilterTrans'].data)

    column_names = filter_table.colnames
    wavelength_angstrom = np.array(filter_table[column_names[0]], dtype=float)
    filter_columns = [name for name in column_names[1:] if name != 'W146-W']

    roman_filters = {}
    for filter_name in filter_columns:
        response = np.array(filter_table[filter_name], dtype=float)
        pyphot_filter = pyphot.Filter(
            wavelength_angstrom * pyphot.config.units.U('AA'),
            response,
            name=filter_name,
            dtype='photon',
        )
        leff_quantity = pyphot_filter.leff.to('AA')
        lambda_eff = float(getattr(leff_quantity, 'value', getattr(leff_quantity, 'magnitude', leff_quantity)))
        roman_filters[filter_name] = {
            'filter': pyphot_filter,
            'lambda_eff': lambda_eff,
        }
    return roman_filters


def compute_roman_ab_magnitudes(wavelength_observed, flux_observed, roman_filters):
    wavelength_with_units = np.asarray(wavelength_observed, dtype=float) * pyphot.config.units.U('AA')
    flux_with_units = np.asarray(flux_observed, dtype=float) * pyphot.config.units.U('flam')

    ab_magnitudes = {}
    for filter_name, filter_data in roman_filters.items():
        pyphot_filter = filter_data['filter']
        synthetic_flux = pyphot_filter.get_flux(wavelength_with_units, flux_with_units, axis=-1)
        flux_value = float(synthetic_flux.value if hasattr(synthetic_flux, 'value') else synthetic_flux.magnitude)
        if flux_value <= 0 or not np.isfinite(flux_value):
            ab_magnitudes[filter_name] = np.nan
        else:
            ab_magnitudes[filter_name] = -2.5 * np.log10(flux_value) - pyphot_filter.AB_zero_mag
    return ab_magnitudes


# ---------------------------------------------------------------------------
# New: dataset generation driver
# ---------------------------------------------------------------------------

def build_redshift_grid(redshift_min, redshift_max, number_of_redshifts, spacing):
    if spacing == 'linear':
        return np.linspace(redshift_min, redshift_max, number_of_redshifts)
    if spacing == 'log':
        if redshift_min <= 0:
            raise ValueError('log spacing requires redshift_min > 0')
        return np.geomspace(redshift_min, redshift_max, number_of_redshifts)
    raise ValueError(f'unknown spacing: {spacing!r}')


def select_row_group_indices(parquet_path, max_row_groups, random_seed):
    """Optional row-group subsampling for smoke tests."""
    pf = pq.ParquetFile(parquet_path)
    total = pf.num_row_groups
    if max_row_groups is None or max_row_groups >= total:
        return list(range(total))
    rng = np.random.default_rng(random_seed)
    return sorted(int(i) for i in rng.choice(total, size=max_row_groups, replace=False))


def _presample_dust(z_grid, av_pool, rv_pool, ebv_pool, seed):
    """One dust draw per redshift, shared across all angles at that z."""
    n_z = len(z_grid)
    rng = np.random.default_rng(seed)
    av_arr = rng.choice(np.asarray(av_pool, dtype=float), size=n_z)
    rv_arr = rng.choice(np.asarray(rv_pool, dtype=float), size=n_z)
    ebv_arr = rng.choice(np.asarray(ebv_pool, dtype=float), size=n_z)
    return av_arr, rv_arr, ebv_arr


def _transmission(wl_aa, av, rv):
    """F99 transmission curve over wl_aa; 1.0 outside F99 valid range."""
    model = F99(Rv=float(rv))
    t = np.ones(len(wl_aa))
    mask = (wl_aa >= F99_VALID_RANGE[0]) & (wl_aa <= F99_VALID_RANGE[1])
    if mask.any():
        t[mask] = model.extinguish(wl_aa[mask] * u.AA, Av=float(av))
    return t


def _mags_batch(wl_obs_aa, flux_matrix, roman_filters):
    """Batch AB magnitudes for (n_t, n_wl) flux at fixed observed wavelengths."""
    wl_u = wl_obs_aa * pyphot.config.units.U('AA')
    flux_u = flux_matrix * pyphot.config.units.U('flam')
    n_t = flux_matrix.shape[0]
    mags = np.full((n_t, len(roman_filters)), np.nan)
    for f_idx, (_fn, fd) in enumerate(roman_filters.items()):
        pyf = fd['filter']
        fnu = pyf.get_flux(wl_u, flux_u, axis=-1)
        fval = np.asarray(getattr(fnu, 'magnitude', fnu.value), dtype=float).ravel()
        good = (fval > 0) & np.isfinite(fval)
        mags[good, f_idx] = -2.5 * np.log10(fval[good]) - pyf.AB_zero_mag
    return mags


def _lc_stop_idx_perband(mags, mag_limit, patience):
    """Per-band STOP-LC truncation index; keep mags[:result].

    Each band keeps a counter of consecutive non-detected steps in decay.
    The truncation fires when ALL bands reach `patience`. Decay rule
    between consecutive steps for band b:
        finite prev, finite curr -> in_decay iff curr > prev (fainter)
        finite prev, NaN  curr   -> in_decay (band dropped off)
        NaN   prev, finite curr  -> NOT in_decay (band came online)
        NaN   prev, NaN   curr   -> in_decay (stayed off)
    Detected steps reset the counter for that band.
    """
    n_t, n_f = mags.shape
    count = np.zeros(n_f, dtype=int)
    prev = np.full(n_f, np.nan)
    for i in range(n_t):
        curr = mags[i]
        if i == 0:
            prev = curr
            continue
        detected_b = np.isfinite(curr) & (curr <= mag_limit)
        fp, fc = np.isfinite(prev), np.isfinite(curr)
        in_decay = np.where(
            fp & fc, curr > prev,
            np.where(fp & ~fc, True,
                     np.where(~fp & fc, False, True))
        )
        advance = (~detected_b) & in_decay
        count = np.where(detected_b, 0,
                np.where(advance, count + 1, count))
        prev = curr
        if (count >= patience).all():
            return i + 1
    return n_t


def process_row_group(
    parquet_path,
    row_group_index,
    redshift_grid,
    av_pool,
    rv_pool,
    ebv_pool,
    kcor_path,
    detection_mag_limit,
    lc_patience,
    z_patience,
    worker_seed,
):
    """Apply the full extinction pipeline to one cached row group.

    Loop order: redshift (near->far) -> angle -> time (vectorised batch).
    Dust is sampled once per redshift and shared across all angles at that z.
    STOP-LC truncates the time axis per (z, angle); STOP-A breaks the z-loop
    after `z_patience` consecutive redshifts with no detection at any angle.
    """
    roman_filters = load_roman_filters(kcor_path)
    filter_names = list(roman_filters.keys())
    n_filters = len(filter_names)

    pf = pq.ParquetFile(parquet_path)
    lam_aa = np.frombuffer(
        pf.schema_arrow.metadata[b'wavelength_rest_aa'],
        dtype=np.float32,
    ).astype(np.float64)

    table = pf.read_row_group(row_group_index)
    df = table.to_pandas()

    z_sorted = np.sort(np.asarray(redshift_grid, dtype=float))
    av_arr, rv_arr, ebv_arr = _presample_dust(
        z_sorted, av_pool, rv_pool, ebv_pool, worker_seed
    )

    lum_dist_pc = Planck18.luminosity_distance(z_sorted).to(u.pc).value
    lum_dist_pc = np.where(z_sorted > 0, lum_dist_pc, 10.0)

    mag_limit_arr = np.full(n_filters, float(detection_mag_limit))

    angles = sorted(int(a) for a in df['angle_index'].unique())
    per_angle = {}
    for angle_index in angles:
        grp = (df[df['angle_index'] == angle_index]
               .sort_values('time_index').reset_index(drop=True))
        per_angle[angle_index] = dict(
            t_indices=grp['time_index'].values.astype(int),
            t_days=grp['time_days'].values.astype(float),
            flux_mat=np.maximum(
                np.array(grp['flux_rest'].tolist(), dtype=np.float64), 0.0
            ),
            meta=grp.iloc[0],
        )

    output_rows = []
    z_miss_streak = 0

    for z_idx, redshift in enumerate(z_sorted):
        av_host = float(av_arr[z_idx])
        rv_host = float(rv_arr[z_idx])
        ebv_mw = float(ebv_arr[z_idx])
        av_mw = MILKY_WAY_RV * ebv_mw

        trans_host = _transmission(lam_aa, av_host, rv_host)
        wl_obs = lam_aa * (1.0 + float(redshift))
        trans_mw = _transmission(wl_obs, av_mw, MILKY_WAY_RV)
        dim = (10.0 / float(lum_dist_pc[z_idx])) ** 2

        any_detected_at_z = False

        for angle_index in angles:
            data = per_angle[angle_index]
            flux_obs = (data['flux_mat'] * trans_host
                        * (dim / (1.0 + float(redshift))) * trans_mw)
            mags = _mags_batch(wl_obs, flux_obs, roman_filters)
            detected = np.any(np.isfinite(mags) & (mags <= mag_limit_arr), axis=1)

            stop_idx = _lc_stop_idx_perband(mags, mag_limit_arr, lc_patience)
            if stop_idx == 0:
                continue
            if detected[:stop_idx].any():
                any_detected_at_z = True

            meta = data['meta']
            t_indices = data['t_indices'][:stop_idx]
            t_days = data['t_days'][:stop_idx]
            for j in range(stop_idx):
                row_out = {
                    'simulation_id': int(meta['simulation_id']),
                    'run_type': str(meta['run_type']),
                    'wind': str(meta['wind']),
                    'mass_dynamical': float(meta['mass_dynamical']),
                    'velocity_dynamical': float(meta['velocity_dynamical']),
                    'mass_wind': float(meta['mass_wind']),
                    'velocity_wind': float(meta['velocity_wind']),
                    'time_index': int(t_indices[j]),
                    'time_days': float(t_days[j]),
                    'angle_index': int(angle_index),
                    'redshift': float(redshift),
                    'av_host': av_host,
                    'rv_host': rv_host,
                    'ebv_milky_way': ebv_mw,
                    'av_milky_way': float(av_mw),
                    'luminosity_distance_parsec': float(lum_dist_pc[z_idx]),
                    'detected': bool(detected[j]),
                }
                for f_idx, filter_name in enumerate(filter_names):
                    row_out[f'mag_ab_{filter_name}'] = float(mags[j, f_idx])
                output_rows.append(row_out)

        if any_detected_at_z:
            z_miss_streak = 0
        else:
            z_miss_streak += 1
            if z_miss_streak >= z_patience:
                break

    return output_rows


_process_row_group_remote = ray.remote(process_row_group)


def build_parquet_schema(filter_names):
    fields = [
        ('simulation_id', pa.int64()),
        ('run_type', pa.string()),
        ('wind', pa.string()),
        ('mass_dynamical', pa.float64()),
        ('velocity_dynamical', pa.float64()),
        ('mass_wind', pa.float64()),
        ('velocity_wind', pa.float64()),
        ('time_index', pa.int64()),
        ('time_days', pa.float64()),
        ('angle_index', pa.int64()),
        ('redshift', pa.float64()),
        ('av_host', pa.float64()),
        ('rv_host', pa.float64()),
        ('ebv_milky_way', pa.float64()),
        ('av_milky_way', pa.float64()),
        ('luminosity_distance_parsec', pa.float64()),
        ('detected', pa.bool_()),
    ]
    for filter_name in filter_names:
        fields.append((f'mag_ab_{filter_name}', pa.float64()))
    return pa.schema(fields)


def run_parallel(
    spectra_cache_path,
    output_path,
    redshift_grid,
    av_pool,
    rv_pool,
    ebv_pool,
    kcor_path,
    detection_mag_limit,
    lc_patience,
    z_patience,
    num_workers,
    random_seed,
    row_group_indices,
    max_in_flight_multiplier=2,
):
    roman_filters = load_roman_filters(kcor_path)
    filter_names = list(roman_filters.keys())
    schema = build_parquet_schema(filter_names)

    ray.init(num_cpus=num_workers, ignore_reinit_error=True, log_to_driver=False)
    av_ref = ray.put(np.asarray(av_pool, dtype=float))
    rv_ref = ray.put(np.asarray(rv_pool, dtype=float))
    ebv_ref = ray.put(np.asarray(ebv_pool, dtype=float))
    redshift_ref = ray.put(np.asarray(redshift_grid, dtype=float))

    max_in_flight = max(num_workers * max_in_flight_multiplier, num_workers)
    iterator = iter(row_group_indices)
    pending = []
    future_to_rg = {}

    def submit(rg_index):
        worker_seed = int(random_seed) * 1_000_003 + int(rg_index)
        future = _process_row_group_remote.remote(
            str(spectra_cache_path),
            int(rg_index),
            redshift_ref,
            av_ref,
            rv_ref,
            ebv_ref,
            str(kcor_path),
            float(detection_mag_limit),
            int(lc_patience),
            int(z_patience),
            worker_seed,
        )
        future_to_rg[future] = int(rg_index)
        pending.append(future)

    for _ in range(min(max_in_flight, len(row_group_indices))):
        try:
            submit(next(iterator))
        except StopIteration:
            break

    writer = None
    total_rows = 0
    files_done = 0
    n_total = len(row_group_indices)
    start_time = time.perf_counter()

    try:
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            future = ready[0]
            rg_index = future_to_rg.pop(future)
            rows = ray.get(future)
            files_done += 1

            num_rows_this = 0
            if rows:
                table = pa.Table.from_pylist(rows, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, schema, compression='zstd')
                writer.write_table(table)
                num_rows_this = table.num_rows
                total_rows += num_rows_this

            elapsed = time.perf_counter() - start_time
            rate = files_done / elapsed if elapsed > 0 else 0.0
            eta_h = ((n_total - files_done) / rate / 3600.0) if rate > 0 else float('inf')
            print(
                f'  [{files_done}/{n_total}] rg={rg_index} +{num_rows_this} rows '
                f'-> {total_rows:,} total | {rate*3600:.0f} rg/h | ETA {eta_h:.2f} h',
                flush=True,
            )

            try:
                submit(next(iterator))
            except StopIteration:
                pass
    finally:
        if writer is not None:
            writer.close()
        ray.shutdown()

    return total_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    script_directory = Path(__file__).resolve().parent
    parser.add_argument('--spectra-cache', type=Path,
                        default=script_directory / 'lanl_spectra.parquet',
                        help='Cached rest-frame spectra parquet built by cache_lanl_spectra.py.')
    parser.add_argument('--hourglass-parquet', type=Path,
                        default=script_directory / 'hourglass_objects.parquet',
                        help='Hourglass objects parquet (used to sample Milky Way E(B-V) at real survey coordinates).')
    parser.add_argument('--output', type=Path,
                        default=script_directory / 'lanl_extinguished_photometry.parquet',
                        help='Output parquet path.')
    parser.add_argument('--kcor-path', type=Path, default=DEFAULT_KCOR_PATH,
                        help='SNANA kcor FITS with Roman FilterTrans extension.')
    parser.add_argument('--detection-mag-limit', type=float, default=28.0,
                        help='AB-mag faint limit used for STOP rules (uniform across Roman bands).')

    parser.add_argument('--redshift-min', type=float, default=0.003)
    parser.add_argument('--redshift-max', type=float, default=1.0)
    parser.add_argument('--n-redshift', type=int, default=50)
    parser.add_argument('--redshift-spacing', choices=('linear', 'log'), default='linear')

    parser.add_argument('--lc-patience', type=int, default=3,
                        help='Per-band consecutive non-detected decaying steps before STOP-LC truncates the (z, angle) light curve.')
    parser.add_argument('--z-patience', type=int, default=3,
                        help='Consecutive redshifts with no detection at any (angle, time) before STOP-A breaks the z-loop.')

    parser.add_argument('--num-workers', type=int, default=32,
                        help='Number of Ray workers (one task per row group at a time).')
    parser.add_argument('--max-in-flight-multiplier', type=int, default=2,
                        help='Max in-flight ray tasks = num-workers * this multiplier.')
    parser.add_argument('--max-row-groups', type=int, default=None,
                        help='If set, only process this many row groups (smoke test).')
    parser.add_argument('--n-pool-samples', type=int, default=10000,
                        help='Pool size for Av/Rv/EBV samplers.')
    parser.add_argument('--random-seed', type=int, default=42)

    arguments = parser.parse_args()

    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    if not arguments.spectra_cache.exists():
        raise FileNotFoundError(
            f'Spectra cache not found at {arguments.spectra_cache}. '
            f'Run cache_lanl_spectra.py first.'
        )

    print(f'Spectra cache: {arguments.spectra_cache}', flush=True)
    print(f'Loading Roman filters from {arguments.kcor_path}...', flush=True)
    roman_filters = load_roman_filters(arguments.kcor_path)
    filter_names = list(roman_filters.keys())
    print(f'  {len(filter_names)} filters: {filter_names}', flush=True)

    print('Sampling extinction pools...', flush=True)
    np.random.seed(arguments.random_seed)
    av_pool = sample_extinction_av(arguments.n_pool_samples)
    rv_pool = sample_extinction_rv(arguments.n_pool_samples)
    hourglass = sample_hourglass_ebv(
        parquet_path=arguments.hourglass_parquet,
        number_of_samples=arguments.n_pool_samples,
        random_seed=arguments.random_seed,
    )
    ebv_pool = np.asarray(hourglass['ebv_samples'], dtype=float)
    print(f'  Av median = {np.median(av_pool):.3f}  '
          f'Rv median = {np.median(rv_pool):.3f}  '
          f'EBV_MW median = {np.median(ebv_pool):.4f}', flush=True)

    redshift_grid = build_redshift_grid(
        arguments.redshift_min,
        arguments.redshift_max,
        arguments.n_redshift,
        arguments.redshift_spacing,
    )
    print(f'Redshift grid ({arguments.redshift_spacing}, N={len(redshift_grid)}): '
          f'{redshift_grid[0]:.4f} -> {redshift_grid[-1]:.4f}', flush=True)

    row_group_indices = select_row_group_indices(
        arguments.spectra_cache,
        arguments.max_row_groups,
        arguments.random_seed,
    )
    total_rg = pq.ParquetFile(arguments.spectra_cache).num_row_groups
    print(f'Processing {len(row_group_indices)} row groups (of {total_rg} total)',
          flush=True)

    print(f'STOP rules: STOP-A patience={arguments.z_patience} (z-loop across all angles); '
          f'STOP-LC per-band patience={arguments.lc_patience} in decay; '
          f'mag-limit={arguments.detection_mag_limit}', flush=True)
    print(f'Launching Ray with {arguments.num_workers} workers -> {arguments.output}',
          flush=True)

    total = run_parallel(
        spectra_cache_path=arguments.spectra_cache,
        output_path=arguments.output,
        redshift_grid=redshift_grid,
        av_pool=av_pool,
        rv_pool=rv_pool,
        ebv_pool=ebv_pool,
        kcor_path=arguments.kcor_path,
        detection_mag_limit=arguments.detection_mag_limit,
        lc_patience=arguments.lc_patience,
        z_patience=arguments.z_patience,
        num_workers=arguments.num_workers,
        random_seed=arguments.random_seed,
        row_group_indices=row_group_indices,
        max_in_flight_multiplier=arguments.max_in_flight_multiplier,
    )
    print(f'Done. Wrote {total:,} rows to {arguments.output}', flush=True)


if __name__ == '__main__':
    main()
