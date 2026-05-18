"""Generate a parquet of LANL kilonova spectra attenuated by host + MW extinction.

All physics functions are taken verbatim from Extinction.ipynb (already validated).
For each (LANL spectrum) x (redshift in the grid) we draw one realization of
(Av_host, Rv_host, EBV_MW) and apply the full observational pipeline:

    rest spectrum --> F99 host extinction (Av_host, Rv_host)
                  --> redshift (specutils)
                  --> cosmological distance dimming (Planck18 luminosity distance)
                  --> F99 Milky Way extinction (Av_MW = Rv_MW * EBV_MW)

Output: one parquet row per (spectrum_id, redshift) with wavelength and flux as
list<float64> columns (observer frame).
"""

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyphot
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
DEFAULT_KCOR_PATH = Path(__file__).resolve().parent.parent / 'kcor' / 'kcor_ROMAN.fits'

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


def select_spectrum_ids(catalog, max_spectra, random_seed):
    if max_spectra is None or max_spectra >= len(catalog):
        return catalog.index.to_numpy()
    rng = np.random.default_rng(random_seed)
    return np.sort(rng.choice(catalog.index.to_numpy(), size=max_spectra, replace=False))


def _is_detected_in_any_band(ab_magnitudes, detection_mag_limit):
    for magnitude in ab_magnitudes.values():
        if np.isfinite(magnitude) and magnitude <= detection_mag_limit:
            return True
    return False


def iterate_attenuated_rows(
    catalog,
    spectrum_ids,
    redshift_grid,
    av_pool,
    rv_pool,
    ebv_pool,
    roman_filters,
    random_seed,
    detection_mag_limit,
):
    rng = np.random.default_rng(random_seed)
    selected_ids = set(int(value) for value in spectrum_ids)
    catalog_subset = catalog.loc[catalog.index.isin(selected_ids)]
    filter_names = list(roman_filters.keys())
    redshift_grid_sorted = np.sort(np.asarray(redshift_grid, dtype=float))

    for filepath, file_catalog in catalog_subset.groupby('filepath'):
        times_file, lam_AA, flux_cube = parse_spec(filepath)

        for angle_index, angle_catalog in file_catalog.groupby('angle_index'):
            angle_catalog_sorted = angle_catalog.sort_values('time_index')
            stop_light_curve = False

            for spectrum_id, row in angle_catalog_sorted.iterrows():
                if stop_light_curve:
                    break

                time_index = int(row['time_index'])
                spectrum_slice = flux_cube[time_index, :, int(angle_index)]
                valid_mask = spectrum_slice > 0
                if not valid_mask.any():
                    continue
                wavelength_rest = lam_AA[valid_mask]
                flux_rest = spectrum_slice[valid_mask]

                detected_at_smallest_redshift = False

                for redshift in redshift_grid_sorted:
                    av_host = float(rng.choice(av_pool))
                    rv_host = float(rng.choice(rv_pool))
                    ebv_mw = float(rng.choice(ebv_pool))

                    observed = generate_observed_kilonova_spectrum(
                        wavelength_rest=wavelength_rest,
                        spectrum_input=flux_rest,
                        extinction_av_host=av_host,
                        extinction_rv_host=rv_host,
                        redshift=float(redshift),
                        ebv_milky_way=ebv_mw,
                    )
                    parameters = observed['parameters']

                    ab_magnitudes = compute_roman_ab_magnitudes(
                        observed['wavelength_observed'],
                        observed['flux_observed'],
                        roman_filters,
                    )

                    detected = _is_detected_in_any_band(ab_magnitudes, detection_mag_limit)

                    output_row = {
                        'spectrum_id': int(spectrum_id),
                        'simulation_id': int(row['simulation_id']),
                        'run_type': str(row['run_type']),
                        'wind': str(row['wind']),
                        'mass_dynamical': float(row['mass_dynamical']),
                        'velocity_dynamical': float(row['velocity_dynamical']),
                        'mass_wind': float(row['mass_wind']),
                        'velocity_wind': float(row['velocity_wind']),
                        'time_index': time_index,
                        'time_days': float(row['time_days']),
                        'angle_index': int(angle_index),
                        'redshift': float(redshift),
                        'av_host': av_host,
                        'rv_host': rv_host,
                        'ebv_milky_way': ebv_mw,
                        'av_milky_way': parameters['extinction_av_milky_way'],
                        'luminosity_distance_parsec': parameters['luminosity_distance_parsec'],
                        'detected': bool(detected),
                    }
                    for filter_name in filter_names:
                        output_row[f'mag_ab_{filter_name}'] = float(ab_magnitudes[filter_name])
                    yield output_row

                    if redshift == redshift_grid_sorted[0]:
                        detected_at_smallest_redshift = detected

                    if not detected:
                        break

                if not detected_at_smallest_redshift:
                    stop_light_curve = True


def build_parquet_schema(filter_names):
    fields = [
        ('spectrum_id', pa.int64()),
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


def write_dataset(row_iterator, output_path, batch_size, schema):
    writer = None
    batch = []
    total_rows = 0
    try:
        for row in row_iterator:
            batch.append(row)
            if len(batch) >= batch_size:
                table = pa.Table.from_pylist(batch, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, schema, compression='zstd')
                writer.write_table(table)
                total_rows += len(batch)
                print(f'  wrote {total_rows} rows', flush=True)
                batch = []
        if batch:
            table = pa.Table.from_pylist(batch, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(output_path, schema, compression='zstd')
            writer.write_table(table)
            total_rows += len(batch)
            print(f'  wrote {total_rows} rows', flush=True)
    finally:
        if writer is not None:
            writer.close()
    return total_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    script_directory = Path(__file__).resolve().parent
    parser.add_argument('--lanl-dir', type=Path,
                        default=script_directory / 'kn_sim_cube_v1',
                        help='Directory containing LANL *_spec_*.dat files (relative to the script by default).')
    parser.add_argument('--catalog-path', type=Path,
                        default=script_directory / 'lanl_catalog.parquet',
                        help='Path to the LANL catalog parquet. Built from --lanl-dir if missing.')
    parser.add_argument('--hourglass-parquet', type=Path,
                        default=script_directory / 'hourglass_objects.parquet',
                        help='Hourglass objects parquet (used to sample Milky Way E(B-V) at real survey coordinates).')
    parser.add_argument('--output', type=Path,
                        default=script_directory / 'lanl_extinguished_photometry.parquet',
                        help='Output parquet path.')
    parser.add_argument('--kcor-path', type=Path, default=DEFAULT_KCOR_PATH,
                        help='SNANA kcor FITS with Roman FilterTrans extension.')
    parser.add_argument('--detection-mag-limit', type=float, default=27.0,
                        help='AB-mag faint limit; STOP triggers when no band reaches this depth.')

    parser.add_argument('--redshift-min', type=float, default=0.01)
    parser.add_argument('--redshift-max', type=float, default=0.5)
    parser.add_argument('--n-redshift', type=int, default=20)
    parser.add_argument('--redshift-spacing', choices=('linear', 'log'), default='linear')

    parser.add_argument('--max-spectra', type=int, default=None,
                        help='If set, randomly subsample this many LANL spectra (otherwise use all).')
    parser.add_argument('--n-pool-samples', type=int, default=10000,
                        help='Pool size for Av/Rv/EBV samplers.')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Rows per parquet write batch.')
    parser.add_argument('--random-seed', type=int, default=42)

    arguments = parser.parse_args()

    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    print('Building / loading LANL catalog...', flush=True)
    if not arguments.catalog_path.exists() and not arguments.lanl_dir.exists():
        raise FileNotFoundError(
            f'Neither catalog ({arguments.catalog_path}) nor LANL grid directory '
            f'({arguments.lanl_dir}) exists. Mount the external drive or provide '
            f'a prebuilt catalog parquet.'
        )
    catalog = load_lanl_catalog(arguments.catalog_path, arguments.lanl_dir)
    print(f'  catalog rows: {len(catalog):,}', flush=True)

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
    ebv_pool = hourglass['ebv_samples']
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
          f'{redshift_grid[0]:.4f} - {redshift_grid[-1]:.4f}', flush=True)

    spectrum_ids = select_spectrum_ids(catalog, arguments.max_spectra, arguments.random_seed)
    expected_rows = len(spectrum_ids) * len(redshift_grid)
    print(f'Generating {len(spectrum_ids):,} spectra x {len(redshift_grid)} redshifts '
          f'= {expected_rows:,} rows -> {arguments.output}', flush=True)

    row_iterator = iterate_attenuated_rows(
        catalog=catalog,
        spectrum_ids=spectrum_ids,
        redshift_grid=redshift_grid,
        av_pool=av_pool,
        rv_pool=rv_pool,
        ebv_pool=ebv_pool,
        roman_filters=roman_filters,
        random_seed=arguments.random_seed,
        detection_mag_limit=arguments.detection_mag_limit,
    )

    schema = build_parquet_schema(filter_names)
    total = write_dataset(row_iterator, arguments.output, arguments.batch_size, schema)
    print(f'Done. Wrote {total:,} rows to {arguments.output}', flush=True)


if __name__ == '__main__':
    main()
