"""Dump the LANL kilonova .dat grid into a single parquet of rest-frame spectra.

One row per (simulation_id, time_index, angle_index) with the positive-flux
subset of the wavelength grid and the corresponding flux_lambda (float32).
Parallelized with Ray: one remote task per .dat file.
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import ray

from generate_extinguished_spectra import (
    load_lanl_catalog,
    parse_spec,
)


SPECTRA_SCHEMA = pa.schema([
    ('simulation_id', pa.int64()),
    ('run_type', pa.string()),
    ('wind', pa.string()),
    ('mass_dynamical', pa.float32()),
    ('velocity_dynamical', pa.float32()),
    ('mass_wind', pa.float32()),
    ('velocity_wind', pa.float32()),
    ('time_index', pa.int32()),
    ('time_days', pa.float32()),
    ('angle_index', pa.int32()),
    ('wavelength_rest', pa.list_(pa.float32())),
    ('flux_rest', pa.list_(pa.float32())),
])


@ray.remote
def parse_file_to_table(filepath, file_metadata, time_days_lookup):
    times_file, lam_AA, flux_cube = parse_spec(filepath)
    n_times, n_wavelengths, n_angles = flux_cube.shape

    flux_per_cell = np.transpose(flux_cube, (0, 2, 1)).reshape(n_times * n_angles, n_wavelengths)
    valid_mask = flux_per_cell > 0
    cell_has_data = valid_mask.any(axis=1)

    time_grid, angle_grid = np.meshgrid(np.arange(n_times), np.arange(n_angles), indexing='ij')
    time_indices_flat = time_grid.reshape(-1)[cell_has_data]
    angle_indices_flat = angle_grid.reshape(-1)[cell_has_data]
    valid_mask = valid_mask[cell_has_data]
    flux_per_cell = flux_per_cell[cell_has_data].astype(np.float32)
    lam_AA = lam_AA.astype(np.float32)

    wavelength_lists = [lam_AA[mask].tolist() for mask in valid_mask]
    flux_lists = [flux_per_cell[row][valid_mask[row]].tolist() for row in range(len(valid_mask))]

    time_days_array = np.array(
        [time_days_lookup.get(int(time_index), float(times_file[time_index])) for time_index in time_indices_flat],
        dtype=np.float32,
    )
    n_rows = len(time_indices_flat)

    return pa.Table.from_pydict({
        'simulation_id': np.full(n_rows, int(file_metadata['simulation_id']), dtype=np.int64),
        'run_type': [str(file_metadata['run_type'])] * n_rows,
        'wind': [str(file_metadata['wind'])] * n_rows,
        'mass_dynamical': np.full(n_rows, float(file_metadata['mass_dynamical']), dtype=np.float32),
        'velocity_dynamical': np.full(n_rows, float(file_metadata['velocity_dynamical']), dtype=np.float32),
        'mass_wind': np.full(n_rows, float(file_metadata['mass_wind']), dtype=np.float32),
        'velocity_wind': np.full(n_rows, float(file_metadata['velocity_wind']), dtype=np.float32),
        'time_index': time_indices_flat.astype(np.int32),
        'time_days': time_days_array,
        'angle_index': angle_indices_flat.astype(np.int32),
        'wavelength_rest': wavelength_lists,
        'flux_rest': flux_lists,
    }, schema=SPECTRA_SCHEMA)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    script_directory = Path(__file__).resolve().parent
    parser.add_argument('--lanl-dir', type=Path,
                        default=script_directory / 'kn_sim_cube_v1')
    parser.add_argument('--catalog-path', type=Path,
                        default=script_directory / 'lanl_catalog.parquet')
    parser.add_argument('--output', type=Path,
                        default=script_directory / 'lanl_spectra.parquet')
    parser.add_argument('--num-cpus', type=int, default=None,
                        help='Ray workers (default: all cores).')
    parser.add_argument('--max-in-flight', type=int, default=8,
                        help='Max concurrent ray tasks (bounds memory).')
    arguments = parser.parse_args()

    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    print('Loading LANL catalog...', flush=True)
    catalog = load_lanl_catalog(arguments.catalog_path, arguments.lanl_dir)
    metadata_columns = ['simulation_id', 'run_type', 'wind', 'mass_dynamical', 'velocity_dynamical', 'mass_wind', 'velocity_wind']
    file_groups = catalog.groupby('filepath')
    file_metadata_lookup = file_groups[metadata_columns].first().to_dict('index')
    time_days_lookup_per_file = {
        filepath: dict(zip(group['time_index'].astype(int), group['time_days'].astype(float)))
        for filepath, group in file_groups
    }
    filepaths = list(file_metadata_lookup.keys())
    print(f'  catalog rows: {len(catalog):,}  files: {len(filepaths)}', flush=True)

    ray.init(num_cpus=arguments.num_cpus, ignore_reinit_error=True)

    pending_futures = []
    filepath_iterator = iter(filepaths)
    for _ in range(min(arguments.max_in_flight, len(filepaths))):
        filepath = next(filepath_iterator)
        pending_futures.append(parse_file_to_table.remote(filepath, file_metadata_lookup[filepath], time_days_lookup_per_file[filepath]))

    writer = None
    files_done = 0
    total_rows = 0
    try:
        while pending_futures:
            ready, pending_futures = ray.wait(pending_futures, num_returns=1)
            table = ray.get(ready[0])
            if writer is None:
                writer = pq.ParquetWriter(arguments.output, SPECTRA_SCHEMA, compression='zstd')
            writer.write_table(table)
            files_done += 1
            total_rows += table.num_rows
            print(f'  [{files_done}/{len(filepaths)}] +{table.num_rows} rows -> {total_rows} total', flush=True)

            next_filepath = next(filepath_iterator, None)
            if next_filepath is not None:
                pending_futures.append(parse_file_to_table.remote(next_filepath, file_metadata_lookup[next_filepath], time_days_lookup_per_file[next_filepath]))
    finally:
        if writer is not None:
            writer.close()
        ray.shutdown()

    print(f'Done. Wrote {total_rows:,} rows to {arguments.output}', flush=True)


if __name__ == '__main__':
    main()
