"""Dump the LANL kilonova .dat grid into a single parquet of rest-frame spectra.

One row per (simulation_id, time_index, angle_index). Flux is stored as a
FixedSizeList<float32, 1024>; the shared wavelength grid is written once to the
parquet schema metadata as raw float32 bytes (key ``wavelength_rest_aa``).
Parallelized with Ray: one remote task per .dat file.

Read back the wavelength grid with::

    table = pyarrow.parquet.read_table('lanl_spectra.parquet')
    lam_aa = numpy.frombuffer(table.schema.metadata[b'wavelength_rest_aa'], dtype=numpy.float32)
    flux = table.column('flux_rest').to_numpy_ndarray().reshape(-1, lam_aa.size)
"""

import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import ray

from kilonova.simulation.extinction import (
    CM_TO_ANG,
    _read_header_times,
    load_lanl_catalog,
)

logger = logging.getLogger(__name__)

N_WAVELENGTHS = 1024  # invariant across kn_sim_cube_v1/*_spec_*.dat

SCHEMA_FIELDS = [
    ("simulation_id", pa.int64()),
    ("run_type", pa.dictionary(pa.int8(), pa.string())),
    ("wind", pa.dictionary(pa.int8(), pa.string())),
    ("mass_dynamical", pa.float32()),
    ("velocity_dynamical", pa.float32()),
    ("mass_wind", pa.float32()),
    ("velocity_wind", pa.float32()),
    ("time_index", pa.int32()),
    ("time_days", pa.float32()),
    ("angle_index", pa.int32()),
    ("flux_rest", pa.list_(pa.float32(), N_WAVELENGTHS)),
]


def build_schema(wavelength_grid_aa):
    schema = pa.schema(SCHEMA_FIELDS)
    return schema.with_metadata(
        {
            b"wavelength_rest_aa": wavelength_grid_aa.astype(np.float32).tobytes(),
            b"wavelength_rest_n": str(wavelength_grid_aa.size).encode(),
            b"wavelength_rest_dtype": b"float32",
            b"wavelength_unit": b"angstrom",
            b"flux_unit": b"erg s-1 cm-2 angstrom-1",
        }
    )


def parse_spec_fast(filepath, n_times):
    """Fast numpy text parser. Returns (lam_aa[N_WAVELENGTHS], flux[n_times, n_wave, n_angles])."""
    data = np.loadtxt(filepath, comments="#", dtype=np.float32)
    n_wavelengths = data.shape[0] // n_times
    data = data.reshape(n_times, n_wavelengths, -1)
    lam_aa = (0.5 * (data[0, :, 0] + data[0, :, 1]) * CM_TO_ANG).astype(np.float32)
    flux = data[:, :, 2:]  # (n_times, n_wavelengths, n_angles)
    return lam_aa, flux


def _dict_string_column(value, n_rows):
    """Single-value dictionary array matching pa.dictionary(int8, string)."""
    indices = pa.array(np.zeros(n_rows, dtype=np.int8), type=pa.int8())
    dictionary = pa.array([str(value)], type=pa.string())
    return pa.DictionaryArray.from_arrays(indices, dictionary)


@ray.remote
def parse_file_to_table(filepath, file_metadata, time_days_lookup, schema):
    times = _read_header_times(filepath)
    n_times = len(times)
    lam_aa, flux = parse_spec_fast(filepath, n_times)
    n_wavelengths, n_angles = flux.shape[1], flux.shape[2]

    if n_wavelengths != N_WAVELENGTHS:
        raise RuntimeError(f"{filepath}: expected {N_WAVELENGTHS} wavelength bins, got {n_wavelengths}")

    # (n_times, n_wavelengths, n_angles) -> (n_times, n_angles, n_wavelengths) -> flat (rows, n_wavelengths)
    flux_per_cell = np.ascontiguousarray(
        np.transpose(flux, (0, 2, 1)).reshape(n_times * n_angles, n_wavelengths),
        dtype=np.float32,
    )
    n_rows = flux_per_cell.shape[0]

    time_indices = np.repeat(np.arange(n_times, dtype=np.int32), n_angles)
    angle_indices = np.tile(np.arange(n_angles, dtype=np.int32), n_times)

    time_days_array = np.array(
        [time_days_lookup.get(int(time_index), float(times[time_index])) for time_index in time_indices],
        dtype=np.float32,
    )

    flux_values = pa.array(flux_per_cell.reshape(-1), type=pa.float32())
    flux_fixed_list = pa.FixedSizeListArray.from_arrays(flux_values, n_wavelengths)

    columns = {
        "simulation_id": pa.array(np.full(n_rows, int(file_metadata["simulation_id"]), dtype=np.int64)),
        "run_type": _dict_string_column(file_metadata["run_type"], n_rows),
        "wind": _dict_string_column(file_metadata["wind"], n_rows),
        "mass_dynamical": pa.array(np.full(n_rows, float(file_metadata["mass_dynamical"]), dtype=np.float32)),
        "velocity_dynamical": pa.array(
            np.full(n_rows, float(file_metadata["velocity_dynamical"]), dtype=np.float32)
        ),
        "mass_wind": pa.array(np.full(n_rows, float(file_metadata["mass_wind"]), dtype=np.float32)),
        "velocity_wind": pa.array(np.full(n_rows, float(file_metadata["velocity_wind"]), dtype=np.float32)),
        "time_index": pa.array(time_indices),
        "time_days": pa.array(time_days_array),
        "angle_index": pa.array(angle_indices),
        "flux_rest": flux_fixed_list,
    }
    return pa.Table.from_pydict(columns, schema=schema), lam_aa


def write_lanl_spectra_parquet(lanl_dir, catalog_path, output_path, num_cpus=None, max_in_flight=8):
    """Parse the whole .dat grid into one zstd parquet at `output_path` (Ray-parallel)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading LANL catalog...")
    catalog = load_lanl_catalog(catalog_path, lanl_dir)
    metadata_columns = [
        "simulation_id",
        "run_type",
        "wind",
        "mass_dynamical",
        "velocity_dynamical",
        "mass_wind",
        "velocity_wind",
    ]
    file_groups = catalog.groupby("filepath")
    file_metadata_lookup = file_groups[metadata_columns].first().to_dict("index")
    time_days_lookup_per_file = {
        filepath: dict(zip(group["time_index"].astype(int), group["time_days"].astype(float), strict=True))
        for filepath, group in file_groups
    }
    filepaths = list(file_metadata_lookup.keys())
    logger.info("  catalog rows: %s  files: %d", f"{len(catalog):,}", len(filepaths))

    # Bootstrap: parse one file to capture the wavelength grid for the schema metadata.
    bootstrap_path = filepaths[0]
    bootstrap_times = _read_header_times(bootstrap_path)
    bootstrap_lam, _ = parse_spec_fast(bootstrap_path, len(bootstrap_times))
    if bootstrap_lam.size != N_WAVELENGTHS:
        raise RuntimeError(f"expected {N_WAVELENGTHS} wavelength bins, got {bootstrap_lam.size}")
    schema = build_schema(bootstrap_lam)

    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
    schema_ref = ray.put(schema)

    pending_futures = []
    filepath_iterator = iter(filepaths)
    for _ in range(min(max_in_flight, len(filepaths))):
        filepath = next(filepath_iterator)
        pending_futures.append(
            parse_file_to_table.remote(
                filepath,
                file_metadata_lookup[filepath],
                time_days_lookup_per_file[filepath],
                schema_ref,
            )
        )

    writer = None
    files_done = 0
    total_rows = 0
    try:
        while pending_futures:
            ready, pending_futures = ray.wait(pending_futures, num_returns=1)
            table, lam_aa = ray.get(ready[0])
            if not np.array_equal(lam_aa, bootstrap_lam):
                raise RuntimeError(
                    "wavelength grid mismatch between files; cannot share grid in schema metadata"
                )
            if writer is None:
                writer = pq.ParquetWriter(output_path, schema, compression="zstd")
            writer.write_table(table)
            files_done += 1
            total_rows += table.num_rows
            logger.info(
                "  [%d/%d] +%d rows -> %d total", files_done, len(filepaths), table.num_rows, total_rows
            )

            next_filepath = next(filepath_iterator, None)
            if next_filepath is not None:
                pending_futures.append(
                    parse_file_to_table.remote(
                        next_filepath,
                        file_metadata_lookup[next_filepath],
                        time_days_lookup_per_file[next_filepath],
                        schema_ref,
                    )
                )
    finally:
        if writer is not None:
            writer.close()
        ray.shutdown()

    logger.info("Done. Wrote %s rows to %s", f"{total_rows:,}", output_path)
