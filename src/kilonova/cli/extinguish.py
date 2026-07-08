"""kn-extinguish: Roman AB photometry of the LANL kilonova grid with host + Milky Way extinction.

Reads the cached rest-frame spectra parquet (kn-cache-lanl), applies host extinction
(Av/Rv pools) and Milky Way extinction (E(B-V) sampled at real Hourglass survey
coordinates), integrates through the Roman filters of the kcor FITS, and writes one
photometry row per (simulation, redshift, angle, time, band). STOP rules truncate
light curves that fall below the detection limit. Ray-parallel, one task per row group.
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from kilonova.config import load_paths, require
from kilonova.log import setup_logging
from kilonova.simulation.extinction import (
    build_redshift_grid,
    load_roman_filters,
    run_parallel,
    sample_extinction_av,
    sample_extinction_rv,
    sample_hourglass_ebv,
    select_row_group_indices,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--spectra-cache", type=Path, help="Cached rest-frame spectra parquet built by kn-cache-lanl."
    )
    parser.add_argument(
        "--hourglass-parquet",
        type=Path,
        help="Hourglass objects parquet (samples Milky Way E(B-V) at real coordinates).",
    )
    parser.add_argument("--output", type=Path, help="Output parquet path.")
    parser.add_argument("--kcor-path", type=Path, help="SNANA kcor FITS with Roman FilterTrans extension.")
    parser.add_argument(
        "--detection-mag-limit",
        type=float,
        default=28.0,
        help="AB-mag faint limit used for STOP rules (uniform across Roman bands).",
    )

    parser.add_argument("--redshift-min", type=float, default=0.003)
    parser.add_argument("--redshift-max", type=float, default=1.0)
    parser.add_argument("--n-redshift", type=int, default=50)
    parser.add_argument("--redshift-spacing", choices=("linear", "log"), default="linear")

    parser.add_argument(
        "--lc-patience",
        type=int,
        default=3,
        help="Per-band consecutive non-detected decaying steps before STOP-LC "
        "truncates the (z, angle) light curve.",
    )
    parser.add_argument(
        "--z-patience",
        type=int,
        default=3,
        help="Consecutive redshifts with no detection at any (angle, time) before STOP-A breaks the z-loop.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=32,
        help="Number of Ray workers (one task per row group at a time).",
    )
    parser.add_argument(
        "--max-in-flight-multiplier",
        type=int,
        default=2,
        help="Max in-flight ray tasks = num-workers * this multiplier.",
    )
    parser.add_argument(
        "--max-row-groups",
        type=int,
        default=None,
        help="If set, only process this many row groups (smoke test).",
    )
    parser.add_argument("--n-pool-samples", type=int, default=10000, help="Pool size for Av/Rv/EBV samplers.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")

    arguments = parser.parse_args(argv)
    setup_logging(arguments.verbose)
    paths = load_paths()

    spectra_cache = require(arguments.spectra_cache or paths.lanl_spectra, "lanl_spectra")
    hourglass_parquet = require(arguments.hourglass_parquet or paths.hourglass_objects, "hourglass_objects")
    kcor_path = require(arguments.kcor_path or paths.kcor, "kcor")
    output_path = arguments.output
    if output_path is None:
        if paths.output_dir is None:
            raise SystemExit("output is not configured (--output or output_dir in configs/paths.yaml)")
        output_path = Path(paths.output_dir) / "lanl_extinguished_photometry.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Spectra cache: {spectra_cache}", flush=True)
    print(f"Loading Roman filters from {kcor_path}...", flush=True)
    roman_filters = load_roman_filters(kcor_path)
    filter_names = list(roman_filters.keys())
    print(f"  {len(filter_names)} filters: {filter_names}", flush=True)

    print("Sampling extinction pools...", flush=True)
    np.random.seed(arguments.random_seed)
    av_pool = sample_extinction_av(arguments.n_pool_samples)
    rv_pool = sample_extinction_rv(arguments.n_pool_samples)
    hourglass = sample_hourglass_ebv(
        parquet_path=hourglass_parquet,
        number_of_samples=arguments.n_pool_samples,
        random_seed=arguments.random_seed,
    )
    ebv_pool = np.asarray(hourglass["ebv_samples"], dtype=float)
    print(
        f"  Av median = {np.median(av_pool):.3f}  "
        f"Rv median = {np.median(rv_pool):.3f}  "
        f"EBV_MW median = {np.median(ebv_pool):.4f}",
        flush=True,
    )

    redshift_grid = build_redshift_grid(
        arguments.redshift_min,
        arguments.redshift_max,
        arguments.n_redshift,
        arguments.redshift_spacing,
    )
    print(
        f"Redshift grid ({arguments.redshift_spacing}, N={len(redshift_grid)}): "
        f"{redshift_grid[0]:.4f} -> {redshift_grid[-1]:.4f}",
        flush=True,
    )

    row_group_indices = select_row_group_indices(
        spectra_cache,
        arguments.max_row_groups,
        arguments.random_seed,
    )
    total_rg = pq.ParquetFile(spectra_cache).num_row_groups
    print(f"Processing {len(row_group_indices)} row groups (of {total_rg} total)", flush=True)

    print(
        f"STOP rules: STOP-A patience={arguments.z_patience} (z-loop across all angles); "
        f"STOP-LC per-band patience={arguments.lc_patience} in decay; "
        f"mag-limit={arguments.detection_mag_limit}",
        flush=True,
    )
    print(f"Launching Ray with {arguments.num_workers} workers -> {output_path}", flush=True)

    total = run_parallel(
        spectra_cache_path=spectra_cache,
        output_path=output_path,
        redshift_grid=redshift_grid,
        av_pool=av_pool,
        rv_pool=rv_pool,
        ebv_pool=ebv_pool,
        kcor_path=kcor_path,
        detection_mag_limit=arguments.detection_mag_limit,
        lc_patience=arguments.lc_patience,
        z_patience=arguments.z_patience,
        num_workers=arguments.num_workers,
        random_seed=arguments.random_seed,
        row_group_indices=row_group_indices,
        max_in_flight_multiplier=arguments.max_in_flight_multiplier,
    )
    print(f"Done. Wrote {total:,} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
