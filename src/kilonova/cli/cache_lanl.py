"""kn-cache-lanl: dump the LANL kilonova .dat grid into one parquet of rest-frame spectra.

One row per (simulation_id, time_index, angle_index); the shared wavelength grid goes into
the parquet schema metadata (key ``wavelength_rest_aa``). Ray-parallel, one task per file.
"""

import argparse
from pathlib import Path

from kilonova.config import load_paths, require
from kilonova.log import setup_logging
from kilonova.simulation.lanl_cache import write_lanl_spectra_parquet


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lanl-dir", type=Path, help="raw kn_sim_cube_v1 grid of .dat spectra")
    parser.add_argument("--catalog-path", type=Path, help="LANL grid index parquet (built if missing)")
    parser.add_argument("--output", type=Path, help="output lanl_spectra.parquet")
    parser.add_argument("--num-cpus", type=int, default=None, help="Ray workers (default: all cores)")
    parser.add_argument(
        "--max-in-flight", type=int, default=8, help="max concurrent Ray tasks (bounds memory)"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    arguments = parser.parse_args(argv)

    setup_logging(arguments.verbose)
    paths = load_paths()
    lanl_dir = require(arguments.lanl_dir or paths.lanl_grid_dir, "lanl_grid_dir")
    catalog_path = arguments.catalog_path or paths.lanl_catalog
    output_path = arguments.output or paths.lanl_spectra
    if output_path is None:
        raise SystemExit("lanl_spectra is not configured (configs/paths.yaml, KN_LANL_SPECTRA or --output)")

    write_lanl_spectra_parquet(
        lanl_dir,
        catalog_path,
        output_path,
        num_cpus=arguments.num_cpus,
        max_in_flight=arguments.max_in_flight,
    )


if __name__ == "__main__":
    main()
