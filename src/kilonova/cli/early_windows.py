"""kn-early-windows: early-window light curves (contaminants) for ONE OpenUniverse field.

Reads the field's SED hdf5 + object catalog, applies the Roman HLTDS noise recipe and
cadence, and writes one early_windows_{tier}.parquet per tier into the output directory.
For the full 33-field run use kn-run-openuniverse.
"""

import argparse
from pathlib import Path

from kilonova.config import load_paths, require
from kilonova.log import setup_logging
from kilonova.simulation import early_windows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hdf5", type=Path, help="OpenUniverse SED hdf5 of the field")
    parser.add_argument("--catalog", type=Path, help="object catalog parquet of the field")
    parser.add_argument("--output-dir", type=Path, help="where early_windows_{tier}.parquet is written")
    parser.add_argument("--tier", choices=["deep", "wide", "both"], default="both")
    parser.add_argument(
        "--limit-ou", type=int, default=None, help="process only the first N transients (smoke tests)"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    arguments = parser.parse_args(argv)

    setup_logging(arguments.verbose)
    paths = load_paths()
    hdf5_path = require(arguments.hdf5 or paths.openuniverse_hdf5, "openuniverse_hdf5")
    catalog_path = require(arguments.catalog or paths.openuniverse_catalog, "openuniverse_catalog")
    output_dir = arguments.output_dir or paths.output_dir
    if output_dir is None:
        raise SystemExit("output_dir is not configured (configs/paths.yaml, KN_OUTPUT_DIR or --output-dir)")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    object_records = early_windows.collect_object_records(catalog_path, limit=arguments.limit_ou)
    tiers = ["deep", "wide"] if arguments.tier == "both" else [arguments.tier]

    summary = {}
    for tier in tiers:
        output_path = output_dir / f"early_windows_{tier}.parquet"
        summary[tier] = early_windows.run_tier(tier, object_records, hdf5_path, output_path)

    for tier, n_ou_detected in summary.items():
        print(f"{tier}: transients with >=1 detection = {n_ou_detected} / {len(object_records)}")


if __name__ == "__main__":
    main()
