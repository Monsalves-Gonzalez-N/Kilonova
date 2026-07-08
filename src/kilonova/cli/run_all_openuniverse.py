"""kn-run-openuniverse: early windows for ALL OpenUniverse fields, one combined parquet per tier.

For every snana_XXXXX.hdf5 (with matching snana_XXXXX.parquet) in the source directory,
extracts the early-window light curves with the same noise recipe and cadence as
kn-early-windows, then concatenates all fields into:

    {output_dir}/early_windows_deep.parquet
    {output_dir}/early_windows_wide.parquet

object_id is prefixed with the snana_id (e.g. "snana_10307_12345678") to stay unique across
fields. A snana_id column is added for traceability. Tiers are processed sequentially to
bound peak memory (one tier in RAM at a time).
"""

import argparse
import glob
import logging
import os
import time
import traceback
from pathlib import Path

import h5py
import pandas as pd

from kilonova.config import load_paths, require
from kilonova.log import setup_logging
from kilonova.simulation import early_windows

logger = logging.getLogger(__name__)


def process_tier(tier, hdf5_paths, limit_ou=None):
    """One pass over all HDF5 files for a single tier -> list of window DataFrames."""
    constants = early_windows.build_tier_constants(tier)
    logger.info(
        "[%s] bands=%s  noise_floor_variance: %s",
        tier,
        constants["bands"],
        ", ".join(f"{band}={variance:.0f}" for band, variance in constants["noise_floor_variance"].items()),
    )

    all_windows = []
    total_detected = 0
    tier_start = time.time()

    for file_index, hdf5_path in enumerate(hdf5_paths, start=1):
        snana_id = os.path.basename(hdf5_path).replace(".hdf5", "")
        catalog_path = hdf5_path.replace(".hdf5", ".parquet")
        if not os.path.exists(catalog_path):
            logger.warning("[%s] no catalog parquet, skipping", snana_id)
            continue

        object_records = early_windows.collect_object_records(catalog_path, limit=limit_ou)
        file_windows = []
        file_start = time.time()

        with h5py.File(hdf5_path, "r") as hdf5:
            for object_id, redshift, gentype in object_records:
                if str(object_id) not in hdf5:
                    continue
                window = early_windows.build_early_window(
                    object_id, hdf5[str(object_id)], constants, redshift, gentype
                )
                if window is None:
                    continue
                window = window.copy()
                window["object_id"] = snana_id + "_" + window["object_id"].astype(str)
                window["snana_id"] = snana_id
                file_windows.append(window)

        n_detected = len(file_windows)
        total_detected += n_detected
        all_windows.extend(file_windows)
        logger.info(
            "[%s] [%d/%d] %s: detected=%d/%d (%.0fs)",
            tier,
            file_index,
            len(hdf5_paths),
            snana_id,
            n_detected,
            len(object_records),
            time.time() - file_start,
        )

    logger.info(
        "[%s] all files done: total_detected=%d elapsed=%.0fs", tier, total_detected, time.time() - tier_start
    )
    return all_windows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source-dir", type=Path, help="directory with the snana_*.hdf5 + snana_*.parquet pairs"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--limit-ou",
        type=int,
        default=None,
        help="process only the first N transients per field (smoke tests)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    arguments = parser.parse_args(argv)

    setup_logging(arguments.verbose)
    paths = load_paths()
    source_dir = require(arguments.source_dir or paths.openuniverse_source, "openuniverse_source")
    output_dir = arguments.output_dir or paths.output_dir
    if output_dir is None:
        raise SystemExit("output_dir is not configured (configs/paths.yaml, KN_OUTPUT_DIR or --output-dir)")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf5_paths = sorted(glob.glob(str(source_dir / "snana_*.hdf5")))
    if not hdf5_paths:
        raise SystemExit(f"No snana_*.hdf5 files found in {source_dir}")

    logger.info("source_dir : %s", source_dir)
    logger.info("output_dir : %s", output_dir)
    logger.info("HDF5 files : %d", len(hdf5_paths))

    total_start = time.time()
    for tier in ("deep", "wide"):
        output_path = output_dir / f"early_windows_{tier}.parquet"
        if output_path.exists():
            logger.info("[%s] output already exists, skipping: %s", tier, output_path)
            continue

        try:
            windows = process_tier(tier, hdf5_paths, limit_ou=arguments.limit_ou)
            if not windows:
                logger.warning("[%s] no windows produced, skipping write", tier)
                continue
            combined = pd.concat(windows, ignore_index=True)
            combined.to_parquet(output_path, index=False)
            logger.info("[%s] -> %s rows=%d", tier, output_path, len(combined))
            del combined, windows
        except Exception:
            logger.error("[%s] ERROR:\n%s", tier, traceback.format_exc())

    logger.info("DONE total=%.0fs", time.time() - total_start)


if __name__ == "__main__":
    main()
