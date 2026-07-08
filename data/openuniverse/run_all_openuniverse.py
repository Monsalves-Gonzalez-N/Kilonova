"""Run generate_early_windows for all HDF5/parquet pairs and produce ONE combined parquet per tier.

For every snana_XXXXX.hdf5 (with matching snana_XXXXX.parquet) in SOURCE_DIR, extracts the
early-window light curves using the same noise recipe and cadence as generate_early_windows.py,
then concatenates all 33 fields into two output files:

    {output_dir}/early_windows_deep.parquet
    {output_dir}/early_windows_wide.parquet

object_id is prefixed with the snana_id (e.g. "snana_10307_12345678") to stay unique across
fields. A snana_id column is added for traceability.

Tiers are processed sequentially to bound peak memory (one tier in RAM at a time).

Usage:
    python run_all_openuniverse.py [source_dir] [output_dir]

Defaults:
    source_dir = /Volumes/T7/openuniverse2025
    output_dir = /Users/bhianca/Kilonova/data/openuniverse
"""

import glob
import os
import sys
import time
import traceback

import h5py
import numpy as np
import pandas as pd

import generate_early_windows as gew

SOURCE_DIR = "/Volumes/T7/openuniverse2025"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def collect_object_records(catalog_path):
    catalog = pd.read_parquet(catalog_path)
    transients = catalog[catalog["gentype"] != 99]
    records = [
        (int(row.id), float(row.z_CMB), int(row.gentype))
        for row in transients.itertuples()
    ]
    if gew.OU_LIMIT is not None:
        records = records[:gew.OU_LIMIT]
    return records


def process_tier(tier, hdf5_paths, source_dir):
    """One pass over all HDF5 files for a single tier -> list of window DataFrames."""
    constants = gew.build_tier_constants(tier)
    print(
        f"\n[{tier}] bands={constants['bands']}  noise_floor_variance: "
        + ", ".join(f"{band}={variance:.0f}" for band, variance in constants["noise_floor_variance"].items()),
        flush=True,
    )

    all_windows = []
    total_detected = 0
    tier_start = time.time()

    for file_index, hdf5_path in enumerate(hdf5_paths, start=1):
        snana_id = os.path.basename(hdf5_path).replace(".hdf5", "")
        catalog_path = hdf5_path.replace(".hdf5", ".parquet")
        if not os.path.exists(catalog_path):
            print(f"  [{snana_id}] no catalog parquet, skipping", flush=True)
            continue

        object_records = collect_object_records(catalog_path)
        file_windows = []
        file_start = time.time()

        with h5py.File(hdf5_path, "r") as hdf5:
            for counter, (object_id, redshift, gentype) in enumerate(object_records):
                if str(object_id) not in hdf5:
                    continue
                window = gew.build_early_window(
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
        print(
            f"  [{tier}] [{file_index}/{len(hdf5_paths)}] {snana_id}: "
            f"detected={n_detected}/{len(object_records)}  "
            f"({time.time() - file_start:.0f}s)",
            flush=True,
        )

    print(
        f"[{tier}] all files done: total_detected={total_detected}  "
        f"elapsed={time.time() - tier_start:.0f}s",
        flush=True,
    )
    return all_windows


def main():
    source_dir = sys.argv[1] if len(sys.argv) > 1 else SOURCE_DIR
    output_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR

    hdf5_paths = sorted(glob.glob(os.path.join(source_dir, "snana_*.hdf5")))
    if not hdf5_paths:
        print(f"No snana_*.hdf5 files found in {source_dir}", flush=True)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"source_dir : {source_dir}", flush=True)
    print(f"output_dir : {output_dir}", flush=True)
    print(f"HDF5 files : {len(hdf5_paths)}", flush=True)

    total_start = time.time()

    for tier in ("deep", "wide"):
        output_path = os.path.join(output_dir, f"early_windows_{tier}.parquet")
        if os.path.exists(output_path):
            print(f"\n[{tier}] output already exists, skipping: {output_path}", flush=True)
            continue

        try:
            windows = process_tier(tier, hdf5_paths, source_dir)
            if not windows:
                print(f"[{tier}] WARNING: no windows produced, skipping write", flush=True)
                continue
            combined = pd.concat(windows, ignore_index=True)
            combined.to_parquet(output_path, index=False)
            print(f"[{tier}] -> {output_path}  rows={len(combined)}", flush=True)
            del combined, windows
        except Exception:
            print(f"[{tier}] ERROR:", flush=True)
            traceback.print_exc()

    print(f"\nDONE  total={time.time() - total_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
