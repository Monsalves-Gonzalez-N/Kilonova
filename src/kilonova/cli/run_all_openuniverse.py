"""kn-run-openuniverse: early windows for ALL OpenUniverse fields, one combined parquet per tier.

For every snana_XXXXX.hdf5 (with matching snana_XXXXX.parquet) in the source directory,
extracts the early-window light curves with the same noise recipe and cadence as
kn-early-windows, then concatenates all fields into:

    {output_dir}/early_windows_deep.parquet
    {output_dir}/early_windows_wide.parquet

object_id is prefixed with the snana_id (e.g. "snana_10307_12345678") to stay unique across
fields. A snana_id column is added for traceability. Tiers are processed sequentially.

EVERY FIELD IS WRITTEN AS ITS OWN SHARD and the tier is stitched from the shards, which is what
makes `--workers` possible and bounds peak memory at the same time. Holding a whole tier in RAM
took 13 GB and, worse, lost everything if the run died: there was no partial output at all.
A shard that already exists is skipped, so an interrupted run resumes where it stopped.

PARALLELISING IS SAFE because the noise is not drawn from a shared stream: `build_early_window`
seeds it with `int(object_id)`, so a field gives the same rows whatever process renders it and in
whatever order. Fields are independent -- one HDF5 each -- so the split is over files.
"""

import argparse
import glob
import logging
import os
import shutil
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import h5py
import pandas as pd
import pyarrow.parquet as pq

from kilonova.config import load_paths, require
from kilonova.log import setup_logging
from kilonova.simulation import early_windows

logger = logging.getLogger(__name__)


_TIER_CONSTANTS = {}


def _constants(tier):
    """`build_tier_constants` per process: it is the same for every field and not free to build."""
    if tier not in _TIER_CONSTANTS:
        _TIER_CONSTANTS[tier] = early_windows.build_tier_constants(tier)
    return _TIER_CONSTANTS[tier]


def shard_path(shard_dir, tier, snana_id):
    return Path(shard_dir) / f"{tier}__{snana_id}.parquet"


def process_field(payload):
    """One HDF5 -> one shard parquet. Returns a row of the log, never raises into the pool."""
    tier, hdf5_path, limit_ou, shard_dir = payload
    snana_id = os.path.basename(hdf5_path).replace(".hdf5", "")
    output = shard_path(shard_dir, tier, snana_id)
    if output.exists():
        return snana_id, None, None, 0.0, "ya estaba"

    catalog_path = hdf5_path.replace(".hdf5", ".parquet")
    if not os.path.exists(catalog_path):
        return snana_id, 0, 0, 0.0, "sin catalogo parquet"

    started = time.time()
    try:
        constants = _constants(tier)
        object_records = early_windows.collect_object_records(catalog_path, limit=limit_ou)
        field_windows = []
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
                field_windows.append(window)
        if not field_windows:
            return snana_id, 0, len(object_records), time.time() - started, "sin detecciones"
        # Escritura atomica: el shard aparece entero o no aparece, para que reanudar no lea un
        # parquet a medias de una corrida que murio.
        partial = output.with_suffix(".partial")
        pd.concat(field_windows, ignore_index=True).to_parquet(partial, index=False)
        partial.replace(output)
        return snana_id, len(field_windows), len(object_records), time.time() - started, None
    except Exception:
        return snana_id, None, None, time.time() - started, traceback.format_exc()


def process_tier(tier, hdf5_paths, output_path, shard_dir, limit_ou=None, workers=1):
    """Every field of one tier -> shards -> one parquet. True if the parquet was written."""
    constants = early_windows.build_tier_constants(tier)
    logger.info(
        "[%s] bands=%s  noise_floor_variance: %s",
        tier,
        constants["bands"],
        ", ".join(f"{band}={variance:.0f}" for band, variance in constants["noise_floor_variance"].items()),
    )
    Path(shard_dir).mkdir(parents=True, exist_ok=True)
    payloads = [(tier, path, limit_ou, str(shard_dir)) for path in hdf5_paths]

    tier_start = time.time()
    total_detected = 0
    failures = []

    def record(index, result):
        nonlocal total_detected
        snana_id, n_detected, n_records, elapsed, note = result
        if note and note not in ("ya estaba", "sin detecciones", "sin catalogo parquet"):
            failures.append((snana_id, note))
            logger.error("[%s] [%d/%d] %s FALLO:\n%s", tier, index, len(payloads), snana_id, note)
            return
        if n_detected is not None:
            total_detected += n_detected
        logger.info(
            "[%s] [%d/%d] %s: %s (%.0fs)",
            tier,
            index,
            len(payloads),
            snana_id,
            note if note else f"detected={n_detected}/{n_records}",
            elapsed,
        )

    if workers > 1:
        with Pool(workers) as pool:
            for index, result in enumerate(pool.imap_unordered(process_field, payloads), start=1):
                record(index, result)
    else:
        for index, payload in enumerate(payloads, start=1):
            record(index, process_field(payload))

    if failures:
        raise RuntimeError(f"[{tier}] {len(failures)} campos fallaron: {[one for one, _ in failures]}")

    shards = sorted(Path(shard_dir).glob(f"{tier}__*.parquet"))
    if not shards:
        logger.warning("[%s] ningun shard producido, no se escribe nada", tier)
        return False

    # Se cose por row group en vez de concatenar en pandas: el tier entero en memoria era el pico
    # de 13 GB, y aqui nunca hay mas de un shard cargado.
    writer = None
    rows = 0
    try:
        for shard in shards:
            table = pq.read_table(shard)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    logger.info(
        "[%s] all files done: total_detected=%d shards=%d rows=%d elapsed=%.0fs",
        tier,
        total_detected,
        len(shards),
        rows,
        time.time() - tier_start,
    )
    return True


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes over which to split the fields of a tier (1 = sequential)",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="keep the per-field shards after stitching (default: delete them)",
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

    logger.info("workers    : %d", arguments.workers)
    shard_dir = output_dir / ".early_windows_shards"

    total_start = time.time()
    for tier in ("deep", "wide"):
        output_path = output_dir / f"early_windows_{tier}.parquet"
        if output_path.exists():
            logger.info("[%s] output already exists, skipping: %s", tier, output_path)
            continue

        try:
            written = process_tier(
                tier,
                hdf5_paths,
                output_path,
                shard_dir,
                limit_ou=arguments.limit_ou,
                workers=arguments.workers,
            )
            if not written:
                continue
            logger.info("[%s] -> %s", tier, output_path)
            if not arguments.keep_shards:
                for shard in Path(shard_dir).glob(f"{tier}__*.parquet"):
                    shard.unlink()
        except Exception:
            # Los shards SOBREVIVEN a un fallo a proposito: relanzar reanuda donde quedo.
            logger.error("[%s] ERROR:\n%s", tier, traceback.format_exc())

    if not arguments.keep_shards and shard_dir.exists() and not any(shard_dir.iterdir()):
        shutil.rmtree(shard_dir)

    logger.info("DONE total=%.0fs", time.time() - total_start)


if __name__ == "__main__":
    main()
