"""kn-afterglow-windows: early-window light curves for GRB afterglow contaminants.

Draws afterglows over the parameter space of configs/afterglow.yaml, evaluates them with
afterglowpy at the visits of the 5 d cadence, discards the configurations brighter than the
saturation limit and writes, with the same noise recipe, cadence and window logic as the
kilonovae, one {name}_windows_{tier}.parquet per tier plus {name}_parameters.parquet (one row
per drawn afterglow, with what happened to it).

The run is split in chunks over --workers processes and is resumable: rerun the same command
after an interruption and only the missing chunks are computed. Use a different --seed and
--name for a held-out sample; its afterglow ids never overlap those of another seed.
"""

import argparse
import os
from pathlib import Path

import numpy as np

from kilonova.config import load_paths
from kilonova.log import setup_logging
from kilonova.simulation import afterglow_contaminants


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-afterglows",
        type=int,
        required=True,
        help="afterglows to draw; roughly 0.2 of them leave a window in each tier",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="afterglow", help="prefix of the output files")
    parser.add_argument("--output-dir", type=Path, help="where the parquet files are written")
    parser.add_argument("--tier", choices=["deep", "wide", "both"], default="both")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--chunk-size", type=int, default=500, help="afterglows per part file")
    parser.add_argument(
        "--config",
        type=Path,
        default=afterglow_contaminants.AFTERGLOW_CONFIG_FILE,
        help="parameter space yaml",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    arguments = parser.parse_args(argv)

    setup_logging(arguments.verbose)
    if arguments.n_afterglows > afterglow_contaminants.AFTERGLOW_IDS_PER_SEED:
        raise SystemExit(
            f"--n-afterglows is limited to {afterglow_contaminants.AFTERGLOW_IDS_PER_SEED} per seed"
        )
    output_dir = arguments.output_dir or load_paths().output_dir
    if output_dir is None:
        raise SystemExit("output_dir is not configured (configs/paths.yaml, KN_OUTPUT_DIR or --output-dir)")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = afterglow_contaminants.load_afterglow_config(arguments.config)
    population = afterglow_contaminants.sample_afterglow_parameters(
        arguments.n_afterglows,
        np.random.default_rng(arguments.seed),
        config,
        first_afterglow_id=arguments.seed * afterglow_contaminants.AFTERGLOW_IDS_PER_SEED,
    )
    tiers = ["deep", "wide"] if arguments.tier == "both" else [arguments.tier]
    windows_by_tier = afterglow_contaminants.run_afterglow_generation(
        population,
        tiers,
        output_dir,
        arguments.name,
        arguments.workers,
        arguments.chunk_size,
        arguments.config,
    )
    for tier, number_of_windows in windows_by_tier.items():
        print(f"{tier}: afterglows with a window = {number_of_windows} / {len(population)}")


if __name__ == "__main__":
    main()
