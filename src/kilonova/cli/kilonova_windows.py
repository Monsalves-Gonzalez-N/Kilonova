"""kn-kilonova-windows: early-window light curves for LANL kilonovas over a redshift grid.

Samples realizations (simulation, viewing angle, explosion offset) at every redshift of the
grid, synthesizes the 6-band Roman photometry from the cached LANL rest-frame spectra, and
writes one kn_windows_{tier}.parquet per tier with the same noise recipe, cadence and window
logic as the OpenUniverse contaminants (kn-early-windows). Realizations are shared across
tiers: the same kilonova is observed in deep and wide.
"""

import argparse
from pathlib import Path

import numpy as np

from kilonova.config import load_paths, require
from kilonova.log import setup_logging
from kilonova.simulation import early_windows
from kilonova.simulation.extinction import build_redshift_grid


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--lanl-spectra", type=Path, help="Cached rest-frame spectra parquet built by kn-cache-lanl."
    )
    parser.add_argument("--output-dir", type=Path, help="where kn_windows_{tier}.parquet is written")
    parser.add_argument("--tier", choices=["deep", "wide", "both"], default="both")

    parser.add_argument("--redshift-min", type=float, default=0.01)
    parser.add_argument("--redshift-max", type=float, default=1.0)
    parser.add_argument("--n-redshift", type=int, default=50)
    parser.add_argument("--redshift-spacing", choices=("linear", "log"), default="log")
    parser.add_argument(
        "--realizations-per-redshift",
        type=int,
        default=200,
        help="kilonova realizations (simulation x angle x explosion offset) drawn at each redshift",
    )
    parser.add_argument("--seed", type=int, default=early_windows.KN_REALIZATION_SEED)
    parser.add_argument(
        "--limit-kn", type=int, default=None, help="process only the first N realizations (smoke tests)"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    arguments = parser.parse_args(argv)

    setup_logging(arguments.verbose)
    paths = load_paths()
    lanl_spectra_path = require(arguments.lanl_spectra or paths.lanl_spectra, "lanl_spectra")
    output_dir = arguments.output_dir or paths.output_dir
    if output_dir is None:
        raise SystemExit("output_dir is not configured (configs/paths.yaml, KN_OUTPUT_DIR or --output-dir)")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    redshift_grid = build_redshift_grid(
        arguments.redshift_min, arguments.redshift_max, arguments.n_redshift, arguments.redshift_spacing
    )
    print(
        f"Redshift grid ({arguments.redshift_spacing}, N={len(redshift_grid)}): "
        f"{redshift_grid[0]:.4f} -> {redshift_grid[-1]:.4f}, "
        f"{arguments.realizations_per_redshift} realizations each"
    )

    lanl_catalog = early_windows.load_lanl_catalog_metadata(lanl_spectra_path)
    simulation_pool = np.sort(lanl_catalog["simulation_id"].unique())
    simulation_time_grids = early_windows.build_simulation_time_grids(lanl_catalog)
    wavelength_rest_aa = early_windows.load_lanl_wavelength_grid(lanl_spectra_path)

    realizations = early_windows.sample_kn_realizations_on_grid(
        redshift_grid,
        arguments.realizations_per_redshift,
        simulation_pool,
        np.random.default_rng(arguments.seed),
    )
    if arguments.limit_kn is not None:
        realizations = dict(list(realizations.items())[: arguments.limit_kn])

    kn_models = early_windows.build_kn_models(
        realizations, simulation_time_grids, wavelength_rest_aa, lanl_spectra_path
    )

    tiers = ["deep", "wide"] if arguments.tier == "both" else [arguments.tier]
    summary = {}
    for tier in tiers:
        output_path = output_dir / f"kn_windows_{tier}.parquet"
        summary[tier] = early_windows.run_kn_tier(tier, kn_models, output_path)

    for tier, n_detected in summary.items():
        print(f"{tier}: kilonovas with >=1 detection = {n_detected} / {len(realizations)}")


if __name__ == "__main__":
    main()
