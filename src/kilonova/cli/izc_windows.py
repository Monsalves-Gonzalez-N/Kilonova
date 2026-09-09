"""kn-izc-windows: OpenUniverse contaminants re-rendered at the redshifts the survey has none at.

Reads the 33 healpix catalogues, counts the redshift deficit of the training set against its own
kilonovae, draws a parent object per missing contaminant, measures that parent's brightness off its
full light curve in the release's hdf5, and re-renders it at the drawn redshift through the same
noise recipe, cadence and window logic as `kn-run-openuniverse` and `kn-kilonova-windows`:

    {output_dir}/izc_windows_deep.parquet
    {output_dir}/izc_windows_wide.parquet

`object_id` is `izc_{parent_key}_{z}_{index}` and the parent key is the OpenUniverse `object_id` of
the object it was re-rendered from, which is what puts every copy of a parent, and the parent
itself, on one side of the leakage-aware split.

The hdf5 files are the expensive input and the only one that has to be mounted: `--source` points
at the directory of snana_*.hdf5. The catalogues are 135 MB and live in the repository's own data
directory.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from kilonova.config import load_paths, require
from kilonova.log import setup_logging
from kilonova.simulation import intermediate_z_contaminants as izc
from kilonova.simulation import openuniverse_parents

logger = logging.getLogger(__name__)


def window_redshifts(path):
    """The redshift of every object of a window parquet, one entry per object."""
    table = pq.read_table(path, columns=["object_id", "z_CMB"]).to_pandas()
    return table.drop_duplicates("object_id")["z_CMB"].to_numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source", type=Path, help="directory with snana_*.hdf5 (default: openuniverse_source)"
    )
    parser.add_argument(
        "--catalogs", type=Path, help="directory with snana_*.parquet (default: openuniverse_catalogs)"
    )
    parser.add_argument(
        "--windows-dir",
        type=Path,
        help="where early_windows_{tier}.parquet and kn_windows_{tier}.parquet are read from, "
        "for the redshift deficit (default: output_dir)",
    )
    parser.add_argument("--output-dir", type=Path, help="where izc_windows_{tier}.parquet is written")
    parser.add_argument("--tier", choices=["deep", "wide", "both"], default="both")
    parser.add_argument(
        "--deficit-scale",
        type=float,
        default=1.0,
        help="multiply the deficit of every bin by this before drawing (1.0 = fill it exactly)",
    )
    parser.add_argument(
        "--limit-objects",
        type=int,
        default=None,
        help="generate only this many objects, sampled from the deficit's own shape (smoke tests)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers", type=int, default=1, help="processes over which to split the healpix (1 = sequential)"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    arguments = parser.parse_args(argv)

    setup_logging(arguments.verbose)
    paths = load_paths()
    source = require(arguments.source or paths.openuniverse_source, "openuniverse_source")
    catalogs = require(arguments.catalogs or paths.openuniverse_catalogs, "openuniverse_catalogs")
    output_dir = Path(arguments.output_dir or paths.output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = Path(arguments.windows_dir or paths.output_dir or output_dir)
    tiers = ["deep", "wide"] if arguments.tier == "both" else [arguments.tier]

    # --- what the training set is missing --------------------------------------------------------
    kilonova_redshifts, contaminant_redshifts = {}, {}
    for tier in ("deep", "wide"):
        kilonova_redshifts[tier] = window_redshifts(windows_dir / f"kn_windows_{tier}.parquet")
        contaminant_redshifts[tier] = window_redshifts(windows_dir / f"early_windows_{tier}.parquet")
    edges, deficit = izc.redshift_deficit(kilonova_redshifts, contaminant_redshifts)
    scale = arguments.deficit_scale
    if arguments.limit_objects is not None:
        scale = arguments.limit_objects / deficit.sum()
    random_generator = np.random.default_rng(arguments.seed)
    redshifts = izc.draw_redshifts_from_deficit(edges, deficit, random_generator, scale)
    logger.info(
        "deficit %d objects over %d bins, z = %.3f to %.3f; generating %d",
        deficit.sum(),
        len(deficit),
        edges[0],
        edges[-1],
        len(redshifts),
    )
    if not len(redshifts):
        raise SystemExit("the deficit is empty: nothing to generate")

    # --- the parents ----------------------------------------------------------------------------
    catalog = openuniverse_parents.read_parent_catalog(catalogs)
    source_by_template_index = izc.core_collapse_source_by_template_index(catalog)
    logger.info(
        "%d parents over %d healpix, %d core-collapse templates",
        len(catalog),
        catalog["healpix"].nunique(),
        len(source_by_template_index),
    )
    population = izc.draw_population_from_parents(
        catalog, redshifts, random_generator, source_by_template_index
    )
    parents_used = len({one["parent_key"] for one in population})
    logger.info("%d objects from %d distinct parents", len(population), parents_used)

    output_paths = {tier: output_dir / f"izc_windows_{tier}.parquet" for tier in tiers}
    totals, per_tier = izc.run_izc_tiers(population, source, tiers, output_paths, arguments.workers)

    print(
        f"objects={totals['objects']}  parents={parents_used}  "
        f"without a measurable brightness={totals['unmeasured']}"
    )
    for tier in tiers:
        summary = per_tier[tier]
        print(
            f"{tier}: windows={summary['windows']}  dropped: no band coverage={summary['coverage']}  "
            f"never detected={summary['undetected']}  (saturated but kept={summary['saturated_kept']})"
        )


if __name__ == "__main__":
    main()
