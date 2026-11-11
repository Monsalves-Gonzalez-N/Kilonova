"""Class light-curve figure of the paper with a GRB afterglow block next to the kilonova.

Same objects, photometry and drawing code as cell 12 of openuniverse_hdf5_lightcurve_error.ipynb
(loaded as in plot_class_lightcurves_stacked.py). The OpenUniverse objects are the ones listed in
paper_class_lightcurves_objects.md, read from the public hdf5 by HTTP range requests, so the 16 GB
file of the healpix is not needed. The afterglow comes from AFTERGLOW_ATTEMPTS draws of the
training parameter space placed at the kilonova redshift. Only on-axis draws (viewing angle inside
the jet core) that are not saturated and fade below the deep limit within MAXIMUM_DETECTED_DAYS
are kept, and the one best detected in its worse tier is drawn: without those two conditions the
well-detected draws are off-axis or very energetic bursts that stay visible for more than a year.
The rows are cached as a pickle; --rebuild redoes them all, --rebuild-afterglow only the afterglow.
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import fsspec
import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_class_lightcurves_stacked import load_notebook_namespace  # noqa: E402

from kilonova.config import load_paths  # noqa: E402
from kilonova.photometry.roman_noise import BASE_CADENCE_DAYS  # noqa: E402
from kilonova.photometry.spectra import ALL_ROMAN_BANDS, spectrum_to_roman_magnitudes  # noqa: E402
from kilonova.simulation.afterglow_contaminants import (  # noqa: E402
    AFTERGLOW_LABEL,
    AFTERGLOW_SEED_OFFSET,
    WAVELENGTH_GRID_AA,
    afterglow_spectra,
    load_afterglow_config,
    sample_afterglow_parameters,
)
from kilonova.simulation.early_windows import model_from_hdf5_group  # noqa: E402

REMOTE_HDF5_URL = (
    "https://nasa-irsa-simulations.s3.amazonaws.com/openuniverse2024/roman/full/"
    "roman_rubin_cats_v1.1.2_faint/snana_10307.hdf5"
)
OPENUNIVERSE_OBJECTS = [
    ("SN II", 30556708),
    ("SN Ia", 20168320),
    ("SN Ic", 50012864),
    ("SN Ib", 40065344),
    ("SN Iax", 60151783),
    ("TDE", 110002345),
    ("SLSN-I", 70000922),
    ("PISN-H", 80000086),
    ("PISN-He", 90000105),
]
SELECTION_SEED = 20260728  # build_class_lightcurve_rows default: reproduces the kilonova of the paper
KILONOVA_REDSHIFT = 0.5
KILONOVA_MINIMUM_EJECTA_MASS = 0.2
AFTERGLOW_SEED = 20261111
AFTERGLOW_ATTEMPTS = 300
AFTERGLOW_VISITS = 60
MAXIMUM_DETECTED_DAYS = 60.0
FIGURE_HEIGHT_INCHES = 9.6  # six rows of blocks instead of five


def afterglow_model(parameters):
    """{band: (days after the burst, true AB magnitude)} at every visit of the 5 d cadence, or
    None when afterglowpy fails. Visits where a band carries no flux are left out."""
    visit_days = float(parameters["first_visit_time_days"]) + BASE_CADENCE_DAYS * np.arange(AFTERGLOW_VISITS)
    flux_lambda = afterglow_spectra(parameters, visit_days)
    if flux_lambda is None:
        return None, visit_days
    magnitudes = {band: np.empty(AFTERGLOW_VISITS) for band in ALL_ROMAN_BANDS}
    for visit_index in range(AFTERGLOW_VISITS):
        visit_magnitudes = spectrum_to_roman_magnitudes(
            WAVELENGTH_GRID_AA, flux_lambda[visit_index], ALL_ROMAN_BANDS
        )
        for band in ALL_ROMAN_BANDS:
            magnitudes[band][visit_index] = visit_magnitudes[band]
    model = {}
    for band in ALL_ROMAN_BANDS:
        finite = np.isfinite(magnitudes[band])
        if finite.any():
            model[band] = (visit_days[finite], magnitudes[band][finite])
    return model, visit_days


def build_afterglow_example(namespace, redshift):
    config = load_afterglow_config()
    bright_limit = float(config["saturation"]["bright_limit_magnitude"])
    population = sample_afterglow_parameters(
        AFTERGLOW_ATTEMPTS, np.random.default_rng(AFTERGLOW_SEED), config
    )
    population["redshift"] = redshift
    candidates = []
    for _, parameters in population.iterrows():
        if parameters["viewing_angle_rad"] >= parameters["core_angle_rad"]:
            continue
        model, visit_days = afterglow_model(parameters)
        if model is None or len(model) == 0:
            continue
        if min(band_magnitudes.min() for _, band_magnitudes in model.values()) < bright_limit:
            continue
        by_tier = namespace["photometry_in_both_tiers"](
            model, visit_days, AFTERGLOW_SEED_OFFSET + int(parameters["afterglow_id"])
        )
        if by_tier is None:
            continue
        deep = by_tier["deep"]
        if deep.loc[deep["detected"], "days_since_detection"].max() > MAXIMUM_DETECTED_DAYS:
            continue
        detections = min(int(by_tier[tier]["detected"].sum()) for tier in namespace["TIER_COLUMNS"])
        if detections > 0:
            candidates.append((detections, int(parameters["afterglow_id"]), parameters, by_tier))
    if len(candidates) == 0:
        raise RuntimeError(f"no fast on-axis afterglow detected in both tiers at z={redshift}")
    candidates.sort(key=lambda candidate: candidate[:2])
    print(f"afterglow candidates: {len(candidates)} of {AFTERGLOW_ATTEMPTS} draws")
    _, _, parameters, by_tier = candidates[-1]
    return parameters, by_tier


def afterglow_row(namespace):
    parameters, afterglow_by_tier = build_afterglow_example(namespace, KILONOVA_REDSHIFT)
    print("afterglow parameters:")
    print(parameters.to_string())
    return {
        "label": AFTERGLOW_LABEL,
        "object_id": f"afterglow_{int(parameters['afterglow_id']):07d}",
        "z_CMB": KILONOVA_REDSHIFT,
        "by_tier": afterglow_by_tier,
        "parameters": parameters.to_dict(),
    }


def build_rows(namespace):
    realization, kilonova_by_tier = namespace["build_kilonova_example"](
        KILONOVA_REDSHIFT,
        np.random.default_rng(SELECTION_SEED),
        minimum_ejecta_mass=KILONOVA_MINIMUM_EJECTA_MASS,
    )
    rows = [
        {
            "label": "KN",
            "object_id": f"sim{realization['simulation_id']}_angle{realization['angle_index']}",
            "z_CMB": KILONOVA_REDSHIFT,
            "by_tier": kilonova_by_tier,
        }
    ]

    rows.append(afterglow_row(namespace))

    catalog = pd.read_parquet(namespace["CATALOG_PATH"]).set_index("id")
    with fsspec.open(REMOTE_HDF5_URL, block_size=2**20).open() as remote_file:
        with h5py.File(remote_file, "r") as hdf5:
            for label, object_id in OPENUNIVERSE_OBJECTS:
                model_6band = model_from_hdf5_group(hdf5[str(object_id)], {"bands": ALL_ROMAN_BANDS})
                by_tier = namespace["photometry_in_both_tiers"](model_6band, None, object_id)
                rows.append(
                    {
                        "label": label,
                        "object_id": object_id,
                        "z_CMB": float(catalog.loc[object_id, "z_CMB"]),
                        "by_tier": by_tier,
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="ignore the cached rows")
    parser.add_argument("--rebuild-afterglow", action="store_true", help="redo only the afterglow row")
    arguments = parser.parse_args()

    paths = load_paths()
    namespace = load_notebook_namespace(paths.openuniverse_hdf5, paths.openuniverse_catalog)
    cache_path = Path(paths.output_dir) / "paper_class_lightcurves_rows.pkl"
    if cache_path.exists() and not arguments.rebuild:
        rows = pickle.loads(cache_path.read_bytes())
        if arguments.rebuild_afterglow:
            afterglow_position = [entry["label"] for entry in rows].index(AFTERGLOW_LABEL)
            rows[afterglow_position] = afterglow_row(namespace)
            cache_path.write_bytes(pickle.dumps(rows))
    else:
        rows = build_rows(namespace)
        cache_path.write_bytes(pickle.dumps(rows))

    summary_rows = []
    for letter, entry in zip(namespace["panel_letters"](len(rows)), rows, strict=True):
        summary_rows.append(
            {
                "row": letter,
                "label": entry["label"],
                "object_id": entry["object_id"],
                "z_CMB": entry["z_CMB"],
                "detections_deep": int(entry["by_tier"]["deep"]["detected"].sum()),
                "detections_wide": int(entry["by_tier"]["wide"]["detected"].sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))

    catalog = pd.read_parquet(namespace["CATALOG_PATH"])
    catalog["label"] = catalog["gentype"].map(namespace["GENTYPE_LABEL"])
    census = (
        catalog[catalog["label"] != "FIXMAG"]
        .groupby("label")
        .size()
        .sort_values(ascending=False)
        .rename("objects")
        .reset_index()
    )

    paper_style = namespace["FIGURE_STYLES"]["paper"]
    paper_style["figsize"] = (paper_style["figsize"][0], FIGURE_HEIGHT_INCHES)
    namespace["draw_class_lightcurves"](rows, summary, census, "paper")


if __name__ == "__main__":
    main()
