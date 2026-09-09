"""`salt2-extended` against `salt3-nir`, on the same SNe Ia and the same drawn population.

`scripts/compare_csp_lightcurves.py` measures the izc SN Ia against the Carnegie Supernova Project
and finds one defect that survives every control: the near-infrared decline rate is too slow by
0.21, 0.23 and 0.38 mag in Y, J and H. That residual is immune to the distance (it is a difference
of two magnitudes of the same object) and immune to host dust (a dust screen does not change a
decline rate), which is what makes it the firmest result of that comparison.

It is also, as measured there, a statement about only half of the generator. The generator used to
give a SN Ia `salt3-nir` above z = 0.05 and `salt2-extended` below, because `salt3-nir` stops at
20000 A rest-frame and cannot reach the red edge of F184 at lower redshift. Every CSP SN Ia with a
near-infrared decline measurement sits below z = 0.05, so the whole of that +0.38 mag was measured
on the fallback, and the model that carries most of the generated sample was never tested.

Nothing forces that split here. The constraint that imposes it is Roman's F184, and CSP's reddest
band is H, which ends at 18676 A -- inside `salt3-nir` at every redshift in this sample. So both
sources can be put under the same photometry, and this script does exactly that.

The comparison is paired, which is the point. Each realization is used twice, with its source name
overwritten and everything else -- the parent's own x1 and colour, and the reference magnitude both
sources are normalised to -- left as `izc.draw_class_population` built it. Both are normalised in
rest-frame B through the same `set_source_peakabsmag` call, so the two models are the same supernova
with the same B magnitude and differ only in the spectral shape that carries it into the
near-infrared. Any difference in the residuals is that shape and nothing else.
"""

import argparse
from pathlib import Path

import compare_csp_lightcurves as comparison
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kilonova.config import load_paths, require
from kilonova.simulation import intermediate_z_contaminants as izc
from kilonova.simulation import openuniverse_parents
from kilonova.validation import csp

SALT_SOURCES = ("salt2-extended", "salt3-nir")
SOURCE_COLOUR = {"salt2-extended": "#b2182b", "salt3-nir": "#2166ac"}
SOURCE_MARKER = {"salt2-extended": "o", "salt3-nir": "s"}


def paired_realizations(realizations, source_name):
    """The same population, told to use one source. See the module docstring on why it is paired."""
    paired = []
    for realization in realizations:
        copy = dict(realization)
        copy["source_name"] = source_name
        paired.append(copy)
    return paired


def residual_rows_of_supernova(supernova, observed, bandpasses, redshift, realizations):
    """One row per (supernova, bandpass, source): the two statistics and their residuals."""
    curves_by_source = {}
    for source_name in SALT_SOURCES:
        curves_by_source[source_name] = comparison.model_band_curves(
            paired_realizations(realizations, source_name), bandpasses, redshift, "bmax"
        )

    brightness = comparison.BRIGHTNESS_STATISTIC["bmax"]
    shape = comparison.SHAPE_STATISTIC["bmax"]
    rows = []
    for bandpass in bandpasses:
        points = observed[observed["bandpass"] == bandpass]
        days = points["days"].to_numpy()
        magnitudes = points["mag_corrected"].to_numpy()
        observed_peak = brightness(days, magnitudes, redshift)
        observed_decline = shape(days, magnitudes, redshift)
        for source_name in SALT_SOURCES:
            model_days, model_curves = curves_by_source[source_name]
            median, _, _ = comparison.percentile_curve(model_curves[bandpass])
            model_peak = brightness(model_days, median, redshift)
            model_decline = shape(model_days, median, redshift)
            rows.append(
                {
                    "sn": supernova["sn"],
                    "source": source_name,
                    "letter": csp.BANDPASS_LETTER[bandpass],
                    "bandpass": bandpass,
                    "redshift": redshift,
                    "observed_peak": observed_peak,
                    "model_peak": model_peak,
                    "peak_residual": observed_peak - model_peak,
                    "observed_decline": observed_decline,
                    "model_decline": model_decline,
                    "decline_residual": observed_decline - model_decline,
                }
            )
    return rows


def paired_summary(residuals, column):
    """Median residual per band and source, and the paired difference on the SNe both measure.

    The difference is taken supernova by supernova rather than between the two medians: the two
    sources do not always reach the statistic on the same objects -- a decline rate needs the model
    to be defined across the whole interval -- and a difference of medians over different samples
    would be a statement about which objects survived."""
    wide = residuals.pivot_table(index=["sn", "letter"], columns="source", values=column, aggfunc="first")
    both = wide.dropna()
    rows = []
    for letter in comparison.LETTER_ORDER:
        of_letter = (
            wide.xs(letter, level="letter") if letter in wide.index.get_level_values("letter") else None
        )
        if of_letter is None:
            continue
        paired = both.xs(letter, level="letter") if letter in both.index.get_level_values("letter") else None
        row = {"letter": letter}
        for source_name in SALT_SOURCES:
            values = of_letter[source_name].dropna()
            row[source_name] = np.median(values) if len(values) else np.nan
            row[f"n_{source_name}"] = len(values)
        if paired is not None and len(paired):
            difference = paired["salt3-nir"] - paired["salt2-extended"]
            row["nir_minus_extended"] = np.median(difference)
            row["n_paired"] = len(difference)
        else:
            row["nir_minus_extended"] = np.nan
            row["n_paired"] = 0
        rows.append(row)
    return pd.DataFrame(rows).set_index("letter")


def plot_summary(residuals):
    figure, axes_grid = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    panels = (
        ("peak_residual", "obs - model, peak magnitude"),
        ("decline_residual", "obs - model, m(+15 d) - m(max)"),
    )
    letters = [letter for letter in comparison.LETTER_ORDER if letter in set(residuals["letter"])]
    for axes, (column, ylabel) in zip(axes_grid, panels, strict=True):
        for offset, letter in enumerate(letters):
            for shift, source_name in zip((-0.16, 0.16), SALT_SOURCES, strict=True):
                values = residuals.loc[
                    (residuals["letter"] == letter) & (residuals["source"] == source_name), column
                ].to_numpy()
                values = values[np.isfinite(values)]
                if not len(values):
                    continue
                axes.scatter(
                    np.full(len(values), offset + shift)
                    + np.random.default_rng(offset).normal(0.0, 0.03, len(values)),
                    values,
                    s=10,
                    color=SOURCE_COLOUR[source_name],
                    alpha=0.35,
                    edgecolors="none",
                )
                median = np.median(values)
                axes.errorbar(
                    offset + shift,
                    median,
                    yerr=[
                        [median - np.percentile(values, 16)],
                        [np.percentile(values, 84) - median],
                    ],
                    marker=SOURCE_MARKER[source_name],
                    markersize=7,
                    color=SOURCE_COLOUR[source_name],
                    elinewidth=1.4,
                    capsize=4,
                    zorder=5,
                    label=source_name if offset == 0 else None,
                )
        axes.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axes.set_ylabel(ylabel)
        axes.set_xticks(range(len(letters)))
        axes.set_xticklabels(letters)
    axes_grid[0].legend(loc="upper left", fontsize=9, frameon=False)
    axes_grid[0].set_title(
        "izc SN Ia against CSP-I: the same drawn population through both SALT sources", fontsize=11
    )
    axes_grid[-1].set_xlabel("CSP band")
    figure.tight_layout()
    return figure


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=Path("data/csp/ia_salt_sources.pdf"))
    parser.add_argument("--realizations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--catalogs",
        type=Path,
        default=None,
        help="directory of the OpenUniverse snana_*.parquet (default: openuniverse_catalogs)",
    )
    arguments = parser.parse_args()

    paths = load_paths()
    catalog = openuniverse_parents.read_parent_catalog(
        require(arguments.catalogs or paths.openuniverse_catalogs, "openuniverse_catalogs")
    )
    source_by_template_index = izc.core_collapse_source_by_template_index(catalog)

    metadata = csp.load_metadata()
    photometry = csp.load_photometry()
    reddening = csp.milky_way_reddening(metadata)
    extinction_ratio = csp.milky_way_extinction_ratios(csp.BANDPASS_ORDER)
    metadata["redshift_cmb"] = csp.cmb_frame_redshift(
        metadata["right_ascension"], metadata["declination"], metadata["redshift"]
    )

    sample = metadata[metadata["label"] == "SN Ia"].reset_index(drop=True)
    if arguments.limit:
        sample = sample.iloc[: arguments.limit]
    random_generator = np.random.default_rng(arguments.seed)

    residual_rows = []
    for _, supernova in sample.iterrows():
        of_supernova = photometry[photometry["sn"] == supernova["sn"]]
        if not len(of_supernova):
            continue
        origin_mjd = supernova["epoch_mjd"]
        estimated_origin = not np.isfinite(origin_mjd)
        if estimated_origin:
            origin_mjd = comparison.estimate_bmax(of_supernova)
            if not np.isfinite(origin_mjd):
                continue
        published_distance = np.isfinite(supernova["distance_modulus"])
        redshift = float(supernova["redshift"] if published_distance else supernova["redshift_cmb"])
        # The same two cuts the main comparison applies before it summarises; see its `too_close`.
        if not published_distance and redshift < comparison.MINIMUM_HUBBLE_FLOW_REDSHIFT:
            continue
        if estimated_origin:
            continue
        observed = comparison.supernova_photometry(
            of_supernova, extinction_ratio, float(reddening[supernova["sn"]]), origin_mjd
        )
        bandpasses = sorted(set(observed["bandpass"]), key=csp.BANDPASS_ORDER.index)
        realizations = comparison.draw_class_population(
            catalog,
            "SN Ia",
            redshift,
            arguments.realizations,
            random_generator,
            source_by_template_index,
        )
        residual_rows.extend(
            residual_rows_of_supernova(supernova, observed, bandpasses, redshift, realizations)
        )

    residuals = pd.DataFrame(residual_rows)
    residuals.to_csv(arguments.output.with_suffix(".csv"), index=False)

    for column, name in (
        ("peak_residual", "peak magnitude"),
        ("decline_residual", "m(+15 d) - m(max)"),
    ):
        print(f"\n=== {name}: median obs - model ===")
        print(paired_summary(residuals, column).round(3).to_string())

    figure = plot_summary(residuals)
    figure.savefig(arguments.output)
    plt.close(figure)
    print(f"\n{arguments.output}")


if __name__ == "__main__":
    main()
