"""Visual validation of the low-redshift contaminant generator, one page per class and redshift.

Reading the parquet is not enough to see whether a light curve is a light curve. Two of the bugs
this figure was built to look for -- the near-infrared shoulder of `salt2-extended` and the
half-magnitude Y106 deficit of the core-collapse classes -- are invisible in any aggregate and
obvious in a plot.

Left panel: the model as `roman_light_curve` returns it, the redshifted SED integrated through the
six Roman bandpasses, on days from maximum. Right panel: the four epochs the classifier actually
receives in the deep tier, on days from the first detection, with the unobserved band-epoch slots
left empty. The two axes are deliberately separate -- the model runs on days from maximum and the
window on days from a detection, and there is no honest way to put them on one axis.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from kilonova.config import load_paths, require
from kilonova.photometry.roman_noise import build_tier_constants
from kilonova.photometry.spectra import ALL_ROMAN_BANDS
from kilonova.simulation import openuniverse_parents
from kilonova.simulation.intermediate_z_contaminants import (
    CLASS_FRACTION,
    build_izc_windows,
    core_collapse_source_by_template_index,
    draw_class_population,
    measure_population_brightness,
    roman_light_curve,
    saturation_magnitude,
)

BAND_COLOUR = {
    "R062": "#4b0082",
    "Z087": "#1f6fd0",
    "Y106": "#1a9850",
    "J129": "#d9a300",
    "H158": "#e2591b",
    "F184": "#a50026",
}
REDSHIFTS = (0.02, 0.05, 0.10, 0.20, 0.40)


def measure_brightness(population, source_directory):
    """Read every parent's own brightness out of the release, one hdf5 at a time.

    Without it the figure plots objects at the generator's reference magnitude, which is a plausible
    supernova but not THIS supernova: the right-hand panel is a detection and a saturation check, so
    the magnitude has to be the one the sample will carry."""
    by_healpix = {}
    for realization in population:
        by_healpix.setdefault(realization["parent_healpix"], []).append(realization)
    for healpix, block in sorted(by_healpix.items()):
        measure_population_brightness(block, Path(source_directory) / f"snana_{healpix}.hdf5")


def plot_model(axes, curves, realization):
    for band in ALL_ROMAN_BANDS:
        days, magnitudes = curves[band]
        axes.plot(days, magnitudes, color=BAND_COLOUR[band], label=band, linewidth=1.4)
    axes.invert_yaxis()
    axes.set_xlabel("observer days from maximum")
    axes.set_ylabel("AB magnitude")
    axes.set_title(
        f"{realization['label']}  z = {realization['redshift']:.2f}\n"
        f"{realization['source_name']}, M = {realization['peak_absolute_magnitude']:.2f}",
        fontsize=9,
    )
    axes.legend(fontsize=7, ncol=2, frameon=False)


def plot_window(axes, window, bright_limit):
    observed = window[window["observed"]]
    for band, rows in observed.groupby("band"):
        axes.errorbar(
            rows["days_since_detection"],
            rows["mag_observed"],
            yerr=rows["mag_err"],
            marker="o",
            markersize=5,
            linestyle="none",
            color=BAND_COLOUR[band],
            label=band,
        )
        axes.plot(
            rows["days_since_detection"],
            rows["mag_true"],
            marker="_",
            markersize=11,
            linestyle="none",
            color=BAND_COLOUR[band],
            alpha=0.5,
        )
        axes.axhline(bright_limit[band], color=BAND_COLOUR[band], linewidth=0.6, alpha=0.25)
    axes.invert_yaxis()
    axes.set_xlabel("days since first detection")
    axes.set_ylabel("AB magnitude")
    first_epoch = observed[observed["epoch"] == 1]["band"]
    axes.set_title(
        "deep tier window, epochs 1-4\nfirst epoch: " + ", ".join(sorted(first_epoch)),
        fontsize=9,
    )
    axes.legend(fontsize=7, ncol=2, frameon=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/openuniverse/izc_lightcurve_validation.pdf")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--catalogs", type=Path, default=None, help="directory of the OpenUniverse snana_*.parquet"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="directory of the snana_*.hdf5; without it every object is plotted at the generator's "
        "reference magnitude instead of at its parent's own",
    )
    arguments = parser.parse_args()

    paths = load_paths()
    catalog = openuniverse_parents.read_parent_catalog(
        require(arguments.catalogs or paths.openuniverse_catalogs, "openuniverse_catalogs")
    )
    source_by_template_index = core_collapse_source_by_template_index(catalog)

    random_generator = np.random.default_rng(arguments.seed)
    bright_limit = saturation_magnitude("deep")
    deep_bands = build_tier_constants("deep")["bands"]

    population = {}
    for label in CLASS_FRACTION:
        drawn = draw_class_population(catalog, label, REDSHIFTS, random_generator, source_by_template_index)
        population.update({(label, one["redshift"]): one for one in drawn})
    if arguments.source:
        measure_brightness(list(population.values()), arguments.source)

    with PdfPages(arguments.output) as pdf:
        for label in CLASS_FRACTION:
            for redshift in REDSHIFTS:
                realization = population[(label, redshift)]
                curves = roman_light_curve(realization)
                if not curves:
                    continue
                windows, _ = build_izc_windows([realization], "deep")
                figure, (model_axes, window_axes) = plt.subplots(1, 2, figsize=(11, 4.2))
                plot_model(model_axes, curves, realization)
                if len(windows):
                    plot_window(window_axes, windows, bright_limit)
                else:
                    window_axes.text(0.5, 0.5, "never detected", ha="center", va="center")
                    window_axes.set_axis_off()
                figure.tight_layout()
                pdf.savefig(figure)
                plt.close(figure)
    print(f"{arguments.output}  ({len(CLASS_FRACTION)} classes x {len(REDSHIFTS)} redshifts)")
    print("deep bands:", ", ".join(deep_bands))


if __name__ == "__main__":
    main()
