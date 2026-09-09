"""The izc model light curves, continuous, against the Carnegie Supernova Project I photometry.

The generator is checked here as an SED, not as a window. `scripts/plot_izc_validation.py` already
shows what the classifier receives -- four epochs, five bands, the cadence -- and that part works.
What no Roman-side figure can show is whether the underlying spectral time series is the light
curve of a real supernova, because Roman has never observed one of these at z = 0.02. The Carnegie
Supernova Project has: 134 white dwarf explosions, 34 stripped-envelope supernovae and 94 type II,
in a natural system sncosmo carries transmissions for, and with the near-infrared YJH that is the
extrapolated half of every template in the module. Five of the six izc classes have a counterpart
there; only the TDE has none.

So each figure puts the same two things side by side, per band:

  * the model, continuous: the drawn SED of that izc class at that supernova's redshift, integrated
    through the actual CSP filter it was observed in. Not a fit to the supernova -- a population.
    The line is the median of `--realizations` draws of the class and the shaded band is their
    16-84 percentile range, so the question the figure asks is whether the observed supernova is a
    plausible member of the population the generator produces, not whether one template matches it.
  * the observation, discrete: the released CSP magnitudes, corrected only for Milky Way
    extinction, on the same time axis -- days from B maximum, or days from the explosion for the
    type II, which have no maximum worth aligning on.

Two differences are expected and are not failures. The observed supernova carries host-galaxy
extinction and the model carries none -- the izc module applies no dust of any kind, deliberately,
because the OpenUniverse contaminants it sits next to hide theirs inside the templates -- so the
data may sit fainter and redder than the model, one-sidedly. And the observed object is one draw of
its class, so it is expected to leave the 16-84 band roughly a third of the time.

What would be a failure, and what the figure is read for: the model brighter than the data in a
band where dust cannot help, a colour that runs the wrong way, a decline rate that misses, or a
near-infrared shape -- the second maximum of a SN Ia above all -- that the template does not have.
"""

import argparse
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18
from matplotlib.backends.backend_pdf import PdfPages

from kilonova.config import load_paths, require
from kilonova.simulation import intermediate_z_contaminants as izc
from kilonova.simulation import openuniverse_parents
from kilonova.validation import csp

# Rest-frame phases the model is drawn on, per kind of time origin. Fine enough that the curve
# reads as a curve and not as a polyline, which is the whole point of putting it under discrete
# data. The type II range runs past the plateau; the other one brackets maximum.
MODEL_PHASE_LIMITS = {"bmax": (-20.0, 80.0), "explosion": (0.0, 115.0)}
MODEL_PHASE_STEP = 0.5

# The model's own epoch of explosion, for the type II comparison: the first phase at which the
# template carries this fraction of its peak B flux. The templates do not agree on where their
# phase zero is -- the Nugent sources put it at the explosion and the SNANA ones at B maximum, and
# a few of the SNANA ones simply start after the rise -- so the origin has to be read off the light
# curve rather than taken from the file. A SN II rises through five magnitudes in B in a day or
# two, which is the accuracy of this definition and is far below anything the comparison measures.
EXPLOSION_FLUX_FRACTION = 0.01

# Days from B maximum, observer frame, that the peak magnitude of a band is measured inside. Wide
# enough on the red side to hold the near-infrared maximum of a stripped-envelope supernova, which
# lags B by up to two weeks, and narrow enough to exclude the second maximum of a SN Ia.
PEAK_WINDOW_DAYS = (-10.0, 20.0)
MINIMUM_PEAK_POINTS = 3

# A type II has no peak to measure. Its brightness is quoted mid-plateau, at 50 rest-frame days
# after the explosion, and its shape by the plateau slope between 40 and 80 days -- the s2 of
# Anderson et al. (2014), over the interval where the plateau is established and has not yet
# fallen off the end.
PLATEAU_EPOCH_DAYS = 50.0
PLATEAU_SLOPE_DAYS = (40.0, 80.0)
# A wider gap than the one the decline rate allows, because the quantity interpolated across it is
# a plateau and not a decline: the near-infrared cadence of the type II release runs to a fortnight
# and refusing to cross it would throw away most of the sample to protect a flat curve.
PLATEAU_MAXIMUM_GAP_DAYS = 15.0

# A phase where fewer than this fraction of the drawn models have flux is not drawn: the percentile
# there would be a statement about which templates happen to be defined, not about the class.
MINIMUM_DEFINED_FRACTION = 0.9

# Below this the heliocentric redshift stops being a distance: the residual peculiar velocity of the
# host is 300 km/s and at z = 0.01 that is 0.22 mag of distance modulus, which is most of the
# scatter the comparison is trying to measure. Those supernovae are still plotted and are excluded
# from the population summary.
MINIMUM_HUBBLE_FLOW_REDSHIFT = 0.01

LETTER_ORDER = ("u", "B", "g", "V", "r", "i", "Y", "J", "H")
CLASS_ORDER = ("SN Ia", "SN Iax", "SN Ib", "SN Ic", "SN II")
LETTER_COLOUR = {
    "u": "#4b0082",
    "B": "#3b4cc0",
    "g": "#1f9e89",
    "V": "#4daf4a",
    "r": "#d9a300",
    "i": "#e2591b",
    "Y": "#c1272d",
    "J": "#8c2155",
    "H": "#5c1a3a",
}


# The type II release gives no subtype and the generator splits the class in three. Rejecting on
# the set rather than on one name keeps the subtypes in the generator's own proportions, which is
# also the mix the classifier sees: all three carry OpenUniverse gentype 32.
# OpenUniverse drew no SN IIn -- see the SOURCES_BY_LABEL block of the generator -- so the class is
# gone from both sides and the type II release maps onto the two subtypes that exist.
LABELS_OF = {"SN II": ("SN IIP", "SN IIL")}


def draw_class_population(catalog, label, redshift, count, random_generator, source_by_template_index):
    """`count` realizations of one izc class at one redshift.

    The brightness comes out at the generator's reference magnitude rather than measured off a
    parent's light curve, and this comparison is built to be indifferent to it: every residual here
    is a difference of two magnitudes of the same object, and the one free scale left -- the host
    dust screen -- is fitted per class over the optical. See `izc.draw_class_population`."""
    return izc.draw_class_population(
        catalog,
        LABELS_OF.get(label, (label,)),
        np.full(count, redshift),
        random_generator,
        source_by_template_index,
    )


@cache
def explosion_phase(source_name):
    """The template's own phase of explosion; see EXPLOSION_FLUX_FRACTION."""
    import sncosmo

    source = izc._registry_source(source_name)
    phases = np.arange(source.minphase(), source.maxphase(), 0.5)
    model = sncosmo.Model(source=source)
    with np.errstate(all="ignore"):
        flux = np.array([model.bandflux("bessellb", phase) for phase in phases])
    flux = np.where(np.isfinite(flux), flux, 0.0)
    risen = np.flatnonzero(flux >= EXPLOSION_FLUX_FRACTION * flux.max())
    return float(phases[risen[0]]) if len(risen) else float(source.minphase())


def origin_phase(realization, epoch_kind):
    if epoch_kind == "explosion":
        return explosion_phase(realization["source_name"])
    return izc.peak_phase(realization)


def model_band_curves(realizations, bandpasses, redshift, epoch_kind):
    """{bandpass: (observer days from B maximum, 2D array of CSP natural-system magnitudes)}.

    One row per realization. A phase the model does not cover, or covers with no flux, comes back
    NaN: `MODEL_FLUX_FLOOR_MAGNITUDE` is the module's own threshold for "the template has no flux
    here", and it matters in exactly this comparison -- `salt2-extended` has essentially none
    redward of Y before rest-frame phase -10, so a SN Ia model that simply stopped there would
    otherwise be drawn as a magnitude 59 light curve."""
    limits = MODEL_PHASE_LIMITS[epoch_kind]
    rest_phases = np.arange(limits[0], limits[1] + MODEL_PHASE_STEP, MODEL_PHASE_STEP)
    observer_days = rest_phases * (1.0 + redshift)
    curves = {band: np.full((len(realizations), len(rest_phases)), np.nan) for band in bandpasses}
    for row, realization in enumerate(realizations):
        # Planck18, not the module's OpenUniverse default: this comparison puts the models at the
        # distance of a real supernova, and `csp.load_type_ii_metadata` inverts the published
        # distance moduli through Planck18. The two have to be the same cosmology or the model
        # lands 0.07 mag off at these redshifts for no reason but bookkeeping.
        model = izc.build_model(realization, cosmology=Planck18)
        source = model.source
        phases = origin_phase(realization, epoch_kind) + rest_phases
        inside = (phases >= source.minphase()) & (phases <= source.maxphase())
        if not inside.any():
            continue
        times = phases[inside] * (1.0 + redshift)
        for band in bandpasses:
            with np.errstate(divide="ignore", invalid="ignore"):
                magnitudes = model.bandmag(band, "csp", times)
            magnitudes = np.where(
                np.isfinite(magnitudes) & (magnitudes < izc.MODEL_FLUX_FLOOR_MAGNITUDE),
                magnitudes,
                np.nan,
            )
            curves[band][row, inside] = magnitudes
    return observer_days, curves


def percentile_curve(magnitudes):
    """(median, low, high) over the realizations, blanked where too few of them are defined."""
    defined = np.isfinite(magnitudes).mean(axis=0)
    with np.errstate(invalid="ignore"):
        median, low, high = np.nanpercentile(magnitudes, [50.0, 16.0, 84.0], axis=0)
    enough = defined >= MINIMUM_DEFINED_FRACTION
    return (
        np.where(enough, median, np.nan),
        np.where(enough, low, np.nan),
        np.where(enough, high, np.nan),
    )


# The decline rate is measured over this many rest-frame days after B maximum, the interval dm15
# is conventionally defined on. It is only measured where the band's light curve brackets that
# interval with no gap wider than DECLINE_MAXIMUM_GAP_DAYS inside it -- interpolating across a
# fortnight of missing near-infrared would turn the cadence into a decline rate.
DECLINE_INTERVAL_DAYS = 15.0
DECLINE_MAXIMUM_GAP_DAYS = 8.0


def _interpolate(days, magnitudes, epoch, maximum_gap):
    """The magnitude at one epoch, or NaN unless the light curve brackets it closely enough."""
    finite = np.isfinite(magnitudes) & np.isfinite(days)
    days, magnitudes = np.asarray(days)[finite], np.asarray(magnitudes)[finite]
    if len(days) < 2 or days.min() > epoch or days.max() < epoch:
        return np.nan
    order = np.argsort(days)
    days, magnitudes = days[order], magnitudes[order]
    after = np.searchsorted(days, epoch)
    if days[after] - days[after - 1] > maximum_gap:
        return np.nan
    return float(np.interp(epoch, days, magnitudes))


def plateau_magnitude(days, magnitudes, redshift):
    """The magnitude 50 rest-frame days after the explosion; see PLATEAU_EPOCH_DAYS."""
    return _interpolate(days, magnitudes, PLATEAU_EPOCH_DAYS * (1.0 + redshift), PLATEAU_MAXIMUM_GAP_DAYS)


def plateau_slope(days, magnitudes, redshift):
    """m(+80 d) - m(+40 d) from the explosion, both rest-frame; the s2 of Anderson et al. (2014)."""
    early, late = (
        _interpolate(days, magnitudes, epoch * (1.0 + redshift), PLATEAU_MAXIMUM_GAP_DAYS)
        for epoch in PLATEAU_SLOPE_DAYS
    )
    return late - early


def decline_rate(days, magnitudes, redshift):
    """m(+15 rest-frame days) - m(B maximum), in one band, or NaN where the data do not reach.

    The peak magnitude alone cannot separate a model that is too bright from a model that is too
    slow, and in the near-infrared -- where these templates are extrapolations -- the shape is the
    thing least constrained by anything that was ever observed. Measured by interpolation rather
    than by fitting, and measured the same way on the model, so the comparison never depends on the
    two sides being sampled alike."""
    finite = np.isfinite(magnitudes) & np.isfinite(days)
    days, magnitudes = np.asarray(days)[finite], np.asarray(magnitudes)[finite]
    if len(days) < MINIMUM_PEAK_POINTS:
        return np.nan
    target = DECLINE_INTERVAL_DAYS * (1.0 + redshift)
    if days.min() > 0.0 or days.max() < target:
        return np.nan
    order = np.argsort(days)
    days, magnitudes = days[order], magnitudes[order]
    spanning = days[(days >= days[days <= 0.0].max()) & (days <= days[days >= target].min())]
    if np.diff(spanning).max(initial=0.0) > DECLINE_MAXIMUM_GAP_DAYS:
        return np.nan
    order = np.argsort(days)
    return float(
        np.interp(target, days[order], magnitudes[order]) - np.interp(0.0, days[order], magnitudes[order])
    )


def peak_magnitude(days, magnitudes):
    """The brightest magnitude inside PEAK_WINDOW_DAYS, or NaN if the window is not sampled."""
    inside = (days >= PEAK_WINDOW_DAYS[0]) & (days <= PEAK_WINDOW_DAYS[1]) & np.isfinite(magnitudes)
    if inside.sum() < MINIMUM_PEAK_POINTS:
        return np.nan
    return float(np.min(magnitudes[inside]))


# Brightness and shape, one pair per kind of time origin, applied identically to the model and to
# the data. The comparison never subtracts two different statistics.
BRIGHTNESS_STATISTIC = {
    "bmax": lambda days, magnitudes, redshift: peak_magnitude(days, magnitudes),
    "explosion": plateau_magnitude,
}
SHAPE_STATISTIC = {"bmax": decline_rate, "explosion": plateau_slope}
BRIGHTNESS_NAME = {"bmax": "peak magnitude", "explosion": "plateau magnitude at +50 d"}
SHAPE_NAME = {"bmax": "m(+15 d) - m(max)", "explosion": "m(+80 d) - m(+40 d)"}
TIME_AXIS_LABEL = {"bmax": "days from B maximum", "explosion": "days from explosion"}


def supernova_photometry(photometry, extinction_ratio, reddening, t_bmax):
    """The observed light curve of one supernova: MW-corrected, on days from B maximum."""
    frame = photometry.copy()
    frame["days"] = frame["mjd"] - t_bmax
    frame["mag_corrected"] = frame["mag"] - reddening * frame["bandpass"].map(extinction_ratio)
    frame["letter"] = frame["bandpass"].map(csp.BANDPASS_LETTER)
    return frame


def estimate_bmax(frame):
    """B maximum from the data, for the supernovae DR3 does not give an epoch for."""
    in_b = frame[frame["bandpass"] == "cspb"]
    if len(in_b) < MINIMUM_PEAK_POINTS:
        return np.nan
    return float(in_b.loc[in_b["mag"].idxmin(), "mjd"])


def plot_supernova(supernova, observed, model_days, model_curves, distance_note, reddening, sources):
    letters = [letter for letter in LETTER_ORDER if letter in set(observed["letter"])]
    columns = 3
    rows = int(np.ceil(len(letters) / columns))
    figure, axes_grid = plt.subplots(
        rows, columns, figsize=(4.0 * columns, 2.9 * rows), squeeze=False, sharex=True
    )
    for position, letter in enumerate(letters):
        axes = axes_grid[position // columns][position % columns]
        points = observed[observed["letter"] == letter]
        for bandpass, rows_of_band in points.groupby("bandpass"):
            median, low, high = percentile_curve(model_curves[bandpass])
            axes.fill_between(model_days, low, high, color=LETTER_COLOUR[letter], alpha=0.18, linewidth=0)
            axes.plot(model_days, median, color=LETTER_COLOUR[letter], linewidth=1.4)
            axes.errorbar(
                rows_of_band["days"],
                rows_of_band["mag_corrected"],
                yerr=rows_of_band["mag_err"],
                linestyle="none",
                marker="o",
                markersize=3.2,
                markerfacecolor="white",
                markeredgewidth=0.9,
                color=LETTER_COLOUR[letter],
                elinewidth=0.8,
            )
        # The vertical range is set by the data, not by the model: several templates have a
        # near-vertical rise from nothing, and letting it set the limits compresses the part of the
        # panel the comparison is actually read in.
        faintest = points["mag_corrected"].max()
        brightest = points["mag_corrected"].min()
        axes.set_ylim(faintest + 1.0, brightest - 1.0)
        limits = MODEL_PHASE_LIMITS[supernova["epoch_kind"]]
        axes.set_xlim(
            min(limits[0], points["days"].min()) - 3.0,
            max(limits[1], points["days"].max()) + 3.0,
        )
        axes.text(
            0.04,
            0.10,
            letter,
            transform=axes.transAxes,
            fontsize=13,
            fontweight="bold",
            color=LETTER_COLOUR[letter],
        )
        if position % columns == 0:
            axes.set_ylabel("magnitude (CSP natural)")
        if position // columns == rows - 1:
            axes.set_xlabel(TIME_AXIS_LABEL[supernova["epoch_kind"]])
    for empty in range(len(letters), rows * columns):
        axes_grid[empty // columns][empty % columns].set_axis_off()
    figure.suptitle(
        f"{supernova['sn']}   {supernova['subtype']}   {distance_note}   "
        f"E(B-V)_MW = {reddening:.3f}\n"
        f"izc model: {supernova['label']}   sources: {sources}",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return figure


def plot_residual_summary(residuals, column, names, xlabel):
    """One residual, per class and band, over the whole sample: every supernova and the median.

    The statistic is not the same for every class -- a type II has a plateau where a SN Ia has a
    maximum -- so each panel says which one it is rather than letting one axis label speak for all
    five."""
    classes = [label for label in CLASS_ORDER if label in set(residuals["label"])]
    figure, axes_grid = plt.subplots(
        len(classes), 1, figsize=(9.0, 2.6 * len(classes)), squeeze=False, sharex=True
    )
    for position, label in enumerate(classes):
        axes = axes_grid[position][0]
        of_class = residuals[residuals["label"] == label]
        letters = [
            letter
            for letter in LETTER_ORDER
            if np.isfinite(of_class.loc[of_class["letter"] == letter, column]).any()
        ]
        for offset, letter in enumerate(letters):
            values = of_class.loc[of_class["letter"] == letter, column].to_numpy()
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            axes.scatter(
                np.full(len(values), offset) + np.random.default_rng(offset).normal(0.0, 0.06, len(values)),
                values,
                s=12,
                color=LETTER_COLOUR[letter],
                alpha=0.55,
                edgecolors="none",
            )
            median = np.median(values)
            axes.errorbar(
                offset,
                median,
                yerr=[[median - np.percentile(values, 16)], [np.percentile(values, 84) - median]],
                marker="s",
                markersize=7,
                color="black",
                elinewidth=1.4,
                capsize=4,
                zorder=5,
            )
        axes.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axes.set_xticks(range(len(letters)))
        axes.set_xticklabels(letters)
        axes.set_ylabel("obs - model")
        statistic = names[of_class["epoch_kind"].iloc[0]]
        axes.set_title(f"{label}   ({of_class['sn'].nunique()} supernovae)   {statistic}", fontsize=10)
    axes_grid[-1][0].set_xlabel(xlabel)
    figure.tight_layout()
    return figure


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=Path("data/csp/izc_vs_csp.pdf"))
    parser.add_argument("--realizations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classes", nargs="*", default=list(CLASS_ORDER))
    parser.add_argument("--limit", type=int, default=None, help="only the first N supernovae")
    parser.add_argument("--sn", nargs="*", default=None, help="only these supernovae, by name")
    parser.add_argument(
        "--catalogs",
        type=Path,
        default=None,
        help="directory of the OpenUniverse snana_*.parquet the generator draws its parents from "
        "(default: the configured openuniverse_catalogs)",
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

    sample = metadata[metadata["label"].isin(arguments.classes)].reset_index(drop=True)
    if arguments.sn:
        sample = sample[sample["sn"].isin(arguments.sn)].reset_index(drop=True)
    if arguments.limit:
        sample = sample.iloc[: arguments.limit]
    random_generator = np.random.default_rng(arguments.seed)

    residual_rows = []
    with PdfPages(arguments.output) as pdf:
        for _, supernova in sample.iterrows():
            of_supernova = photometry[photometry["sn"] == supernova["sn"]]
            if not len(of_supernova):
                continue
            epoch_kind = supernova["epoch_kind"]
            origin_mjd = supernova["epoch_mjd"]
            estimated_origin = not np.isfinite(origin_mjd)
            if estimated_origin:
                origin_mjd = estimate_bmax(of_supernova)
                if not np.isfinite(origin_mjd):
                    continue
            observed = supernova_photometry(
                of_supernova, extinction_ratio, float(reddening[supernova["sn"]]), origin_mjd
            )
            bandpasses = sorted(set(observed["bandpass"]), key=csp.BANDPASS_ORDER.index)
            # The redshift is the model's, and for the type II release it is the one inverted from
            # the published distance modulus, so the model comes out at that distance without the
            # brightness ever passing through a Hubble-flow redshift. See `load_type_ii_metadata`.
            published_distance = np.isfinite(supernova["distance_modulus"])
            redshift = float(supernova["redshift"] if published_distance else supernova["redshift_cmb"])
            realizations = draw_class_population(
                catalog,
                supernova["label"],
                redshift,
                arguments.realizations,
                random_generator,
                source_by_template_index,
            )
            model_days, model_curves = model_band_curves(realizations, bandpasses, redshift, epoch_kind)
            drawn_sources = sorted({one["source_name"] for one in realizations})
            sources = (
                ", ".join(drawn_sources)
                if len(drawn_sources) <= 4
                else f"{len(drawn_sources)} templates, {drawn_sources[0]} ... {drawn_sources[-1]}"
            )
            distance_note = (
                f"mu = {supernova['distance_modulus']:.2f} (published)"
                if published_distance
                else f"z = {supernova['redshift']:.4f} (CMB {redshift:.4f})"
            )
            figure = plot_supernova(
                supernova,
                observed,
                model_days,
                model_curves,
                distance_note,
                float(reddening[supernova["sn"]]),
                sources,
            )
            pdf.savefig(figure)
            plt.close(figure)

            # A supernova whose time origin had to be read off the data keeps its figure and leaves
            # the summary: the brightness survives a two-day error in that origin, the shape does
            # not, and a residual that mixes the two measures nothing. The Hubble-flow cut applies
            # only where the distance came from the redshift.
            too_close = not published_distance and redshift < MINIMUM_HUBBLE_FLOW_REDSHIFT
            if too_close or estimated_origin:
                continue
            brightness = BRIGHTNESS_STATISTIC[epoch_kind]
            shape = SHAPE_STATISTIC[epoch_kind]
            for bandpass in bandpasses:
                points = observed[observed["bandpass"] == bandpass]
                days = points["days"].to_numpy()
                magnitudes = points["mag_corrected"].to_numpy()
                median, _, _ = percentile_curve(model_curves[bandpass])
                observed_brightness = brightness(days, magnitudes, redshift)
                model_brightness = brightness(model_days, median, redshift)
                residual_rows.append(
                    {
                        "sn": supernova["sn"],
                        "label": supernova["label"],
                        "epoch_kind": epoch_kind,
                        "letter": csp.BANDPASS_LETTER[bandpass],
                        "bandpass": bandpass,
                        "redshift": redshift,
                        "observed_peak": observed_brightness,
                        "model_peak": model_brightness,
                        "residual": observed_brightness - model_brightness,
                        "observed_decline": shape(days, magnitudes, redshift),
                        "model_decline": shape(model_days, median, redshift),
                    }
                )
            print(
                f"  {supernova['sn']:10s} {supernova['label']:7s} z={redshift:.4f}",
                flush=True,
            )

        residuals = pd.DataFrame(residual_rows)
        residuals["decline_residual"] = residuals["observed_decline"] - residuals["model_decline"]
        residuals.to_csv(arguments.output.with_suffix(".csv"), index=False)
        for column, names, title in (
            (
                "residual",
                BRIGHTNESS_NAME,
                "brightness residual, observed minus the median of the izc population\n"
                "positive = the observation is fainter than the model, which is where host "
                "extinction pushes it",
            ),
            (
                "decline_residual",
                SHAPE_NAME,
                "shape residual, observed minus model\n"
                "positive = the observation fades faster than the model",
            ),
        ):
            summary = plot_residual_summary(residuals, column, names, title)
            pdf.savefig(summary)
            plt.close(summary)

    table = (
        residuals.groupby(["label", "letter"])
        .agg(
            supernovae=("residual", "count"),
            peak_median=("residual", "median"),
            peak_scatter=("residual", "std"),
            declines=("decline_residual", "count"),
            decline_median=("decline_residual", "median"),
        )
        .reset_index()
    )
    table["letter"] = pd.Categorical(table["letter"], LETTER_ORDER, ordered=True)
    print(table.sort_values(["label", "letter"]).to_string(index=False))
    print(f"\n{arguments.output}")


if __name__ == "__main__":
    main()
