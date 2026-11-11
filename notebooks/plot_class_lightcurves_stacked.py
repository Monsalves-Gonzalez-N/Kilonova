"""Stacked edition of the class light-curve figure: every class in ONE panel per tier, each class
shifted by a constant magnitude offset (spectra-atlas style), instead of one block per class.

Reuses the selection and photometry of openuniverse_hdf5_lightcurve_error.ipynb (cell 12) exactly
as make_class_lightcurves_gif.py does, so the objects are the same ones as in
paper_class_lightcurves. Selection costs minutes, so the rows are cached as a pickle.
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FormatStrFormatter

NOTEBOOK_PATH = Path(__file__).with_name("openuniverse_hdf5_lightcurve_error.ipynb")
NOTEBOOK_CELLS = [1, 3, 11, 12]
CELL_12_STOP_MARKER = "class_lightcurve_rows, paper_figure_objects, class_census"

OFFSET_GAP_MAG = 1.0
SYMLOG_LINEAR_DAYS = 20.0


def load_notebook_namespace(hdf5_path, catalog_path):
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    namespace = {"__name__": "__notebook__", "display": lambda *args, **kwargs: None}
    for cell_index in NOTEBOOK_CELLS:
        source = "".join(notebook["cells"][cell_index]["source"])
        if CELL_12_STOP_MARKER in source:
            source = source[: source.index(CELL_12_STOP_MARKER)]
        exec(compile(source, f"<cell {cell_index}>", "exec"), namespace)
        if cell_index == 1:
            namespace["HDF5_PATH"] = str(hdf5_path)
            namespace["CATALOG_PATH"] = str(catalog_path)
    plt.close("all")
    return namespace


def visible_photometry(namespace, photometry, tier):
    """Same trimming as plot_class_panel: observed rows, cut after the last detection and after the
    fade of each band, with at most MAX_UPPER_LIMITS_PER_RUN consecutive upper limits."""
    observed = namespace["cut_after_last_detection"](photometry[photometry["observed"]])
    bands = namespace["CONSTANTS_BY_TIER"][tier]["bands"]
    detections_by_band = {}
    limits_by_band = {}
    for band in bands:
        band_rows = namespace["cut_after_fade"](observed[observed["band"] == band])
        if band_rows.empty:
            continue
        detections_by_band[band] = band_rows[band_rows["detected"] & band_rows["mag_observed"].notna()]
        limits_by_band[band] = namespace["trim_upper_limits"](band_rows)
    return detections_by_band, limits_by_band


def class_extent(detections_by_band, limits_by_band):
    magnitudes = pd.concat(
        [frame["mag_observed"] for frame in detections_by_band.values()]
        + [frame["mag_limit_5sigma"] for frame in limits_by_band.values()]
    ).dropna()
    days = pd.concat(
        [frame["days_since_detection"] for frame in detections_by_band.values()]
        + [frame["days_since_detection"] for frame in limits_by_band.values()]
    )
    return magnitudes.min(), magnitudes.max(), days.min(), days.max()


def draw_stacked(namespace, rows, style, time_scale, output_path):
    tiers = namespace["TIER_COLUMNS"]
    band_color = namespace["BAND_COLOR"]
    prepared = {
        tier: [visible_photometry(namespace, entry["by_tier"][tier], tier) for entry in rows]
        for tier in tiers
    }
    # One offset per class shared by both tiers, so deep and wide of the same object sit at the same
    # height and the depth difference stays readable. The normalisation is the brightest deep
    # detection; the stack grows downward with the full faint range of the class plus a gap.
    peak_by_class = []
    span_by_class = []
    for class_index in range(len(rows)):
        extents = [class_extent(*prepared[tier][class_index]) for tier in tiers]
        peak = min(extent[0] for extent in extents)
        faintest = max(extent[1] for extent in extents)
        peak_by_class.append(peak)
        span_by_class.append(faintest - peak)
    offsets = []
    running = 0.0
    for span in span_by_class:
        offsets.append(running)
        running += span + OFFSET_GAP_MAG

    figure, axes = plt.subplots(1, len(tiers), figsize=style["figsize"], sharex=True, sharey=True)
    day_max = 0.0
    for tier, ax in zip(tiers, axes, strict=True):
        for class_index in range(len(rows)):
            detections_by_band, limits_by_band = prepared[tier][class_index]
            shift = offsets[class_index] - peak_by_class[class_index]
            for band, detections in detections_by_band.items():
                if len(detections) == 0:
                    continue
                ax.errorbar(
                    detections["days_since_detection"]
                    + (style["time_shift"] if time_scale == "log" else 0.0),
                    detections["mag_observed"] + shift,
                    yerr=detections["mag_err"],
                    fmt="o",
                    ms=style["marker_size"],
                    color=band_color[band],
                    ecolor=band_color[band],
                    elinewidth=style["error_linewidth"],
                    capsize=0,
                    mec="k",
                    mew=style["marker_edgewidth"],
                    zorder=3,
                )
                day_max = max(day_max, detections["days_since_detection"].max())
            for band, limits in limits_by_band.items():
                if len(limits) == 0:
                    continue
                ax.scatter(
                    limits["days_since_detection"] + (style["time_shift"] if time_scale == "log" else 0.0),
                    limits["mag_limit_5sigma"] + shift,
                    marker="v",
                    s=style["limit_size"],
                    facecolor="none",
                    edgecolor=band_color[band],
                    linewidth=style["limit_linewidth"],
                    alpha=0.8,
                    zorder=3,
                )
                day_max = max(day_max, limits["days_since_detection"].max())
            ax.axhline(offsets[class_index], color="0.85", lw=0.6, ls=":", zorder=1)
        ax.set_title(f"{tier.capitalize()} tier", fontsize=style["title_size"], pad=8)
        ax.minorticks_on()
        ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=style["tick_labelsize"])

    for class_index, entry in enumerate(rows):
        axes[-1].annotate(
            f"{entry['label']}, z={entry['z_CMB']:.2f}",
            xy=(1.0, offsets[class_index]),
            xycoords=("axes fraction", "data"),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=style["panel_labelsize"],
            annotation_clip=False,
        )

    axes[0].set_ylim(offsets[-1] + span_by_class[-1] + 0.6, -0.8)
    axes[0].set_ylabel("AB magnitude, normalised to the deep peak + offset", fontsize=style["axis_labelsize"])
    if time_scale == "log":
        axes[0].set_xscale("log")
        axes[0].set_xlim(style["time_shift"] * 0.8, (day_max + style["time_shift"]) * 1.15)
        xlabel = f"Days since first detection (deep) + {style['time_shift']:g}"
    elif time_scale == "symlog":
        # Linear up to SYMLOG_LINEAR_DAYS (the whole KN) and logarithmic beyond: the linear stretch
        # gets the width of one decade, so the KN is as wide as 20-200 d of a SN II.
        axes[0].set_xscale("symlog", linthresh=SYMLOG_LINEAR_DAYS, linscale=1.0)
        axes[0].set_xlim(-0.08 * SYMLOG_LINEAR_DAYS, day_max * 1.15)
        axes[0].xaxis.set_major_locator(FixedLocator([0, 5, 10, 20, 50, 100, 200, 400]))
        axes[0].xaxis.set_major_formatter(FormatStrFormatter("%g"))
        axes[0].xaxis.set_minor_locator(FixedLocator([15, 30, 40, 60, 70, 80, 90, 150, 300]))
        for ax in axes:
            ax.axvline(SYMLOG_LINEAR_DAYS, color="0.75", lw=0.6, ls="--", zorder=1)
        xlabel = f"Days since first detection (deep); linear to {SYMLOG_LINEAR_DAYS:g} d, log beyond"
    else:
        axes[0].set_xlim(-0.03 * day_max, day_max * 1.03)
        xlabel = "Days since first detection (deep)"
    figure.supxlabel(xlabel, fontsize=style["axis_labelsize"], y=0.004)

    band_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=band_color[band],
            mec="k",
            mew=0.3,
            ms=style["marker_size"] + 1.5,
            label=band,
        )
        for band in namespace["ROMAN_BANDS_BY_WAVELENGTH"]
    ]
    symbol_handles = [
        plt.Line2D(
            [], [], marker="o", ls="", color="0.4", ms=style["marker_size"] + 1.5, label="detection (S/N ≥ 5)"
        ),
        plt.Line2D(
            [],
            [],
            marker="v",
            ls="",
            mfc="none",
            color="0.4",
            ms=style["marker_size"] + 1.5,
            label="5σ upper limit",
        ),
    ]
    figure.legend(
        handles=band_handles + symbol_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=style["legend_columns"],
        frameon=False,
        fontsize=style["legend_size"],
    )
    figure.tight_layout(rect=(0, 0.015, 0.9, 0.95))
    figure.savefig(output_path, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=style["dpi"], bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", default=None)
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--cache", default=None, help="pickle with the selected rows")
    parser.add_argument("--output-stem", default="paper_class_lightcurves_stacked")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--time-scale", choices=["linear", "log", "symlog", "all"], default="all")
    parser.add_argument("--time-shift", type=float, default=1.0, help="added to days on the log axis")
    arguments = parser.parse_args()

    from kilonova.config import load_paths

    paths = load_paths()
    hdf5_path = arguments.hdf5 or paths.openuniverse_hdf5
    catalog_path = arguments.catalog or paths.openuniverse_catalog
    output_dir = Path(arguments.output_dir or paths.output_dir)
    namespace = load_notebook_namespace(hdf5_path, catalog_path)

    cache_path = Path(arguments.cache) if arguments.cache else None
    if cache_path is not None and cache_path.exists():
        rows = pickle.loads(cache_path.read_bytes())
    else:
        rows, summary, census = namespace["build_class_lightcurve_rows"](
            kilonova_redshift=0.5, kilonova_minimum_ejecta_mass=0.2
        )
        print(summary.to_markdown(index=False))
        if cache_path is not None:
            cache_path.write_bytes(pickle.dumps(rows))

    style = dict(namespace["FIGURE_STYLES"]["paper"])
    style.update({"figsize": (180 / 25.4, 9.5), "time_shift": arguments.time_shift})
    scales = ["linear", "log", "symlog"] if arguments.time_scale == "all" else [arguments.time_scale]
    for time_scale in scales:
        draw_stacked(
            namespace, rows, style, time_scale, output_dir / f"{arguments.output_stem}_{time_scale}.pdf"
        )


if __name__ == "__main__":
    main()
