"""Kilonova version of the Hourglass token diagram (hourglass_token_diagram.pdf).

Left panel keeps that diagram's style: one panel, epoch shading, filled circles with 1 sigma
error bars for detections, open downward triangles for 5 sigma upper limits, filled squares for
visits the survey cadence skipped. Right panel is redrawn from scratch for this project's own
token schema (``kilonova.datasets.openuniverse``): band names are the current Roman filters
(Z087/Y106/J129/H158/F184, not the old R/Z/Y/J placeholders), token types are d/u/n (detection /
5 sigma upper limit / instrumental gap), and the global token is [Z] with the continuous redshift
plus its low/high regime bin.

Rather than one of the 15 canned draws in kilonova_windows_demo.hdf5 (all fairly faint), the
example here is chosen deliberately from the LANL grid: the "massive, fast ejecta" simulation
(mass_dynamical=0.1, velocity_dynamical=0.3, mass_wind=0.1, velocity_wind=0.3 -- panel (b) of
kilonova_examples.png, the brightest of the four grid extremes) seen face-on, redshifted to
z=0.03. Its photometry is computed with the same recipe as plot_kilonova_examples.py
(``photometry_for_angle``, parametrised here on redshift instead of that script's module-level
constant). The per-epoch "observed" mask (which band gets skipped at which visit) is copied from
a real deep-tier injection (kilonova_windows_demo.hdf5 group "9") so the cadence stays realistic
even though the brightness is picked by hand.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, Rectangle
from plot_kilonova_examples import load_ejecta_catalog, select_simulation

from kilonova.config import load_paths
from kilonova.datasets.openuniverse import REDSHIFT_THRESHOLD
from kilonova.photometry.roman_noise import (
    SNR_DETECTION,
    build_tier_constants,
    collecting_area_cm2,
    flux_error_electrons,
    limiting_magnitude_5sigma,
    source_flux_electrons,
)
from kilonova.photometry.spectra import ALL_ROMAN_BANDS, magnitudes_for_bands
from kilonova.simulation.early_windows import (
    load_lanl_catalog_metadata,
    load_lanl_wavelength_grid,
    load_simulation_spectra,
)

REDSHIFT = 0.05
NOISE_SEED = 20260722
TIER = "deep"
ANGLE_INDEX = 0  # face-on: brightest viewing angle
EJECTA_PARAMETERS = dict(mass_dynamical=0.1, velocity_dynamical=0.3, mass_wind=0.1, velocity_wind=0.3)
EJECTA_RUN_TYPE, EJECTA_WIND = "TS", "wind1"

# Deep-tier visit cadence copied from a real injection (kilonova_windows_demo.hdf5 group "9"):
# which (band, epoch) pairs the survey actually visited, independent of what the source is doing.
CADENCE_MASK = {
    ("F184", 0.0): True,
    ("H158", 0.0): False,
    ("J129", 0.0): True,
    ("Y106", 0.0): False,
    ("Z087", 0.0): True,
    ("F184", 5.0): True,
    ("H158", 5.0): False,
    ("J129", 5.0): True,
    ("Y106", 5.0): False,
    ("Z087", 5.0): True,
    ("F184", 10.0): False,
    ("H158", 10.0): True,
    ("J129", 10.0): False,
    ("Y106", 10.0): True,
    ("Z087", 10.0): True,
}

BAND_COLORS = {
    band: plt.cm.turbo(position)
    for band, position in zip(ALL_ROMAN_BANDS, np.linspace(0.05, 0.95, len(ALL_ROMAN_BANDS)), strict=True)
}


def photometry_for_angle(spectra, time_days, wavelength_rest_aa, angle_index, constants, redshift, rng):
    """Roman photometry of one (simulation, angle) at ``redshift`` over the full LANL time grid.

    Copy of the function of the same name in plot_kilonova_examples.py, parametrised on redshift
    instead of that script's module-level constant, so this diagram can pick its own z.
    """
    rows = []
    for time_index, rest_time in enumerate(time_days):
        flux_rest = spectra.get((angle_index, time_index))
        if flux_rest is None:
            continue
        magnitudes = magnitudes_for_bands(wavelength_rest_aa, flux_rest, redshift, ALL_ROMAN_BANDS)
        for band in constants["bands"]:
            mag_true = float(magnitudes[band])
            if not np.isfinite(mag_true):
                continue
            exposure = constants["exposure_time"][band]
            zeropoint = constants["zeropoint"][band]
            flux_true = source_flux_electrons(mag_true, exposure, zeropoint)
            flux_error = flux_error_electrons(flux_true, constants["noise_floor_variance"][band])
            snr = flux_true / flux_error
            flux_observed = flux_true + rng.normal(0.0, flux_error)
            rows.append(
                {
                    "days_since_merger": float(rest_time) * (1.0 + redshift),
                    "band_letter": band[0],
                    "mag_observed": (
                        zeropoint - 2.5 * np.log10(flux_observed / exposure / collecting_area_cm2())
                        if flux_observed > 0
                        else np.nan
                    ),
                    "mag_err": 1.0857 / snr,
                    "detected": bool(snr >= SNR_DETECTION),
                    "mag_limit_5sigma": limiting_magnitude_5sigma(flux_error, exposure, zeropoint),
                }
            )
    return pd.DataFrame(rows)


def build_example(lanl_spectra_path):
    catalog = load_lanl_catalog_metadata(lanl_spectra_path)
    ejecta_catalog = load_ejecta_catalog(lanl_spectra_path)
    wavelength_rest_aa = load_lanl_wavelength_grid(lanl_spectra_path)
    constants = build_tier_constants(TIER)
    rng = np.random.default_rng(NOISE_SEED)

    simulation_id = select_simulation(ejecta_catalog, EJECTA_RUN_TYPE, EJECTA_WIND, EJECTA_PARAMETERS)
    simulation_rows = catalog[catalog["simulation_id"] == simulation_id]
    time_days = (
        simulation_rows.drop_duplicates("time_index").sort_values("time_index")["time_days"].to_numpy()
    )
    spectra = load_simulation_spectra(simulation_id, lanl_spectra_path)
    photometry = photometry_for_angle(
        spectra, time_days, wavelength_rest_aa, ANGLE_INDEX, constants, REDSHIFT, rng
    )

    band_column, epoch_column = [], []
    mag_observed, mag_err, mag_limit_5sigma, detected, observed = [], [], [], [], []
    for (band, epoch), is_observed in CADENCE_MASK.items():
        band_column.append(band)
        epoch_column.append(epoch)
        observed.append(is_observed)
        if not is_observed:
            mag_observed.append(np.nan)
            mag_err.append(np.nan)
            mag_limit_5sigma.append(np.nan)
            detected.append(False)
            continue
        band_rows = photometry[photometry["band_letter"] == band[0]]
        nearest = band_rows.iloc[(band_rows["days_since_merger"] - epoch).abs().argsort().iloc[0]]
        mag_observed.append(nearest["mag_observed"])
        mag_err.append(nearest["mag_err"])
        mag_limit_5sigma.append(nearest["mag_limit_5sigma"])
        detected.append(bool(nearest["detected"] and np.isfinite(nearest["mag_observed"])))

    return {
        "band": np.array(band_column),
        "days_since_detection": np.array(epoch_column),
        "mag_observed": np.array(mag_observed, dtype=float),
        "mag_err": np.array(mag_err, dtype=float),
        "mag_limit_5sigma": np.array(mag_limit_5sigma, dtype=float),
        "detected": np.array(detected, dtype=bool),
        "observed": np.array(observed, dtype=bool),
        "redshift": REDSHIFT,
    }


def token_types(example):
    return np.where(~example["observed"], "n", np.where(example["detected"], "d", "u"))


NOT_OBSERVED_STEP_MAG = 0.2  # vertical spacing between stacked "not observed" squares


def plot_light_curve(ax, example, band_order):
    token_type = token_types(example)
    epochs = np.unique(example["days_since_detection"])
    epoch_width = 0.6
    for epoch in epochs:
        ax.axvspan(epoch - epoch_width / 2, epoch + epoch_width / 2, color="0.92", zorder=0)

    brightest_mag = np.nanmin(example["mag_observed"][token_type == "d"])

    for band in band_order:
        color = BAND_COLORS[band]
        band_mask = example["band"] == band
        for epoch_index in np.where(band_mask)[0]:
            time = example["days_since_detection"][epoch_index]
            kind = token_type[epoch_index]
            if kind == "d":
                ax.errorbar(
                    time,
                    example["mag_observed"][epoch_index],
                    yerr=example["mag_err"][epoch_index],
                    fmt="o",
                    ms=8.8,
                    color=color,
                    ecolor=color,
                    elinewidth=1.8,
                    capsize=0,
                    mec="k",
                    mew=0.9,
                )
            elif kind == "u":
                ax.scatter(
                    time,
                    example["mag_limit_5sigma"][epoch_index],
                    marker="v",
                    s=117,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=1.8,
                )

    # "Not observed" squares carry no magnitude information, so they are stacked at fixed offsets
    # above the brightest detected point instead of at mag_true: all squares of the same stacking
    # rank sit at the same height across epochs, and a second square sharing an epoch with the
    # first is offset by one more step (0.2 mag, so 0.4 mag total) to avoid overlapping it.
    for epoch in epochs:
        not_observed_indices = [
            index
            for index in np.where(example["days_since_detection"] == epoch)[0]
            if token_type[index] == "n"
        ]
        not_observed_indices.sort(key=lambda index: band_order.index(example["band"][index]))
        for rank, index in enumerate(not_observed_indices, start=1):
            ax.scatter(
                epoch,
                brightest_mag - rank * NOT_OBSERVED_STEP_MAG,
                marker="s",
                s=61,
                color=BAND_COLORS[example["band"][index]],
                edgecolor="k",
                linewidth=0.8,
            )

    ax.invert_yaxis()
    ax.set_xlabel("days since first detection")
    ax.set_ylabel("AB magnitude")
    ax.set_title("Observed light curve", pad=40)
    label_transform = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    for epoch_index, epoch in enumerate(epochs, start=1):
        ax.text(
            epoch,
            1.03,
            f"epoch {epoch_index}",
            transform=label_transform,
            ha="center",
            va="bottom",
            fontsize=9,
            color="0.3",
        )
    return [
        Line2D([], [], marker="o", ls="", color=BAND_COLORS[band], mec="k", mew=0.9, ms=8.8, label=band)
        for band in band_order
    ]


def cell_text(example, token_type, index):
    if token_type[index] == "d":
        return f"{example['mag_observed'][index]:.1f}±{example['mag_err'][index]:.1f}"
    if token_type[index] == "u":
        return f"<{example['mag_limit_5sigma'][index]:.1f}"
    return "×"


def draw_token_table(ax, example, band_order):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Tokenised input to transformer")

    token_type = token_types(example)
    epochs = np.unique(example["days_since_detection"])
    deltas = epochs - epochs[0]

    row_label_width = 1.3
    column_width = 2.3
    column_gap = 0.05
    column_left = [row_label_width + column_index * column_width for column_index in range(len(epochs))]
    global_gap = 0.6
    global_column_left = column_left[-1] + column_width + global_gap

    table_top = 8.6
    table_bottom = 3.0
    row_height = (table_top - table_bottom) / len(band_order)
    row_gap = 0.05
    row_top = [table_top - row_index * row_height for row_index in range(len(band_order))]

    for column_index, delta in enumerate(deltas):
        ax.text(
            column_left[column_index] + column_width / 2,
            9.5,
            f"epoch {column_index + 1}\nΔt={delta:.0f} d",
            ha="center",
            va="top",
            fontsize=10,
        )
    ax.text(global_column_left + column_width / 2, 9.5, "global\ntoken", ha="center", va="top", fontsize=10)

    for row_index, band in enumerate(band_order):
        color = BAND_COLORS[band]
        top = row_top[row_index]
        ax.text(
            row_label_width - 0.2,
            top - row_height / 2,
            band,
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        band_mask = example["band"] == band
        for column_index, epoch in enumerate(epochs):
            matches = np.where(band_mask & (example["days_since_detection"] == epoch))[0]
            if len(matches) == 0:
                continue
            index = matches[0]
            rectangle = Rectangle(
                (column_left[column_index] + column_gap / 2, top - row_height + row_gap / 2),
                column_width - column_gap,
                row_height - row_gap,
                facecolor="none",
                edgecolor=color,
                linewidth=2.2,
            )
            ax.add_patch(rectangle)
            ax.text(
                column_left[column_index] + column_width / 2,
                top - row_height / 2,
                cell_text(example, token_type, index),
                ha="center",
                va="center",
                fontsize=11,
            )

    redshift_regime = "low" if example["redshift"] < REDSHIFT_THRESHOLD else "high"
    global_box_size = 1.8
    global_top = (table_top + table_bottom) / 2 + global_box_size / 2
    global_box = Rectangle(
        (global_column_left + (column_width - global_box_size) / 2, global_top - global_box_size),
        global_box_size,
        global_box_size,
        facecolor="none",
        edgecolor="0.2",
        linewidth=2.2,
    )
    ax.add_patch(global_box)
    ax.text(
        global_column_left + column_width / 2,
        global_top - global_box_size / 2,
        f"[Z]\nz={example['redshift']:.3f}\n{redshift_regime}",
        ha="center",
        va="center",
        fontsize=10,
    )

    draw_token_schema(ax, example, token_type, band_order)


def draw_token_schema(ax, example, token_type, band_order):
    example_index = np.where(token_type == "d")[0][0]
    fields = [
        ("Δt", f"{example['days_since_detection'][example_index]:.0f} d"),
        ("band", example["band"][example_index]),
        ("type (d/u/n)", token_type[example_index]),
        ("mag", f"{example['mag_observed'][example_index]:.1f}"),
        ("σ_mag", f"{example['mag_err'][example_index]:.2f}"),
    ]
    ax.text(0.4, 2.35, "per-visit token (one per band × epoch)", fontsize=9, color="0.3", style="italic")

    box_width = 1.5
    gap = 0.5
    start_x = 0.4
    for field_index, (label, value) in enumerate(fields):
        left = start_x + field_index * (box_width + gap)
        ax.text(left + box_width / 2, 1.95, label, ha="center", va="bottom", fontsize=9, color="0.3")
        rectangle = Rectangle((left, 0.7), box_width, 1.1, facecolor="0.95", edgecolor="0.4", linewidth=1.4)
        ax.add_patch(rectangle)
        ax.text(left + box_width / 2, 1.25, value, ha="center", va="center", fontsize=10)
        if field_index < len(fields) - 1:
            arrow_x = left + box_width
            ax.add_patch(
                FancyArrow(
                    arrow_x + 0.08,
                    1.25,
                    gap - 0.16,
                    0,
                    width=0.03,
                    head_width=0.2,
                    head_length=0.12,
                    length_includes_head=True,
                    color="0.4",
                )
            )

    token_left = start_x + len(fields) * (box_width + gap)
    ax.add_patch(
        FancyArrow(
            token_left - gap + 0.08,
            1.25,
            gap - 0.16,
            0,
            width=0.03,
            head_width=0.2,
            head_length=0.12,
            length_includes_head=True,
            color="0.4",
        )
    )
    token_box = Rectangle((token_left, 0.7), 1.4, 1.1, facecolor="#c9d9f2", edgecolor="0.2", linewidth=1.4)
    ax.add_patch(token_box)
    ax.text(token_left + 0.7, 1.25, "token", ha="center", va="center", fontsize=10, fontweight="bold")


def main():
    paths = load_paths()
    example = build_example(str(paths.lanl_spectra))
    band_order = [band for band in ALL_ROMAN_BANDS if band in example["band"]]

    figure, (left_axis, right_axis) = plt.subplots(
        1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.0, 1.1]}
    )
    band_handles = plot_light_curve(left_axis, example, band_order)
    draw_token_table(right_axis, example, band_order)

    marker_handles = [
        Line2D([], [], marker="o", ls="", color="0.3", mec="k", mew=0.9, ms=8.8, label="detection"),
        Line2D(
            [],
            [],
            marker="v",
            ls="",
            markerfacecolor="none",
            markeredgecolor="0.3",
            mew=1.8,
            ms=10.8,
            label="5σ upper limit",
        ),
        Line2D(
            [], [], marker="s", ls="", color="0.3", mec="k", mew=0.8, ms=7.8, label="not observed (cadence)"
        ),
    ]

    left_axis.text(
        0.5,
        1.54,
        f"Kilonova token diagram, z={example['redshift']:.3f}",
        transform=left_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )
    band_legend = left_axis.legend(
        handles=band_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.36),
        bbox_transform=left_axis.transAxes,
        ncol=len(band_handles),
        frameon=False,
        fontsize=9,
    )
    left_axis.add_artist(band_legend)
    left_axis.legend(
        handles=marker_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.22),
        bbox_transform=left_axis.transAxes,
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    figure.subplots_adjust(left=0.05, right=0.98, top=0.60, bottom=0.08, wspace=0.28)

    output_path = paths.output_dir / "kilonova_token_diagram.png"
    figure.savefig(output_path, dpi=180)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
