"""Kilonova example light curves for the presentation: LANL grid extremes at fixed redshift.

Same visual style as the OpenUniverse contaminant grid (one panel per object, AB magnitude on an
inverted axis, one turbo colour per Roman band, filled circles = detections with 1 sigma error bars,
open triangles = 5 sigma upper limits). Rows are the ejecta-parameter extremes of the grid, columns
are the viewing angle.

The x axis is days since merger in the observer frame, not days from peak: the kilonova peaks inside
the first day in the blue bands and several days later in the red ones, so a single "peak" epoch is
not well defined across bands.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from kilonova.config import load_paths
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

REDSHIFT = 0.01
NOISE_SEED = 20260721
MAX_UPPER_LIMITS_PER_RUN = 2  # once a band is gone, two triangles already say so

BAND_ORDER = ["R", "Z", "Y", "J", "H", "F"]
BAND_COLORS = {
    band: plt.cm.turbo(position)
    for band, position in zip(BAND_ORDER, np.linspace(0.05, 0.95, len(BAND_ORDER)), strict=True)
}

# Four panels spanning the LANL grid (masses in Msun, velocities in c): two per survey tier, and
# one per ejecta morphology (run_type = dynamical shape, wind = wind shape) since the injection
# samples all four combinations uniformly.
PANELS = [
    (
        "faint, slow ejecta",
        "deep",
        0,
        "TP",
        "wind1",
        dict(mass_dynamical=0.001, velocity_dynamical=0.05, mass_wind=0.001, velocity_wind=0.05),
    ),
    (
        "massive, fast ejecta",
        "deep",
        0,
        "TS",
        "wind1",
        dict(mass_dynamical=0.1, velocity_dynamical=0.3, mass_wind=0.1, velocity_wind=0.3),
    ),
    (
        "light dynamical, massive wind",
        "wide",
        30,
        "TP",
        "wind2",
        dict(mass_dynamical=0.003, velocity_dynamical=0.3, mass_wind=0.1, velocity_wind=0.05),
    ),
    (
        "massive dynamical, light wind",
        "wide",
        40,
        "TS",
        "wind2",
        dict(mass_dynamical=0.1, velocity_dynamical=0.05, mass_wind=0.003, velocity_wind=0.3),
    ),
]


EJECTA_COLUMNS = [
    "simulation_id",
    "run_type",
    "wind",
    "mass_dynamical",
    "velocity_dynamical",
    "mass_wind",
    "velocity_wind",
]


def load_ejecta_catalog(lanl_spectra_path):
    """One row per simulation with its ejecta parameters (load_lanl_catalog_metadata only returns
    the time/angle columns)."""
    table = pq.read_table(lanl_spectra_path, columns=EJECTA_COLUMNS)
    return table.to_pandas().drop_duplicates("simulation_id").reset_index(drop=True)


def select_simulation(ejecta_catalog, run_type, wind, parameters):
    selection = ejecta_catalog[
        (ejecta_catalog["run_type"] == run_type)
        & (ejecta_catalog["wind"] == wind)
        & np.isclose(ejecta_catalog["mass_dynamical"], parameters["mass_dynamical"])
        & np.isclose(ejecta_catalog["velocity_dynamical"], parameters["velocity_dynamical"])
        & np.isclose(ejecta_catalog["mass_wind"], parameters["mass_wind"])
        & np.isclose(ejecta_catalog["velocity_wind"], parameters["velocity_wind"])
    ]
    if selection.empty:
        raise KeyError(f"no LANL simulation matches {run_type}/{wind} {parameters}")
    return int(selection["simulation_id"].iloc[0])


def photometry_for_angle(spectra, time_days, wavelength_rest_aa, angle_index, constants, rng):
    """Roman photometry of one (simulation, angle) at REDSHIFT over the full LANL time grid.

    The noise recipe is the pipeline one (build_window_from_model), applied here at every LANL epoch
    instead of only at the survey visits, so the example curves are densely sampled.
    """
    rows = []
    for time_index, rest_time in enumerate(time_days):
        flux_rest = spectra.get((angle_index, time_index))
        if flux_rest is None:
            continue
        magnitudes = magnitudes_for_bands(wavelength_rest_aa, flux_rest, REDSHIFT, ALL_ROMAN_BANDS)
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
                    "days_since_merger": float(rest_time) * (1.0 + REDSHIFT),
                    "band_letter": band[0],
                    "mag_true": mag_true,
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


def trim_upper_limits(band_rows):
    """Keep only the first MAX_UPPER_LIMITS_PER_RUN triangles of each contiguous run of
    non-detections: a long tail of identical limits carries no extra information."""
    band_rows = band_rows.sort_values("days_since_merger")
    is_upper_limit = ~band_rows["detected"].to_numpy()
    position_in_run = np.zeros(len(band_rows), dtype=int)
    counter = 0
    for index, upper_limit in enumerate(is_upper_limit):
        counter = counter + 1 if upper_limit else 0
        position_in_run[index] = counter
    return band_rows[is_upper_limit & (position_in_run <= MAX_UPPER_LIMITS_PER_RUN)]


def cut_after_fade(band_rows):
    """Drop everything past the first run of MAX_UPPER_LIMITS_PER_RUN non-detections that happens
    after the band peaked: once it has faded, the late points that scatter back above 5 sigma are
    noise, not signal. Runs before the peak are kept (the rise starts below the limit)."""
    band_rows = band_rows.sort_values("days_since_merger")
    detections = band_rows[band_rows["detected"]]
    if detections.empty:
        return band_rows
    peak_position = band_rows.index.get_loc(detections["mag_observed"].idxmin())
    is_upper_limit = ~band_rows["detected"].to_numpy()
    counter = 0
    for position in range(peak_position, len(band_rows)):
        counter = counter + 1 if is_upper_limit[position] else 0
        if counter == MAX_UPPER_LIMITS_PER_RUN:
            return band_rows.iloc[: position + 1]
    return band_rows


def plot_panel(photometry, ax, title):
    for band in BAND_ORDER:
        band_rows = photometry[photometry["band_letter"] == band]
        if band_rows.empty:
            continue
        band_rows = cut_after_fade(band_rows)
        color = BAND_COLORS[band]
        detections = band_rows[band_rows["detected"] & band_rows["mag_observed"].notna()]
        upper_limits = trim_upper_limits(band_rows)
        if len(detections) > 0:
            ax.errorbar(
                detections["days_since_merger"],
                detections["mag_observed"],
                yerr=detections["mag_err"],
                fmt="o",
                ms=3,
                color=color,
                ecolor=color,
                elinewidth=0.6,
                capsize=0,
                mec="k",
                mew=0.25,
            )
        if len(upper_limits) > 0:
            ax.scatter(
                upper_limits["days_since_merger"],
                upper_limits["mag_limit_5sigma"],
                marker="v",
                s=16,
                facecolor="none",
                edgecolor=color,
                linewidth=0.7,
                alpha=0.7,
            )
    visible = pd.concat(
        [
            cut_after_fade(photometry[photometry["band_letter"] == band])
            for band in BAND_ORDER
            if (photometry["band_letter"] == band).any()
        ]
    )
    shown = pd.concat(
        [
            visible.loc[visible["detected"], "mag_observed"],
            pd.concat(
                [
                    trim_upper_limits(visible[visible["band_letter"] == band])["mag_limit_5sigma"]
                    for band in BAND_ORDER
                ]
            ),
        ]
    ).dropna()
    margin = max(0.2, 0.05 * (shown.max() - shown.min()))
    ax.set_ylim(shown.max() + margin, shown.min() - margin)
    # Cut the time axis where the last plotted point is: past it there is nothing left to show.
    last_day = visible["days_since_merger"].max()
    ax.set_xlim(-0.03 * last_day, last_day * 1.05)
    ax.text(0.97, 0.95, title, transform=ax.transAxes, ha="right", va="top", fontsize=9)


def main():
    paths = load_paths()
    lanl_spectra_path = str(paths.lanl_spectra)
    catalog = load_lanl_catalog_metadata(lanl_spectra_path)
    ejecta_catalog = load_ejecta_catalog(lanl_spectra_path)
    wavelength_rest_aa = load_lanl_wavelength_grid(lanl_spectra_path)
    rng = np.random.default_rng(NOISE_SEED)
    constants_by_tier = {tier: build_tier_constants(tier) for tier in {panel[1] for panel in PANELS}}

    # No shared axes: each kilonova spans a different magnitude range and a common scale flattens
    # the faint ones.
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    parameter_rows = []
    for panel_letter, ax, (panel_name, tier, angle_index, run_type, wind, parameters) in zip(
        "abcd", axes.ravel(), PANELS, strict=True
    ):
        constants = constants_by_tier[tier]
        simulation_id = select_simulation(ejecta_catalog, run_type, wind, parameters)
        simulation_rows = catalog[catalog["simulation_id"] == simulation_id]
        time_days = (
            simulation_rows.drop_duplicates("time_index").sort_values("time_index")["time_days"].to_numpy()
        )
        spectra = load_simulation_spectra(simulation_id, lanl_spectra_path)
        photometry = photometry_for_angle(spectra, time_days, wavelength_rest_aa, angle_index, constants, rng)
        # Only a short tag in the panel; the full parameters go to the companion markdown table.
        plot_panel(photometry, ax, f"({panel_letter}) {panel_name}\n{tier.upper()}, {run_type}/{wind}")
        parameter_rows.append(
            {
                "panel": panel_letter,
                "name": panel_name,
                "tier": tier.upper(),
                "simulation_id": simulation_id,
                "angle_index": angle_index,
                "run_type": run_type,
                "wind": wind,
                **parameters,
            }
        )
        ax.set_ylabel("AB magnitude")
        ax.set_xlabel("days since merger (observer frame)")

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BAND_COLORS[band], mec="k", mew=0.3, label=band)
        for band in BAND_ORDER
    ]
    figure.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(BAND_ORDER), frameon=False
    )
    figure.suptitle(f"LANL kilonovae through Roman, z={REDSHIFT:.2f}", y=0.998, fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output_path = paths.output_dir / "kilonova_examples.png"
    figure.savefig(output_path, dpi=180)

    # Backup table: what each panel letter actually is, for the questions after the talk.
    parameters_path = output_path.with_name("kilonova_examples_parameters.md")
    parameters_path.write_text(pd.DataFrame(parameter_rows).to_markdown(index=False) + "\n")
    print(f"wrote {output_path}\nwrote {parameters_path}")


if __name__ == "__main__":
    main()
