"""Observed light curve of the SAME kilonova as seen by the two HLTDS tiers, side by side.

Reuses the left panel of plot_kilonova_token_diagram.py (same event, same viewing angle, same
redshift, same noise seed) and drops the token table: one row, two columns, deep on the left and
wide on the right. The point of the pair is the tier difference alone -- band set (deep swaps the
wide R062 for F184), exposure time, and therefore depth -- so everything else is held fixed and
both panels share a magnitude axis.

The per-visit cadence mask is derived from bands_observed_at_visit instead of being written out by
hand as in the deep-only diagram: the anchor band differs between tiers (R062 wide, Z087 deep), so
the observed sequences are RZY/RJH and ZYJ/ZHF respectively.
"""

import matplotlib.pyplot as plt
import numpy as np
import plot_kilonova_token_diagram as token_diagram_module
from matplotlib.lines import Line2D
from plot_kilonova_examples import load_ejecta_catalog, select_simulation
from plot_kilonova_token_diagram import (
    ANGLE_INDEX,
    BAND_COLORS,
    EJECTA_PARAMETERS,
    EJECTA_RUN_TYPE,
    EJECTA_WIND,
    NOISE_SEED,
    REDSHIFT,
    photometry_for_angle,
    plot_light_curve,
)

from kilonova.config import load_paths
from kilonova.photometry.roman_noise import (
    BASE_CADENCE_DAYS,
    bands_observed_at_visit,
    build_tier_constants,
)
from kilonova.photometry.spectra import ALL_ROMAN_BANDS
from kilonova.simulation.early_windows import (
    load_lanl_catalog_metadata,
    load_lanl_wavelength_grid,
    load_simulation_spectra,
)

TIERS = ("deep", "wide")
EPOCHS_PER_WINDOW = 3

# The stacked "not observed" squares sit closer together in the deep-only diagram, whose panel is
# taller in magnitude units; here the two panels share a tighter axis, so the stack is opened up.
# Overridden on the module because plot_light_curve reads these as globals.
token_diagram_module.NOT_OBSERVED_STEP_MAG = 0.34
token_diagram_module.NOT_OBSERVED_BASE_OFFSET_MAG = 0.30


def cadence_mask_for_tier(constants):
    """{(band, epoch_day): observed} over the first EPOCHS_PER_WINDOW visits of the tier."""
    mask = {}
    for visit_index in range(EPOCHS_PER_WINDOW):
        epoch_day = visit_index * BASE_CADENCE_DAYS
        observed_bands = bands_observed_at_visit(visit_index, constants["bands"], constants["anchor_band"])
        for band in constants["bands"]:
            mask[(band, epoch_day)] = band in observed_bands
    return mask


def build_example(tier, simulation_spectra, time_days, wavelength_rest_aa):
    constants = build_tier_constants(tier)
    rng = np.random.default_rng(NOISE_SEED)
    photometry = photometry_for_angle(
        simulation_spectra, time_days, wavelength_rest_aa, ANGLE_INDEX, constants, REDSHIFT, rng
    )

    band_column, epoch_column = [], []
    mag_observed, mag_err, mag_limit_5sigma, detected, observed = [], [], [], [], []
    for (band, epoch), is_observed in cadence_mask_for_tier(constants).items():
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


def main():
    paths = load_paths()
    lanl_spectra_path = str(paths.lanl_spectra)
    catalog = load_lanl_catalog_metadata(lanl_spectra_path)
    ejecta_catalog = load_ejecta_catalog(lanl_spectra_path)
    wavelength_rest_aa = load_lanl_wavelength_grid(lanl_spectra_path)

    simulation_id = select_simulation(ejecta_catalog, EJECTA_RUN_TYPE, EJECTA_WIND, EJECTA_PARAMETERS)
    simulation_rows = catalog[catalog["simulation_id"] == simulation_id]
    time_days = (
        simulation_rows.drop_duplicates("time_index").sort_values("time_index")["time_days"].to_numpy()
    )
    simulation_spectra = load_simulation_spectra(simulation_id, lanl_spectra_path)

    examples = {
        tier: build_example(tier, simulation_spectra, time_days, wavelength_rest_aa) for tier in TIERS
    }

    figure, axes = plt.subplots(2, 1, figsize=(4.9, 9.6), sharex=True)
    for axis, tier in zip(axes, TIERS, strict=True):
        example = examples[tier]
        band_order = [band for band in ALL_ROMAN_BANDS if band in example["band"]]
        plot_light_curve(axis, example, band_order)
        axis.set_title("")  # plot_light_curve sets its own; the tier label goes inside the axes
        axis.set_box_aspect(1)

    # both panels on one magnitude scale, so the depth difference between the tiers is readable off
    # the figure. The limits are computed from the data rather than left to autoscale: the top of
    # the range is set by the square stack, which carries no magnitude, so it only needs enough
    # headroom to be seen, and any extra is dead space between the two panels.
    faintest = max(
        np.nanmax(np.where(example["detected"], example["mag_observed"], example["mag_limit_5sigma"]))
        for example in examples.values()
    )
    highest_square = min(
        np.nanmin(example["mag_observed"][example["detected"]])
        - token_diagram_module.NOT_OBSERVED_BASE_OFFSET_MAG
        - 2 * token_diagram_module.NOT_OBSERVED_STEP_MAG
        for example in examples.values()
    )
    for axis in axes:
        axis.set_ylim(faintest + 0.20, highest_square - 0.22)
    axes[0].set_xlabel("")
    # the epoch banners are the same on both panels now that x is shared, so only the top keeps them
    for text in list(axes[1].texts):
        text.remove()
    # tier name inside the axes, in the empty faint corner between the epoch columns: a title band
    # above each panel would cost vertical space no data uses.
    for axis, tier in zip(axes, TIERS, strict=True):
        axis.text(
            0.03,
            0.06,
            f"{tier.upper()} tier",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    band_handles = [
        Line2D([], [], marker="o", ls="", color=BAND_COLORS[band], mec="k", mew=0.9, ms=8.8, label=band)
        for band in ALL_ROMAN_BANDS
    ]
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
    figure.legend(
        handles=band_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(band_handles),
        frameon=False,
        fontsize=7.5,
        columnspacing=0.9,
        handletextpad=0.3,
    )
    figure.legend(
        handles=marker_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.973),
        ncol=len(marker_handles),
        frameon=False,
        fontsize=7.5,
        columnspacing=0.9,
        handletextpad=0.3,
    )
    figure.subplots_adjust(left=0.15, right=0.97, top=0.905, bottom=0.055, hspace=0.07)

    output_path = paths.output_dir / "kilonova_lightcurve_deep_vs_wide.pdf"
    figure.savefig(output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
