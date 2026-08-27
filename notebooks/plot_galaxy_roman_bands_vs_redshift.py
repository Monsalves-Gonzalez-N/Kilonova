"""The same galaxy imaged in the six Roman bands across redshift: how the angular size AND the
per-band brightness change together, and whether the six-band colour alone separates z < 1 from z > 1.

The morphology is the synthetic spiral of plot_galaxy_angular_size_vs_redshift.py. What is new here
is that the surface brightness of every cell is set by real synthetic photometry: a rest-frame galaxy
SED is redshifted, dimmed and integrated through the galsim.roman bandpasses with the pipeline's own
kilonova.photometry.spectra, so the (1+z)^-4 dimming, the K-correction and the band-dependent
break-crossing all come out of one calculation instead of being drawn by hand.

The SED is a SCHEMATIC template, not a fitted one: a broken power law in f_nu with a 4000 A break, a
1.6 um stellar bump, Lyman-alpha forest suppression and a hard Lyman limit. It reproduces the two
features that carry the redshift information (the 4000 A break and the Lyman dropout) but it has no
emission lines, no age/metallicity sequence and NO DUST -- which is exactly the idealisation the
z < 1 / z > 1 question was posed under. Swap it for a real template (Brown+14, SWIRE, CWW) before
quoting any number as a photo-z performance.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18
from matplotlib.colors import LinearSegmentedColormap
from plot_galaxy_angular_size_vs_redshift import (
    FWHM_TO_SIGMA,
    PIXEL_SCALE_ARCSEC,
    ROMAN_PSF_FWHM_ARCSEC,
    galaxy_surface_brightness,
)
from scipy.ndimage import gaussian_filter

from kilonova.config import load_paths
from kilonova.photometry.roman_noise import (
    build_tier_constants,
    flux_error_electrons,
    limiting_magnitude_5sigma,
)
from kilonova.photometry.spectra import ALL_ROMAN_BANDS, magnitudes_for_bands

GRID_REDSHIFTS = [0.2, 0.5, 1.0, 2.0, 3.0]
CURVE_REDSHIFTS = np.geomspace(0.05, 4.0, 36)
TIER = "deep"

CANVAS_HALF_WIDTH_ARCSEC = 4.0
STRETCH_FLOOR = 2e-4  # fraction of the brightest cell where the arcsinh stretch turns over
ABSOLUTE_MAGNITUDE_AB = -21.0  # L* galaxy, normalised at rest-frame 5500 A

BREAK_WAVELENGTH_AA = 4000.0
STELLAR_BUMP_WAVELENGTH_AA = 16000.0
NEAR_INFRARED_SLOPE = -1.5
# (break amplitude, f_nu slope redward of the break, marker): an old red population has a strong
# 4000 A break, a young star-forming one barely has any. The grid images use the first.
TEMPLATES = {
    "old population": (1.6, 0.4, "o"),
    "star forming": (1.15, -0.2, "s"),
}
LYMAN_ALPHA_AA = 1216.0
LYMAN_LIMIT_AA = 912.0
FOREST_TRANSMISSION = 0.3

SPEED_OF_LIGHT_AA_PER_S = 2.99792458e18

COLOR_BLUE_BAND = "R062"
COLOR_MIDDLE_BAND = "Y106"
COLOR_RED_BAND = "F184"

BAND_COLORS = {
    band: plt.cm.turbo(position)
    for band, position in zip(ALL_ROMAN_BANDS, np.linspace(0.05, 0.95, len(ALL_ROMAN_BANDS)), strict=True)
}


def galaxy_rest_frame_spectrum(break_amplitude, red_slope):
    """Schematic rest-frame galaxy SED at 10 pc, in f_lambda, normalised to ABSOLUTE_MAGNITUDE_AB
    at 5500 A. See the module docstring: the shape carries the 4000 A break and the Lyman dropout
    and nothing else."""
    wavelength_aa = np.geomspace(300.0, 60000.0, 6000)

    flux_nu = np.where(
        wavelength_aa < BREAK_WAVELENGTH_AA,
        1.0 / break_amplitude,
        (wavelength_aa / BREAK_WAVELENGTH_AA) ** red_slope,
    )
    beyond_bump = wavelength_aa > STELLAR_BUMP_WAVELENGTH_AA
    bump_amplitude = (STELLAR_BUMP_WAVELENGTH_AA / BREAK_WAVELENGTH_AA) ** red_slope
    flux_nu[beyond_bump] = (
        bump_amplitude * (wavelength_aa[beyond_bump] / STELLAR_BUMP_WAVELENGTH_AA) ** NEAR_INFRARED_SLOPE
    )

    flux_nu[wavelength_aa < LYMAN_ALPHA_AA] *= FOREST_TRANSMISSION
    flux_nu[wavelength_aa < LYMAN_LIMIT_AA] = 0.0

    reference_flux_nu = np.interp(5500.0, wavelength_aa, flux_nu)
    target_flux_nu = 10 ** (-0.4 * (ABSOLUTE_MAGNITUDE_AB + 48.60))
    flux_nu = flux_nu * target_flux_nu / reference_flux_nu

    flux_lambda = flux_nu * SPEED_OF_LIGHT_AA_PER_S / wavelength_aa**2
    return wavelength_aa, flux_lambda


def band_colormap(band):
    return LinearSegmentedColormap.from_list(band, ["black", BAND_COLORS[band], "white"])


def galaxy_shape(redshift, cosmology=Planck18):
    """Unit-total-flux angular profile of the galaxy at `redshift`, per arcsec^2, PSF convolved."""
    arcsec_per_kpc = cosmology.arcsec_per_kpc_proper(redshift).value
    axis = np.arange(
        -CANVAS_HALF_WIDTH_ARCSEC, CANVAS_HALF_WIDTH_ARCSEC + PIXEL_SCALE_ARCSEC, PIXEL_SCALE_ARCSEC
    )
    x_arcsec, y_arcsec = np.meshgrid(axis, axis)

    profile = galaxy_surface_brightness(x_arcsec, y_arcsec, arcsec_per_kpc)
    profile = profile / (profile.sum() * PIXEL_SCALE_ARCSEC**2)

    psf_sigma_pixels = ROMAN_PSF_FWHM_ARCSEC * FWHM_TO_SIGMA / PIXEL_SCALE_ARCSEC
    return gaussian_filter(profile, psf_sigma_pixels), arcsec_per_kpc


def plot_image_grid(figure, grid, magnitudes_by_redshift):
    shapes = {redshift: galaxy_shape(redshift) for redshift in GRID_REDSHIFTS}

    surface_brightness = {}
    for redshift in GRID_REDSHIFTS:
        shape, _ = shapes[redshift]
        for band in ALL_ROMAN_BANDS:
            magnitude = magnitudes_by_redshift[redshift][band]
            band_flux = 10 ** (-0.4 * magnitude) if np.isfinite(magnitude) else 0.0
            surface_brightness[(redshift, band)] = shape * band_flux

    peak = max(image.max() for image in surface_brightness.values())
    normalization = STRETCH_FLOOR * peak
    vmax = np.arcsinh(peak / normalization)

    extent = [
        -CANVAS_HALF_WIDTH_ARCSEC,
        CANVAS_HALF_WIDTH_ARCSEC,
        -CANVAS_HALF_WIDTH_ARCSEC,
        CANVAS_HALF_WIDTH_ARCSEC,
    ]
    for row, redshift in enumerate(GRID_REDSHIFTS):
        _, arcsec_per_kpc = shapes[redshift]
        for column, band in enumerate(ALL_ROMAN_BANDS):
            ax = figure.add_subplot(grid[row, column])
            ax.imshow(
                np.arcsinh(surface_brightness[(redshift, band)] / normalization),
                origin="lower",
                extent=extent,
                cmap=band_colormap(band),
                vmin=0.0,
                vmax=vmax,
            )
            magnitude = magnitudes_by_redshift[redshift][band]
            label = f"{magnitude:.1f}" if np.isfinite(magnitude) else "no flux"
            ax.text(
                0.05,
                0.95,
                label,
                transform=ax.transAxes,
                color="white",
                fontsize=9,
                va="top",
                ha="left",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(band, fontsize=12, color=BAND_COLORS[band])
            if column == 0:
                ax.set_ylabel(
                    f"$z = {redshift:g}$\n{1 / arcsec_per_kpc:.1f} kpc/″",
                    fontsize=11,
                )


def point_source_depths(tier=TIER):
    """5 sigma point-source depth per band of `tier`, at the background-limited end (source flux 0)."""
    constants = build_tier_constants(tier)
    depths = {}
    for band in constants["bands"]:
        exposure = constants["exposure_time"][band]
        zeropoint = constants["zeropoint"][band]
        flux_error = flux_error_electrons(0.0, constants["noise_floor_variance"][band])
        depths[band] = limiting_magnitude_5sigma(flux_error, exposure, zeropoint)
    return depths


def plot_magnitude_curves(ax, magnitudes_by_curve_redshift):
    depths = point_source_depths()
    ax.axhspan(
        min(depths.values()),
        max(depths.values()),
        color="0.85",
        zorder=0,
    )
    ax.text(
        0.055,
        np.mean(list(depths.values())),
        f"{TIER} tier single-visit 5σ (point source)",
        fontsize=8,
        color="0.35",
        va="center",
    )

    for band in ALL_ROMAN_BANDS:
        magnitudes = np.array([magnitudes_by_curve_redshift[redshift][band] for redshift in CURVE_REDSHIFTS])
        finite = np.isfinite(magnitudes)
        ax.plot(
            CURVE_REDSHIFTS[finite],
            magnitudes[finite],
            color=BAND_COLORS[band],
            linewidth=2.0,
            label=band,
        )
        if not finite.all():
            first_lost = CURVE_REDSHIFTS[~finite][0]
            ax.plot(first_lost, magnitudes[finite][-1], "v", color=BAND_COLORS[band], markersize=9)

    ax.axvline(1.0, color="0.4", linestyle="--", linewidth=1.4)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("redshift")
    ax.set_ylabel("observed AB magnitude")
    ax.set_title(f"$M_{{AB}} = {ABSOLUTE_MAGNITUDE_AB:g}$ galaxy, no dust", fontsize=11)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)


def color_track(magnitudes_by_curve_redshift):
    blue_color = []
    red_color = []
    kept_redshifts = []
    for redshift in CURVE_REDSHIFTS:
        magnitudes = magnitudes_by_curve_redshift[redshift]
        first = magnitudes[COLOR_BLUE_BAND] - magnitudes[COLOR_MIDDLE_BAND]
        second = magnitudes[COLOR_MIDDLE_BAND] - magnitudes[COLOR_RED_BAND]
        if np.isfinite(first) and np.isfinite(second):
            blue_color.append(first)
            red_color.append(second)
            kept_redshifts.append(redshift)
    return np.array(red_color), np.array(blue_color), np.array(kept_redshifts)


def plot_color_track(ax, magnitudes_by_template):
    for template_name, (_break_amplitude, _red_slope, marker) in TEMPLATES.items():
        red_color, blue_color, kept_redshifts = color_track(magnitudes_by_template[template_name])
        ax.plot(red_color, blue_color, color="0.7", linewidth=1.2, zorder=2)
        scatter = ax.scatter(
            red_color,
            blue_color,
            c=kept_redshifts,
            cmap="viridis",
            marker=marker,
            s=55,
            edgecolor="0.2",
            linewidth=0.6,
            zorder=3,
            label=template_name,
        )

        for redshift in [0.2, 1.0, 3.0]:
            index = int(np.argmin(np.abs(kept_redshifts - redshift)))
            ax.annotate(
                f"$z = {kept_redshifts[index]:.1f}$",
                (red_color[index], blue_color[index]),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=9,
            )

    plt.colorbar(scatter, ax=ax, label="redshift")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlabel(f"{COLOR_MIDDLE_BAND} - {COLOR_RED_BAND}")
    ax.set_ylabel(f"{COLOR_BLUE_BAND} - {COLOR_MIDDLE_BAND}")
    ax.set_title("colour tracks, no dust: two templates already widen the locus", fontsize=11)
    ax.grid(alpha=0.3)


def main():
    paths = load_paths()

    magnitudes_by_template = {}
    for template_name, (break_amplitude, red_slope, _) in TEMPLATES.items():
        wavelength_rest_aa, flux_rest_lambda = galaxy_rest_frame_spectrum(break_amplitude, red_slope)
        magnitudes_by_template[template_name] = {
            redshift: magnitudes_for_bands(wavelength_rest_aa, flux_rest_lambda, redshift)
            for redshift in np.concatenate([CURVE_REDSHIFTS, GRID_REDSHIFTS])
        }

    reference_template = next(iter(TEMPLATES))
    magnitudes_by_redshift = magnitudes_by_template[reference_template]
    magnitudes_by_curve_redshift = magnitudes_by_template[reference_template]

    figure = plt.figure(figsize=(15.0, 16.5))
    outer = figure.add_gridspec(2, 1, height_ratios=[len(GRID_REDSHIFTS), 1.6], hspace=0.09)
    image_grid = outer[0].subgridspec(len(GRID_REDSHIFTS), len(ALL_ROMAN_BANDS), hspace=0.06, wspace=0.06)

    plot_image_grid(figure, image_grid, magnitudes_by_redshift)

    bottom = outer[1].subgridspec(1, 2, wspace=0.24)
    plot_magnitude_curves(figure.add_subplot(bottom[0, 0]), magnitudes_by_curve_redshift)
    plot_color_track(figure.add_subplot(bottom[0, 1]), magnitudes_by_template)

    figure.suptitle(
        "One galaxy in the six Roman bands: angular size set by $D_A$, brightness by the redshifted SED "
        f"(each panel is {2 * CANVAS_HALF_WIDTH_ARCSEC:.0f}″ across, label = observed AB mag)",
        fontsize=14,
        y=0.905,
    )

    output_path = paths.output_dir / "galaxy_roman_bands_vs_redshift.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
