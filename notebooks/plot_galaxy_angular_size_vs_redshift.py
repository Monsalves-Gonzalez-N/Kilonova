"""Schematic of the same galaxy seen at increasing redshift: one synthetic spiral of fixed physical
size (bulge + exponential disk + two logarithmic arms) painted on a common angular grid, so the only
things that change between panels are the angular scale kpc -> arcsec set by the angular diameter
distance, the (1+z)^-4 surface brightness dimming, and the Roman PSF blur.

The bottom panel is the reason the shrinking saturates: D_A peaks near z ~ 1.6, so beyond that a
galaxy of fixed physical size stops getting smaller on the sky (it only keeps getting fainter).
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter

from kilonova.config import load_paths

REDSHIFTS = [0.1, 0.3, 0.6, 1.0]

DISK_SCALE_LENGTH_KPC = 3.5
BULGE_EFFECTIVE_RADIUS_KPC = 0.9
BULGE_TO_DISK_FLUX = 0.35
ARM_NUMBER = 2
ARM_PITCH_ANGLE_DEGREES = 18.0
ARM_CONTRAST = 0.6
ARM_WIDTH = 0.45

GALAXY_DIAMETER_KPC = 4.0 * DISK_SCALE_LENGTH_KPC

CANVAS_HALF_WIDTH_ARCSEC = 8.0
PIXEL_SCALE_ARCSEC = 0.02
ROMAN_PSF_FWHM_ARCSEC = 0.11
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

SCALE_BAR_KPC = 10.0


def galaxy_surface_brightness(x_arcsec, y_arcsec, arcsec_per_kpc):
    """Face-on spiral in units of the central disk surface brightness, evaluated on an angular grid."""
    x_kpc = x_arcsec / arcsec_per_kpc
    y_kpc = y_arcsec / arcsec_per_kpc
    radius_kpc = np.hypot(x_kpc, y_kpc)
    azimuth = np.arctan2(y_kpc, x_kpc)

    disk = np.exp(-radius_kpc / DISK_SCALE_LENGTH_KPC)

    # Logarithmic spiral: the arm crest sits where the phase residual vanishes.
    winding = np.tan(np.radians(ARM_PITCH_ANGLE_DEGREES))
    arm_phase = ARM_NUMBER * (
        azimuth - np.log(np.maximum(radius_kpc, 1e-3) / DISK_SCALE_LENGTH_KPC) / winding
    )
    arm_residual = np.angle(np.exp(1j * arm_phase))
    arms = np.exp(-0.5 * (arm_residual / ARM_WIDTH) ** 2)
    disk = disk * (1.0 + ARM_CONTRAST * arms)

    sersic_radius = np.maximum(radius_kpc, 1e-3) / BULGE_EFFECTIVE_RADIUS_KPC
    bulge = np.exp(-7.669 * (sersic_radius**0.25 - 1.0))
    bulge = BULGE_TO_DISK_FLUX * bulge / bulge.max()

    return disk + bulge


def render_galaxy(redshift, cosmology=Planck18):
    arcsec_per_kpc = cosmology.arcsec_per_kpc_proper(redshift).value

    axis = np.arange(
        -CANVAS_HALF_WIDTH_ARCSEC, CANVAS_HALF_WIDTH_ARCSEC + PIXEL_SCALE_ARCSEC, PIXEL_SCALE_ARCSEC
    )
    x_arcsec, y_arcsec = np.meshgrid(axis, axis)

    image = galaxy_surface_brightness(x_arcsec, y_arcsec, arcsec_per_kpc)
    image = image / (1.0 + redshift) ** 4

    psf_sigma_pixels = ROMAN_PSF_FWHM_ARCSEC * FWHM_TO_SIGMA / PIXEL_SCALE_ARCSEC
    image = gaussian_filter(image, psf_sigma_pixels)

    return image, arcsec_per_kpc


def plot_panel(ax, redshift, image, arcsec_per_kpc, normalization, cosmology=Planck18):
    extent = [
        -CANVAS_HALF_WIDTH_ARCSEC,
        CANVAS_HALF_WIDTH_ARCSEC,
        -CANVAS_HALF_WIDTH_ARCSEC,
        CANVAS_HALF_WIDTH_ARCSEC,
    ]
    ax.imshow(
        np.arcsinh(image / normalization),
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=np.arcsinh(1.0 / normalization),
    )

    scale_bar_arcsec = SCALE_BAR_KPC * arcsec_per_kpc
    bar_y = -CANVAS_HALF_WIDTH_ARCSEC * 0.84
    bar_left = -scale_bar_arcsec / 2
    ax.plot([bar_left, bar_left + scale_bar_arcsec], [bar_y, bar_y], color="white", linewidth=2.5)
    ax.text(
        0.0,
        bar_y + 0.25,
        f"{SCALE_BAR_KPC:.0f} kpc = {scale_bar_arcsec:.2f}″",
        color="white",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    psf_center = (CANVAS_HALF_WIDTH_ARCSEC * 0.75, -CANVAS_HALF_WIDTH_ARCSEC * 0.75)
    ax.add_patch(
        Circle(psf_center, ROMAN_PSF_FWHM_ARCSEC / 2, facecolor="white", edgecolor="white", linewidth=1.0)
    )
    ax.text(
        psf_center[0],
        psf_center[1] + 0.35,
        "Roman PSF",
        color="white",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    angular_diameter_distance = cosmology.angular_diameter_distance(redshift).value
    ax.set_title(
        f"$z = {redshift:g}$\n"
        f"$D_A = {angular_diameter_distance:.0f}$ Mpc, {1 / arcsec_per_kpc:.2f} kpc/″",
        fontsize=11,
    )
    ax.set_xlabel("offset (arcsec)")
    ax.set_xlim(-CANVAS_HALF_WIDTH_ARCSEC, CANVAS_HALF_WIDTH_ARCSEC)
    ax.set_ylim(-CANVAS_HALF_WIDTH_ARCSEC, CANVAS_HALF_WIDTH_ARCSEC)
    ax.set_xticks([-8, -4, 0, 4, 8])
    ax.set_yticks([-8, -4, 0, 4, 8])


def plot_angular_size_curve(ax, cosmology=Planck18):
    redshift_grid = np.geomspace(0.02, 5.0, 400)
    angular_diameter = GALAXY_DIAMETER_KPC * cosmology.arcsec_per_kpc_proper(redshift_grid).value
    ax.plot(redshift_grid, angular_diameter, color="0.25", linewidth=2.0)

    for redshift in REDSHIFTS:
        size = GALAXY_DIAMETER_KPC * cosmology.arcsec_per_kpc_proper(redshift).value
        ax.plot(redshift, size, "o", color="crimson", markersize=8, zorder=3)
        ax.annotate(
            f"$z = {redshift:g}$\n{size:.1f}″",
            (redshift, size),
            textcoords="offset points",
            xytext=(10, 6),
            fontsize=9,
            color="crimson",
        )

    turnover_redshift = redshift_grid[np.argmin(angular_diameter)]
    ax.axvline(turnover_redshift, color="steelblue", linestyle="--", linewidth=1.4)
    ax.text(
        turnover_redshift * 1.06,
        angular_diameter.max() * 0.5,
        f"$D_A$ maximum, $z = {turnover_redshift:.2f}$:\nbeyond this the galaxy stops shrinking",
        color="steelblue",
        fontsize=9,
        va="center",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("redshift")
    ax.set_ylabel(f"angular diameter of a {GALAXY_DIAMETER_KPC:.0f} kpc galaxy (arcsec)")
    ax.grid(alpha=0.3, which="both")


def main():
    paths = load_paths()

    renders = [render_galaxy(redshift) for redshift in REDSHIFTS]
    normalization = 0.03 * max(image.max() for image, _ in renders)

    figure = plt.figure(figsize=(4.0 * len(REDSHIFTS), 8.6))
    grid = figure.add_gridspec(2, len(REDSHIFTS), height_ratios=[1.0, 0.62], hspace=0.32)

    for column, (redshift, (image, arcsec_per_kpc)) in enumerate(zip(REDSHIFTS, renders, strict=False)):
        ax = figure.add_subplot(grid[0, column])
        plot_panel(ax, redshift, image, arcsec_per_kpc, normalization)
        if column == 0:
            ax.set_ylabel("offset (arcsec)")

    plot_angular_size_curve(figure.add_subplot(grid[1, :]))

    figure.suptitle(
        "The same galaxy (exponential disk, $h = 3.5$ kpc) at four redshifts: fixed physical size, "
        "shrinking angular size, $(1+z)^{-4}$ surface brightness dimming",
        fontsize=13,
    )

    output_path = paths.output_dir / "galaxy_angular_size_vs_redshift.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
