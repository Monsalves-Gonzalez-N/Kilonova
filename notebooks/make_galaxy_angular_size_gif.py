"""Animated version of plot_galaxy_angular_size_vs_redshift.py: the same synthetic spiral swept
continuously in redshift from z = 0.1 to z = 6 and back.

Left panel keeps a fixed stretch, so the galaxy both shrinks and fades as it recedes. Middle panel
re-stretches every frame to its own peak, isolating the pure geometry: the angular size stops
falling at the D_A maximum (z ~ 1.6) and then grows again, because in an expanding universe the
object was closer in proper distance when the light we see now left it.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from plot_galaxy_angular_size_vs_redshift import (
    CANVAS_HALF_WIDTH_ARCSEC,
    DISK_SCALE_LENGTH_KPC,
    GALAXY_DIAMETER_KPC,
    SCALE_BAR_KPC,
    render_galaxy,
)

from kilonova.config import load_paths

MINIMUM_REDSHIFT = 0.1
MAXIMUM_REDSHIFT = 6.0
FORWARD_FRAMES = 70
FRAMES_PER_SECOND = 14


def redshift_sequence():
    forward = np.geomspace(MINIMUM_REDSHIFT, MAXIMUM_REDSHIFT, FORWARD_FRAMES)
    backward = forward[-2:0:-1]
    return np.concatenate([forward, backward])


ZOOM_HALF_WIDTH_ARCSEC = 3.5


def setup_image_axis(ax, title, half_width, ticks):
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_width, half_width)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("offset (arcsec)")


def main():
    paths = load_paths()
    cosmology = Planck18

    redshifts = redshift_sequence()
    renders = {
        redshift: render_galaxy(redshift, cosmology)
        for redshift in np.geomspace(MINIMUM_REDSHIFT, MAXIMUM_REDSHIFT, FORWARD_FRAMES)
    }

    reference_image, _ = renders[MINIMUM_REDSHIFT]
    fixed_normalization = 0.03 * reference_image.max()

    figure = plt.figure(figsize=(14.0, 5.4))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25], wspace=0.28)
    ax_observed = figure.add_subplot(grid[0, 0])
    ax_stretched = figure.add_subplot(grid[0, 1])
    ax_curve = figure.add_subplot(grid[0, 2])

    extent = [
        -CANVAS_HALF_WIDTH_ARCSEC,
        CANVAS_HALF_WIDTH_ARCSEC,
        -CANVAS_HALF_WIDTH_ARCSEC,
        CANVAS_HALF_WIDTH_ARCSEC,
    ]
    observed_image = ax_observed.imshow(
        np.zeros_like(reference_image),
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=np.arcsinh(reference_image.max() / fixed_normalization),
    )
    stretched_image = ax_stretched.imshow(
        np.zeros_like(reference_image),
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )

    setup_image_axis(
        ax_observed, "as observed: shrinks and fades", CANVAS_HALF_WIDTH_ARCSEC, [-8, -4, 0, 4, 8]
    )
    setup_image_axis(
        ax_stretched,
        "contrast stretched and zoomed: geometry only",
        ZOOM_HALF_WIDTH_ARCSEC,
        [-3, 0, 3],
    )
    ax_observed.set_ylabel("offset (arcsec)")

    (scale_bar,) = ax_observed.plot([], [], color="white", linewidth=2.5)
    scale_bar_label = ax_observed.text(
        0.0,
        -CANVAS_HALF_WIDTH_ARCSEC * 0.84 + 0.25,
        "",
        color="white",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    dimming_label = ax_observed.text(
        -CANVAS_HALF_WIDTH_ARCSEC * 0.92,
        CANVAS_HALF_WIDTH_ARCSEC * 0.86,
        "",
        color="white",
        ha="left",
        va="top",
        fontsize=9,
    )

    redshift_grid = np.geomspace(MINIMUM_REDSHIFT * 0.6, MAXIMUM_REDSHIFT * 1.2, 500)
    angular_diameter_grid = GALAXY_DIAMETER_KPC * cosmology.arcsec_per_kpc_proper(redshift_grid).value
    turnover_redshift = redshift_grid[np.argmin(angular_diameter_grid)]

    ax_curve.plot(redshift_grid, angular_diameter_grid, color="0.25", linewidth=2.0)
    ax_curve.axvline(turnover_redshift, color="steelblue", linestyle="--", linewidth=1.4)
    ax_curve.text(
        turnover_redshift * 1.08,
        angular_diameter_grid.max() * 0.55,
        f"$D_A$ maximum\n$z = {turnover_redshift:.2f}$",
        color="steelblue",
        fontsize=10,
        va="center",
    )
    # Fixed ruler in the zoom panel: no galaxy of this physical size can ever look smaller.
    minimum_angular_diameter = angular_diameter_grid.min()
    ax_stretched.add_patch(
        Circle(
            (0.0, 0.0),
            minimum_angular_diameter / 2,
            facecolor="none",
            edgecolor="steelblue",
            linestyle="--",
            linewidth=1.4,
        )
    )
    ax_stretched.text(
        0.0,
        minimum_angular_diameter / 2 + 0.12,
        f"smallest it can ever look ({minimum_angular_diameter:.2f}″)",
        color="steelblue",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    (curve_marker,) = ax_curve.plot([], [], "o", color="crimson", markersize=11, zorder=3)
    regime_label = ax_curve.text(
        0.03,
        0.06,
        "",
        transform=ax_curve.transAxes,
        fontsize=11,
        va="bottom",
        ha="left",
    )
    ax_curve.set_xscale("log")
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("redshift")
    ax_curve.set_ylabel(f"angular diameter of a {GALAXY_DIAMETER_KPC:.0f} kpc galaxy (arcsec)")
    ax_curve.grid(alpha=0.3, which="both")

    figure_title = figure.suptitle("", fontsize=13)

    def draw_frame(frame_index):
        redshift = redshifts[frame_index]
        image, arcsec_per_kpc = renders[redshift]

        observed_image.set_data(np.arcsinh(image / fixed_normalization))
        stretched_image.set_data(np.arcsinh(image / (0.03 * image.max())) / np.arcsinh(1.0 / 0.03))

        scale_bar_arcsec = SCALE_BAR_KPC * arcsec_per_kpc
        bar_y = -CANVAS_HALF_WIDTH_ARCSEC * 0.84
        scale_bar.set_data(
            [-scale_bar_arcsec / 2, scale_bar_arcsec / 2],
            [bar_y, bar_y],
        )
        scale_bar_label.set_text(f"{SCALE_BAR_KPC:.0f} kpc = {scale_bar_arcsec:.2f}″")
        dimming_label.set_text(f"surface brightness $\\times (1+z)^{{-4}}$ = {1 / (1 + redshift) ** 4:.3f}")

        angular_diameter = GALAXY_DIAMETER_KPC * arcsec_per_kpc
        curve_marker.set_data([redshift], [angular_diameter])
        if redshift < turnover_redshift:
            regime_label.set_text("still shrinking")
            regime_label.set_color("0.25")
        else:
            regime_label.set_text("growing again: $D_A$ now decreases with $z$")
            regime_label.set_color("crimson")

        angular_diameter_distance = cosmology.angular_diameter_distance(redshift).value
        figure_title.set_text(
            f"Same galaxy ($h = {DISK_SCALE_LENGTH_KPC}$ kpc, {GALAXY_DIAMETER_KPC:.0f} kpc across) "
            f"at $z = {redshift:.2f}$   |   "
            f"$D_A = {angular_diameter_distance:.0f}$ Mpc, {1 / arcsec_per_kpc:.2f} kpc/″, "
            f"angular size {angular_diameter:.2f}″"
        )

        return observed_image, stretched_image, scale_bar, curve_marker

    animation = FuncAnimation(figure, draw_frame, frames=len(redshifts), blit=False)

    output_path = paths.output_dir / "galaxy_angular_size_vs_redshift.gif"
    animation.save(str(output_path), writer=PillowWriter(fps=FRAMES_PER_SECOND), dpi=90)
    print(output_path)


if __name__ == "__main__":
    main()
