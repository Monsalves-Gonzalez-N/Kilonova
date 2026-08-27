"""Future-work slide: the host galaxy as a three-band Roman deep-tier image cube, drawn in the
stacked-feature-map style of a CNN diagram, animated across redshift.

Each sheared plane is one deep-tier band of the same galaxy, with its real synthetic photometry
(kilonova.photometry.spectra) setting the surface brightness and the angular diameter distance
setting the size. Sweeping z shows what the cube would hand to an image encoder: the galaxy shrinks,
fades, and the blue plane fades faster than the red one as the 4000 A break crosses the filters --
that differential is the redshift information a host cutout adds on top of the light-curve tokens.

The SED template is the schematic one of plot_galaxy_roman_bands_vs_redshift.py, with no dust.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
from plot_galaxy_roman_bands_vs_redshift import (
    BAND_COLORS,
    STRETCH_FLOOR,
    TEMPLATES,
    band_colormap,
    galaxy_rest_frame_spectrum,
    galaxy_shape,
)

from kilonova.config import load_paths
from kilonova.photometry.spectra import magnitudes_for_bands

CUBE_BANDS = ["Z087", "J129", "F184"]  # blue / middle / red of the deep tier
FORWARD_REDSHIFTS = np.geomspace(0.1, 3.0, 44)
FRAMES_PER_SECOND = 12

SHEAR_DEGREES = 24.0
PLANE_WIDTH = 0.70
PLANE_HEIGHT = 0.68
PLANE_SPACING = 0.75


def plane_transform(ax, level):
    return (
        Affine2D()
        .scale(PLANE_WIDTH, PLANE_HEIGHT)
        .skew_deg(0.0, SHEAR_DEGREES)
        .translate(level * PLANE_SPACING, 0.0)
        + ax.transData
    )


def redshift_sequence():
    return np.concatenate([FORWARD_REDSHIFTS, FORWARD_REDSHIFTS[-2:0:-1]])


def main():
    paths = load_paths()
    cosmology = Planck18

    break_amplitude, red_slope, _ = next(iter(TEMPLATES.values()))
    wavelength_rest_aa, flux_rest_lambda = galaxy_rest_frame_spectrum(break_amplitude, red_slope)

    magnitudes_by_redshift = {}
    surface_brightness = {}
    for redshift in FORWARD_REDSHIFTS:
        shape, _ = galaxy_shape(redshift, cosmology)
        magnitudes = magnitudes_for_bands(wavelength_rest_aa, flux_rest_lambda, redshift, bands=CUBE_BANDS)
        magnitudes_by_redshift[redshift] = magnitudes
        for band in CUBE_BANDS:
            band_flux = 10 ** (-0.4 * magnitudes[band]) if np.isfinite(magnitudes[band]) else 0.0
            surface_brightness[(redshift, band)] = shape * band_flux

    peak = max(image.max() for image in surface_brightness.values())
    normalization = STRETCH_FLOOR * peak
    vmax = np.arcsinh(peak / normalization)

    horizontal_limits = (-0.10, 2.32)
    vertical_limits = (-0.19, 0.98)
    aspect = (vertical_limits[1] - vertical_limits[0]) / (horizontal_limits[1] - horizontal_limits[0])

    figure = plt.figure(figsize=(11.0, 11.0 * aspect))
    ax_cube = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    ax_cube.set_xlim(*horizontal_limits)
    ax_cube.set_ylim(*vertical_limits)
    ax_cube.set_aspect("equal")
    ax_cube.axis("off")

    images = {}
    for level, band in enumerate(CUBE_BANDS):
        transform = plane_transform(ax_cube, level)
        images[band] = ax_cube.imshow(
            np.zeros((2, 2)),
            origin="lower",
            extent=[0.0, 1.0, 0.0, 1.0],
            transform=transform,
            cmap=band_colormap(band),
            vmin=0.0,
            vmax=vmax,
            zorder=level,
        )
        ax_cube.add_patch(
            Polygon(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                closed=True,
                facecolor="none",
                edgecolor=BAND_COLORS[band],
                linewidth=2.0,
                transform=transform,
                zorder=level + 0.5,
            )
        )
        label_corner = ax_cube.transData.inverted().transform(transform.transform([0.5, 1.0]))
        ax_cube.text(
            label_corner[0],
            label_corner[1] + 0.06,
            band,
            color=BAND_COLORS[band],
            fontsize=16,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    redshift_label = ax_cube.text(
        (horizontal_limits[0] + horizontal_limits[1]) / 2,
        -0.06,
        "",
        fontsize=20,
        ha="center",
        va="top",
        color="0.2",
    )

    redshifts = redshift_sequence()

    def draw_frame(frame_index):
        redshift = redshifts[frame_index]
        for band in CUBE_BANDS:
            images[band].set_data(np.arcsinh(surface_brightness[(redshift, band)] / normalization))
        redshift_label.set_text(f"$z = {redshift:.2f}$")
        return tuple(images.values())

    animation = FuncAnimation(figure, draw_frame, frames=len(redshifts), blit=False)

    output_path = paths.output_dir / "galaxy_band_cube_vs_redshift.gif"
    animation.save(str(output_path), writer=PillowWriter(fps=FRAMES_PER_SECOND), dpi=100)
    print(output_path)


if __name__ == "__main__":
    main()
