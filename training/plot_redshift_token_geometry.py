"""Where does the learned no-redshift token sit relative to the locus of known-redshift tokens?

The [Z] token of GlobalTokens is redshift_projection(z) + bias, an affine map of a single
scalar: [Z](z) = z * w + b. As z sweeps, the token traces a
straight line in R^d_model. The no_redshift_token is an unconstrained point of the same space,
so "what did the model learn for a missing redshift?" reduces to a decomposition:

    no_redshift_token - b = z_effective * w  +  perpendicular

A large parallel part and a vanishing perpendicular part would mean the model imputes an
effective redshift (it would sit ON the line). A vanishing parallel part and a large
perpendicular part mean it learned a direction no real redshift can produce -- a genuine
"missing" flag rather than an imputed value.

Both panels use the checkpoint at CHECKPOINT_PATH. Lengths are reported in units of ||w||, the
displacement produced by one unit of redshift, which makes the perpendicular offset directly
comparable to a redshift difference.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

CHECKPOINT_PATH = "checkpoints/kilonova_transformer-soup.ckpt"
OUTPUT_PATH = "plots/redshift_token_geometry.png"

PROJECTION_WEIGHT_KEY = "model.global_tokens.redshift_projection.weight"
PROJECTION_BIAS_KEY = "model.global_tokens.redshift_projection.bias"
NO_REDSHIFT_TOKEN_KEY = "model.global_tokens.no_redshift_token"

REDSHIFT_MAXIMUM = 2.0
REDSHIFT_MARKERS = [0.0, 0.5, 1.0, 1.5, 2.0]
KILONOVA_REDSHIFT_THRESHOLD = 0.5

KNOWN_REDSHIFT_COLOR = "#2166ac"
NO_REDSHIFT_COLOR = "#b2182b"
ANNOTATION_COLOR = "#4d4d4d"
GRID_COLOR = "#dcdcdc"


def load_redshift_geometry(checkpoint_path):
    """Return the decomposition of no_redshift_token in the (w, perpendicular) plane."""
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    projection_weight = state_dict[PROJECTION_WEIGHT_KEY].detach().float()
    bias = state_dict[PROJECTION_BIAS_KEY].detach().float()
    no_redshift_token = state_dict[NO_REDSHIFT_TOKEN_KEY].detach().float().flatten()

    redshift_direction = projection_weight[:, 0]
    offset = no_redshift_token - bias
    effective_redshift = torch.dot(offset, redshift_direction) / redshift_direction.pow(2).sum()
    parallel_component = effective_redshift * redshift_direction
    perpendicular_component = offset - parallel_component

    return {
        "redshift_direction_norm": float(redshift_direction.norm()),
        "effective_redshift": float(effective_redshift),
        "perpendicular_norm": float(perpendicular_component.norm()),
        "offset_norm": float(offset.norm()),
        "token_norm": float(no_redshift_token.norm()),
        "bias_norm": float(bias.norm()),
    }


def plot_redshift_token_geometry(geometry, axes_pair):
    """Left: the plane spanned by w and the perpendicular offset. Right: distance to the line."""
    plane_axis, distance_axis = axes_pair

    direction_norm = geometry["redshift_direction_norm"]
    effective_redshift = geometry["effective_redshift"]
    perpendicular_in_redshift_units = geometry["perpendicular_norm"] / direction_norm

    plane_axis.axhline(0.0, color=KNOWN_REDSHIFT_COLOR, linewidth=2.0, zorder=2)
    for redshift in REDSHIFT_MARKERS:
        plane_axis.plot(
            redshift,
            0.0,
            marker="o",
            markersize=8,
            color=KNOWN_REDSHIFT_COLOR,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=3,
        )
        plane_axis.annotate(
            f"z = {redshift:g}",
            xy=(redshift, 0.0),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=ANNOTATION_COLOR,
        )

    plane_axis.plot(
        [effective_redshift, effective_redshift],
        [0.0, perpendicular_in_redshift_units],
        linestyle="--",
        linewidth=1.2,
        color=ANNOTATION_COLOR,
        zorder=2,
    )
    plane_axis.plot(
        effective_redshift,
        perpendicular_in_redshift_units,
        marker="o",
        markersize=11,
        color=NO_REDSHIFT_COLOR,
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=4,
    )
    plane_axis.annotate(
        "no_redshift_token",
        xy=(effective_redshift, perpendicular_in_redshift_units),
        xytext=(14, 6),
        textcoords="offset points",
        fontsize=10,
        color=NO_REDSHIFT_COLOR,
        fontweight="bold",
    )
    plane_axis.annotate(
        f"perpendicular = {perpendicular_in_redshift_units:.2f} $\\|w\\|$",
        xy=(effective_redshift, perpendicular_in_redshift_units / 2.0),
        xytext=(14, -4),
        textcoords="offset points",
        fontsize=9,
        color=ANNOTATION_COLOR,
    )
    plane_axis.annotate(
        f"$z_{{effective}}$ = {effective_redshift:.3f}",
        xy=(effective_redshift, 0.0),
        xytext=(10, -34),
        textcoords="offset points",
        fontsize=9,
        color=ANNOTATION_COLOR,
    )
    plane_axis.annotate(
        "locus of known-redshift [Z] tokens",
        xy=(1.25, 0.0),
        xytext=(0, 16),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        color=KNOWN_REDSHIFT_COLOR,
        fontweight="bold",
    )

    plane_axis.set_xlim(-0.35, REDSHIFT_MAXIMUM + 0.25)
    plane_axis.set_ylim(-0.16, 0.62)
    plane_axis.set_xlabel("component along $w$   [redshift units]")
    plane_axis.set_ylabel("component perpendicular to $w$   [$\\|w\\|$]")
    plane_axis.set_title("The no-redshift token is orthogonal to the redshift axis", fontsize=11)

    redshift_grid = np.linspace(0.0, REDSHIFT_MAXIMUM, 400)
    distance_to_token = direction_norm * np.sqrt(
        (redshift_grid - effective_redshift) ** 2 + perpendicular_in_redshift_units**2
    )
    distance_axis.plot(
        redshift_grid,
        distance_to_token,
        color=NO_REDSHIFT_COLOR,
        linewidth=2.0,
        zorder=3,
        label="$\\|[Z](z) - $ no_redshift_token$\\|$",
    )
    distance_axis.axhline(
        geometry["perpendicular_norm"],
        color=ANNOTATION_COLOR,
        linestyle=":",
        linewidth=1.2,
        zorder=2,
        label=f"floor = {geometry['perpendicular_norm']:.2f}  (never reaches 0)",
    )
    distance_axis.axvline(
        KILONOVA_REDSHIFT_THRESHOLD,
        color=KNOWN_REDSHIFT_COLOR,
        linestyle="--",
        linewidth=1.2,
        zorder=2,
        label=f"kilonova threshold z = {KILONOVA_REDSHIFT_THRESHOLD:g}",
    )
    distance_axis.set_xlim(0.0, REDSHIFT_MAXIMUM)
    distance_axis.set_ylim(0.0, None)
    distance_axis.set_xlabel("redshift $z$")
    distance_axis.set_ylabel("distance in $\\mathbb{R}^{192}$")
    distance_axis.set_title("No redshift reproduces the token", fontsize=11)
    distance_axis.legend(loc="upper left", fontsize=8, frameon=False)

    for axis in axes_pair:
        axis.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(ANNOTATION_COLOR)
            axis.spines[side].set_linewidth(0.8)


def main():
    geometry = load_redshift_geometry(CHECKPOINT_PATH)

    print(f"||w||  (redshift channel)      = {geometry['redshift_direction_norm']:.4f}")
    print(f"||no_redshift_token||          = {geometry['token_norm']:.4f}")
    print(f"z_effective                    = {geometry['effective_redshift']:.4f}")
    print(f"||perpendicular||              = {geometry['perpendicular_norm']:.4f}")
    print(
        "perpendicular / ||w||          = "
        f"{geometry['perpendicular_norm'] / geometry['redshift_direction_norm']:.4f}"
    )

    figure, axes_pair = plt.subplots(1, 2, figsize=(12.5, 4.8))
    plot_redshift_token_geometry(geometry, axes_pair)
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=200, facecolor="white")
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
