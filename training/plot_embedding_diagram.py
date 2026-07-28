"""Neural-network view of the two dense embedding layers of TokenEmbedding.

Left: the time encoding, nn.Linear(1, d_model) applied to Delta t / DELTA_TIME_SCALE (with
TIME2VEC_FREQUENCIES = 0 the Time2Vec block collapses to this single affine map). Right: the
photometry columns of content_projection -- the 4 columns of nn.Linear(D_BAND + D_TYPE + 4,
d_model) that multiply [mag, sigma_mag, mag_mask, sigma_mask]. The band and token-type columns
are left out of the drawing: they multiply embedding-table outputs, not interpretable inputs.

Edges carry the trained weights of checkpoints/kilonova_transformer-soup.ckpt: colour is the sign,
width and opacity the magnitude. Only a subset of the d_model output units is drawn (the column
would be unreadable otherwise); the ellipsis marks the ones left out.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

CHECKPOINT_PATH = "checkpoints/kilonova_transformer-soup.ckpt"
OUTPUT_PATH = "plots/embedding_linear_layers.png"

TIME_WEIGHT_KEY = "model.token_embedding.time_encoding.projection.weight"
TIME_SCALAR_KEY = "model.token_embedding.time_encoding.linear_weight"
CONTENT_WEIGHT_KEY = "model.token_embedding.content_projection.weight"

D_BAND = 6
D_TYPE = 3
NUM_MAGNITUDE_FEATURES = 4

UNITS_SHOWN_TOP = 9
UNITS_SHOWN_BOTTOM = 3

POSITIVE_COLOR = "#b2182b"
NEGATIVE_COLOR = "#2166ac"

INPUT_X = 0.0
OUTPUT_X = 1.0


def load_weights():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    # linear_weight is a redundant scalar in front of the projection; fold it in so the drawn
    # edges are the effective weight the token actually sees.
    time_weight = (state_dict[TIME_WEIGHT_KEY] * state_dict[TIME_SCALAR_KEY]).numpy()
    magnitude_start = D_BAND + D_TYPE
    magnitude_weight = state_dict[CONTENT_WEIGHT_KEY][
        :, magnitude_start : magnitude_start + NUM_MAGNITUDE_FEATURES
    ].numpy()
    return time_weight, magnitude_weight


def output_layout():
    """y positions of the drawn output units, plus the y of the ellipsis gap."""
    total_shown = UNITS_SHOWN_TOP + UNITS_SHOWN_BOTTOM
    step = 1.0
    positions = []
    for rank in range(UNITS_SHOWN_TOP):
        positions.append(-rank * step)
    gap_y = -(UNITS_SHOWN_TOP - 1 + 1.5) * step
    for rank in range(UNITS_SHOWN_BOTTOM):
        positions.append(-(UNITS_SHOWN_TOP + 2 + rank) * step)
    span = positions[0] - positions[-1]
    centered = [position + span / 2 for position in positions]
    return np.array(centered), gap_y + span / 2, total_shown


def drawn_unit_indices(d_model):
    top = list(range(UNITS_SHOWN_TOP))
    bottom = list(range(d_model - UNITS_SHOWN_BOTTOM, d_model))
    return top + bottom


def draw_layer(ax, weight, input_labels, title, subtitle):
    d_model = weight.shape[0]
    unit_y, gap_y, _ = output_layout()
    unit_indices = drawn_unit_indices(d_model)

    input_y = np.linspace(unit_y.max() * 0.42, unit_y.min() * 0.42, len(input_labels))
    scale = np.abs(weight).max()

    for input_index, y_in in enumerate(input_y):
        for drawn_rank, unit_index in enumerate(unit_indices):
            value = weight[unit_index, input_index]
            strength = abs(value) / scale
            ax.plot(
                [INPUT_X, OUTPUT_X],
                [y_in, unit_y[drawn_rank]],
                color=POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR,
                linewidth=0.35 + 2.4 * strength**1.6,
                alpha=0.18 + 0.72 * strength**1.3,
                zorder=1,
                solid_capstyle="round",
            )

    ax.scatter(
        np.full(len(input_y), INPUT_X),
        input_y,
        s=560,
        facecolor="#f2f2f2",
        edgecolor="0.2",
        linewidth=1.6,
        zorder=3,
    )
    for label, y_in in zip(input_labels, input_y, strict=True):
        ax.text(INPUT_X - 0.09, y_in, label, ha="right", va="center", fontsize=11)

    ax.scatter(
        np.full(len(unit_y), OUTPUT_X),
        unit_y,
        s=150,
        facecolor="#c9d9f2",
        edgecolor="0.2",
        linewidth=1.2,
        zorder=3,
    )
    for drawn_rank, unit_index in enumerate(unit_indices):
        ax.text(
            OUTPUT_X + 0.07,
            unit_y[drawn_rank],
            f"{unit_index + 1}",
            ha="left",
            va="center",
            fontsize=7.5,
            color="0.45",
        )
    ax.text(OUTPUT_X, gap_y, "⋮", ha="center", va="center", fontsize=17, color="0.35", zorder=3)

    ax.text(INPUT_X, unit_y.max() + 1.5, "entrada", ha="center", va="bottom", fontsize=10, color="0.35")
    ax.text(
        OUTPUT_X,
        unit_y.max() + 1.5,
        f"{d_model} unidades",
        ha="center",
        va="bottom",
        fontsize=10,
        color="0.35",
    )
    ax.set_title(title, fontsize=13, pad=26)
    ax.text(
        0.5,
        unit_y.min() - 1.7,
        subtitle,
        ha="center",
        va="top",
        fontsize=10,
        color="0.3",
    )

    ax.set_xlim(INPUT_X - 0.55, OUTPUT_X + 0.32)
    ax.set_ylim(unit_y.min() - 3.4, unit_y.max() + 2.6)
    ax.axis("off")


def main():
    time_weight, magnitude_weight = load_weights()
    d_model = time_weight.shape[0]

    figure, (time_axis, magnitude_axis) = plt.subplots(
        1, 2, figsize=(13.5, 8.0), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )

    draw_layer(
        time_axis,
        time_weight,
        [r"$\Delta t\,/\,5$"],
        f"Tiempo:  nn.Linear(1, {d_model})",
        f"{d_model} pesos + {d_model} bias = {2 * d_model} parámetros\n"
        f"entrada 1-D: todas las salidas caen sobre una recta en $\\mathbb{{R}}^{{{d_model}}}$",
    )
    draw_layer(
        magnitude_axis,
        magnitude_weight,
        [r"$m$", r"$\sigma_m$", r"$\mu_m$", r"$\mu_\sigma$"],
        "Fotometría:  columnas de magnitud de content_projection",
        f"{NUM_MAGNITUDE_FEATURES} de las {D_BAND + D_TYPE + NUM_MAGNITUDE_FEATURES} columnas "
        f"de nn.Linear({D_BAND + D_TYPE + NUM_MAGNITUDE_FEATURES}, {d_model})\n"
        "entrada 4-D: las 4 columnas se mezclan según el valor de cada campo",
    )

    legend_handles = [
        Line2D([], [], color=POSITIVE_COLOR, linewidth=2.4, label="peso > 0"),
        Line2D([], [], color=NEGATIVE_COLOR, linewidth=2.4, label="peso < 0"),
        Line2D([], [], color="0.45", linewidth=2.4, label="grosor ∝ |peso|"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.suptitle(
        "Las dos capas densas del embedding (pesos entrenados, checkpoint soup)",
        fontsize=14,
        y=0.965,
    )
    figure.subplots_adjust(left=0.06, right=0.97, top=0.86, bottom=0.09, wspace=0.22)
    figure.savefig(OUTPUT_PATH, dpi=180)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
