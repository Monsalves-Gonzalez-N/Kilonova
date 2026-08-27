"""Bare marker strip for the kilonova token diagram: one row of markers per epoch, colored by band,
shaped by token type (filled circle = detection, open downward triangle = 5 sigma upper limit,
filled square = not observed). No axes, no magnitude values -- just the per-visit token shapes,
reusing the same example as plot_kilonova_token_diagram.py.
"""

import matplotlib.pyplot as plt
import numpy as np
from plot_kilonova_token_diagram import BAND_COLORS, build_example, token_types

from kilonova.config import load_paths
from kilonova.photometry.spectra import ALL_ROMAN_BANDS

MARKER_SIZE = 260
WITHIN_EPOCH_GAP = 1.0
BETWEEN_EPOCH_GAP = 2.5


def plot_token_strip(ax, example, band_order):
    token_type = token_types(example)
    epochs = np.unique(example["days_since_detection"])

    cursor = 0.0
    for epoch_index, epoch in enumerate(epochs, start=1):
        epoch_indices = [index for index in np.where(example["days_since_detection"] == epoch)[0]]
        epoch_indices.sort(key=lambda index: band_order.index(example["band"][index]))

        start_cursor = cursor
        for index in epoch_indices:
            band = example["band"][index]
            color = BAND_COLORS[band]
            kind = token_type[index]
            if kind == "d":
                ax.scatter(cursor, 0.0, marker="o", s=MARKER_SIZE, color=color, edgecolor="k", linewidth=1.0)
            elif kind == "u":
                ax.scatter(
                    cursor, 0.0, marker="v", s=MARKER_SIZE, facecolor="none", edgecolor=color, linewidth=2.2
                )
            else:
                ax.scatter(
                    cursor, 0.0, marker="s", s=MARKER_SIZE * 0.85, color=color, edgecolor="k", linewidth=1.0
                )
            cursor += WITHIN_EPOCH_GAP

        epoch_center = (start_cursor + cursor - WITHIN_EPOCH_GAP) / 2
        ax.text(epoch_center, 1.0, f"Epoch {epoch_index}", ha="center", va="bottom", fontsize=13)
        cursor += BETWEEN_EPOCH_GAP

    ax.set_xlim(-WITHIN_EPOCH_GAP / 2, cursor - BETWEEN_EPOCH_GAP + WITHIN_EPOCH_GAP / 2)
    ax.set_ylim(-1.0, 1.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.7")


def main():
    paths = load_paths()
    example = build_example(str(paths.lanl_spectra))
    band_order = [band for band in ALL_ROMAN_BANDS if band in example["band"]]

    figure, axis = plt.subplots(figsize=(13, 2.2))
    plot_token_strip(axis, example, band_order)
    figure.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.05)

    output_path = paths.output_dir / "kilonova_token_strip.png"
    figure.savefig(output_path, dpi=180)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
