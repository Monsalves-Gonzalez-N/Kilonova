"""Conceptual (not real-weight) diagram of self-attention over the kilonova token sequence.

Shows a single head so the mechanism reads clearly: the [CLS] token -- the one whose final state
feeds classification_head and therefore the one that decides {other, KN} -- queries every other
token in the sequence and pools their evidence, with arc width/opacity as an illustrative attention
weight. Detections weigh more than 5 sigma upper limits, which weigh more than visits the cadence
skipped, so the pooled evidence grows as epochs arrive.

The token sequence is the same one drawn by plot_kilonova_token_strip.py (see TOKEN_TYPES), so the
two figures can be shown side by side: a z=0.2 kilonova that fades out of reach, its detections
turning into upper limits by epoch 3.

Default output is a GIF where the epochs fade in one after another (1, then 2, then 3), so the
growing set of arcs into [CLS] is read as an animation; ``--static-png`` instead writes the same
three states as stacked panels of a single figure. Axis limits are always those of the full
3-epoch sequence, so tokens appear in place instead of the layout rescaling between frames.

Each token is drawn the way plot_kilonova_token_strip.py and plot_architecture_diagram.py draw it:
a marker on top saying what the visit was (filled circle = detection, open triangle = upper limit,
filled square = not observed, coloured by band), and below it the bar standing for the d_model
vector the transformer actually operates on. A caption note (not a second head drawn) reminds the
reader that with several heads each one can pool a different kind of evidence.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

from kilonova.config import load_paths
from kilonova.photometry.spectra import ALL_ROMAN_BANDS

ORANGE_FILL, ORANGE_EDGE = "#f6cb8e", "#d99a2b"
PURPLE_FILL, PURPLE_EDGE = "#c7bfe3", "#8d7fc4"
MAROON_FILL, MAROON_EDGE = "#9c4569", "#7a3252"
ARC_COLOR = "#2a9d8f"

BAND_COLORS = {
    band: plt.cm.turbo(position)
    for band, position in zip(ALL_ROMAN_BANDS, np.linspace(0.05, 0.95, len(ALL_ROMAN_BANDS)), strict=True)
}

MARKER_SIZE = 220
WITHIN_EPOCH_GAP = 1.0
BETWEEN_EPOCH_GAP = 2.5

STRIP_Y = 0.0
GLOBAL_TOKEN_GAP = 1.6  # extra gap between [CLS]/[Z] and the first per-visit token
GLOBAL_TOKEN_SPACING = 1.15
ARC_CURVATURE = 0.3

BAR_WIDTH = 0.30
BAR_TOP = -0.55
BAR_BOTTOM = -1.90

EPOCH_LABEL_Y = -2.30
CLASSIFICATION_ARROW_TOP = -2.45
CLASSIFICATION_ARROW_BOTTOM = -2.95
CLASSIFICATION_BOX_Y = -3.35

BAND_ORDER = ["Z087", "Y106", "J129", "H158", "F184"]
EPOCHS = [1, 2, 3]

# An epoch counts as "arrived" for the running detection tally once it is more than half faded in,
# so the tally flips over during the fade instead of after it.
EPOCH_ARRIVED_ALPHA = 0.5

# Transcribed from kilonova_token_strip.png so the two figures show the SAME event: the d/u/n
# pattern of the z=0.2 LANL example of plot_kilonova_token_diagram.py, whose "n" entries follow the
# deep tier's cadence (CADENCE_MASK there: ZYJ at even visits, ZHF at odd ones). The source fades,
# so detections turn into upper limits as the epochs advance -- that decline is the signal, not an
# accident of the example.
TOKEN_TYPES = {
    ("Z087", 1): "d",
    ("Y106", 1): "d",
    ("J129", 1): "d",
    ("H158", 1): "n",
    ("F184", 1): "n",
    ("Z087", 2): "u",
    ("Y106", 2): "n",
    ("J129", 2): "n",
    ("H158", 2): "d",
    ("F184", 2): "d",
    ("Z087", 3): "u",
    ("Y106", 3): "u",
    ("J129", 3): "u",
    ("H158", 3): "n",
    ("F184", 3): "n",
}

# Illustrative rule for the single head drawn: the classification token leans on the visits that
# actually carry photometry, and keeps a standing link to the redshift token.
TOKEN_TYPE_WEIGHT = {"d": 1.0, "u": 0.45, "n": 0.08}
REDSHIFT_TOKEN_WEIGHT = 0.6

SUPTITLE = (
    "The [CLS] token is the one that classifies: it attends to every token and pools their evidence\n"
    "arc width/opacity = illustrative attention weight — detections weigh more than upper limits,\n"
    "which weigh more than visits the cadence skipped\n"
    "(with several heads, each head can pool a different kind of evidence)"
)


def attention_weight(key):
    if key["kind"] == "z":
        return REDSHIFT_TOKEN_WEIGHT
    return TOKEN_TYPE_WEIGHT[key["token_type"]]


def build_token_layout(epochs):
    """x position and metadata (kind/band/epoch) for every token in the drawn sequence."""
    tokens = [
        {"kind": "cls", "x": 0.0, "band": None, "epoch": None, "token_type": None},
        {"kind": "z", "x": GLOBAL_TOKEN_SPACING, "band": None, "epoch": None, "token_type": None},
    ]

    cursor = GLOBAL_TOKEN_SPACING + GLOBAL_TOKEN_GAP
    for epoch in epochs:
        for band in BAND_ORDER:
            tokens.append(
                {
                    "kind": "visit",
                    "x": cursor,
                    "band": band,
                    "epoch": epoch,
                    "token_type": TOKEN_TYPES[(band, epoch)],
                }
            )
            cursor += WITHIN_EPOCH_GAP
        cursor += BETWEEN_EPOCH_GAP - WITHIN_EPOCH_GAP
    return tokens


def sequence_extent():
    """x limits of the full 3-epoch sequence, shared by every frame so the layout never rescales."""
    x_left = -1.4
    x_right = 3 * (BETWEEN_EPOCH_GAP + len(BAND_ORDER) * WITHIN_EPOCH_GAP)
    return x_left, x_right


def draw_token_bar(ax, x_center, face_color, edge_color, alpha):
    ax.add_patch(
        Rectangle(
            (x_center - BAR_WIDTH / 2, BAR_BOTTOM),
            BAR_WIDTH,
            BAR_TOP - BAR_BOTTOM,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=1.0,
            alpha=alpha,
            zorder=3,
        )
    )


def draw_global_tokens(ax, tokens):
    for token in tokens:
        if token["kind"] == "cls":
            face_color, edge_color, label = ORANGE_FILL, ORANGE_EDGE, "CLS"
        elif token["kind"] == "z":
            face_color, edge_color, label = PURPLE_FILL, PURPLE_EDGE, "Z"
        else:
            continue
        box = FancyBboxPatch(
            (token["x"] - 0.40, STRIP_Y - 0.40),
            0.80,
            0.80,
            boxstyle="round,pad=0,rounding_size=0.40",
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=2.4 if token["kind"] == "cls" else 1.6,
            zorder=4,
        )
        ax.add_patch(box)
        ax.text(
            token["x"], STRIP_Y, label, ha="center", va="center", fontsize=9.5, fontweight="bold", zorder=5
        )
        draw_token_bar(ax, token["x"], face_color, edge_color, alpha=1.0)


def draw_visit_tokens(ax, tokens, epoch_alpha):
    for token in tokens:
        if token["kind"] != "visit":
            continue
        alpha = epoch_alpha.get(token["epoch"], 0.0)
        if alpha <= 0.0:
            continue
        color = BAND_COLORS[token["band"]]
        kind = token["token_type"]
        if kind == "d":
            ax.scatter(
                token["x"],
                STRIP_Y,
                marker="o",
                s=MARKER_SIZE,
                color=color,
                edgecolor="k",
                linewidth=1.0,
                alpha=alpha,
                zorder=4,
            )
        elif kind == "u":
            ax.scatter(
                token["x"],
                STRIP_Y,
                marker="v",
                s=MARKER_SIZE,
                facecolor="none",
                edgecolor=color,
                linewidth=2.0,
                alpha=alpha,
                zorder=4,
            )
        else:
            ax.scatter(
                token["x"],
                STRIP_Y,
                marker="s",
                s=MARKER_SIZE * 0.85,
                color=color,
                edgecolor="k",
                linewidth=1.0,
                alpha=alpha,
                zorder=4,
            )
        draw_token_bar(ax, token["x"], MAROON_FILL, MAROON_EDGE, alpha)


def draw_epoch_labels(ax, tokens, epoch_alpha):
    """ "Epoch N" centred under each visit group, as in plot_kilonova_token_strip.py.

    Below the bars rather than above them: the arcs into [CLS] sweep over the whole strip and would
    cross a label placed on top.
    """
    for epoch in EPOCHS:
        alpha = epoch_alpha.get(epoch, 0.0)
        if alpha <= 0.0:
            continue
        epoch_x = [token["x"] for token in tokens if token["kind"] == "visit" and token["epoch"] == epoch]
        ax.text(
            (min(epoch_x) + max(epoch_x)) / 2,
            EPOCH_LABEL_Y,
            f"Epoch {epoch}",
            ha="center",
            va="center",
            fontsize=12,
            alpha=alpha,
        )


def find_classification_token(tokens):
    for token in tokens:
        if token["kind"] == "cls":
            return token
    raise ValueError("no [CLS] token in the layout")


def draw_attention_arcs(ax, tokens, query, epoch_alpha):
    pooled = {"d": 0, "u": 0, "n": 0}
    for key in tokens:
        if key is query:
            continue
        alpha = 1.0 if key["kind"] == "z" else epoch_alpha.get(key["epoch"], 0.0)
        if alpha <= 0.0:
            continue
        if key["kind"] == "visit" and alpha > EPOCH_ARRIVED_ALPHA:
            pooled[key["token_type"]] += 1
        weight = attention_weight(key)
        arrow = FancyArrowPatch(
            (query["x"], STRIP_Y + 0.30),
            (key["x"], STRIP_Y + 0.30),
            connectionstyle=f"arc3,rad={-ARC_CURVATURE}",
            arrowstyle="-",
            color=ARC_COLOR,
            linewidth=0.4 + 3.2 * weight,
            alpha=(0.15 + 0.75 * weight) * alpha,
            zorder=1,
        )
        ax.add_patch(arrow)
    return pooled


def draw_classification_readout(ax, query):
    ax.annotate(
        "",
        xy=(query["x"], CLASSIFICATION_ARROW_BOTTOM),
        xytext=(query["x"], CLASSIFICATION_ARROW_TOP),
        arrowprops={"arrowstyle": "-|>", "color": "0.35", "linewidth": 1.6},
    )
    box = FancyBboxPatch(
        (query["x"] - 0.90, CLASSIFICATION_BOX_Y - 0.30),
        3.40,
        0.60,
        boxstyle="round,pad=0,rounding_size=0.12",
        facecolor="#eef2f7",
        edgecolor="0.35",
        linewidth=1.4,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        query["x"] + 0.80,
        CLASSIFICATION_BOX_Y,
        "P(KN)  vs  P(other)",
        ha="center",
        va="center",
        fontsize=9,
        zorder=4,
    )


def plot_attention_state(ax, epoch_alpha):
    """One state of the diagram: every epoch drawn at its own opacity (0 = not yet arrived)."""
    tokens = build_token_layout(EPOCHS)
    query = find_classification_token(tokens)

    pooled = draw_attention_arcs(ax, tokens, query, epoch_alpha)
    draw_global_tokens(ax, tokens)
    draw_visit_tokens(ax, tokens, epoch_alpha)
    draw_classification_readout(ax, query)
    draw_epoch_labels(ax, tokens, epoch_alpha)

    x_left, x_right = sequence_extent()
    # An arc3 connection of curvature r over a chord of length L peaks r*L/2 above the chord; size
    # the headroom from the widest chord the full 3-epoch sequence can produce so no arc is clipped.
    arc_headroom = ARC_CURVATURE * (x_right - x_left) / 2
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(CLASSIFICATION_BOX_Y - 0.7, STRIP_Y + arc_headroom + 0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    epochs_arrived = sum(1 for alpha in epoch_alpha.values() if alpha > EPOCH_ARRIVED_ALPHA)
    ax.set_title(
        f"{epochs_arrived} epoch{'s' if epochs_arrived != 1 else ''} of context"
        f"  —  pooled: {pooled['d']} detection{'s' if pooled['d'] != 1 else ''},"
        f" {pooled['u']} upper limit{'s' if pooled['u'] != 1 else ''},"
        f" {pooled['n']} not observed",
        fontsize=12,
    )


def build_frame_plan(fade_frames, fade_ms, hold_ms):
    """(epoch_alpha, duration_ms) per frame: each epoch ramps in, then the figure holds."""
    plan = []
    epoch_alpha = {}
    for epoch in EPOCHS:
        for step in range(1, fade_frames + 1):
            epoch_alpha[epoch] = step / fade_frames
            plan.append((dict(epoch_alpha), fade_ms))
        plan.append((dict(epoch_alpha), hold_ms))
    # The complete sequence is the point of the animation, so it lingers before the loop restarts.
    plan[-1] = (plan[-1][0], hold_ms * 2)
    return plan


def figure_to_image(figure):
    figure.canvas.draw()
    image = Image.frombuffer(
        "RGBA", figure.canvas.get_width_height(), figure.canvas.buffer_rgba(), "raw", "RGBA", 0, 1
    ).convert("RGB")
    plt.close(figure)
    return image


def draw_gif_frame(epoch_alpha, figsize, dpi):
    figure, axis = plt.subplots(figsize=figsize, dpi=dpi)
    plot_attention_state(axis, epoch_alpha)
    figure.suptitle(SUPTITLE, fontsize=11, y=0.985, va="top")
    figure.subplots_adjust(left=0.03, right=0.97, top=0.80, bottom=0.03)
    return figure_to_image(figure)


def write_gif(output_path, fade_frames, fade_ms, hold_ms, pixels, aspect):
    figsize = (pixels / 100.0, pixels / 100.0 / aspect)
    plan = build_frame_plan(fade_frames, fade_ms, hold_ms)
    images = [draw_gif_frame(epoch_alpha, figsize, dpi=100.0) for epoch_alpha, _duration in plan]
    durations = [duration for _epoch_alpha, duration in plan]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"wrote {output_path} ({len(images)} frames)")


def write_static_png(output_path):
    figure, axes = plt.subplots(3, 1, figsize=(12, 13))
    for axis, epoch_count in zip(axes, EPOCHS, strict=True):
        plot_attention_state(axis, {epoch: 1.0 for epoch in EPOCHS[:epoch_count]})
    figure.suptitle(SUPTITLE, fontsize=12)
    figure.subplots_adjust(left=0.03, right=0.97, top=0.85, bottom=0.02, hspace=0.42)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-png", action="store_true", help="write the 3-panel figure instead of the GIF"
    )
    parser.add_argument("--fade-frames", type=int, default=8, help="frames each epoch takes to fade in")
    parser.add_argument("--fade-ms", type=int, default=70, help="duration of one fade frame")
    parser.add_argument("--hold-ms", type=int, default=1600, help="pause once an epoch has fully arrived")
    parser.add_argument("--pixels", type=int, default=1400, help="GIF width in pixels")
    parser.add_argument("--aspect", type=float, default=2.1, help="GIF width/height")
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()

    paths = load_paths()
    if arguments.static_png:
        output_path = Path(arguments.output or paths.output_dir / "kilonova_attention_diagram.png")
        write_static_png(output_path)
    else:
        output_path = Path(arguments.output or paths.output_dir / "kilonova_attention_diagram.gif")
        write_gif(
            output_path,
            arguments.fade_frames,
            arguments.fade_ms,
            arguments.hold_ms,
            arguments.pixels,
            arguments.aspect,
        )


if __name__ == "__main__":
    main()
