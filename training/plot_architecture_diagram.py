"""Block diagram of the KilonovaTransformer forward pass (model.py), bottom to top:
raw per-token observations -> TokenEmbedding (+ Time2Vec) -> [CLS, Z, token_1..N] sequence ->
6x pre-norm encoder block (opened up) -> CLS -> classification head -> P(KN).
Purely illustrative (no trained weights involved).
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

OUTPUT_PATH = "plots/architecture_diagram"

ORANGE_FILL, ORANGE_EDGE = "#f6cb8e", "#d99a2b"
PURPLE_FILL, PURPLE_EDGE = "#c7bfe3", "#8d7fc4"
PINK_FILL, PINK_EDGE = "#eeb3c1", "#c26b81"
MAROON_FILL, MAROON_EDGE = "#9c4569", "#7a3252"
YELLOW_FILL, YELLOW_EDGE = "#f2eaa8", "#c9bd5a"
BLUE_FILL, BLUE_EDGE = "#a9d5ec", "#5f9ec4"
GREEN_FILL, GREEN_EDGE = "#b7dfb9", "#6aa86d"
PANEL_FILL, PANEL_EDGE = "#ececec", "#3c3c3c"
LINE_COLOR = "#222222"

D_MODEL = 192
NUM_HEADS = 6
NUM_LAYERS = 6
D_FEEDFORWARD = 768

COLUMN_X = 0.0
BOX_WIDTH = 3.1
BOX_HEIGHT = 0.78
PANEL_HALF_WIDTH = 2.25
RESIDUAL_X = -1.95

INPUT_SLOTS = ["bar", "bar", "bar", "bar", "dots", "bar"]
INPUT_SPACING = 0.68


def draw_box(
    axes, y_center, lines, face_color, edge_color, width=BOX_WIDTH, height=BOX_HEIGHT, font_size=11.5
):
    box = FancyBboxPatch(
        (COLUMN_X - width / 2, y_center - height / 2),
        width,
        height,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=face_color,
        edgecolor=edge_color,
        linewidth=1.4,
        zorder=3,
    )
    axes.add_patch(box)
    axes.text(
        COLUMN_X,
        y_center,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=font_size,
        zorder=4,
        linespacing=1.35,
    )


def draw_bar(axes, x_center, y_center, width, height, face_color, edge_color):
    axes.add_patch(
        Rectangle(
            (x_center - width / 2, y_center - height / 2),
            width,
            height,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=1.2,
            zorder=3,
        )
    )


def draw_arrow(axes, y_bottom, y_top, x_center=COLUMN_X):
    axes.annotate(
        "",
        xy=(x_center, y_top),
        xytext=(x_center, y_bottom),
        arrowprops={
            "arrowstyle": "-|>",
            "color": LINE_COLOR,
            "linewidth": 1.5,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 14,
        },
        zorder=2,
    )


def draw_polyline_arrow(axes, points):
    """Right-angle connector; only the final segment carries the arrowhead."""
    for start, end in zip(points[:-1], points[1:-1], strict=False):
        axes.plot([start[0], end[0]], [start[1], end[1]], color=LINE_COLOR, linewidth=1.5, zorder=2)
    axes.annotate(
        "",
        xy=points[-1],
        xytext=points[-2],
        arrowprops={
            "arrowstyle": "-|>",
            "color": LINE_COLOR,
            "linewidth": 1.5,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 14,
        },
        zorder=2,
    )


def draw_sum_symbol(axes, x_center, y_center, radius=0.19):
    axes.add_patch(
        Circle((x_center, y_center), radius, facecolor="white", edgecolor=LINE_COLOR, linewidth=1.5, zorder=5)
    )
    axes.plot(
        [x_center - radius, x_center + radius],
        [y_center, y_center],
        color=LINE_COLOR,
        linewidth=1.4,
        zorder=6,
    )
    axes.plot(
        [x_center, x_center],
        [y_center - radius, y_center + radius],
        color=LINE_COLOR,
        linewidth=1.4,
        zorder=6,
    )


def draw_time_encoding_glyph(axes, x_center, y_center, radius=0.32):
    """A straight ramp, not a sine: TIME2VEC_FREQUENCIES = 0 leaves only the linear term."""
    axes.add_patch(
        Circle((x_center, y_center), radius, facecolor="white", edgecolor=LINE_COLOR, linewidth=1.5, zorder=5)
    )
    axes.plot(
        [x_center - radius * 0.6, x_center + radius * 0.6],
        [y_center - radius * 0.55, y_center + radius * 0.55],
        color=LINE_COLOR,
        linewidth=1.6,
        zorder=6,
    )


def draw_residual(axes, y_split, y_join, x_join=COLUMN_X):
    """Skip connection: leaves the trunk below the sub-layer, returns into the sum above it."""
    draw_polyline_arrow(
        axes,
        [
            (COLUMN_X, y_split),
            (RESIDUAL_X, y_split),
            (RESIDUAL_X, y_join),
            (x_join - 0.19, y_join),
        ],
    )


def draw_input_row(axes, y_center):
    x_start = COLUMN_X - INPUT_SPACING * (len(INPUT_SLOTS) - 1) / 2
    for index, slot in enumerate(INPUT_SLOTS):
        x_position = x_start + index * INPUT_SPACING
        if slot == "dots":
            axes.text(x_position, y_center, "...", ha="center", va="center", fontsize=15)
        else:
            draw_bar(axes, x_position, y_center, 0.26, 0.78, MAROON_FILL, MAROON_EDGE)
    return x_start


def plot_architecture_diagram(plot=True):
    plt.rcParams["font.family"] = "serif"
    figure, axes = plt.subplots(figsize=(5.2, 13.2))

    y_input = 0.85
    draw_input_row(axes, y_input)
    axes.text(COLUMN_X, y_input - 0.85, "Inputs", ha="center", va="center", fontsize=13)

    y_embedding = 2.55
    draw_arrow(axes, y_input + 0.42, y_embedding - BOX_HEIGHT / 2 - 0.22)
    draw_box(axes, y_embedding, ["Token Embedding"], PINK_FILL, PINK_EDGE)

    y_time_sum = 3.75
    draw_arrow(axes, y_embedding + 0.5, y_time_sum - 0.19)
    draw_sum_symbol(axes, COLUMN_X, y_time_sum)
    draw_time_encoding_glyph(axes, COLUMN_X - 1.05, y_time_sum)
    draw_polyline_arrow(axes, [(COLUMN_X - 0.73, y_time_sum), (COLUMN_X - 0.19, y_time_sum)])
    axes.text(COLUMN_X - 1.5, y_time_sum, "Time2Vec", ha="right", va="center", fontsize=12)

    y_prepend = 4.75
    draw_arrow(axes, y_time_sum + 0.19, y_prepend - BOX_HEIGHT / 2)
    draw_box(
        axes, y_prepend, ["Prepend [CLS] and $z$ tokens"], PURPLE_FILL, PURPLE_EDGE, width=3.7, font_size=11
    )

    y_panel_bottom = 5.55
    y_panel_top = 11.30
    axes.add_patch(
        FancyBboxPatch(
            (COLUMN_X - PANEL_HALF_WIDTH, y_panel_bottom),
            2 * PANEL_HALF_WIDTH,
            y_panel_top - y_panel_bottom,
            boxstyle="round,pad=0,rounding_size=0.22",
            facecolor=PANEL_FILL,
            edgecolor=PANEL_EDGE,
            linewidth=1.6,
            zorder=1,
        )
    )
    axes.text(
        COLUMN_X - PANEL_HALF_WIDTH - 0.35,
        (y_panel_bottom + y_panel_top) / 2,
        f"${NUM_LAYERS}\\times$",
        ha="right",
        va="center",
        fontsize=15,
    )

    draw_arrow(axes, y_prepend + BOX_HEIGHT / 2, y_panel_bottom)

    y_attention_norm = 6.15
    y_attention = 7.30
    y_attention_sum = 8.25
    y_feedforward_norm = 9.10
    y_feedforward = 10.15
    y_feedforward_sum = 10.95

    draw_arrow(axes, y_panel_bottom, y_attention_norm - BOX_HEIGHT / 2)
    draw_box(axes, y_attention_norm, ["Layer Norm"], YELLOW_FILL, YELLOW_EDGE)

    draw_arrow(axes, y_attention_norm + BOX_HEIGHT / 2, y_attention - 0.5)
    draw_box(
        axes,
        y_attention,
        ["Multi-Head Attention", f"{NUM_HEADS} heads"],
        ORANGE_FILL,
        ORANGE_EDGE,
        height=1.0,
        font_size=11,
    )
    for x_offset in (-0.85, 0.0, 0.85):
        draw_polyline_arrow(
            axes,
            [
                (COLUMN_X, y_attention_norm + BOX_HEIGHT / 2 + 0.12),
                (COLUMN_X + x_offset, y_attention_norm + BOX_HEIGHT / 2 + 0.12),
                (COLUMN_X + x_offset, y_attention - 0.5),
            ],
        )

    draw_arrow(axes, y_attention + 0.5, y_attention_sum - 0.19)
    draw_sum_symbol(axes, COLUMN_X, y_attention_sum)
    draw_residual(axes, y_panel_bottom + 0.15, y_attention_sum)

    draw_arrow(axes, y_attention_sum + 0.19, y_feedforward_norm - BOX_HEIGHT / 2)
    draw_box(axes, y_feedforward_norm, ["Layer Norm"], YELLOW_FILL, YELLOW_EDGE)

    draw_arrow(axes, y_feedforward_norm + BOX_HEIGHT / 2, y_feedforward - 0.5)
    draw_box(
        axes,
        y_feedforward,
        ["Feed Forward", f"$d_{{ff}} = {D_FEEDFORWARD}$"],
        BLUE_FILL,
        BLUE_EDGE,
        height=1.0,
        font_size=11,
    )

    draw_arrow(axes, y_feedforward + 0.5, y_feedforward_sum - 0.19)
    draw_sum_symbol(axes, COLUMN_X, y_feedforward_sum)
    draw_residual(axes, y_attention_sum + 0.35, y_feedforward_sum)

    draw_arrow(axes, y_feedforward_sum + 0.19, y_panel_top)

    y_take_cls = 12.00
    draw_arrow(axes, y_panel_top, y_take_cls - BOX_HEIGHT / 2)
    draw_box(axes, y_take_cls, ["[CLS] Pooling"], GREEN_FILL, GREEN_EDGE)

    y_head_norm = 13.00
    draw_arrow(axes, y_take_cls + BOX_HEIGHT / 2, y_head_norm - BOX_HEIGHT / 2)
    draw_box(axes, y_head_norm, ["Layer Norm"], YELLOW_FILL, YELLOW_EDGE)

    y_linear = 14.00
    draw_arrow(axes, y_head_norm + BOX_HEIGHT / 2, y_linear - BOX_HEIGHT / 2)
    draw_box(axes, y_linear, [f"Linear  (${D_MODEL} \\rightarrow 2$)"], PURPLE_FILL, PURPLE_EDGE)

    y_softmax = 15.00
    draw_arrow(axes, y_linear + BOX_HEIGHT / 2, y_softmax - BOX_HEIGHT / 2)
    draw_box(axes, y_softmax, ["Softmax"], GREEN_FILL, GREEN_EDGE)

    draw_arrow(axes, y_softmax + BOX_HEIGHT / 2, y_softmax + 1.05)
    axes.text(
        COLUMN_X,
        y_softmax + 1.65,
        "Output\nProbabilities",
        ha="center",
        va="center",
        fontsize=13,
        linespacing=1.35,
    )

    axes.set_xlim(-3.25, 2.45)
    axes.set_ylim(-0.45, 16.85)
    axes.set_aspect("equal")
    axes.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH + ".pdf", bbox_inches="tight")
    figure.savefig(OUTPUT_PATH + ".png", dpi=220, bbox_inches="tight")
    if plot:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    plot_architecture_diagram(plot=False)
