"""Training overview diagram: one figure that follows a kilonova from its light curve to the classifier.

Five panels in pipeline order. The flow snakes through the figure so that every arrow between
panels is short. Top row, left to right: (a) input light curve, (b) token tensor. Right column,
top to bottom: (c) encoding of one token of that tensor, (d) input sequence. Left column, bottom
to top: (e) encoder block and classifier, under the light curve.

The example is one kilonova of the LANL grid, same event, viewing angle and noise seed as
plot_kilonova_token_diagram.py. Panel (a) draws the noiseless light curve of every Roman band and
the detections of the three deep-tier visits. Panel (b) holds the raw, un-normalised token values
of visit 1, tokenised with the live _vectorize_tokens. Panels (c) to (e) follow TokenEmbedding,
GlobalTokens and the encoder of training/model.py.
"""

import os
import sys

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from plot_kilonova_examples import load_ejecta_catalog, select_simulation
from plot_kilonova_token_diagram import (
    ANGLE_INDEX,
    BAND_COLORS,
    EJECTA_PARAMETERS,
    EJECTA_RUN_TYPE,
    EJECTA_WIND,
    NOISE_SEED,
    photometry_for_angle,
)

from kilonova.config import load_paths
from kilonova.photometry.roman_noise import BASE_CADENCE_DAYS, bands_observed_at_visit, build_tier_constants
from kilonova.photometry.spectra import ALL_ROMAN_BANDS, magnitudes_for_bands
from kilonova.simulation.early_windows import (
    load_lanl_catalog_metadata,
    load_lanl_wavelength_grid,
    load_simulation_spectra,
)

REPOSITORY_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(REPOSITORY_ROOT, "training"))
from openuniverse_data import BAND_TO_INDEX, TOKEN_TYPE_TO_INDEX, _vectorize_tokens  # noqa: E402
from plot_architecture_diagram import (  # noqa: E402
    BLUE_EDGE,
    BLUE_FILL,
    BOX_HEIGHT,
    COLUMN_X,
    GREEN_EDGE,
    GREEN_FILL,
    LINE_COLOR,
    ORANGE_EDGE,
    ORANGE_FILL,
    PANEL_EDGE,
    PANEL_FILL,
    PANEL_HALF_WIDTH,
    YELLOW_EDGE,
    YELLOW_FILL,
    draw_arrow,
    draw_box,
    draw_polyline_arrow,
    draw_residual,
    draw_sum_symbol,
)

OUTPUT_PATH = os.path.join(REPOSITORY_ROOT, "data", "openuniverse", "training_diagram.pdf")

# The module constants in training/model.py are defaults that the training command line can
# override, so the checkpoint is the only record of what was trained.
CHECKPOINT_PATH = os.path.join(REPOSITORY_ROOT, "training", "checkpoints", "kilonova_transformer-soup.ckpt")
TRAINED_HYPERPARAMETERS = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)[
    "hyper_parameters"
]
D_MODEL = TRAINED_HYPERPARAMETERS["d_model"]
D_FEEDFORWARD = TRAINED_HYPERPARAMETERS["d_feedforward"]
NUM_HEADS = TRAINED_HYPERPARAMETERS["num_heads"]
NUM_LAYERS = TRAINED_HYPERPARAMETERS["num_layers"]

# ---- the example kilonova
TIER = "deep"
# Closer than the token diagram (z=0.2) so this fast-fading event keeps detections at the third visit.
REDSHIFT = 0.05
NUMBER_OF_VISITS = 3
FIRST_VISIT_DAYS_SINCE_MERGER = 1.0
BAND_LETTERS = ["R", "Z", "Y", "J", "H", "F"]
# the token that panel (c) encodes and panel (d) highlights
ENCODED_BAND_LETTER = "Z"
FIELD_NAMES = [r"$\Delta t$", "band", "type", r"$m$", "$\\sigma_m$\n[mmag]", r"mask$_m$", r"mask$_\sigma$"]
MILLIMAGNITUDES_PER_MAGNITUDE = 1000
INDEX_TO_TOKEN_TYPE = {index: token_type for token_type, index in TOKEN_TYPE_TO_INDEX.items()}
INDEX_TO_BAND_LETTER = {index: letter for letter, index in BAND_TO_INDEX.items()}

# ---- panel (a): light curve
NUMBER_OF_MAGNITUDE_TICKS = 7
TIME_PADDING_DAYS = 1.5
TIME_TICK_SPACING_DAYS = 2

# ---- panel (b): token tensor
CELL = 1.0
# the magnitude needs five characters and the other cells one, so the columns are not equally wide
COLUMN_WIDTHS = [0.8, 1.0, 0.95, 1.7, 1.35, 1.4, 1.4]
COLUMN_EDGES = [sum(COLUMN_WIDTHS[:index]) for index in range(len(COLUMN_WIDTHS) + 1)]
COLUMN_CENTERS = [(left + right) / 2 for left, right in zip(COLUMN_EDGES[:-1], COLUMN_EDGES[1:], strict=True)]
GRID_WIDTH = COLUMN_EDGES[-1]
# shift of each visit grid behind the previous one, in cell units (back grids recede down-right)
CHANNEL_OFFSET = (0.55, -0.45)

# ---- panel (c): token encoding
RAW_CELL_WIDTH = 0.78
RAW_CELL_HEIGHT = 0.5
# room above the token row for its field names, the tallest of which takes two lines
FIELD_NAMES_HEIGHT = 0.72
OPERATION_BOX_HEIGHT = 0.5
VECTOR_CELL = 0.3
# cells drawn before and after the ellipsis of the D_MODEL vector
SHOWN_VECTOR_CELLS = (5, 2)
THIN_ARROW_STYLE = dict(arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color="black")

# ---- panel (d): input sequence
SEQUENCE_BAR_WIDTH = 0.5
SEQUENCE_BAR_HEIGHT = 1.6
GLOBAL_FILL = "0.88"

# ---- composition
FIGURE_WIDTH = 11.5
FIGURE_HEIGHT = 11.5
# panel rectangles in inches: (left, bottom, width, height). The token tensor is then shifted
# sideways until it sits on the vertical line of the token row below it.
INPUT_RECTANGLE = (0.85, 7.7, 3.4, 3.4)
TOKENS_RECTANGLE = (5.6, 7.25, 5.6, 3.8)
TOKEN_ENCODING_RECTANGLE = (5.4, 2.85, 5.2, 4.0)
SEQUENCE_RECTANGLE = (4.25, 0.1, 7.05, 2.15)
ENCODER_RECTANGLE = (0.4, 1.95, 4.0, 4.8)
# text sizes, chosen so that every panel reads like the encoder
INPUT_TICK_LABEL_SIZE = 13
INPUT_AXIS_LABEL_SIZE = 14
INPUT_BAND_LABEL_SCALE = 1.33
TOKENS_TEXT_SCALE = 1.6
# the field names are as wide as their column, so they cannot grow with the cell values
TOKENS_FIELD_NAME_SCALE = 1.1
TOKENS_VISIT_LABEL_SCALE = 1.3
TOKEN_ENCODING_TEXT_SCALE = 1.4
TOKEN_ENCODING_FIELD_NAME_SCALE = 1.1
SEQUENCE_TEXT_SCALE = 1.6
ENCODER_TEXT_SCALE = 1.1
PANEL_LABEL_SIZE = 16
CONNECTOR_COLOR = "0.35"
CONNECTOR_WIDTH = 2.2
# inches
BAND_LABEL_CLEARANCE = 1.0
CONNECTOR_GAP = 0.1
ENCODING_TO_SEQUENCE_DROP = 0.3


def load_example_kilonova():
    """Spectra, time grid and one noisy photometry realisation of the example kilonova."""
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

    constants = build_tier_constants(TIER)
    photometry = photometry_for_angle(
        simulation_spectra,
        time_days,
        wavelength_rest_aa,
        ANGLE_INDEX,
        constants,
        REDSHIFT,
        np.random.default_rng(NOISE_SEED),
    )
    return {
        "simulation_spectra": simulation_spectra,
        "time_days": time_days,
        "wavelength_rest_aa": wavelength_rest_aa,
        "constants": constants,
        "photometry": photometry,
    }


def light_curve_panel_data(example):
    # all six Roman filters on the curves (the model's band vocabulary is tier-agnostic); the
    # observed points follow the deep cadence, so R062 draws a curve and no point
    bands = list(ALL_ROMAN_BANDS)
    curves = true_light_curves(
        example["simulation_spectra"], example["time_days"], example["wavelength_rest_aa"], bands, REDSHIFT
    )
    visits = observed_visits(example["photometry"], example["constants"])
    return curves, visits, bands


def raw_tokens_first_visit(example):
    """Per-band raw token values of visit 1, one row per band in FIELD_NAMES order."""
    constants = example["constants"]
    photometry = example["photometry"]

    # the six-band vocabulary of the model: a band the tier never observes (R062 in deep) is shown
    # as an 'n' token here, although the live deep windows simply carry no R062 row
    bands = list(ALL_ROMAN_BANDS)
    visit_bands = bands_observed_at_visit(0, list(constants["bands"]), constants["anchor_band"])
    band_index, observed, detected, mag_observed, mag_err, mag_limit = [], [], [], [], [], []
    for band in bands:
        band_index.append(BAND_TO_INDEX[band[0]])
        observed.append(band in visit_bands)
        band_rows = photometry[photometry["band_letter"] == band[0]]
        if band_rows.empty:
            detected.append(False)
            mag_observed.append(np.nan)
            mag_err.append(np.nan)
            mag_limit.append(np.nan)
            continue
        days_from_visit = (band_rows["days_since_merger"] - FIRST_VISIT_DAYS_SINCE_MERGER).abs()
        nearest = band_rows.iloc[days_from_visit.argsort().iloc[0]]
        detected.append(
            bool(nearest["detected"] and np.isfinite(nearest["mag_observed"])) and band in visit_bands
        )
        mag_observed.append(nearest["mag_observed"])
        mag_err.append(nearest["mag_err"])
        mag_limit.append(nearest["mag_limit_5sigma"])
    tokens, _ = _vectorize_tokens(
        object_code=np.zeros(len(bands), dtype=np.int64),
        band_index=np.array(band_index),
        day=np.zeros(len(bands)),
        observed=np.array(observed),
        detected=np.array(detected),
        snr=None,
        mag_observed=np.array(mag_observed, dtype=float),
        mag_err=np.array(mag_err, dtype=float),
        mag_limit_5sigma=np.array(mag_limit, dtype=float),
    )
    rows = []
    for position in range(len(bands)):
        has_magnitude = np.isfinite(tokens["mag"][position])
        has_sigma = np.isfinite(tokens["sigma_mag"][position])
        sigma_millimagnitudes = tokens["sigma_mag"][position] * MILLIMAGNITUDES_PER_MAGNITUDE
        rows.append(
            [
                f"{tokens['day'][position]:.0f}",
                INDEX_TO_BAND_LETTER[int(tokens["band_index"][position])],
                INDEX_TO_TOKEN_TYPE[int(tokens["token_type_index"][position])],
                f"{tokens['mag'][position]:.2f}" if has_magnitude else "",
                f"{sigma_millimagnitudes:.0f}" if has_sigma else "",
                "1" if has_magnitude else "0",
                "1" if has_sigma else "0",
            ]
        )
    return rows


def true_light_curves(simulation_spectra, time_days, wavelength_rest_aa, bands, redshift):
    """{band: (days_since_merger, mag_true)} without noise, over the full LANL time grid."""
    days, magnitudes_per_band = [], {band: [] for band in bands}
    for time_index, rest_time in enumerate(time_days):
        flux_rest = simulation_spectra.get((ANGLE_INDEX, time_index))
        if flux_rest is None:
            continue
        magnitudes = magnitudes_for_bands(wavelength_rest_aa, flux_rest, redshift, bands)
        days.append(float(rest_time) * (1.0 + redshift))
        for band in bands:
            magnitudes_per_band[band].append(float(magnitudes[band]))
    days = np.array(days)
    return {band: (days, np.array(magnitudes_per_band[band])) for band in bands}


def observed_visits(photometry, constants):
    """[(visit_index, band, day, mag, mag_err)] for the detected bands among those the deep cadence
    observes at each visit."""
    rows = []
    for visit_index in range(NUMBER_OF_VISITS):
        day = FIRST_VISIT_DAYS_SINCE_MERGER + visit_index * BASE_CADENCE_DAYS
        for band in bands_observed_at_visit(visit_index, constants["bands"], constants["anchor_band"]):
            band_rows = photometry[photometry["band_letter"] == band[0]]
            nearest = band_rows.iloc[(band_rows["days_since_merger"] - day).abs().argsort().iloc[0]]
            if nearest["detected"] and np.isfinite(nearest["mag_observed"]):
                rows.append((visit_index, band, day, nearest["mag_observed"], nearest["mag_err"]))
    return rows


def draw_input_panel(axes, curves, visits, bands):
    time_min = FIRST_VISIT_DAYS_SINCE_MERGER - TIME_PADDING_DAYS
    time_max = FIRST_VISIT_DAYS_SINCE_MERGER + (NUMBER_OF_VISITS - 1) * BASE_CADENCE_DAYS + TIME_PADDING_DAYS
    magnitudes = np.array([magnitude for _, _, _, magnitude, _ in visits])
    faintest_magnitude = magnitudes.max() + 1.0
    brightest_magnitude = magnitudes.min() - 0.8

    for band in bands:
        days, magnitude = curves[band]
        in_range = (days >= time_min) & (days <= time_max)
        axes.plot(days[in_range], magnitude[in_range], color=BAND_COLORS[band], linewidth=1.6, zorder=1)
        # label where the curve leaves the panel: at the right edge, or where it drops below the y range
        visible_indices = np.where(in_range & (magnitude <= faintest_magnitude))[0]
        last_visible = visible_indices[-1]
        at_right_edge = last_visible == np.where(in_range)[0][-1]
        axes.text(
            time_max + 0.15 if at_right_edge else days[last_visible] + 0.15,
            magnitude[last_visible],
            band,
            color=BAND_COLORS[band],
            fontsize=9,
            ha="left",
            va="center" if at_right_edge else "bottom",
            zorder=2,
            clip_on=False,
            path_effects=[patheffects.withStroke(linewidth=0.6, foreground="k")],
        )

    for _, band, day, magnitude, magnitude_error in visits:
        axes.errorbar(
            day,
            magnitude,
            yerr=magnitude_error,
            fmt="o",
            ms=8,
            color=BAND_COLORS[band],
            ecolor=BAND_COLORS[band],
            elinewidth=1.4,
            mec="k",
            mew=0.8,
            zorder=3,
        )

    axes.set_xlim(time_min, time_max)
    axes.set_xticks(np.arange(0, time_max, TIME_TICK_SPACING_DAYS))
    axes.set_ylim(faintest_magnitude, brightest_magnitude)
    first_tick = np.ceil(brightest_magnitude)
    axes.set_yticks(np.arange(first_tick, first_tick + NUMBER_OF_MAGNITUDE_TICKS))
    axes.set_xlabel("days since merger")
    axes.set_ylabel("AB magnitude")


def draw_grid(axes, origin_x, origin_y, number_of_rows, fill, zorder):
    for row in range(number_of_rows):
        for column, width in enumerate(COLUMN_WIDTHS):
            axes.add_patch(
                Rectangle(
                    (origin_x + COLUMN_EDGES[column], origin_y + row * CELL),
                    width,
                    CELL,
                    facecolor=fill,
                    edgecolor="black",
                    linewidth=0.9,
                    zorder=zorder,
                )
            )
    axes.add_patch(
        Rectangle(
            (origin_x, origin_y),
            GRID_WIDTH,
            number_of_rows * CELL,
            fill=False,
            edgecolor="black",
            linewidth=1.8,
            zorder=zorder + 0.1,
        )
    )


def draw_tokens_panel(axes, front_values):
    number_of_rows = len(BAND_LETTERS)
    axes.set_aspect("equal")
    axes.axis("off")

    # back to front: visit 3 furthest, visit 1 in front
    for depth, visit_number in enumerate(reversed(range(1, NUMBER_OF_VISITS + 1))):
        shift = NUMBER_OF_VISITS - 1 - depth
        origin_x = shift * CHANNEL_OFFSET[0]
        origin_y = shift * CHANNEL_OFFSET[1]
        fill = "white" if visit_number == 1 else "0.97"
        draw_grid(axes, origin_x, origin_y, number_of_rows, fill, zorder=depth)
        # visit label just outside the top-right corner of its own grid
        axes.text(
            origin_x + GRID_WIDTH + 0.12,
            origin_y + number_of_rows * CELL + 0.08,
            f"visit {visit_number}",
            ha="left",
            va="bottom",
            fontsize=9.5,
            zorder=10,
        )

    # raw values of visit 1 in the front grid; rows are drawn top-down (Z first), cell rows bottom-up
    band_column = FIELD_NAMES.index("band")
    for row_position, values in enumerate(front_values):
        cell_row = number_of_rows - 1 - row_position
        band_full_name = [band for band in BAND_COLORS if band[0] == values[band_column]][0]
        for column, text in enumerate(values):
            is_band = column == band_column
            axes.text(
                COLUMN_CENTERS[column],
                (cell_row + 0.5) * CELL,
                text,
                ha="center",
                va="center",
                fontsize=10 if is_band else 8.5,
                zorder=20,
                color=BAND_COLORS[band_full_name] if is_band else "black",
                path_effects=[patheffects.withStroke(linewidth=0.6, foreground="k")] if is_band else None,
            )

    for column, field in enumerate(FIELD_NAMES):
        axes.text(
            COLUMN_CENTERS[column], number_of_rows * CELL + 0.25, field, ha="center", va="bottom", fontsize=10
        )

    width = GRID_WIDTH + (NUMBER_OF_VISITS - 1) * CHANNEL_OFFSET[0]
    axes.set_xlim(-0.2, width + 1.6)
    axes.set_ylim((NUMBER_OF_VISITS - 1) * CHANNEL_OFFSET[1] - 0.3, number_of_rows * CELL + 1.3)
    return {
        "left": (0.0, number_of_rows * CELL / 2),
        "right": (width, number_of_rows * CELL / 2),
        "bottom": (width / 2, (NUMBER_OF_VISITS - 1) * CHANNEL_OFFSET[1]),
    }


def band_color(letter):
    return BAND_COLORS[[band for band in BAND_COLORS if band[0] == letter][0]]


def band_stroke(linewidth):
    return [patheffects.withStroke(linewidth=linewidth, foreground="k")]


def draw_operation_box(axes, x_center, y_bottom, width, text):
    axes.add_patch(
        FancyBboxPatch(
            (x_center - width / 2, y_bottom),
            width,
            OPERATION_BOX_HEIGHT,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            facecolor="0.93",
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )
    )
    axes.text(
        x_center, y_bottom + OPERATION_BOX_HEIGHT / 2, text, ha="center", va="center", fontsize=9, zorder=4
    )


def draw_thin_arrow(axes, start, end):
    axes.add_patch(FancyArrowPatch(start, end, shrinkA=0, shrinkB=0, zorder=2, **THIN_ARROW_STYLE))


def draw_vector(axes, x_left, y_bottom):
    x = x_left
    for number, ellipsis in ((SHOWN_VECTOR_CELLS[0], True), (SHOWN_VECTOR_CELLS[1], False)):
        for _ in range(number):
            axes.add_patch(
                Rectangle(
                    (x, y_bottom),
                    VECTOR_CELL,
                    VECTOR_CELL,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=3,
                )
            )
            x += VECTOR_CELL
        if ellipsis:
            axes.text(
                x + 0.3 * VECTOR_CELL,
                y_bottom + VECTOR_CELL / 2,
                "…",
                ha="center",
                va="center",
                fontsize=9,
                zorder=4,
            )
            x += 0.6 * VECTOR_CELL
    axes.add_patch(
        Rectangle(
            (x_left, y_bottom),
            x - x_left,
            VECTOR_CELL,
            fill=False,
            edgecolor="black",
            linewidth=2.0,
            zorder=5,
        )
    )
    return x


def draw_token_row(axes, y_raw, token_values):
    """One row of the token tensor of panel (b) with its field names; returns its width."""
    for index, (name, value) in enumerate(zip(FIELD_NAMES, token_values, strict=True)):
        x = index * RAW_CELL_WIDTH
        axes.add_patch(
            Rectangle(
                (x, y_raw),
                RAW_CELL_WIDTH,
                RAW_CELL_HEIGHT,
                facecolor="white",
                edgecolor="black",
                linewidth=0.9,
                zorder=3,
            )
        )
        is_band = name == "band"
        axes.text(
            x + RAW_CELL_WIDTH / 2,
            y_raw + RAW_CELL_HEIGHT / 2,
            value,
            ha="center",
            va="center",
            fontsize=10 if is_band else 9,
            zorder=4,
            color=band_color(value) if is_band else "black",
            path_effects=band_stroke(0.6) if is_band else None,
        )
        axes.text(
            x + RAW_CELL_WIDTH / 2,
            y_raw + RAW_CELL_HEIGHT + 0.08,
            name,
            ha="center",
            va="bottom",
            fontsize=9.5,
        )
    raw_width = len(FIELD_NAMES) * RAW_CELL_WIDTH
    axes.add_patch(
        Rectangle(
            (0.0, y_raw), raw_width, RAW_CELL_HEIGHT, fill=False, edgecolor="black", linewidth=1.6, zorder=4
        )
    )
    return raw_width


def draw_token_encoding(axes, token_values):
    """One raw token row -> Time2Vec and embedding + linear -> the token encoding vector."""
    # ---- the raw token row
    y_raw = 2.9
    raw_width = draw_token_row(axes, y_raw, token_values)

    # ---- two operations: delta t -> Time2Vec ; the other six fields -> embedding + linear
    y_box = 1.75
    x_time = RAW_CELL_WIDTH / 2
    x_content = (RAW_CELL_WIDTH + raw_width) / 2
    y_bracket = y_raw - 0.12
    axes.plot(
        [RAW_CELL_WIDTH + 0.06, RAW_CELL_WIDTH + 0.06, raw_width, raw_width],
        [y_raw - 0.04, y_bracket, y_bracket, y_raw - 0.04],
        color="0.25",
        linewidth=0.8,
    )
    draw_thin_arrow(axes, (x_time, y_raw - 0.04), (x_time, y_box + OPERATION_BOX_HEIGHT))
    draw_thin_arrow(axes, (x_content, y_bracket), (x_content, y_box + OPERATION_BOX_HEIGHT))
    draw_operation_box(axes, x_time, y_box, 1.15, "Time2Vec")
    draw_operation_box(axes, x_content, y_box, 2.6, "embedding + linear")

    # ---- sum -> token encoding
    y_sum = 1.05
    axes.text(x_content, y_sum, r"$\oplus$", ha="center", va="center", fontsize=15, zorder=4)
    draw_thin_arrow(axes, (x_content, y_box), (x_content, y_sum + 0.16))
    axes.plot([x_time, x_time], [y_box, y_sum], color="black", linewidth=1.0, zorder=2)
    draw_thin_arrow(axes, (x_time, y_sum), (x_content - 0.16, y_sum))

    y_vector = 0.2
    vector_width = (SHOWN_VECTOR_CELLS[0] + SHOWN_VECTOR_CELLS[1] + 0.6) * VECTOR_CELL
    x_vector = x_content - vector_width / 2
    draw_thin_arrow(axes, (x_content, y_sum - 0.16), (x_content, y_vector + VECTOR_CELL))
    x_vector_right = draw_vector(axes, x_vector, y_vector)
    axes.text(
        x_content,
        y_vector - 0.1,
        f"token encoding  $\\mathbb{{R}}^{{{D_MODEL}}}$",
        ha="center",
        va="top",
        fontsize=9,
    )
    return {
        "raw_width": raw_width,
        "y_raw": y_raw,
        "x_content": x_content,
        "x_vector_right": x_vector_right,
        "y_vector": y_vector,
    }


def draw_token_encoding_panel(axes, token_values):
    axes.set_aspect("equal")
    axes.axis("off")
    geometry = draw_token_encoding(axes, token_values)
    y_bottom = geometry["y_vector"] - 0.5
    axes.set_xlim(-0.3, geometry["raw_width"] + 0.1)
    axes.set_ylim(y_bottom, geometry["y_raw"] + RAW_CELL_HEIGHT + FIELD_NAMES_HEIGHT + 0.05)
    return {
        "token_row_top": (
            geometry["raw_width"] / 2,
            geometry["y_raw"] + RAW_CELL_HEIGHT + FIELD_NAMES_HEIGHT,
        ),
        "top_left": (0.0, geometry["y_raw"] + RAW_CELL_HEIGHT + FIELD_NAMES_HEIGHT),
        "bottom": (geometry["x_content"], y_bottom),
    }


def draw_input_sequence(axes, x_left=0.0, y_center=0.0, bar_width=SEQUENCE_BAR_WIDTH):
    """The input sequence as one upright bar per token, read left to right."""
    names = ["[CLS]", "$z$"] + [letter for _ in range(NUMBER_OF_VISITS) for letter in BAND_LETTERS]
    y_bottom = y_center - SEQUENCE_BAR_HEIGHT / 2
    y_top = y_center + SEQUENCE_BAR_HEIGHT / 2
    example_top = None
    for index, name in enumerate(names):
        x = x_left + index * bar_width
        is_band = name in BAND_LETTERS
        axes.add_patch(
            Rectangle(
                (x, y_bottom),
                bar_width,
                SEQUENCE_BAR_HEIGHT,
                facecolor="white" if is_band else GLOBAL_FILL,
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )
        )
        axes.text(
            x + bar_width / 2,
            y_bottom - 0.08,
            name,
            ha="center",
            va="top",
            fontsize=8,
            rotation=90 if name == "[CLS]" else 0,
            color=band_color(name) if is_band else "black",
            path_effects=band_stroke(0.5) if is_band else None,
        )
        if index == 2 + BAND_LETTERS.index(ENCODED_BAND_LETTER):
            example_top = (x + bar_width / 2, y_top)
            axes.add_patch(
                Rectangle(
                    (x, y_bottom),
                    bar_width,
                    SEQUENCE_BAR_HEIGHT,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.0,
                    zorder=5,
                )
            )
    total_width = len(names) * bar_width
    axes.add_patch(
        Rectangle(
            (x_left, y_bottom),
            total_width,
            SEQUENCE_BAR_HEIGHT,
            fill=False,
            edgecolor="black",
            linewidth=1.4,
            zorder=4,
        )
    )
    y_brace = y_bottom - 0.5
    for visit_index in range(NUMBER_OF_VISITS):
        x_visit_left = x_left + (2 + visit_index * len(BAND_LETTERS)) * bar_width
        x_visit_right = x_visit_left + len(BAND_LETTERS) * bar_width
        axes.plot([x_visit_left, x_visit_left], [y_bottom, y_top], color="black", linewidth=1.4, zorder=4)
        axes.plot(
            [x_visit_left + 0.04, x_visit_left + 0.04, x_visit_right - 0.04, x_visit_right - 0.04],
            [y_brace + 0.08, y_brace, y_brace, y_brace + 0.08],
            color="0.25",
            linewidth=0.8,
        )
        axes.text(
            (x_visit_left + x_visit_right) / 2,
            y_brace - 0.08,
            f"visit {visit_index + 1}",
            ha="center",
            va="top",
            fontsize=8.5,
        )
    axes.text(x_left + total_width / 2, y_top + 0.1, "input sequence", ha="center", va="bottom", fontsize=9.5)
    return {
        "example_top": example_top,
        "left": (x_left, y_center),
        "right": (x_left + total_width, y_center),
        "top_left": (x_left, y_top),
        "bottom": y_brace - 0.45,
    }


def draw_encoder_panel(axes):
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

    y_attention_norm = 6.15
    y_attention = 7.30
    y_attention_sum = 8.25
    y_feedforward_norm = 9.10
    y_feedforward = 10.15
    y_feedforward_sum = 10.95

    axes.plot(
        [COLUMN_X, COLUMN_X],
        [y_panel_bottom + 0.15, y_attention_norm - BOX_HEIGHT / 2],
        color=LINE_COLOR,
        linewidth=1.5,
        zorder=2,
    )
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

    y_classifier = 12.10
    draw_arrow(axes, y_panel_top, y_classifier - BOX_HEIGHT / 2)
    draw_box(axes, y_classifier, ["Classifier"], GREEN_FILL, GREEN_EDGE)

    axes.set_xlim(-3.25, 2.45)
    axes.set_ylim(y_panel_bottom - 0.2, y_classifier + BOX_HEIGHT / 2 + 0.2)
    axes.set_aspect("equal")
    axes.axis("off")
    return {"input": (COLUMN_X, y_panel_bottom)}


def scale_text(axes, factor, only=None, excluding=()):
    for text in axes.texts:
        content = text.get_text()
        if content in excluding:
            continue
        if only is not None and content not in only:
            continue
        text.set_fontsize(text.get_fontsize() * factor)


def add_axes_in_inches(figure, rectangle):
    left, bottom, width, height = rectangle
    return figure.add_axes(
        [left / FIGURE_WIDTH, bottom / FIGURE_HEIGHT, width / FIGURE_WIDTH, height / FIGURE_HEIGHT]
    )


def inches_point(figure, axes, data_point):
    x_fraction, y_fraction = figure.transFigure.inverted().transform(axes.transData.transform(data_point))
    return x_fraction * FIGURE_WIDTH, y_fraction * FIGURE_HEIGHT


def draw_connector(figure, points_in_inches):
    """A polyline through the points that ends in an arrow head."""
    points = [(x / FIGURE_WIDTH, y / FIGURE_HEIGHT) for x, y in points_in_inches]
    if len(points) > 2:
        x_values = [x for x, _ in points[:-1]]
        y_values = [y for _, y in points[:-1]]
        figure.add_artist(
            Line2D(
                x_values,
                y_values,
                transform=figure.transFigure,
                color=CONNECTOR_COLOR,
                linewidth=CONNECTOR_WIDTH,
                solid_capstyle="projecting",
            )
        )
    figure.add_artist(
        FancyArrowPatch(
            points[-2],
            points[-1],
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=18,
            shrinkA=0,
            shrinkB=0,
            color=CONNECTOR_COLOR,
            linewidth=CONNECTOR_WIDTH,
        )
    )


def add_panel_label(figure, x_inches, y_inches, label):
    figure.text(
        x_inches / FIGURE_WIDTH,
        y_inches / FIGURE_HEIGHT,
        label,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
    )


def shift_axes(figure, axes, shift_inches):
    bounds = axes.get_position()
    axes.set_position([bounds.x0 + shift_inches / FIGURE_WIDTH, bounds.y0, bounds.width, bounds.height])
    figure.canvas.draw()


def main():
    example = load_example_kilonova()
    curves, visits, bands = light_curve_panel_data(example)
    front_values = raw_tokens_first_visit(example)

    plt.rcParams["font.family"] = "serif"
    figure = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    axes_input = add_axes_in_inches(figure, INPUT_RECTANGLE)
    axes_tokens = add_axes_in_inches(figure, TOKENS_RECTANGLE)
    axes_token_encoding = add_axes_in_inches(figure, TOKEN_ENCODING_RECTANGLE)
    axes_sequence = add_axes_in_inches(figure, SEQUENCE_RECTANGLE)
    axes_encoder = add_axes_in_inches(figure, ENCODER_RECTANGLE)

    draw_input_panel(axes_input, curves, visits, bands)
    axes_input.tick_params(labelsize=INPUT_TICK_LABEL_SIZE)
    axes_input.xaxis.label.set_size(INPUT_AXIS_LABEL_SIZE)
    axes_input.yaxis.label.set_size(INPUT_AXIS_LABEL_SIZE)
    scale_text(axes_input, INPUT_BAND_LABEL_SCALE)
    tokens_anchors = draw_tokens_panel(axes_tokens, front_values)
    visit_labels = [f"visit {visit_number}" for visit_number in (1, 2, 3)]
    scale_text(axes_tokens, TOKENS_TEXT_SCALE, excluding=FIELD_NAMES + visit_labels)
    scale_text(axes_tokens, TOKENS_VISIT_LABEL_SCALE, only=visit_labels)
    scale_text(axes_tokens, TOKENS_FIELD_NAME_SCALE, only=FIELD_NAMES)
    token_encoding_anchors = draw_token_encoding_panel(
        axes_token_encoding, front_values[BAND_LETTERS.index(ENCODED_BAND_LETTER)]
    )
    scale_text(axes_token_encoding, TOKEN_ENCODING_TEXT_SCALE, excluding=FIELD_NAMES)
    scale_text(axes_token_encoding, TOKEN_ENCODING_FIELD_NAME_SCALE, only=FIELD_NAMES)
    encoder_anchors = draw_encoder_panel(axes_encoder)
    scale_text(axes_encoder, ENCODER_TEXT_SCALE)

    sequence_anchors = draw_input_sequence(axes_sequence)
    scale_text(axes_sequence, SEQUENCE_TEXT_SCALE)
    axes_sequence.set_xlim(sequence_anchors["left"][0] - 0.3, sequence_anchors["right"][0] + 0.3)
    axes_sequence.set_ylim(sequence_anchors["bottom"], sequence_anchors["top_left"][1] + 0.5)
    axes_sequence.set_aspect("equal")
    axes_sequence.axis("off")

    # equal-aspect panels shrink inside their rectangle, so their final position is only known after a draw
    figure.canvas.draw()

    # the token tensor sits on the vertical line of the token row of panel (c)
    token_row_x, token_row_y = inches_point(
        figure, axes_token_encoding, token_encoding_anchors["token_row_top"]
    )
    tokens_bottom_x, _ = inches_point(figure, axes_tokens, tokens_anchors["bottom"])
    shift_axes(figure, axes_tokens, token_row_x - tokens_bottom_x)

    # (a) -> (b): light curve to the token tensor
    tokens_left_x, tokens_middle_y = inches_point(figure, axes_tokens, tokens_anchors["left"])
    input_right_x = INPUT_RECTANGLE[0] + INPUT_RECTANGLE[2] + BAND_LABEL_CLEARANCE
    draw_connector(
        figure, [(input_right_x, tokens_middle_y), (tokens_left_x - CONNECTOR_GAP, tokens_middle_y)]
    )

    # (b) -> (c): one row of the tensor is the token whose encoding panel (c) follows
    _, tokens_bottom_y = inches_point(figure, axes_tokens, tokens_anchors["bottom"])
    draw_connector(figure, [(token_row_x, tokens_bottom_y - CONNECTOR_GAP), (token_row_x, token_row_y)])

    # (c) -> (d): the encoded token takes its slot in the input sequence
    encoding_x, encoding_y = inches_point(figure, axes_token_encoding, token_encoding_anchors["bottom"])
    slot_x, slot_y = inches_point(figure, axes_sequence, sequence_anchors["example_top"])
    draw_connector(
        figure,
        [
            (encoding_x, encoding_y),
            (encoding_x, encoding_y - ENCODING_TO_SEQUENCE_DROP),
            (slot_x, encoding_y - ENCODING_TO_SEQUENCE_DROP),
            (slot_x, slot_y + 0.03),
        ],
    )

    # (d) -> (e): the sequence enters the first encoder block from below
    sequence_left_x, sequence_middle_y = inches_point(figure, axes_sequence, sequence_anchors["left"])
    encoder_x, encoder_y = inches_point(figure, axes_encoder, encoder_anchors["input"])
    draw_connector(
        figure,
        [
            (sequence_left_x - CONNECTOR_GAP, sequence_middle_y),
            (encoder_x, sequence_middle_y),
            (encoder_x, encoder_y),
        ],
    )

    top_label_height = INPUT_RECTANGLE[1] + INPUT_RECTANGLE[3] + 0.05
    add_panel_label(figure, INPUT_RECTANGLE[0], top_label_height, "(a)")
    add_panel_label(figure, axes_tokens.get_position().x0 * FIGURE_WIDTH, top_label_height, "(b)")
    encoding_left_x, encoding_top_y = inches_point(
        figure, axes_token_encoding, token_encoding_anchors["top_left"]
    )
    add_panel_label(figure, encoding_left_x - 0.55, encoding_top_y - 0.1, "(c)")
    _, sequence_top_y = inches_point(figure, axes_sequence, sequence_anchors["top_left"])
    add_panel_label(figure, sequence_left_x, sequence_top_y + 0.08, "(d)")
    encoder_bounds = axes_encoder.get_position()
    add_panel_label(figure, encoder_bounds.x0 * FIGURE_WIDTH, encoder_bounds.y1 * FIGURE_HEIGHT - 0.3, "(e)")

    figure.savefig(OUTPUT_PATH, bbox_inches="tight")
    print("saved", OUTPUT_PATH)


if __name__ == "__main__":
    main()
