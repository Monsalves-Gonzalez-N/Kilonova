"""3x2 grid of confusion matrices on the test-only cache: rows = 1 / 2 / 3 epochs, columns = non-z / z.

Same mechanism, data, normalization and threshold as the confusion matrix in
run_evaluation_test_only.py: one model, one test set, and OpenUniverseWindowDataset's
force_epochs (truncates the window to the first N visits) and force_redshift (False masks z as NaN).
Writes one PDF per threshold in THRESHOLDS.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import openuniverse_data as openuniverse_data_module
import torch
from openuniverse_data import (
    GROUP_KEY_VERSION,
    GROUP_ORDER,
    OpenUniverseWindowDataset,
    collate_token_windows,
)
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from train_lightning import MODEL_INPUT_KEYS, LitKilonova

DATA_DIR = "data/openuniverse" if os.path.isdir("data/openuniverse") else "../data/openuniverse"
CHECKPOINT = os.environ.get("KN_CHECKPOINT", "checkpoints/izc-trainonly-2026-09-30/kilonova_transformer-soup.ckpt")
PLOTS_DIR = "plots"
THRESHOLDS = (0.5, 0.2)
EPOCHS = (1, 2, 3)
HAS_Z = (False, True)

device = "cuda" if torch.cuda.is_available() else "cpu"

cached = np.load(f"{DATA_DIR}/openuniverse_tokens_test.npz", allow_pickle=False)
cached_version = int(cached["group_key_version"]) if "group_key_version" in cached else 1
assert cached_version == GROUP_KEY_VERSION, "test cache was cut by another split"
big = {key: cached[key] for key in ["day", "band_index", "token_type_index", "mag", "sigma_mag"]}
meta = {
    key: cached[key] for key in ["offsets", "orig_label", "redshift", "group_key", "is_kn", "is_izc"]
}
assert not meta["is_izc"].any(), "izc objects in the test cache"
label_by_index = np.where(meta["is_kn"], 1, 0)
test_index = np.arange(len(meta["is_kn"]))
print("test objects:", len(test_index), "| finite z:", int(np.isfinite(meta["redshift"]).sum()))

with open(f"{DATA_DIR}/normalization.json") as normalization_file:
    normalization = json.load(normalization_file)
for key in ["MAG_MEAN", "MAG_STD", "SIGMA_MAG_MEAN", "SIGMA_MAG_STD"]:
    setattr(openuniverse_data_module, key, normalization[key])

model = LitKilonova.load_from_checkpoint(
    CHECKPOINT, class_weights=torch.ones(len(GROUP_ORDER)), map_location=device
)
model = model.to(device).eval()

results = {}
for epochs in EPOCHS:
    for has_z in HAS_Z:
        dataset = OpenUniverseWindowDataset(
            test_index,
            big=big,
            meta=meta,
            label_by_index=label_by_index,
            data_aug=False,
            force_epochs=epochs,
            force_redshift=has_z,
        )
        loader = DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=collate_token_windows, num_workers=0)
        y_true, kn_probability = [], []
        with torch.no_grad():
            for batch in loader:
                model_input = {key: value.to(device) for key, value in batch.items() if key in MODEL_INPUT_KEYS}
                kn_probability.append(torch.softmax(model(model_input), dim=1)[:, 1].float().cpu().numpy())
                y_true.append(batch["label"].cpu().numpy())
        results[(epochs, has_z)] = (np.concatenate(y_true), np.concatenate(kn_probability))
        y = results[(epochs, has_z)][0]
        print(f"{epochs}ep {'z' if has_z else 'non-z'}: n={len(y)} | KN {int(y.sum())} | other {int((y == 0).sum())}")

# A&A single column: exactly 3.5 in wide, Times-like serif (A&A uses Times; TeX Gyre Termes is the Times clone
# installed here), 8 pt cell text, 9 pt labels, thin horizontal colorbar on top. Axes are placed by hand in inches so
# the page keeps the exact width (no bbox_inches="tight", which would change it).
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "STIXGeneral"], "mathtext.fontset": "stix"})
CELL_FONTSIZE = 8
LABEL_FONTSIZE = 9
FIGURE_WIDTH = 3.5
LEFT_MARGIN, RIGHT_MARGIN, TOP_MARGIN, BOTTOM_MARGIN = 0.80, 0.10, 0.72, 0.42
COLORBAR_HEIGHT, COLORBAR_GAP = 0.07, 0.24  # gap = bar bottom to panel top, leaves room for the column headers
PANEL_GAP = 0.05
COLUMN_HEADERS = {False: "Photometry only", True: "Photometry + z"}
panel_size = (FIGURE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN - PANEL_GAP) / len(HAS_Z)
figure_height = TOP_MARGIN + len(EPOCHS) * panel_size + (len(EPOCHS) - 1) * PANEL_GAP + BOTTOM_MARGIN

for threshold in THRESHOLDS:
    fig = plt.figure(figsize=(FIGURE_WIDTH, figure_height))
    axes = np.empty((len(EPOCHS), len(HAS_Z)), dtype=object)
    for row, epochs in enumerate(EPOCHS):
        for column, has_z in enumerate(HAS_Z):
            left = LEFT_MARGIN + column * (panel_size + PANEL_GAP)
            bottom = BOTTOM_MARGIN + (len(EPOCHS) - 1 - row) * (panel_size + PANEL_GAP)
            ax = fig.add_axes([left / FIGURE_WIDTH, bottom / figure_height, panel_size / FIGURE_WIDTH, panel_size / figure_height])
            axes[row, column] = ax
            y_true, kn_probability = results[(epochs, has_z)]
            matrix = confusion_matrix(y_true, (kn_probability >= threshold).astype(int))
            matrix_percentage = matrix / matrix.sum(axis=1, keepdims=True) * 100
            image = ax.imshow(matrix_percentage, cmap="Blues", vmin=0, vmax=100)
            ax.set_xticks(range(len(GROUP_ORDER)), GROUP_ORDER if row == len(EPOCHS) - 1 else [], fontsize=LABEL_FONTSIZE)
            ax.set_yticks(range(len(GROUP_ORDER)), GROUP_ORDER if column == 0 else [], fontsize=LABEL_FONTSIZE)
            ax.tick_params(length=2, pad=2)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{matrix_percentage[i, j]:.1f}%\n({matrix[i, j]:,})",
                        ha="center",
                        va="center",
                        color="white" if matrix_percentage[i, j] > 50 else "black",
                        fontsize=CELL_FONTSIZE,
                    )
            if row == 0:
                ax.set_title(COLUMN_HEADERS[has_z], fontsize=LABEL_FONTSIZE, pad=3)
            if column == 0:
                # row label sits left of the class tick labels, rotated like a y label
                ax.text(
                    -0.36, 0.5, f"{epochs} epoch{'s' if epochs > 1 else ''}", transform=ax.transAxes,
                    rotation=90, ha="center", va="center", fontsize=LABEL_FONTSIZE,
                )
    panels_left = LEFT_MARGIN / FIGURE_WIDTH
    panels_right = 1 - RIGHT_MARGIN / FIGURE_WIDTH
    panels_bottom = BOTTOM_MARGIN / figure_height
    panels_top = 1 - TOP_MARGIN / figure_height
    colorbar_bottom = (figure_height - TOP_MARGIN + COLORBAR_GAP) / figure_height
    colorbar_ax = fig.add_axes([panels_left, colorbar_bottom, panels_right - panels_left, COLORBAR_HEIGHT / figure_height])
    colorbar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
    colorbar_ax.xaxis.set_ticks_position("top")
    colorbar_ax.xaxis.set_label_position("top")
    colorbar_ax.tick_params(labelsize=CELL_FONTSIZE, length=2, pad=1.5)
    colorbar.set_label("% of true class", fontsize=LABEL_FONTSIZE, labelpad=3)
    fig.text((panels_left + panels_right) / 2, 0.06 / figure_height, "Predicted class", ha="center", va="bottom", fontsize=LABEL_FONTSIZE)
    fig.text(0.04 / FIGURE_WIDTH, (panels_bottom + panels_top) / 2, "True class", rotation=90, ha="left", va="center", fontsize=LABEL_FONTSIZE)
    suffix = "" if threshold == 0.5 else f"_thr{str(threshold).replace('.', 'p')}"
    output_path = os.path.join(PLOTS_DIR, f"04_confusion_matrix_epochs_z_grid{suffix}.pdf")
    fig.savefig(output_path)
    print("saved", output_path)
