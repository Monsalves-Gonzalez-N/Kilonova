"""Contaminants leaking as KN by true class at threshold 0.2, as one A&A column: rows = 1 / 2 / 3 epochs, each bar
outlined (dashed) at the photometry-only count and filled up to the photometry + z count.

Same model, test set and regimes as plot_confusion_epochs_z_grid.py; same bar encoding as the contaminant figure in
run_evaluation_test_only.py.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import openuniverse_data as openuniverse_data_module
import torch
from matplotlib.patches import Patch
from openuniverse_data import (
    GROUP_KEY_VERSION,
    GROUP_ORDER,
    OpenUniverseWindowDataset,
    collate_token_windows,
)
import pandas as pd
from torch.utils.data import DataLoader
from train_lightning import MODEL_INPUT_KEYS, LitKilonova

DATA_DIR = "data/openuniverse" if os.path.isdir("data/openuniverse") else "../data/openuniverse"
CHECKPOINT = os.environ.get("KN_CHECKPOINT", "checkpoints/izc-trainonly-2026-09-30/kilonova_transformer-soup.ckpt")
PLOTS_DIR = "plots"
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

THRESHOLD = 0.2
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
        y_true, kn_probability, cid = [], [], []
        with torch.no_grad():
            for batch in loader:
                model_input = {key: value.to(device) for key, value in batch.items() if key in MODEL_INPUT_KEYS}
                kn_probability.append(torch.softmax(model(model_input), dim=1)[:, 1].float().cpu().numpy())
                y_true.append(batch["label"].cpu().numpy())
                cid.append(batch["cid"].cpu().numpy())
        y_true, kn_probability, cid = np.concatenate(y_true), np.concatenate(kn_probability), np.concatenate(cid)
        false_positive = (y_true == 0) & (kn_probability >= THRESHOLD)
        results[(epochs, has_z)] = pd.Series(meta["orig_label"][cid][false_positive]).value_counts()
        contaminant_total = pd.Series(meta["orig_label"][cid][y_true == 0]).value_counts()
        print(f"{epochs}ep {'z' if has_z else 'non-z'}: contaminants as KN = {int(false_positive.sum())}")

# Class order fixed by total number of contaminants in test (desc), shared by every panel.
CLASS_ORDER = list(contaminant_total.index)
counts = {key: value.reindex(CLASS_ORDER, fill_value=0).to_numpy() for key, value in results.items()}

# A&A single column, same typography and hand-placed axes as the confusion-matrix grid.
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "STIXGeneral"], "mathtext.fontset": "stix"})
COUNT_FONTSIZE = 5.5
TICK_FONTSIZE = 7
LABEL_FONTSIZE = 9
FIGURE_WIDTH = 3.5
LEFT_MARGIN, RIGHT_MARGIN, TOP_MARGIN, BOTTOM_MARGIN = 0.62, 0.08, 0.30, 0.52
PANEL_HEIGHT = 1.45
PANEL_GAP = 0.08
C_CONTAMINANT = "#8338EC"
LINE_LABELS = {False: "Photometry only", True: "Photometry + z"}
panel_width = FIGURE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
figure_height = TOP_MARGIN + len(EPOCHS) * PANEL_HEIGHT + (len(EPOCHS) - 1) * PANEL_GAP + BOTTOM_MARGIN
positions = np.arange(len(CLASS_ORDER))

fig = plt.figure(figsize=(FIGURE_WIDTH, figure_height))
for row, epochs in enumerate(EPOCHS):
    bottom = BOTTOM_MARGIN + (len(EPOCHS) - 1 - row) * (PANEL_HEIGHT + PANEL_GAP)
    ax = fig.add_axes([LEFT_MARGIN / FIGURE_WIDTH, bottom / figure_height, panel_width / FIGURE_WIDTH, PANEL_HEIGHT / figure_height])
    counts_photometry, counts_z = counts[(epochs, False)], counts[(epochs, True)]
    ax.bar(positions, counts_z, width=0.8, color=C_CONTAMINANT, alpha=0.9, linewidth=0)
    ax.bar(positions, counts_photometry, width=0.8, facecolor="none", edgecolor=C_CONTAMINANT, linewidth=0.9, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylim(0.6, 6e4)
    ax.set_xlim(-0.6, len(CLASS_ORDER) - 0.4)
    ax.grid(alpha=0.3, axis="y", lw=0.5)
    ax.tick_params(labelsize=TICK_FONTSIZE, length=2, pad=2)
    ax.tick_params(axis="x", length=0)
    if row == len(EPOCHS) - 1:
        ax.set_xticks(positions, [f"{name}\n({contaminant_total[name]:,})" for name in CLASS_ORDER], rotation=90, fontsize=TICK_FONTSIZE)
    else:
        ax.set_xticks(positions, [])
    ax.set_ylabel(f"{epochs} epoch{'s' if epochs > 1 else ''}", fontsize=LABEL_FONTSIZE, labelpad=2)
    # photometry-only count above the dashed top edge, photometry + z count just inside the top of the fill
    for position, count in zip(positions, counts_photometry):
        if count > 0:
            ax.text(position, count * 1.2, f"{count:,}", ha="center", va="bottom", fontsize=COUNT_FONTSIZE, color=C_CONTAMINANT)
    for position, count in zip(positions, counts_z):
        if count >= 10:
            ax.text(position, count / 1.2, f"{count:,}", ha="center", va="top", fontsize=COUNT_FONTSIZE, color="white", rotation=90)
        elif count > 0:
            ax.text(position, count * 1.2, f"{count:,}", ha="center", va="bottom", fontsize=COUNT_FONTSIZE, color="black")

legend_handles = [
    Patch(facecolor="none", edgecolor=C_CONTAMINANT, linestyle="--", linewidth=0.9, label=LINE_LABELS[False]),
    Patch(facecolor=C_CONTAMINANT, alpha=0.9, label=LINE_LABELS[True]),
]
fig.legend(
    handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5 + (LEFT_MARGIN - RIGHT_MARGIN) / 2 / FIGURE_WIDTH, 1 - 0.02 / figure_height),
    ncol=2, fontsize=TICK_FONTSIZE + 1, frameon=False, handlelength=1.8, columnspacing=1.2, handletextpad=0.5,
)
fig.text(
    0.04 / FIGURE_WIDTH, (BOTTOM_MARGIN + (figure_height - TOP_MARGIN - BOTTOM_MARGIN) / 2) / figure_height,
    f"Contaminants with $P(\\mathrm{{KN}}) \\geq {THRESHOLD}$", rotation=90, ha="left", va="center", fontsize=LABEL_FONTSIZE,
)
output_path = os.path.join(PLOTS_DIR, "05_contaminants_epochs_z.pdf")
fig.savefig(output_path)
print("saved", output_path)
