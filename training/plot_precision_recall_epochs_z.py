"""1x3 column of precision/recall vs threshold on the test-only cache: rows = 1 / 2 / 3 epochs, each panel overlays
photometry only (dashed) and photometry + z (solid).

Same model, test set and regimes as plot_confusion_epochs_z_grid.py: OpenUniverseWindowDataset's force_epochs
(truncates the window to the first N visits) and force_redshift (False masks z as NaN).
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import openuniverse_data as openuniverse_data_module
import torch
from matplotlib.lines import Line2D
from openuniverse_data import (
    GROUP_KEY_VERSION,
    GROUP_ORDER,
    OpenUniverseWindowDataset,
    collate_token_windows,
)
from sklearn.metrics import average_precision_score, precision_recall_curve
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

# A&A single column, same typography and hand-placed axes as the confusion-matrix grid.
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "STIXGeneral"], "mathtext.fontset": "stix"})
TICK_FONTSIZE = 8
LABEL_FONTSIZE = 9
FIGURE_WIDTH = 3.5
LEFT_MARGIN, RIGHT_MARGIN, TOP_MARGIN, BOTTOM_MARGIN = 0.48, 0.10, 0.42, 0.40
PANEL_HEIGHT = 1.55
PANEL_GAP = 0.08
METRIC_COLORS = {"precision": "#3A86FF", "recall": "#E63946"}
LINE_STYLES = {False: "--", True: "-"}
LINE_LABELS = {False: "Photometry only", True: "Photometry + z"}
# per-row y zoom: five ticks from Y_BOTTOM up to 1, and the top one (unlabeled) tick above 1 so curves at 1 clear the frame
Y_BOTTOM = {1: 0.4, 2: 0.6, 3: 0.8}
Y_TICKS_TO_ONE = 5
panel_width = FIGURE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
figure_height = TOP_MARGIN + len(EPOCHS) * PANEL_HEIGHT + (len(EPOCHS) - 1) * PANEL_GAP + BOTTOM_MARGIN

fig = plt.figure(figsize=(FIGURE_WIDTH, figure_height))
for row, epochs in enumerate(EPOCHS):
    bottom = BOTTOM_MARGIN + (len(EPOCHS) - 1 - row) * (PANEL_HEIGHT + PANEL_GAP)
    ax = fig.add_axes([LEFT_MARGIN / FIGURE_WIDTH, bottom / figure_height, panel_width / FIGURE_WIDTH, PANEL_HEIGHT / figure_height])
    average_precision_text = []
    for has_z in HAS_Z:
        y_true, kn_probability = results[(epochs, has_z)]
        precision, recall, threshold = precision_recall_curve(y_true, kn_probability)
        ax.plot(threshold, precision[:-1], color=METRIC_COLORS["precision"], ls=LINE_STYLES[has_z], lw=1.2)
        ax.plot(threshold, recall[:-1], color=METRIC_COLORS["recall"], ls=LINE_STYLES[has_z], lw=1.2)
        average_precision = average_precision_score(y_true, kn_probability)
        average_precision_text.append(f"AP$_{{\\mathrm{{{'z' if has_z else 'phot'}}}}}$ ={average_precision:.3f}")
    ax.set_xlim(0, 1)
    y_step = (1 - Y_BOTTOM[epochs]) / (Y_TICKS_TO_ONE - 1)
    y_ticks = np.round(Y_BOTTOM[epochs] + y_step * np.arange(Y_TICKS_TO_ONE + 1), 3)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    ax.set_yticks(y_ticks, [f"{tick:g}" if tick <= 1 else "" for tick in y_ticks])
    ax.grid(alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=TICK_FONTSIZE, length=2, pad=2)
    if row != len(EPOCHS) - 1:
        ax.set_xticklabels([])
    ax.set_ylabel(f"{epochs} epoch{'s' if epochs > 1 else ''}", fontsize=LABEL_FONTSIZE, labelpad=3)
    ax.text(
        0.55, 0.05, "\n".join(average_precision_text), transform=ax.transAxes,
        ha="center", va="bottom", fontsize=TICK_FONTSIZE,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.8", "lw": 0.5},
    )

legend_handles = [Line2D([], [], color=color, lw=1.2, label=metric.capitalize()) for metric, color in METRIC_COLORS.items()]
legend_handles += [Line2D([], [], color="0.3", lw=1.2, ls=LINE_STYLES[has_z], label=LINE_LABELS[has_z]) for has_z in HAS_Z]
fig.legend(
    handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5 + (LEFT_MARGIN - RIGHT_MARGIN) / 2 / FIGURE_WIDTH, 1 - 0.02 / figure_height),
    ncol=2, fontsize=TICK_FONTSIZE, frameon=False, handlelength=2.2, columnspacing=1.2, handletextpad=0.5,
)
fig.text(
    (LEFT_MARGIN + panel_width / 2) / FIGURE_WIDTH, 0.06 / figure_height, "Threshold on $P(\\mathrm{KN})$",
    ha="center", va="bottom", fontsize=LABEL_FONTSIZE,
)
output_path = os.path.join(PLOTS_DIR, "02_precision_recall_epochs_z.pdf")
fig.savefig(output_path)
print("saved", output_path)
