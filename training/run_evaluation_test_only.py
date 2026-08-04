"""Standalone evaluation on the test-only token cache (no train/val data available locally).

Loads `openuniverse_tokens_test.npz` directly (already the full test split) and the normalization
constants fit on train (`normalization.json`), instead of going through `build_dataloaders`, which
needs the train/val source parquet/hdf5 files.

Because this bypasses `build_dataloaders`, it also bypasses the cache-version check: the file has
to be re-cut from the same split the checkpoint was trained on, so the version is asserted below.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import openuniverse_data as openuniverse_data_module
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from openuniverse_data import (
    GROUP_KEY_VERSION,
    GROUP_ORDER,
    OpenUniverseWindowDataset,
    collate_token_windows,
)
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from torch.utils.data import DataLoader
from train_lightning import MODEL_INPUT_KEYS, LitKilonova


def first_existing(candidates, default):
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return default


# Resolved against the working directory, so this runs from either machine; the absolute path it
# replaces only existed on the Mac.
DATA_DIR = first_existing(["data/openuniverse", "../data/openuniverse"], "../data/openuniverse")
CHECKPOINT = "checkpoints/kilonova_transformer-soup.ckpt"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
C_CONTAMINANT = "#8338EC"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ----------------------------------------------------------------------------- load test cache
cached = np.load(f"{DATA_DIR}/openuniverse_tokens_test.npz", allow_pickle=False)
cached_version = int(cached["group_key_version"]) if "group_key_version" in cached else 1
assert cached_version == GROUP_KEY_VERSION, (
    f"test cache holds group_key v{cached_version}, this revision needs v{GROUP_KEY_VERSION}. "
    "Re-cut it from the current openuniverse_tokens.npz or the scores describe the old split."
)
big = {key: cached[key] for key in ["day", "band_index", "token_type_index", "mag", "sigma_mag"]}
meta = {
    "offsets": cached["offsets"],
    "orig_label": cached["orig_label"],
    "redshift": cached["redshift"],
    "group_key": cached["group_key"],
    "is_kn": cached["is_kn"],
}
label_by_index = np.where(meta["is_kn"], 1, 0)
test_index = np.arange(len(meta["is_kn"]))
print(
    "test objects:",
    len(test_index),
    "| KN:",
    int(meta["is_kn"].sum()),
    "| other:",
    int((~meta["is_kn"]).sum()),
)

# ----------------------------------------------------------------------------- normalization
with open(f"{DATA_DIR}/normalization.json") as normalization_file:
    normalization = json.load(normalization_file)
openuniverse_data_module.MAG_MEAN = normalization["MAG_MEAN"]
openuniverse_data_module.MAG_STD = normalization["MAG_STD"]
openuniverse_data_module.SIGMA_MAG_MEAN = normalization["SIGMA_MAG_MEAN"]
openuniverse_data_module.SIGMA_MAG_STD = normalization["SIGMA_MAG_STD"]
print("normalization:", normalization)

# ----------------------------------------------------------------------------- model
model = LitKilonova.load_from_checkpoint(
    CHECKPOINT, class_weights=torch.ones(len(GROUP_ORDER)), map_location=device
)
model = model.to(device).eval()
print("parameters:", sum(p.numel() for p in model.model.parameters()))

# ----------------------------------------------------------------------------- regime loaders (no redshift)
EPOCHS = (1, 2, 3)
regime_loaders = {}
for epochs in EPOCHS:
    dataset = OpenUniverseWindowDataset(
        test_index,
        big=big,
        meta=meta,
        label_by_index=label_by_index,
        data_aug=False,
        force_epochs=epochs,
        force_redshift=False,
    )
    regime_loaders[epochs] = DataLoader(
        dataset, batch_size=512, shuffle=False, collate_fn=collate_token_windows, num_workers=0
    )

# ----------------------------------------------------------------------------- inference
results = {}
for epochs, loader in regime_loaders.items():
    y_true, kn_probability, cid = [], [], []
    with torch.no_grad():
        for batch in loader:
            model_input = {key: value.to(device) for key, value in batch.items() if key in MODEL_INPUT_KEYS}
            probabilities = torch.softmax(model(model_input), dim=1)[:, 1]
            kn_probability.append(probabilities.float().cpu().numpy())
            y_true.append(batch["label"].cpu().numpy())
            cid.append(batch["cid"].cpu().numpy())
    results[epochs] = {
        "y_true": np.concatenate(y_true),
        "kn_prob": np.concatenate(kn_probability),
        "cid": np.concatenate(cid),
    }
    result = results[epochs]
    print(
        f"{epochs}ep (no z): {len(result['y_true'])} obj | "
        f"KN {int(result['y_true'].sum())} | other {int((result['y_true'] == 0).sum())}"
    )

orig_label = meta["orig_label"]

# --------------------------------------------------------------- precision/recall + PKN histogram
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for ax, epochs in zip(axes, EPOCHS, strict=False):
    result = results[epochs]
    precision, recall, threshold = precision_recall_curve(result["y_true"], result["kn_prob"])
    average_precision = average_precision_score(result["y_true"], result["kn_prob"])
    ax.plot(threshold, precision[:-1], label="precision", lw=2, color="#3A86FF")
    ax.plot(threshold, recall[:-1], label="recall", lw=2, color="#E63946")
    ax.set_title(f"{epochs} epoch{'s' if epochs > 1 else ''}  ·  AP={average_precision:.3f}")
    ax.set_xlabel("threshold on P(KN)")
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel("score")
fig.suptitle("Precision / Recall vs threshold (KN, test, no redshift)")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "02_precision_recall_test_only.pdf"), bbox_inches="tight")
print("saved", os.path.join(PLOTS_DIR, "02_precision_recall_test_only.pdf"))

# ----------------------------------------------------------------------------- confusion matrix
THRESHOLD = 0.5
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True, gridspec_kw={"wspace": 0.02})
for ax, epochs in zip(axes, EPOCHS, strict=False):
    result = results[epochs]
    predicted_label = (result["kn_prob"] >= THRESHOLD).astype(int)
    matrix = confusion_matrix(result["y_true"], predicted_label)
    matrix_percentage = matrix / matrix.sum(axis=1, keepdims=True) * 100
    image = ax.imshow(matrix_percentage, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(GROUP_ORDER)), GROUP_ORDER, fontsize=15)
    ax.set_yticks(range(len(GROUP_ORDER)), GROUP_ORDER, fontsize=15)
    ax.set_title(f"{epochs} epoch{'s' if epochs > 1 else ''}", fontsize=16)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{matrix_percentage[row, column]:.1f}%\n({matrix[row, column]:,})",
                ha="center",
                va="center",
                color="white" if matrix_percentage[row, column] > 50 else "black",
                fontsize=13,
            )
axes[0].set_ylabel("true", fontsize=16)
axes[1].set_xlabel("predicted", fontsize=16)
colorbar = fig.colorbar(image, ax=axes, fraction=0.046, pad=0.02)
colorbar.set_label("% of true class", fontsize=16)
colorbar.ax.tick_params(labelsize=14)
fig.suptitle(f"Confusion matrix (threshold={THRESHOLD}, no redshift)", fontsize=17)
fig.savefig(os.path.join(PLOTS_DIR, "04_confusion_matrix_test_only.pdf"), bbox_inches="tight")
print("saved", os.path.join(PLOTS_DIR, "04_confusion_matrix_test_only.pdf"))

for epochs in EPOCHS:
    result = results[epochs]
    predicted_label = (result["kn_prob"] >= THRESHOLD).astype(int)
    print(f"=== {epochs} epoch(s), no redshift ===")
    print(classification_report(result["y_true"], predicted_label, target_names=GROUP_ORDER, digits=4))

# --------------------------------------------------------------- threshold=0.2 contaminant leakage
THRESHOLD_HIGH_RECALL = 0.2
for epochs in EPOCHS:
    result = results[epochs]
    predicted_label_high_recall = (result["kn_prob"] >= THRESHOLD_HIGH_RECALL).astype(int)
    is_kn = result["y_true"] == 1
    false_positive_count = int(((result["y_true"] == 0) & (predicted_label_high_recall == 1)).sum())
    print(
        f"{epochs}ep (no redshift): "
        f"KN recall={predicted_label_high_recall[is_kn].mean():.4f}  |  "
        f"contaminants as KN={false_positive_count}"
    )

# ------------------------------------------------- same test set, but WITH the true redshift fed in
regime_loaders_with_redshift = {}
for epochs in EPOCHS:
    dataset = OpenUniverseWindowDataset(
        test_index,
        big=big,
        meta=meta,
        label_by_index=label_by_index,
        data_aug=False,
        force_epochs=epochs,
        force_redshift=True,
    )
    regime_loaders_with_redshift[epochs] = DataLoader(
        dataset, batch_size=512, shuffle=False, collate_fn=collate_token_windows, num_workers=0
    )

results_with_redshift = {}
for epochs, loader in regime_loaders_with_redshift.items():
    y_true, kn_probability, cid = [], [], []
    with torch.no_grad():
        for batch in loader:
            model_input = {key: value.to(device) for key, value in batch.items() if key in MODEL_INPUT_KEYS}
            probabilities = torch.softmax(model(model_input), dim=1)[:, 1]
            kn_probability.append(probabilities.float().cpu().numpy())
            y_true.append(batch["label"].cpu().numpy())
            cid.append(batch["cid"].cpu().numpy())
    results_with_redshift[epochs] = {
        "y_true": np.concatenate(y_true),
        "kn_prob": np.concatenate(kn_probability),
        "cid": np.concatenate(cid),
    }
    result = results_with_redshift[epochs]
    predicted_label_high_recall = (result["kn_prob"] >= THRESHOLD_HIGH_RECALL).astype(int)
    is_kn = result["y_true"] == 1
    false_positive_count = int(((result["y_true"] == 0) & (predicted_label_high_recall == 1)).sum())
    print(
        f"{epochs}ep (with redshift): "
        f"KN recall={predicted_label_high_recall[is_kn].mean():.4f}  |  "
        f"contaminants as KN={false_positive_count}"
    )

# ------------------------------------------------------- contaminant leakage per class, both regimes
# Class order fixed by total number of contaminants in test (desc) so every panel shares an x axis.
contaminant_total = pd.Series(orig_label[results[3]["cid"]][results[3]["y_true"] == 0]).value_counts()
CLASS_ORDER = list(contaminant_total.index)
CLASS_TOTAL = contaminant_total.reindex(CLASS_ORDER)


def false_positive_count_by_class(regime_results, threshold):
    counts = {}
    for epochs, result in regime_results.items():
        predicted_label = (result["kn_prob"] >= threshold).astype(int)
        false_positive = (result["y_true"] == 0) & (predicted_label == 1)
        counts[epochs] = (
            pd.Series(orig_label[result["cid"]][false_positive])
            .value_counts()
            .reindex(CLASS_ORDER, fill_value=0)
        )
    return counts


def draw_contaminant_panel(ax, counts_no_redshift, counts_with_redshift, label_fontsize):
    """One dashed rectangle per class at the no-redshift count, filled from the axis floor up to the
    with-redshift count, so both regimes share the same bar outline."""
    positions = np.arange(len(CLASS_ORDER))
    ax.bar(
        positions,
        counts_with_redshift,
        width=0.8,
        color=C_CONTAMINANT,
        alpha=0.9,
        linewidth=0,
    )
    ax.bar(
        positions,
        counts_no_redshift,
        width=0.8,
        facecolor="none",
        edgecolor=C_CONTAMINANT,
        linewidth=1.8,
        linestyle="--",
    )
    ax.set_xticks(
        positions,
        [f"{name}\n({CLASS_TOTAL[name]:,})" for name in CLASS_ORDER],
        fontsize=label_fontsize,
    )
    ax.set_yscale("log")
    ax.set_ylim(0.6, 5e4)
    ax.grid(alpha=0.3, axis="y")
    # The no-redshift count goes above the dashed top edge, the with-redshift one just below the
    # top of the fill, both centered -- they never collide even when the two regimes coincide.
    for position, count in zip(positions, counts_no_redshift, strict=False):
        if count > 0:
            ax.text(
                position,
                count * 1.15,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=label_fontsize,
                color=C_CONTAMINANT,
            )
    for position, count in zip(positions, counts_with_redshift, strict=False):
        if count >= 10:  # a fill shorter than a decade cannot hold the label inside
            ax.text(
                position,
                count / 1.15,
                f"{count:,}",
                ha="center",
                va="top",
                fontsize=label_fontsize,
                color="white",
            )
        elif count > 0:
            ax.text(
                position,
                count * 1.15,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=label_fontsize,
                color="black",
            )


def contaminant_legend_handles():
    return [
        Patch(facecolor="none", edgecolor=C_CONTAMINANT, linestyle="--", linewidth=1.8, label="no redshift"),
        Patch(facecolor=C_CONTAMINANT, alpha=0.9, label="with redshift"),
    ]


counts_no_redshift = false_positive_count_by_class(results, THRESHOLD_HIGH_RECALL)
counts_with_redshift = false_positive_count_by_class(results_with_redshift, THRESHOLD_HIGH_RECALL)

contaminant_pdf = os.path.join(PLOTS_DIR, "05_contaminants_redshift_comparison_test_only.pdf")
with PdfPages(contaminant_pdf) as pdf_pages:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, epochs in zip(axes, EPOCHS, strict=False):
        draw_contaminant_panel(ax, counts_no_redshift[epochs].values, counts_with_redshift[epochs].values, 8)
        ax.set_title(f"{epochs} epoch{'s' if epochs > 1 else ''}", fontsize=14)
    axes[0].set_ylabel("contaminants predicted as KN (log)", fontsize=13)
    axes[1].set_xlabel("true class (total in test)", fontsize=13)
    axes[0].legend(handles=contaminant_legend_handles(), fontsize=11, loc="upper right")
    fig.suptitle(
        f"Contaminants leaking as KN by true class  (threshold={THRESHOLD_HIGH_RECALL})", fontsize=16
    )
    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # Second page: the 2-epoch panel on its own, large — this is the presentation figure.
    PRESENTATION_EPOCHS = 2
    fig, ax = plt.subplots(figsize=(12, 7))
    draw_contaminant_panel(
        ax,
        counts_no_redshift[PRESENTATION_EPOCHS].values,
        counts_with_redshift[PRESENTATION_EPOCHS].values,
        13,
    )
    ax.set_ylabel("contaminants predicted as KN (log)", fontsize=17)
    ax.set_xlabel("true class (total in test)", fontsize=17)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(handles=contaminant_legend_handles(), fontsize=15, loc="upper right")
    ax.set_title(
        f"Contaminants leaking as KN by true class\n"
        f"({PRESENTATION_EPOCHS} epochs, threshold={THRESHOLD_HIGH_RECALL})",
        fontsize=19,
    )
    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)
print("saved", contaminant_pdf)

print("done")
