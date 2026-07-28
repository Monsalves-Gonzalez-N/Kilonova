"""Standalone evaluation on the test-only token cache (no train/val data available locally).

Loads `openuniverse_tokens_test.npz` directly (already the full test split, 93,119 objects)
and the normalization constants fit on train (`normalization.json`), instead of going through
`build_dataloaders`, which needs the train/val source parquet/hdf5 files.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import openuniverse_data as openuniverse_data_module
import torch
from openuniverse_data import (
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

DATA_DIR = "/Users/bhianca/Kilonova/data/openuniverse"
CHECKPOINT = "checkpoints/kilonova_transformer-soup.ckpt"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ----------------------------------------------------------------------------- load test cache
cached = np.load(f"{DATA_DIR}/openuniverse_tokens_test.npz", allow_pickle=False)
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
    ax.plot(threshold, recall[:-1], label="recall", lw=2, color="#2EC4B6")
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
fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
for ax, epochs in zip(axes, EPOCHS, strict=False):
    result = results[epochs]
    predicted_label = (result["kn_prob"] >= THRESHOLD).astype(int)
    matrix = confusion_matrix(result["y_true"], predicted_label)
    matrix_percentage = matrix / matrix.sum(axis=1, keepdims=True) * 100
    image = ax.imshow(matrix_percentage, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(GROUP_ORDER)), GROUP_ORDER)
    ax.set_yticks(range(len(GROUP_ORDER)), GROUP_ORDER)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{epochs} epoch{'s' if epochs > 1 else ''}")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{matrix_percentage[row, column]:.1f}%\n({matrix[row, column]:,})",
                ha="center",
                va="center",
                color="white" if matrix_percentage[row, column] > 50 else "black",
            )
fig.colorbar(image, ax=axes, fraction=0.046, label="% of true class")
fig.suptitle(
    f"Confusion matrix (threshold={THRESHOLD}, row-normalized %, counts in parentheses, no redshift)"
)
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

print("done")
