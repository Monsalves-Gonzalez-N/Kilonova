"""Shared plumbing for the diagnostics that probe the trained classifier on the frozen test split.

These scripts ask what the model does when the input is changed in ways the training set never
contained. They read the same `openuniverse_tokens_test.npz` the published metrics come from and
never write to it: every intervention works on a copy of the token arrays, so a diagnostic can be
re-run in any order without disturbing the evaluation it is auditing.

`training/` is a flat directory of scripts rather than a package, so the path fix-up below is what
lets the diagnostics live one level down and still import the model and the dataloader.
"""

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

TRAINING_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TRAINING_DIRECTORY not in sys.path:
    sys.path.insert(0, TRAINING_DIRECTORY)

import openuniverse_data  # noqa: E402
from openuniverse_data import (  # noqa: E402
    GROUP_ORDER,
    OpenUniverseWindowDataset,
    collate_token_windows,
)
from train_lightning import MODEL_INPUT_KEYS, LitKilonova  # noqa: E402

TOKEN_FIELDS = ["day", "band_index", "token_type_index", "mag", "sigma_mag"]
META_FIELDS = ["offsets", "orig_label", "redshift", "group_key", "is_kn"]
NOT_OBSERVED_TOKEN = 2  # TOKEN_TYPE_ORDER = ["d", "u", "n"]
DETECTION_TOKEN = 0

DEFAULT_DATA_DIRECTORY = "data/openuniverse"
DEFAULT_CHECKPOINT = "training/checkpoints/kilonova_transformer-soup.ckpt"


def load_test_split(data_directory=DEFAULT_DATA_DIRECTORY):
    """The frozen test cache as (tokens, meta). `tokens` is a mutable copy, `meta` is not."""
    cached = np.load(f"{data_directory}/openuniverse_tokens_test.npz", allow_pickle=False)
    tokens = {field: cached[field].copy() for field in TOKEN_FIELDS}
    meta = {field: cached[field] for field in META_FIELDS}
    return tokens, meta


def load_model(data_directory=DEFAULT_DATA_DIRECTORY, checkpoint_path=DEFAULT_CHECKPOINT, device=None):
    """The checkpoint, with the train-split normalization installed as module globals.

    The normalization has to be set before any dataset is built: `OpenUniverseWindowDataset` reads
    it out of `openuniverse_data` at __getitem__ time, not at construction time, so forgetting it
    yields silently wrong magnitudes rather than an error."""
    for name, value in json.load(open(f"{data_directory}/normalization.json")).items():
        setattr(openuniverse_data, name, value)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LitKilonova.load_from_checkpoint(
        checkpoint_path, class_weights=torch.ones(len(GROUP_ORDER)), map_location=device
    )
    return model.to(device).eval(), device


def predict(model, tokens, meta, index, device, epochs=1, has_redshift=False, batch_size=2048):
    """P(KN) for `index`, in the regime the model was selected on (one epoch, no redshift)."""
    dataset = OpenUniverseWindowDataset(
        index,
        big=tokens,
        meta=meta,
        label_by_index=np.where(meta["is_kn"], 1, 0),
        data_aug=False,
        force_epochs=epochs,
        force_redshift=has_redshift,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_token_windows, num_workers=0
    )
    probabilities = []
    with torch.no_grad():
        for batch in loader:
            model_input = {key: value.to(device) for key, value in batch.items() if key in MODEL_INPUT_KEYS}
            probabilities.append(torch.softmax(model(model_input), dim=1)[:, 1].float().cpu().numpy())
    return np.concatenate(probabilities)


def first_epoch_positions(tokens, meta, object_index):
    """Absolute positions in the flat token arrays of the first epoch of one object."""
    low, high = meta["offsets"][object_index], meta["offsets"][object_index + 1]
    day = tokens["day"][low:high]
    return low + np.flatnonzero(day == day.min())
