"""Dataloader for the kilonova transformer (transformer_architecture.ipynb).

Turns the early-window light curves produced upstream into the exact batch contract the
``KilonovaTransformer`` consumes: one token per (band x visit) carrying its time, band,
magnitude and a type flag (detection ``d`` / 5-sigma upper limit ``u`` / instrumental gap
``n``), plus the per-object globals (redshift) and the class label.

Two upstream sources share the same long-format schema and are loaded the same way:

* the OpenUniverse contaminants written by ``kn-run-openuniverse``
  (``early_windows_{deep,wide}.parquet``: gentypes 32/40/57/58, columns ``object_id, gentype,
  label, z_CMB, days_since_detection, band, observed, mag_observed, mag_err, detected,
  mag_limit_5sigma``) -> ``load_early_windows``;
* the injected kilonovas written by ``kn-kilonova-windows``
  (``kn_windows_{deep,wide}.parquet``: gentype 50, same columns minus ``snana_id``)
  -> ``load_kilonova_windows``.

``load_kilonova_windows_hdf5`` reads the older one-group-per-object hdf5 instead. That was the KN
format until June 2026; ``kilonova_windows_{deep,wide}.hdf5`` were retired on 2026-07-31 (three
photometry bugs, no angular factor) and the loader is kept only for hand-made demo files such as
``kilonova_windows_demo.hdf5``. New work reads the parquet.

The model prepends the ``[CLS]`` and ``[Z]`` tokens itself, so ``collate_token_windows`` emits
only the per-token fields, ``padding_mask``, the redshift globals and ``label`` -- identical to
``collate_token_windows`` in the architecture notebook, so ``next(iter(train_loader))`` is a
drop-in for ``make_synthetic_batch()``.
"""

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Token vocabulary -- must match transformer_architecture.ipynb exactly.
BAND_ORDER = ["R", "Z", "Y", "J", "H", "F"]
TOKEN_TYPE_ORDER = ["d", "u", "n"]  # detection / 5-sigma upper limit / instrumental gap
GROUP_ORDER = ["Ia", "II", "other", "KN"]

BAND_TO_INDEX = {band: index for index, band in enumerate(BAND_ORDER)}
TOKEN_TYPE_TO_INDEX = {token_type: index for index, token_type in enumerate(TOKEN_TYPE_ORDER)}
GROUP_TO_INDEX = {group: index for index, group in enumerate(GROUP_ORDER)}

# The authoritative class comes from the ``label`` string column carried by BOTH upstream sources
# (the OpenUniverse parquet's ``GENTYPE_LABEL`` and the KN hdf5 group attr ``label``), so we map
# from it directly rather than re-deriving from ``gentype`` (one source of truth, no dict to keep in
# sync with the generator). Anything not listed -- SN Ib/Ic, SLSN-I, TDE, PISN-H/He -- is 'other'.
LABEL_TO_GROUP = {
    "SN Ia": "Ia",
    "SN Iax": "Ia",  # peculiar Ia subclass; grouped with Ia (flip to 'other' to split it out)
    "SN II": "II",
    "KN": "KN",
}

# Coarse redshift regime for the [Z] token. The exact z is fed continuously, but a low/high/unknown
# bin is fed alongside it: it is the discriminative-and-reproducible part (a KN is undetectable above
# z ~ 0.15, so z >= 0.5 rules KN out; a photo-z lands on the right side of 0.5 even when imprecise).
# 'unknown' replaces the old has_redshift=0 / [NO_Z] case for z-dropout.
REDSHIFT_THRESHOLD = 0.5
REDSHIFT_REGIME_ORDER = ["low", "high", "unknown"]
REDSHIFT_REGIME_TO_INDEX = {regime: index for index, regime in enumerate(REDSHIFT_REGIME_ORDER)}

# Keys emitted per token / per object, in the order the model reads them.
PER_TOKEN_KEYS = [
    "delta_time",
    "band_index",
    "token_type_index",
    "magnitude",
    "sigma_magnitude",
    "magnitude_mask",
    "sigma_mask",
]
GLOBAL_KEYS = ["redshift", "redshift_regime"]


def _read_windows_parquet(path, max_objects, random_seed):
    """One window parquet -> (DataFrame, tier), optionally down to ``max_objects`` random objects.

    The subsample is drawn in a first pass over the ``object_id`` column alone and pushed back into
    the read as a row filter, so a 1.4M-object file never has to be materialized whole just to keep
    a few thousand light curves (that is the difference between a notebook that opens and one that
    dies). Sampling is by OBJECT, never by row: half a light curve is not a light curve.

    The tier is read off the bands, which are disjoint between the two: F184 is deep-only and R062
    wide-only, so the presence of an F band identifies the file without trusting its name.
    """
    if max_objects is None:
        windows = pd.read_parquet(path)
    else:
        object_ids = pd.read_parquet(path, columns=["object_id"])["object_id"].unique()
        if len(object_ids) > max_objects:
            generator = np.random.default_rng(random_seed)
            object_ids = generator.choice(object_ids, size=max_objects, replace=False)
        windows = pd.read_parquet(path, filters=[("object_id", "in", set(object_ids))])
    return windows, ("deep" if windows["band"].str[0].eq("F").any() else "wide")


def load_early_windows(parquet_paths, max_objects=None, random_seed=0):
    """One or more ``early_windows_{tier}.parquet`` files -> a single long DataFrame.

    A ``source`` column tags the tier (deep/wide) so the two tiers can be mixed while still
    being separable downstream. ``object_id`` is suffixed with the source to keep ids unique
    across tiers (the same OpenUniverse transient appears in both).

    ``max_objects`` caps the number of objects taken from EACH file (see ``_read_windows_parquet``);
    None reads everything.
    """
    if isinstance(parquet_paths, (str, bytes)):
        parquet_paths = [parquet_paths]
    frames = []
    for path in parquet_paths:
        windows, tier = _read_windows_parquet(path, max_objects, random_seed)
        windows = windows.assign(
            source=tier,
            tier=tier,
            object_id=windows["object_id"].astype(str) + f"_{tier}",
        )
        frames.append(windows)
    return pd.concat(frames, ignore_index=True)


def load_kilonova_windows(parquet_paths, max_objects=None, random_seed=0):
    """One or more ``kn_windows_{tier}.parquet`` files -> the same long DataFrame as the
    contaminants, with ``source='kn'`` (the tier stays in ``tier``).

    ``object_id`` MUST be namespaced by tier here: unlike the OpenUniverse transients, a kilonova
    id (``{simulation}_{angle}_{offset}_{z}_{parity}_{seed}``) is generated independently of the
    tier and the very same id appears in both files. Without the suffix the deep and wide
    observations of one injection would be grouped into a single 10-band window that no survey ever
    took.

    Note this is the id used to GROUP tokens into a light curve, not to split train/val/test:
    splitting on it leaks one LANL ejecta model across splits, since a single simulation spawns many
    realizations (x angle x offset x redshift x noise). ``training/openuniverse_data.py`` splits on
    the leading ``simulation_id`` instead, which is what a real training run must do.
    """
    if isinstance(parquet_paths, (str, bytes)):
        parquet_paths = [parquet_paths]
    frames = []
    for path in parquet_paths:
        windows, tier = _read_windows_parquet(path, max_objects, random_seed)
        windows = windows.assign(
            source="kn",
            tier=tier,
            object_id="kn_" + windows["object_id"].astype(str) + f"_{tier}",
        )
        frames.append(windows)
    return pd.concat(frames, ignore_index=True)


def load_kilonova_windows_hdf5(hdf5_path):
    """Legacy one-group-per-object hdf5 -> the same long DataFrame schema as ``load_early_windows``.

    Superseded by ``load_kilonova_windows`` (parquet) and kept only for hand-made demo files; the
    generated ``kilonova_windows_{deep,wide}.hdf5`` were retired on 2026-07-31. Cadence gaps are kept
    as ``observed=False`` rows (NaN photometry) just like OpenUniverse, so kilonovas also get ``n``
    tokens; ``gentype`` is forced to 50 (KN) from the group attrs. Older files without an ``observed``
    dataset fall back to all-observed.
    """
    rows = []
    with h5py.File(hdf5_path, "r") as hdf5:
        tier = hdf5.attrs.get("tier", "")
        for object_id in hdf5.keys():
            group = hdf5[object_id]
            number_of_rows = group["days_since_detection"].shape[0]
            observed = group["observed"][:] if "observed" in group else np.ones(number_of_rows, dtype=bool)
            frame = pd.DataFrame(
                {
                    "object_id": f"kn_{tier}_{object_id}" if tier else f"kn_{object_id}",
                    "gentype": int(group.attrs.get("gentype", 50)),
                    "label": group.attrs.get("label", "KN"),
                    "z_CMB": float(group.attrs["redshift"]),
                    "days_since_detection": group["days_since_detection"][:],
                    "band": [band.decode() for band in group["band"][:]],
                    "observed": observed,
                    "mag_observed": group["mag_observed"][:],
                    "mag_err": group["mag_err"][:],
                    "detected": group["detected"][:],
                    "mag_limit_5sigma": group["mag_limit_5sigma"][:],
                    "source": "kn",
                    "tier": tier,
                }
            )
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _token_fields(window):
    """Long rows of one object -> per-token (delta_time, band_index, type_index, magnitude, sigma).

    magnitude is the detection magnitude (``d``), the 5-sigma limit (``u``) or NaN (``n``);
    sigma is the photometric error (``d``) or NaN otherwise. Type comes from observed/detected:
    not observed -> ``n``; observed & detected -> ``d``; observed & not detected -> ``u``.
    """
    detected = window["detected"].to_numpy(dtype=bool)
    observed = window["observed"].to_numpy(dtype=bool)

    token_type = np.where(~observed, "n", np.where(detected, "d", "u"))
    magnitude = np.where(
        token_type == "d",
        window["mag_observed"].to_numpy(dtype=float),
        np.where(token_type == "u", window["mag_limit_5sigma"].to_numpy(dtype=float), np.nan),
    )
    sigma_magnitude = np.where(token_type == "d", window["mag_err"].to_numpy(dtype=float), np.nan)

    return {
        "delta_time": window["days_since_detection"].to_numpy(dtype=float),
        "band_index": window["band"].str[0].map(BAND_TO_INDEX).to_numpy(dtype=np.int64),
        "token_type_index": np.array([TOKEN_TYPE_TO_INDEX[token] for token in token_type], dtype=np.int64),
        "magnitude": magnitude,
        "sigma_magnitude": sigma_magnitude,
    }


def compute_normalization(long_df):
    """Global magnitude / sigma normalization (mean, std) over the given rows.

    Magnitudes pool detection magnitudes and 5-sigma limits (same physical scale, both feed a
    token's magnitude channel); sigmas come from detections only. Fit on the TRAIN split and
    reuse for val/test so the model sees a consistent scale.
    """
    fields = _token_fields(long_df)
    magnitude = fields["magnitude"][np.isfinite(fields["magnitude"])]
    sigma = fields["sigma_magnitude"][np.isfinite(fields["sigma_magnitude"])]
    return {
        "mag_mean": float(magnitude.mean()),
        "mag_std": float(magnitude.std() + 1e-8),
        "sigma_mean": float(sigma.mean()),
        "sigma_std": float(sigma.std() + 1e-8),
    }


class EarlyWindowDataset(Dataset):
    """One early-window light curve -> the per-object dict the transformer's collate expects.

    ``long_df`` is the concatenation of any of ``load_early_windows`` / ``load_kilonova_windows``;
    objects are keyed by ``object_id``. ``normalization`` is the dict from ``compute_normalization``
    (fit on train). Magnitude and sigma are globally normalized; masked channels are set to 0 and
    flagged with ``magnitude_mask`` / ``sigma_mask`` so the model can tell a real 0 from an absent
    channel. The label is read from the authoritative ``label`` column via ``LABEL_TO_GROUP``. The
    redshift token gets the continuous truth z plus its ``redshift_regime`` (low/high) bin; the
    training notebook can flip the regime to 'unknown' (z-dropout) and add photo-z noise on top.
    """

    def __init__(self, long_df, normalization):
        self.normalization = normalization
        self.objects = []
        for _object_id, window in long_df.groupby("object_id", sort=False):
            self.objects.append(self._encode_object(window))

    def _encode_object(self, window):
        fields = _token_fields(window)
        normalization = self.normalization

        magnitude_mask = np.isfinite(fields["magnitude"])
        sigma_mask = np.isfinite(fields["sigma_magnitude"])
        magnitude = np.where(
            magnitude_mask,
            (fields["magnitude"] - normalization["mag_mean"]) / normalization["mag_std"],
            0.0,
        )
        sigma_magnitude = np.where(
            sigma_mask,
            (fields["sigma_magnitude"] - normalization["sigma_mean"]) / normalization["sigma_std"],
            0.0,
        )

        first = window.iloc[0]
        group_index = GROUP_TO_INDEX[LABEL_TO_GROUP.get(str(first["label"]), "other")]

        redshift = float(first["z_CMB"])
        redshift_regime = REDSHIFT_REGIME_TO_INDEX["low" if redshift < REDSHIFT_THRESHOLD else "high"]

        return {
            "delta_time": torch.tensor(fields["delta_time"], dtype=torch.float32),
            "band_index": torch.tensor(fields["band_index"], dtype=torch.long),
            "token_type_index": torch.tensor(fields["token_type_index"], dtype=torch.long),
            "magnitude": torch.tensor(magnitude, dtype=torch.float32),
            "sigma_magnitude": torch.tensor(sigma_magnitude, dtype=torch.float32),
            "magnitude_mask": torch.tensor(magnitude_mask.astype(np.float32)),
            "sigma_mask": torch.tensor(sigma_mask.astype(np.float32)),
            "redshift": torch.tensor(redshift, dtype=torch.float32),
            "redshift_regime": torch.tensor(redshift_regime, dtype=torch.long),
            "label": torch.tensor(group_index, dtype=torch.long),
        }

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index):
        return self.objects[index]


def collate_token_windows(items):
    """Pad a list of per-object dicts to the longest token sequence and stack the globals.

    Identical contract to ``collate_token_windows`` in the architecture notebook: per-token keys
    padded to ``(B, T)``, ``padding_mask`` ``True`` on padded slots, globals and ``label`` stacked
    to ``(B,)``. The model prepends ``[CLS]`` / ``[Z]`` and builds the full attention mask itself.
    """
    batch_size = len(items)
    max_tokens = max(item["delta_time"].shape[0] for item in items)
    padded = {key: torch.zeros(batch_size, max_tokens, dtype=items[0][key].dtype) for key in PER_TOKEN_KEYS}
    padding_mask = torch.ones(batch_size, max_tokens, dtype=torch.bool)
    for row_index, item in enumerate(items):
        number_of_tokens = item["delta_time"].shape[0]
        for key in PER_TOKEN_KEYS:
            padded[key][row_index, :number_of_tokens] = item[key]
        padding_mask[row_index, :number_of_tokens] = False

    collated = dict(padded)
    collated["padding_mask"] = padding_mask
    for key in GLOBAL_KEYS:
        collated[key] = torch.stack([item[key] for item in items])
    collated["label"] = torch.stack([item["label"] for item in items])
    return collated


def split_objects(long_df, fractions=(0.70, 0.15, 0.15), random_seed=42):
    """Split by ``object_id`` into train/validation/test long DataFrames (no object leaks)."""
    object_ids = long_df["object_id"].drop_duplicates().to_numpy()
    shuffled = np.random.default_rng(random_seed).permutation(object_ids)
    train_fraction, validation_fraction, _ = fractions
    train_end = int(round(train_fraction * shuffled.size))
    validation_end = train_end + int(round(validation_fraction * shuffled.size))
    split_ids = {
        "train": set(shuffled[:train_end]),
        "validation": set(shuffled[train_end:validation_end]),
        "test": set(shuffled[validation_end:]),
    }
    return {name: long_df[long_df["object_id"].isin(ids)].copy() for name, ids in split_ids.items()}


def build_dataloaders(long_df, batch_size=32, fractions=(0.70, 0.15, 0.15), random_seed=42, num_workers=0):
    """Convenience: split ``long_df``, fit normalization on TRAIN, return train/val/test loaders.

    Returns ``(loaders, normalization)`` where ``loaders`` is a dict of ``DataLoader`` keyed by
    split. Drop-in for the architecture notebook::

        from kilonova.datasets.openuniverse import (
            load_early_windows, load_kilonova_windows, build_dataloaders)
        long_df = pd.concat([
            load_early_windows(['early_windows_deep.parquet', 'early_windows_wide.parquet'],
                               max_objects=2000),
            load_kilonova_windows(['kn_windows_deep.parquet', 'kn_windows_wide.parquet'],
                                  max_objects=2000),
        ], ignore_index=True)
        loaders, normalization = build_dataloaders(long_df, batch_size=32)
        batch = next(iter(loaders['train']))   # replaces make_synthetic_batch()
    """
    splits = split_objects(long_df, fractions=fractions, random_seed=random_seed)
    normalization = compute_normalization(splits["train"])
    loaders = {}
    for name, split_df in splits.items():
        dataset = EarlyWindowDataset(split_df, normalization)
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            collate_fn=collate_token_windows,
        )
    return loaders, normalization
