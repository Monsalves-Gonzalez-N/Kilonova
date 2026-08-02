"""OpenUniverse dataloader for the kilonova classifier (KN vs. everything else).

Four source files, all long-format parquet with the same window schema:

    * kn_windows_deep.parquet     / kn_windows_wide.parquet      -> the KN class
    * early_windows_deep.parquet  / early_windows_wide.parquet   -> the contaminants

The data already comes windowed (up to 4 visit-day epochs x 5 bands per tier), so there is no
first-detection anchoring / windowing to do — this module only maps each measurement to a token,
fits the magnitude normalization, and builds a leakage-aware train/val/test split.

Task: BINARY {KN, other}.  `other` pools every SNANA contaminant (SN II/Ia/Ib/Ic/Iax,
TDE, SLSN-I, PISN). deep and wide are COMBINED into one model over a 6-band vocabulary
(R062, Z087, Y106, J129, H158, F184); a band a given tier never observes is simply absent
(it never produces a token), exactly like a real missing band.

Leakage control: nothing is split per object, always per GROUP, and groups are GLOBAL across the
deep/wide files, because both tiers observe the same underlying transient.
    * KN  -> group = `simulation_id` (the physical ejecta model), read off the first field of the
      kn_object_id. 900 models each spawn many realizations (x angle_index x redshift x noise); a
      model must live in exactly one split.
    * contaminants -> group = `object_id`, stratified by original class. The deep file is a
      superset of the wide one: 717,863 of the 717,864 wide transients also appear in deep with
      the SAME object_id, so grouping is what keeps the two views of one SNANA light curve on the
      same side of the split. There is NO usable template id in the parquet: `snana_id` has only
      33 values and all 33 span every class, so it is a SNANA batch/file id (healpix), not the SED
      model. SALT2 Ia are continuous (no template leakage); SNANA core-collapse draw from a finite
      SED template library that is NOT recorded here, so template-level leakage among CC cannot be
      blocked from these files. (Object-level grouping is the best available; flag this if CC
      template leakage matters.)

Output contract: collated batch dict with the keys KilonovaTransformer.forward() consumes
(see docs/token_definitions.md), plus GROUP_ORDER / regime-loader metadata for
train_lightning.py.

Usage:
    from openuniverse_data import build_dataloaders, GROUP_ORDER
    data = build_dataloaders(
        kn_deep='.../kn_windows_deep.parquet',
        kn_wide='.../kn_windows_wide.parquet',
        contaminant_deep='.../early_windows_deep.parquet',
        contaminant_wide='.../early_windows_wide.parquet',
        batch_size=1024,
    )
    train_loader = data['train_loader']
"""

import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------------- vocabularies
# Roman bands -> the single-letter vocabulary model.py already uses (NUM_BANDS = 6).
BAND_NAME_TO_LETTER = {
    "R062": "R",
    "Z087": "Z",
    "Y106": "Y",
    "J129": "J",
    "H158": "H",
    "F184": "F",
}
BAND_ORDER = ["R", "Z", "Y", "J", "H", "F"]
BAND_TO_INDEX = {band: index for index, band in enumerate(BAND_ORDER)}

TOKEN_TYPE_TO_INDEX = {"d": 0, "u": 1, "n": 2}

# Binary task. label 0 = other (any contaminant), label 1 = KN (the positive class).
GROUP_ORDER = ["other", "KN"]
GROUP_TO_LABEL = {"other": 0, "KN": 1}

# The model sees at most 3 visit-day epochs. This was set by the old KN hdf5, whose windows only
# had 3; kn_windows_*.parquet carries 4, the same as the contaminants, so 4 is now available and
# would be a model-side decision -- left at 3 so switching sources changes no behaviour.
EPOCHS_PER_WINDOW = 3
SNR_MIN = 5.0  # detection threshold, used as the fallback when no `detected` flag

PER_TOKEN_KEYS = [
    "delta_time",
    "band_index",
    "token_type_index",
    "magnitude",
    "sigma_magnitude",
    "magnitude_mask",
    "sigma_mask",
]
GLOBAL_KEYS = ["redshift", "has_redshift"]

SHIFT_PROBABILITY = 0.20  # slide the window forward by one epoch (late-onset sim)
REDSHIFT_DROPOUT_PROBABILITY = 0.50  # hide z from the model -> learned [NO_Z] token

# Bumped whenever the meaning of meta['group_key'] changes, so a token cache written by an older
# revision is rebuilt instead of silently reproducing its split.
#   v1: contaminant groups were tier-namespaced (deep_<id> / wide_<id>) -> the two tiers of one
#       transient landed in independent splits.
#   v2: contaminant groups are the bare object_id, global across tiers.
GROUP_KEY_VERSION = 2

# Magnitude normalization, fit on the TRAIN split inside build_dataloaders (module globals).
MAG_MEAN = None
MAG_STD = None
SIGMA_MAG_MEAN = None
SIGMA_MAG_STD = None


# ----------------------------------------------------------------------------- raw -> tokens
def _vectorize_tokens(
    object_code, band_index, day, observed, detected, snr, mag_observed, mag_err, mag_limit_5sigma
):
    """Vectorized mapping of a flat column of measurements (object_code = a contiguous 0..K-1
    integer id per row) to sorted big token arrays + per-object token counts.

    token type: 'd' detection (detected / SNR>=5), 'u' upper limit (observed, not detected),
    'n' not observed. magnitude: detection -> mag_observed, upper limit -> mag_limit_5sigma,
    not observed -> NaN. sigma: only detections carry one. Output is sorted by
    (object_code, day, band_index) — the order the model consumes and __getitem__ relies on."""
    if detected is None:
        is_detection = observed & (snr >= SNR_MIN)
    else:
        is_detection = detected
    is_upper = observed & ~is_detection

    token_type = np.full(len(object_code), TOKEN_TYPE_TO_INDEX["n"], dtype=np.int8)
    token_type[is_detection] = TOKEN_TYPE_TO_INDEX["d"]
    token_type[is_upper] = TOKEN_TYPE_TO_INDEX["u"]

    magnitude = np.full(len(object_code), np.nan)
    magnitude[is_detection] = mag_observed[is_detection]
    magnitude[is_upper] = mag_limit_5sigma[is_upper]
    sigma = np.full(len(object_code), np.nan)
    sigma[is_detection] = mag_err[is_detection]

    order = np.lexsort((band_index, day, object_code))
    big = {
        "day": day.astype(np.float32)[order],
        "band_index": band_index.astype(np.int8)[order],
        "token_type_index": token_type[order],
        "mag": magnitude.astype(np.float32)[order],
        "sigma_mag": sigma.astype(np.float32)[order],
    }
    counts = np.bincount(object_code, minlength=int(object_code.max()) + 1).astype(np.int64)
    return big, counts


# ----------------------------------------------------------------------------- source readers
def _read_long_parquet(path, group_key_from_ids):
    """Read every object from one long-format parquet (~20 rows/object), fully vectorized.
    Returns (big, counts, meta).

    Both classes come from the same window schema, so the only thing that differs between them is
    what a "group" is for the leakage-aware split -- hence `group_key_from_ids`, a callable mapping
    the sorted unique object_ids to their split group."""
    table = pq.read_table(
        path,
        columns=[
            "object_id",
            "label",
            "z_CMB",
            "days_since_detection",
            "band",
            "observed",
            "detected",
            "snr",
            "mag_observed",
            "mag_err",
            "mag_limit_5sigma",
        ],
    ).to_pandas()

    letters = table["band"].map(BAND_NAME_TO_LETTER)
    table = table[letters.notna()].copy()
    table["band_index"] = letters[letters.notna()].map(BAND_TO_INDEX).to_numpy()

    # contiguous 0..K-1 code per object_id, in sorted-id order
    object_code, unique_ids = table["object_id"].factorize(sort=True)

    big, counts = _vectorize_tokens(
        object_code=object_code.astype(np.int64),
        band_index=table["band_index"].to_numpy(),
        day=table["days_since_detection"].to_numpy(),
        observed=table["observed"].to_numpy().astype(bool),
        detected=table["detected"].to_numpy().astype(bool),
        snr=table["snr"].to_numpy(),
        mag_observed=table["mag_observed"].to_numpy(),
        mag_err=table["mag_err"].to_numpy(),
        mag_limit_5sigma=table["mag_limit_5sigma"].to_numpy(),
    )
    # per-object meta: first row of each object block (label/z constant within an object)
    first_row = table.groupby("object_id", sort=True).first()
    meta = {
        "orig_label": first_row["label"].to_numpy().astype(str),
        "redshift": first_row["z_CMB"].to_numpy().astype(np.float32),
        "group_key": group_key_from_ids(unique_ids),
    }
    return big, counts, meta


def _read_contaminants_parquet(path, tier):  # tier unused: groups are deliberately not namespaced
    """OpenUniverse contaminants. Group by transient: group_key = object_id, which is already
    global (`snana_{healpix}_{snana_object_id}`). NO tier prefix: the deep file is a superset of
    the wide one, so the same transient appears in both with the same id and its two views must
    stay a single group across the two files -- same underlying SNANA light curve, same z, same
    peak; only bands, exposure, noise and window differ."""
    return _read_long_parquet(path, lambda ids: np.asarray(ids, dtype=str))


def _read_kn_parquet(path, tier):  # tier unused: KN groups are deliberately not namespaced
    """Kilonovas from kn_windows_{tier}.parquet.

    Group by the LANL ejecta model, not by object: one simulation spawns many realizations
    (x angle x redshift x offset x parity x noise) and all of them have to land in the same split,
    or the model sees the same SED in train and test. kn_object_id is
    `{simulation_id}_{angle_index}_{offset:.4f}_{z:.4f}_{cadence_parity}_{noise_id}`
    (kilonova.simulation.early_windows), so the first field is that simulation_id."""
    return _read_long_parquet(path, lambda ids: np.array([f"sim_{i.split('_')[0]}" for i in ids]))


# ----------------------------------------------------------------------------- assembly + cache
def _assemble(kn_deep, kn_wide, contaminant_deep, contaminant_wide, verbose=True):
    """Read all four sources into flat ragged arrays (CSR-style: one big array per field +
    per-object offsets) plus per-object metadata."""
    sources = [
        ("KN", _read_kn_parquet, kn_deep, "deep"),
        ("KN", _read_kn_parquet, kn_wide, "wide"),
        ("contaminant", _read_contaminants_parquet, contaminant_deep, "deep"),
        ("contaminant", _read_contaminants_parquet, contaminant_wide, "wide"),
    ]
    bigs, counts_list = [], []
    orig_label, redshift, group_key, is_kn = [], [], [], []
    for kind, reader, path, tier in sources:
        start = time.time()
        big, counts, meta = reader(path, tier)
        bigs.append(big)
        counts_list.append(counts)
        orig_label.append(meta["orig_label"])
        redshift.append(meta["redshift"])
        group_key.append(meta["group_key"])
        is_kn.append(np.full(len(counts), kind == "KN"))
        if verbose:
            print(
                f"  read {os.path.basename(path):32s} {len(counts):>8,} objects ({time.time() - start:.1f}s)"
            )

    n_tokens = np.concatenate(counts_list)
    offsets = np.zeros(len(n_tokens) + 1, dtype=np.int64)
    np.cumsum(n_tokens, out=offsets[1:])

    big = {
        key: np.concatenate([source[key] for source in bigs])
        for key in ["day", "band_index", "token_type_index", "mag", "sigma_mag"]
    }
    meta = {
        "offsets": offsets,
        "orig_label": np.concatenate(orig_label),
        "redshift": np.concatenate(redshift),
        "group_key": np.concatenate(group_key),
        "is_kn": np.concatenate(is_kn),
    }
    return big, meta


def _load_or_build(cache_path, kn_deep, kn_wide, contaminant_deep, contaminant_wide, verbose=True):
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        cached_version = int(cached["group_key_version"]) if "group_key_version" in cached else 1
        if cached_version == GROUP_KEY_VERSION:
            if verbose:
                print(f"loading cached tokens from {cache_path}")
            big = {k: cached[k] for k in ["day", "band_index", "token_type_index", "mag", "sigma_mag"]}
            meta = {
                "offsets": cached["offsets"],
                "orig_label": cached["orig_label"],
                "redshift": cached["redshift"],
                "group_key": cached["group_key"],
                "is_kn": cached["is_kn"],
            }
            return big, meta
        print(
            f"cache {cache_path} holds group_key v{cached_version}, this revision needs "
            f"v{GROUP_KEY_VERSION} -- rebuilding (its split is not reproducible from here)"
        )
    if verbose:
        print("assembling tokens from source files...")
    big, meta = _assemble(kn_deep, kn_wide, contaminant_deep, contaminant_wide, verbose=verbose)
    if cache_path:
        np.savez(cache_path, **big, **meta, group_key_version=GROUP_KEY_VERSION)
        if verbose:
            print(f"cached tokens to {cache_path}")
    return big, meta


# ----------------------------------------------------------------------------- split
def _leakage_aware_split(meta, fractions, random_seed):
    """Train/val/test object indices, 90/5/5 by default. Whole GROUPS are carved, never objects:
    KN group = simulation_id (a LANL ejecta model lands in one split), contaminant group =
    object_id (a transient lands in one split, both tiers together). Contaminants are additionally
    stratified by original class; the KN pool is carved as a single group population.

    The fractions apply to the group counts, so the resulting object counts drift a little from
    90/5/5 -- groups have unequal numbers of objects (a contaminant seen in both tiers weighs 2,
    one seen only in deep weighs 1)."""
    rng = np.random.default_rng(random_seed)
    train_fraction, validation_fraction, _ = fractions
    is_kn = meta["is_kn"]
    group_key = meta["group_key"]
    orig_label = meta["orig_label"]
    index = np.arange(len(is_kn))

    train, validation, test = [], [], []

    def carve_groups(object_indices):
        unique_groups, group_code = np.unique(group_key[object_indices], return_inverse=True)
        shuffled = rng.permutation(len(unique_groups))
        n_train = int(round(train_fraction * len(unique_groups)))
        n_validation = int(round(validation_fraction * len(unique_groups)))
        split_of_group = np.empty(len(unique_groups), dtype=np.int8)
        split_of_group[shuffled[:n_train]] = 0
        split_of_group[shuffled[n_train : n_train + n_validation]] = 1
        split_of_group[shuffled[n_train + n_validation :]] = 2
        split_of_object = split_of_group[group_code]
        train.append(object_indices[split_of_object == 0])
        validation.append(object_indices[split_of_object == 1])
        test.append(object_indices[split_of_object == 2])

    carve_groups(index[is_kn])

    contaminant_index = index[~is_kn]
    contaminant_labels = orig_label[contaminant_index]
    for class_name in np.unique(contaminant_labels):
        carve_groups(contaminant_index[contaminant_labels == class_name])

    return (np.concatenate(train), np.concatenate(validation), np.concatenate(test))


# ----------------------------------------------------------------------------- dataset
class OpenUniverseWindowDataset(Dataset):
    """One example = one object's sampled token window + binary label + true redshift.

    data_aug=True (training): window shift, random prefix truncation to 1/2/3 epochs, and
    redshift dropout. data_aug=False (val/test): full 3-epoch window with the true z.
    force_epochs / force_redshift override the augmentations to build the deterministic
    {1,2,3 epochs} x {with z, without z} validation regimes."""

    def __init__(
        self,
        object_indices,
        big,
        meta,
        label_by_index,
        data_aug=False,
        random_seed=None,
        force_epochs=None,
        force_redshift=None,
    ):
        self.object_indices = np.asarray(object_indices, dtype=np.int64)
        self.big = big
        self.offsets = meta["offsets"]
        self.redshift = meta["redshift"]
        self.label_by_index = label_by_index
        self.data_aug = data_aug
        self.random_generator = np.random.default_rng(random_seed)
        self.force_epochs = force_epochs
        self.force_redshift = force_redshift

    def __len__(self):
        return len(self.object_indices)

    def __getitem__(self, position):
        object_index = int(self.object_indices[position])
        lo, hi = int(self.offsets[object_index]), int(self.offsets[object_index + 1])
        day = self.big["day"][lo:hi]
        rng = self.random_generator

        visit_days = np.unique(day)  # sorted unique visit days
        # shift eligibility: >=2 visits and the 2nd visit has a detection (so dropping epoch 1
        # still leaves a detection to anchor on)
        token_type = self.big["token_type_index"][lo:hi]
        is_shift_eligible = False
        if len(visit_days) >= 2:
            second = token_type[day == visit_days[1]]
            is_shift_eligible = bool((second == TOKEN_TYPE_TO_INDEX["d"]).any())

        shift_probability = SHIFT_PROBABILITY if self.data_aug else 0.0
        was_shifted = is_shift_eligible and (rng.random() < shift_probability)
        start_epoch = 1 if was_shifted else 0
        window_days = visit_days[start_epoch : start_epoch + EPOCHS_PER_WINDOW]

        if self.force_epochs is not None:
            visible = min(self.force_epochs, len(window_days))
        elif self.data_aug:
            visible = min(int(rng.integers(1, EPOCHS_PER_WINDOW + 1)), len(window_days))
        else:
            visible = len(window_days)
        anchor_day = window_days[0]
        last_day = window_days[visible - 1]

        left = lo + int(np.searchsorted(day, anchor_day, side="left"))
        right = lo + int(np.searchsorted(day, last_day, side="right"))

        delta_time = (self.big["day"][left:right] - anchor_day).astype(np.float32)
        band_index = self.big["band_index"][left:right].astype(np.int64)
        token_type_index = self.big["token_type_index"][left:right].astype(np.int64)
        raw_mag = self.big["mag"][left:right]
        raw_sigma = self.big["sigma_mag"][left:right]
        magnitude_mask = ~np.isnan(raw_mag)
        sigma_mask = ~np.isnan(raw_sigma)
        magnitude = np.where(magnitude_mask, (raw_mag - MAG_MEAN) / MAG_STD, 0.0).astype(np.float32)
        sigma_magnitude = np.where(sigma_mask, (raw_sigma - SIGMA_MAG_MEAN) / SIGMA_MAG_STD, 0.0).astype(
            np.float32
        )

        true_redshift = float(self.redshift[object_index])
        redshift_for_input = true_redshift
        if self.force_redshift is not None:
            if not self.force_redshift:
                redshift_for_input = np.nan
        elif self.data_aug and rng.random() < REDSHIFT_DROPOUT_PROBABILITY:
            redshift_for_input = np.nan
        has_redshift = bool(np.isfinite(redshift_for_input))

        return {
            "delta_time": torch.from_numpy(delta_time),
            "band_index": torch.from_numpy(band_index),
            "token_type_index": torch.from_numpy(token_type_index),
            "magnitude": torch.from_numpy(magnitude),
            "sigma_magnitude": torch.from_numpy(sigma_magnitude),
            "magnitude_mask": torch.from_numpy(magnitude_mask.astype(np.float32)),
            "sigma_mask": torch.from_numpy(sigma_mask.astype(np.float32)),
            "redshift": torch.tensor(redshift_for_input if has_redshift else 0.0, dtype=torch.float32),
            "has_redshift": torch.tensor(float(has_redshift), dtype=torch.float32),
            "label": torch.tensor(self.label_by_index[object_index], dtype=torch.long),
            "cid": torch.tensor(object_index, dtype=torch.long),
            "true_redshift": torch.tensor(true_redshift, dtype=torch.float32),
        }


def collate_token_windows(batch):
    """Pad variable-length token dicts into rectangular batch tensors + a padding mask
    (True = padded slot to ignore). Globals/labels just stack."""
    batch_size = len(batch)
    max_tokens = max(item["delta_time"].shape[0] for item in batch)

    padded = {key: torch.zeros(batch_size, max_tokens, dtype=batch[0][key].dtype) for key in PER_TOKEN_KEYS}
    padding_mask = torch.ones(batch_size, max_tokens, dtype=torch.bool)
    for row, item in enumerate(batch):
        n = item["delta_time"].shape[0]
        for key in PER_TOKEN_KEYS:
            padded[key][row, :n] = item[key]
        padding_mask[row, :n] = False

    collated = dict(padded)
    collated["padding_mask"] = padding_mask
    for key in GLOBAL_KEYS:
        collated[key] = torch.stack([item[key] for item in batch])
    for key in ("label", "cid", "true_redshift"):
        collated[key] = torch.stack([item[key] for item in batch])
    return collated


# ----------------------------------------------------------------------------- entry point
def build_dataloaders(
    kn_deep,
    kn_wide,
    contaminant_deep,
    contaminant_wide,
    batch_size=1024,
    fractions=(0.90, 0.05, 0.05),
    split_seed=42,
    train_seed=0,
    num_workers=8,
    cache_path=None,
    verbose=True,
):
    """Read the four OpenUniverse sources, build the leakage-aware 90/5/5 split, fit magnitude
    normalization on TRAIN detections, and return the dataloaders + metadata."""
    global MAG_MEAN, MAG_STD, SIGMA_MAG_MEAN, SIGMA_MAG_STD

    big, meta = _load_or_build(cache_path, kn_deep, kn_wide, contaminant_deep, contaminant_wide, verbose)

    label_by_index = np.where(meta["is_kn"], GROUP_TO_LABEL["KN"], GROUP_TO_LABEL["other"])

    train_index, validation_index, test_index = _leakage_aware_split(meta, fractions, split_seed)
    assert len(np.intersect1d(train_index, validation_index)) == 0
    assert len(np.intersect1d(train_index, test_index)) == 0
    assert len(np.intersect1d(validation_index, test_index)) == 0
    # no group may straddle two splits: neither a KN simulation_id nor a contaminant object_id
    # (the latter is what the tier-namespaced v1 group key silently allowed).
    for name, left, right in (
        ("train/validation", train_index, validation_index),
        ("train/test", train_index, test_index),
        ("validation/test", validation_index, test_index),
    ):
        straddling = np.intersect1d(meta["group_key"][left], meta["group_key"][right])
        assert len(straddling) == 0, f"{len(straddling)} groups leaked across {name}: {straddling[:5]}"

    # fit magnitude normalization on TRAIN detection tokens only
    offsets = meta["offsets"]
    n_objects = len(offsets) - 1
    token_object = np.repeat(np.arange(n_objects), np.diff(offsets))
    is_train_object = np.zeros(n_objects, dtype=bool)
    is_train_object[train_index] = True
    train_token_mask = is_train_object[token_object]
    detections = (big["token_type_index"] == TOKEN_TYPE_TO_INDEX["d"]) & train_token_mask
    MAG_MEAN = float(np.nanmean(big["mag"][detections]))
    MAG_STD = float(np.nanstd(big["mag"][detections]))
    SIGMA_MAG_MEAN = float(np.nanmean(big["sigma_mag"][detections]))
    SIGMA_MAG_STD = float(np.nanstd(big["sigma_mag"][detections]))

    common = dict(big=big, meta=meta, label_by_index=label_by_index)
    train_dataset = OpenUniverseWindowDataset(train_index, data_aug=True, random_seed=train_seed, **common)
    validation_dataset = OpenUniverseWindowDataset(validation_index, data_aug=False, **common)
    test_dataset = OpenUniverseWindowDataset(test_index, data_aug=False, **common)

    loader_kwargs = dict(
        collate_fn=collate_token_windows,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    # balanced validation: the SAME val objects under every regime {1,2,3 epochs} x {z, no-z}
    validation_regime_loaders = []
    for epochs in (1, 2, 3):
        for has_z in (True, False):
            regime_dataset = OpenUniverseWindowDataset(
                validation_index, data_aug=False, force_epochs=epochs, force_redshift=has_z, **common
            )
            regime_loader = DataLoader(regime_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
            name = f"{epochs}ep_{'z' if has_z else 'noz'}"
            validation_regime_loaders.append(
                {"name": name, "epochs": epochs, "has_z": has_z, "loader": regime_loader}
            )

    def class_counts(object_indices):
        labels = label_by_index[object_indices]
        return {"other": int((labels == 0).sum()), "KN": int((labels == 1).sum())}

    return {
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "validation_regime_loaders": validation_regime_loaders,
        "test_loader": test_loader,
        "normalization": {
            "MAG_MEAN": MAG_MEAN,
            "MAG_STD": MAG_STD,
            "SIGMA_MAG_MEAN": SIGMA_MAG_MEAN,
            "SIGMA_MAG_STD": SIGMA_MAG_STD,
        },
        "split_sizes": {
            "train": len(train_index),
            "validation": len(validation_index),
            "test": len(test_index),
        },
        "class_balance": {
            "train": class_counts(train_index),
            "validation": class_counts(validation_index),
            "test": class_counts(test_index),
        },
    }
