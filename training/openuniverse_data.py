"""OpenUniverse dataloader for the kilonova classifier (KN vs. everything else).

Four source files:

    * kilonova_windows_deep.hdf5  / kilonova_windows_wide.hdf5   -> the KN class
    * early_windows_deep.parquet  / early_windows_wide.parquet   -> the contaminants

The data already comes windowed (up to 4 visit-day epochs x 5 bands for the contaminants,
3 epochs x 5 bands for the KN), so there is no first-detection anchoring / windowing to do —
this module only maps each measurement to a token, fits the magnitude normalization, and builds
a leakage-aware train/val/test split.

Task: BINARY {KN, other}.  `other` pools every SNANA contaminant (SN II/Ia/Ib/Ic/Iax,
TDE, SLSN-I, PISN). deep and wide are COMBINED into one model over a 6-band vocabulary
(R062, Z087, Y106, J129, H158, F184); a band a given tier never observes is simply absent
(it never produces a token), exactly like a real missing band.

Leakage control:
    * KN  -> split by `simulation_id` (the physical ejecta model). ~630 models each spawn many
      realizations (x angle_index x redshift); a model must live in exactly one split, GLOBAL
      across deep+wide (the same simulation_id appears in both tiers).
    * contaminants -> split by `object_id`, stratified by original class. There is NO usable
      template id in the parquet: `snana_id` has only 33 values and all 33 span every class, so
      it is a SNANA batch/file id, not the SED model. SALT2 Ia are continuous (no template
      leakage); SNANA core-collapse draw from a finite SED template library that is NOT recorded
      here, so template-level leakage among CC cannot be blocked from these files. (Object-level
      split is the best available; flag this if CC template leakage matters.)

Output contract: collated batch dict with the keys KilonovaTransformer.forward() consumes
(see docs/token_definitions.md), plus GROUP_ORDER / regime-loader metadata for
train_lightning.py.

Usage:
    from openuniverse_data import build_dataloaders, GROUP_ORDER
    data = build_dataloaders(
        deep_hdf5='.../kilonova_windows_deep.hdf5',
        wide_hdf5='.../kilonova_windows_wide.hdf5',
        deep_parquet='.../early_windows_deep.parquet',
        wide_parquet='.../early_windows_wide.parquet',
        batch_size=1024,
    )
    train_loader = data['train_loader']
"""

import os
import time

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------------- vocabularies
# Roman bands -> the single-letter vocabulary model.py already uses (NUM_BANDS = 6).
BAND_NAME_TO_LETTER = {
    'R062': 'R', 'Z087': 'Z', 'Y106': 'Y', 'J129': 'J', 'H158': 'H', 'F184': 'F',
}
BAND_ORDER = ['R', 'Z', 'Y', 'J', 'H', 'F']
BAND_TO_INDEX = {band: index for index, band in enumerate(BAND_ORDER)}

TOKEN_TYPE_TO_INDEX = {'d': 0, 'u': 1, 'n': 2}

# Binary task. label 0 = other (any contaminant), label 1 = KN (the positive class).
GROUP_ORDER = ['other', 'KN']
GROUP_TO_LABEL = {'other': 0, 'KN': 1}

EPOCHS_PER_WINDOW = 3  # the model sees at most 3 visit-day epochs (KN windows only have 3)
SNR_MIN = 5.0          # detection threshold, used as the fallback when no `detected` flag

PER_TOKEN_KEYS = [
    'delta_time', 'band_index', 'token_type_index',
    'magnitude', 'sigma_magnitude', 'magnitude_mask', 'sigma_mask',
]
GLOBAL_KEYS = ['redshift', 'redshift_error', 'has_redshift']

SHIFT_PROBABILITY = 0.20             # slide the window forward by one epoch (late-onset sim)
REDSHIFT_DROPOUT_PROBABILITY = 0.50  # hide z from the model -> learned [NO_Z] token

# Magnitude normalization, fit on the TRAIN split inside build_dataloaders (module globals).
MAG_MEAN = None
MAG_STD = None
SIGMA_MAG_MEAN = None
SIGMA_MAG_STD = None


# ----------------------------------------------------------------------------- raw -> tokens
def _vectorize_tokens(object_code, band_index, day, observed, detected, snr,
                      mag_observed, mag_err, mag_limit_5sigma):
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

    token_type = np.full(len(object_code), TOKEN_TYPE_TO_INDEX['n'], dtype=np.int8)
    token_type[is_detection] = TOKEN_TYPE_TO_INDEX['d']
    token_type[is_upper] = TOKEN_TYPE_TO_INDEX['u']

    magnitude = np.full(len(object_code), np.nan)
    magnitude[is_detection] = mag_observed[is_detection]
    magnitude[is_upper] = mag_limit_5sigma[is_upper]
    sigma = np.full(len(object_code), np.nan)
    sigma[is_detection] = mag_err[is_detection]

    order = np.lexsort((band_index, day, object_code))
    big = {
        'day': day.astype(np.float32)[order],
        'band_index': band_index.astype(np.int8)[order],
        'token_type_index': token_type[order],
        'mag': magnitude.astype(np.float32)[order],
        'sigma_mag': sigma.astype(np.float32)[order],
    }
    counts = np.bincount(object_code, minlength=int(object_code.max()) + 1).astype(np.int64)
    return big, counts


# ----------------------------------------------------------------------------- source readers
def _read_kn_hdf5(path, tier, verbose=True):
    """Read every KN object from one tier's hdf5 (the per-group read is the slow part; the token
    mapping is then vectorized over the concatenated columns). Returns (big, counts, meta).
    group_key = simulation_id (GLOBAL across deep+wide so a model in both tiers stays one group).

    `band` is identical for every group in a tier (3 epochs x 5 bands, fixed order) so it is read
    once and tiled; `snr` is skipped (KN groups carry the `detected` flag) -> 6 reads/group."""
    days, observed, detected = [], [], []
    mag_observed, mag_err, mag_limit = [], [], []
    object_code, redshift, group_key = [], [], []
    with h5py.File(path, 'r') as handle:
        keys = list(handle.keys())
        band_ref = handle[keys[0]]['band'][()]
        per_object_length = band_ref.shape[0]
        for code, key in enumerate(keys):
            group = handle[key]
            days.append(group['days_since_detection'][()])
            observed.append(group['observed'][()])
            detected.append(group['detected'][()])
            mag_observed.append(group['mag_observed'][()])
            mag_err.append(group['mag_err'][()])
            mag_limit.append(group['mag_limit_5sigma'][()])
            object_code.append(np.full(per_object_length, code, dtype=np.int64))
            redshift.append(float(group.attrs['redshift']))
            group_key.append(f'sim_{int(group.attrs["simulation_id"])}')
            if verbose and code % 50000 == 0 and code:
                print(f'    {os.path.basename(path)}: {code:,} groups...')

    letters = np.array([BAND_NAME_TO_LETTER.get(b.decode() if isinstance(b, bytes) else str(b))
                        for b in band_ref])
    keep_band = letters != None  # noqa: E711  (constant per object -> tile)
    band_index_object = np.array([BAND_TO_INDEX[b] for b in letters[keep_band]], dtype=np.int64)
    keep = np.tile(keep_band, len(redshift))
    band_index = np.tile(band_index_object, len(redshift))

    big, counts = _vectorize_tokens(
        object_code=np.concatenate(object_code)[keep], band_index=band_index,
        day=np.concatenate(days)[keep], observed=np.concatenate(observed).astype(bool)[keep],
        detected=np.concatenate(detected).astype(bool)[keep], snr=None,
        mag_observed=np.concatenate(mag_observed)[keep], mag_err=np.concatenate(mag_err)[keep],
        mag_limit_5sigma=np.concatenate(mag_limit)[keep],
    )
    meta = {
        'orig_label': np.full(len(redshift), 'KN'),
        'redshift': np.array(redshift, dtype=np.float32),
        'group_key': np.array(group_key),
    }
    return big, counts, meta


def _read_contaminants_parquet(path, tier):
    """Read every contaminant object from one tier's parquet (long format, ~20 rows/object),
    fully vectorized. Returns (big, counts, meta). group_key = object_id (object-level split),
    tier-namespaced so deep/wide ids never collide."""
    table = pq.read_table(path, columns=[
        'object_id', 'label', 'z_CMB', 'days_since_detection', 'band',
        'observed', 'detected', 'snr', 'mag_observed', 'mag_err', 'mag_limit_5sigma',
    ]).to_pandas()

    letters = table['band'].map(BAND_NAME_TO_LETTER)
    table = table[letters.notna()].copy()
    table['band_index'] = letters[letters.notna()].map(BAND_TO_INDEX).to_numpy()

    # contiguous 0..K-1 code per object_id, in sorted-id order
    object_code, unique_ids = table['object_id'].factorize(sort=True)

    big, counts = _vectorize_tokens(
        object_code=object_code.astype(np.int64),
        band_index=table['band_index'].to_numpy(),
        day=table['days_since_detection'].to_numpy(),
        observed=table['observed'].to_numpy().astype(bool),
        detected=table['detected'].to_numpy().astype(bool),
        snr=table['snr'].to_numpy(),
        mag_observed=table['mag_observed'].to_numpy(),
        mag_err=table['mag_err'].to_numpy(),
        mag_limit_5sigma=table['mag_limit_5sigma'].to_numpy(),
    )
    # per-object meta: first row of each object block (label/z constant within an object)
    first_row = table.groupby('object_id', sort=True).first()
    meta = {
        'orig_label': first_row['label'].to_numpy().astype(str),
        'redshift': first_row['z_CMB'].to_numpy().astype(np.float32),
        'group_key': np.array([f'{tier}_{i}' for i in unique_ids]),
    }
    return big, counts, meta


# ----------------------------------------------------------------------------- assembly + cache
def _assemble(deep_hdf5, wide_hdf5, deep_parquet, wide_parquet, verbose=True):
    """Read all four sources into flat ragged arrays (CSR-style: one big array per field +
    per-object offsets) plus per-object metadata."""
    sources = [
        ('KN', _read_kn_hdf5, deep_hdf5, 'deep'),
        ('KN', _read_kn_hdf5, wide_hdf5, 'wide'),
        ('contaminant', _read_contaminants_parquet, deep_parquet, 'deep'),
        ('contaminant', _read_contaminants_parquet, wide_parquet, 'wide'),
    ]
    bigs, counts_list = [], []
    orig_label, redshift, group_key, is_kn = [], [], [], []
    for kind, reader, path, tier in sources:
        start = time.time()
        big, counts, meta = reader(path, tier)
        bigs.append(big)
        counts_list.append(counts)
        orig_label.append(meta['orig_label'])
        redshift.append(meta['redshift'])
        group_key.append(meta['group_key'])
        is_kn.append(np.full(len(counts), kind == 'KN'))
        if verbose:
            print(f'  read {os.path.basename(path):32s} {len(counts):>8,} objects '
                  f'({time.time() - start:.1f}s)')

    n_tokens = np.concatenate(counts_list)
    offsets = np.zeros(len(n_tokens) + 1, dtype=np.int64)
    np.cumsum(n_tokens, out=offsets[1:])

    big = {key: np.concatenate([source[key] for source in bigs])
           for key in ['day', 'band_index', 'token_type_index', 'mag', 'sigma_mag']}
    meta = {
        'offsets': offsets,
        'orig_label': np.concatenate(orig_label),
        'redshift': np.concatenate(redshift),
        'group_key': np.concatenate(group_key),
        'is_kn': np.concatenate(is_kn),
    }
    return big, meta


def _load_or_build(cache_path, deep_hdf5, wide_hdf5, deep_parquet, wide_parquet, verbose=True):
    if cache_path and os.path.exists(cache_path):
        if verbose:
            print(f'loading cached tokens from {cache_path}')
        cached = np.load(cache_path, allow_pickle=False)
        big = {k: cached[k] for k in ['day', 'band_index', 'token_type_index', 'mag', 'sigma_mag']}
        meta = {
            'offsets': cached['offsets'], 'orig_label': cached['orig_label'],
            'redshift': cached['redshift'], 'group_key': cached['group_key'],
            'is_kn': cached['is_kn'],
        }
        return big, meta
    if verbose:
        print('assembling tokens from source files...')
    big, meta = _assemble(deep_hdf5, wide_hdf5, deep_parquet, wide_parquet, verbose=verbose)
    if cache_path:
        np.savez(cache_path, **big, **meta)
        if verbose:
            print(f'cached tokens to {cache_path}')
    return big, meta


# ----------------------------------------------------------------------------- split
def _leakage_aware_split(meta, fractions, random_seed):
    """Train/val/test object indices. KN: grouped by simulation_id (a model lands in one split).
    Contaminants: by object, stratified by original class. Both 90/5/5 by default."""
    rng = np.random.default_rng(random_seed)
    train_fraction, validation_fraction, _ = fractions
    is_kn = meta['is_kn']
    group_key = meta['group_key']
    orig_label = meta['orig_label']
    n_objects = len(is_kn)
    index = np.arange(n_objects)

    train, validation, test = [], [], []

    def carve(object_indices):
        object_indices = object_indices.copy()
        rng.shuffle(object_indices)
        n = len(object_indices)
        n_train = int(round(train_fraction * n))
        n_validation = int(round(validation_fraction * n))
        train.append(object_indices[:n_train])
        validation.append(object_indices[n_train:n_train + n_validation])
        test.append(object_indices[n_train + n_validation:])

    # KN: split the unique simulation_ids, then assign all objects of each model to that split.
    kn_index = index[is_kn]
    kn_groups = group_key[kn_index]
    unique_models = np.unique(kn_groups)
    rng.shuffle(unique_models)
    n_models = len(unique_models)
    n_train_models = int(round(train_fraction * n_models))
    n_validation_models = int(round(validation_fraction * n_models))
    train_models = set(unique_models[:n_train_models])
    validation_models = set(unique_models[n_train_models:n_train_models + n_validation_models])
    in_train = np.array([g in train_models for g in kn_groups])
    in_validation = np.array([g in validation_models for g in kn_groups])
    train.append(kn_index[in_train])
    validation.append(kn_index[in_validation])
    test.append(kn_index[~(in_train | in_validation)])

    # contaminants: split by object, stratified within each original class
    contaminant_index = index[~is_kn]
    contaminant_labels = orig_label[contaminant_index]
    for class_name in np.unique(contaminant_labels):
        carve(contaminant_index[contaminant_labels == class_name])

    return (np.concatenate(train), np.concatenate(validation), np.concatenate(test))


# ----------------------------------------------------------------------------- dataset
class OpenUniverseWindowDataset(Dataset):
    """One example = one object's sampled token window + binary label + true redshift.

    data_aug=True (training): window shift, random prefix truncation to 1/2/3 epochs, and
    redshift dropout. data_aug=False (val/test): full 3-epoch window with the true z.
    force_epochs / force_redshift override the augmentations to build the deterministic
    {1,2,3 epochs} x {with z, without z} validation regimes."""

    def __init__(self, object_indices, big, meta, label_by_index,
                 data_aug=False, random_seed=None, force_epochs=None, force_redshift=None):
        self.object_indices = np.asarray(object_indices, dtype=np.int64)
        self.big = big
        self.offsets = meta['offsets']
        self.redshift = meta['redshift']
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
        day = self.big['day'][lo:hi]
        rng = self.random_generator

        visit_days = np.unique(day)  # sorted unique visit days
        # shift eligibility: >=2 visits and the 2nd visit has a detection (so dropping epoch 1
        # still leaves a detection to anchor on)
        token_type = self.big['token_type_index'][lo:hi]
        is_shift_eligible = False
        if len(visit_days) >= 2:
            second = token_type[day == visit_days[1]]
            is_shift_eligible = bool((second == TOKEN_TYPE_TO_INDEX['d']).any())

        shift_probability = SHIFT_PROBABILITY if self.data_aug else 0.0
        was_shifted = is_shift_eligible and (rng.random() < shift_probability)
        start_epoch = 1 if was_shifted else 0
        window_days = visit_days[start_epoch:start_epoch + EPOCHS_PER_WINDOW]

        if self.force_epochs is not None:
            visible = min(self.force_epochs, len(window_days))
        elif self.data_aug:
            visible = min(int(rng.integers(1, EPOCHS_PER_WINDOW + 1)), len(window_days))
        else:
            visible = len(window_days)
        anchor_day = window_days[0]
        last_day = window_days[visible - 1]

        left = lo + int(np.searchsorted(day, anchor_day, side='left'))
        right = lo + int(np.searchsorted(day, last_day, side='right'))

        delta_time = (self.big['day'][left:right] - anchor_day).astype(np.float32)
        band_index = self.big['band_index'][left:right].astype(np.int64)
        token_type_index = self.big['token_type_index'][left:right].astype(np.int64)
        raw_mag = self.big['mag'][left:right]
        raw_sigma = self.big['sigma_mag'][left:right]
        magnitude_mask = ~np.isnan(raw_mag)
        sigma_mask = ~np.isnan(raw_sigma)
        magnitude = np.where(magnitude_mask, (raw_mag - MAG_MEAN) / MAG_STD, 0.0).astype(np.float32)
        sigma_magnitude = np.where(
            sigma_mask, (raw_sigma - SIGMA_MAG_MEAN) / SIGMA_MAG_STD, 0.0).astype(np.float32)

        true_redshift = float(self.redshift[object_index])
        redshift_for_input = true_redshift
        if self.force_redshift is not None:
            if not self.force_redshift:
                redshift_for_input = np.nan
        elif self.data_aug and rng.random() < REDSHIFT_DROPOUT_PROBABILITY:
            redshift_for_input = np.nan
        has_redshift = bool(np.isfinite(redshift_for_input))

        return {
            'delta_time': torch.from_numpy(delta_time),
            'band_index': torch.from_numpy(band_index),
            'token_type_index': torch.from_numpy(token_type_index),
            'magnitude': torch.from_numpy(magnitude),
            'sigma_magnitude': torch.from_numpy(sigma_magnitude),
            'magnitude_mask': torch.from_numpy(magnitude_mask.astype(np.float32)),
            'sigma_mask': torch.from_numpy(sigma_mask.astype(np.float32)),
            'redshift': torch.tensor(redshift_for_input if has_redshift else 0.0, dtype=torch.float32),
            'redshift_error': torch.tensor(0.0, dtype=torch.float32),
            'has_redshift': torch.tensor(float(has_redshift), dtype=torch.float32),
            'label': torch.tensor(self.label_by_index[object_index], dtype=torch.long),
            'cid': torch.tensor(object_index, dtype=torch.long),
            'true_redshift': torch.tensor(true_redshift, dtype=torch.float32),
        }


def collate_token_windows(batch):
    """Pad variable-length token dicts into rectangular batch tensors + a padding mask
    (True = padded slot to ignore). Globals/labels just stack."""
    batch_size = len(batch)
    max_tokens = max(item['delta_time'].shape[0] for item in batch)

    padded = {
        key: torch.zeros(batch_size, max_tokens, dtype=batch[0][key].dtype)
        for key in PER_TOKEN_KEYS
    }
    padding_mask = torch.ones(batch_size, max_tokens, dtype=torch.bool)
    for row, item in enumerate(batch):
        n = item['delta_time'].shape[0]
        for key in PER_TOKEN_KEYS:
            padded[key][row, :n] = item[key]
        padding_mask[row, :n] = False

    collated = dict(padded)
    collated['padding_mask'] = padding_mask
    for key in GLOBAL_KEYS:
        collated[key] = torch.stack([item[key] for item in batch])
    for key in ('label', 'cid', 'true_redshift'):
        collated[key] = torch.stack([item[key] for item in batch])
    return collated


# ----------------------------------------------------------------------------- entry point
def build_dataloaders(deep_hdf5, wide_hdf5, deep_parquet, wide_parquet, batch_size=1024,
                      fractions=(0.90, 0.05, 0.05), split_seed=42, train_seed=0,
                      num_workers=8, cache_path=None, verbose=True):
    """Read the four OpenUniverse sources, build the leakage-aware 90/5/5 split, fit magnitude
    normalization on TRAIN detections, and return the dataloaders + metadata."""
    global MAG_MEAN, MAG_STD, SIGMA_MAG_MEAN, SIGMA_MAG_STD

    big, meta = _load_or_build(cache_path, deep_hdf5, wide_hdf5, deep_parquet, wide_parquet, verbose)

    label_by_index = np.where(meta['is_kn'], GROUP_TO_LABEL['KN'], GROUP_TO_LABEL['other'])

    train_index, validation_index, test_index = _leakage_aware_split(meta, fractions, split_seed)
    assert len(np.intersect1d(train_index, validation_index)) == 0
    assert len(np.intersect1d(train_index, test_index)) == 0
    assert len(np.intersect1d(validation_index, test_index)) == 0
    # no KN simulation_id may straddle splits
    kn_models_train = set(meta['group_key'][train_index[meta['is_kn'][train_index]]])
    kn_models_test = set(meta['group_key'][test_index[meta['is_kn'][test_index]]])
    assert len(kn_models_train & kn_models_test) == 0, 'KN model leaked between train and test'

    # fit magnitude normalization on TRAIN detection tokens only
    offsets = meta['offsets']
    n_objects = len(offsets) - 1
    token_object = np.repeat(np.arange(n_objects), np.diff(offsets))
    is_train_object = np.zeros(n_objects, dtype=bool)
    is_train_object[train_index] = True
    train_token_mask = is_train_object[token_object]
    detections = (big['token_type_index'] == TOKEN_TYPE_TO_INDEX['d']) & train_token_mask
    MAG_MEAN = float(np.nanmean(big['mag'][detections]))
    MAG_STD = float(np.nanstd(big['mag'][detections]))
    SIGMA_MAG_MEAN = float(np.nanmean(big['sigma_mag'][detections]))
    SIGMA_MAG_STD = float(np.nanstd(big['sigma_mag'][detections]))

    common = dict(big=big, meta=meta, label_by_index=label_by_index)
    train_dataset = OpenUniverseWindowDataset(train_index, data_aug=True, random_seed=train_seed, **common)
    validation_dataset = OpenUniverseWindowDataset(validation_index, data_aug=False, **common)
    test_dataset = OpenUniverseWindowDataset(test_index, data_aug=False, **common)

    loader_kwargs = dict(collate_fn=collate_token_windows, num_workers=num_workers,
                         pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    # balanced validation: the SAME val objects under every regime {1,2,3 epochs} x {z, no-z}
    validation_regime_loaders = []
    for epochs in (1, 2, 3):
        for has_z in (True, False):
            regime_dataset = OpenUniverseWindowDataset(
                validation_index, data_aug=False, force_epochs=epochs, force_redshift=has_z, **common)
            regime_loader = DataLoader(regime_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
            name = f'{epochs}ep_{"z" if has_z else "noz"}'
            validation_regime_loaders.append({'name': name, 'epochs': epochs,
                                              'has_z': has_z, 'loader': regime_loader})

    def class_counts(object_indices):
        labels = label_by_index[object_indices]
        return {'other': int((labels == 0).sum()), 'KN': int((labels == 1).sum())}

    return {
        'train_loader': train_loader,
        'validation_loader': validation_loader,
        'validation_regime_loaders': validation_regime_loaders,
        'test_loader': test_loader,
        'normalization': {
            'MAG_MEAN': MAG_MEAN, 'MAG_STD': MAG_STD,
            'SIGMA_MAG_MEAN': SIGMA_MAG_MEAN, 'SIGMA_MAG_STD': SIGMA_MAG_STD,
        },
        'split_sizes': {
            'train': len(train_index), 'validation': len(validation_index), 'test': len(test_index),
        },
        'class_balance': {
            'train': class_counts(train_index),
            'validation': class_counts(validation_index),
            'test': class_counts(test_index),
        },
    }
