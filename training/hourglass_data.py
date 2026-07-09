"""Hourglass survey dataloader for the kilonova classifier.

Extracted verbatim from hourglass_eda.ipynb (the data cells only — all EDA/plotting cells
are left in the notebook). Builds the train/val/test PyTorch DataLoaders whose collated
batch matches the dict KilonovaTransformer.forward() consumes.

The class label is the binary {Ia, CCSN} -- a methodological test on the two well-sampled
classes (SN_Ia vs core-collapse). The Ia-peculiars and rare exotics are dropped; KN is
reserved for the (not-yet-included) injected-signal scenario.

Usage:
    from hourglass_data import build_dataloaders
    data = build_dataloaders(
        objects_path='data/dust_generation/hourglass_objects.parquet',
        photometry_path='data/dust_generation/hourglass_photometry.parquet',
        batch_size=64,
    )
    train_loader = data['train_loader']
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader

SNR_MIN = 5.0
N_EPOCHS = 4
ZP_FLUXCAL = 27.5  # mag_calib = 27.5 - 2.5 * log10(fluxcal)

BAND_ORDER = ['R', 'Z', 'Y', 'J', 'H', 'F']
SURVEY_BANDS = {
    'WIDE': ['R', 'Z', 'Y', 'J'],
    'DEEP': ['Y', 'J', 'H', 'F'],
}

EPOCHS_PER_WINDOW = 3  # the model sees at most 3 visit-day epochs; N_EPOCHS=4 keeps a shift buffer

BAND_TO_INDEX = {band: index for index, band in enumerate(BAND_ORDER)}
TOKEN_TYPE_TO_INDEX = {'d': 0, 'u': 1, 'n': 2}

# Methodological binary test: SN_Ia vs CCSN (the two well-sampled classes, ~21.7k / ~39.2k).
# Everything else -- the Ia-peculiars (Iax, 91bg, ~1.3k each) and the rare exotics (SLSN-I,
# TDE, ILOT, PISN, <70 each) -- is dropped rather than diluting the two clean classes with
# poorly sampled tails. KN stays reserved for the (not-yet-included) injected-signal scenario.
DROP_CLASSES = ('AGN', 'Fixed_mag', 'KN', 'SN_Iax', 'SNIa-91bg', 'SLSN-I', 'TDE', 'ILOT', 'PISN')
CLASS_TO_GROUP = {
    'SN_Ia': 'Ia',
    'CCSN': 'CCSN',
}
GROUP_ORDER = ['Ia', 'CCSN']
GROUP_TO_LABEL = {group_name: index for index, group_name in enumerate(GROUP_ORDER)}

PER_TOKEN_KEYS = [
    'delta_time', 'band_index', 'token_type_index',
    'magnitude', 'sigma_magnitude', 'magnitude_mask', 'sigma_mask',
]
GLOBAL_KEYS = ['redshift', 'redshift_error', 'has_redshift']
SCENARIO_KEYS = ['flux_true', 'fluxcal_err', 'zp']

SHIFT_PROBABILITY = 0.20             # slide the window to {2,3,4} (late-onset simulation)
REDSHIFT_DROPOUT_PROBABILITY = 0.50  # hide z from the model -> learned [NO_Z] token

# Magnitude normalization constants. Fit on the TRAIN split only inside build_dataloaders and
# read by encode_window_to_tokens (module globals, exactly as in the notebook).
MAG_MEAN = None
MAG_STD = None
SIGMA_MAG_MEAN = None
SIGMA_MAG_STD = None


def load_photometry(objects_path, photometry_path):
    """Load and clean the Hourglass photometry into the per-token table `phot`."""
    phot = pq.read_table(
        photometry_path,
        columns=['cid', 'mjd', 'band', 'fluxcal', 'fluxcal_err', 'sim_mag_obs', 'zp'],
    ).to_pandas()

    # 1. drop the simulation sentinel sim_mag_obs == 99 (no model SED at this phase)
    phot = phot[phot['sim_mag_obs'] != 99.0]

    # 2. token type: 'd' detection (SNR >= 5), 'u' upper limit (SNR < 5)
    signal_to_noise = phot['fluxcal'] / phot['fluxcal_err']
    phot['token_type'] = np.where(signal_to_noise >= SNR_MIN, 'd', 'u')

    # 3. anchor each object at its first detection and keep its first N_EPOCHS visit days
    first_detection_mjd = phot[phot['token_type'] == 'd'].groupby('cid')['mjd'].min()
    phot = phot[phot['mjd'] >= phot['cid'].map(first_detection_mjd)]
    window_end_mjd = (
        phot.groupby('cid')['mjd']
        .apply(lambda mjd: np.sort(mjd.unique())[:N_EPOCHS][-1])
    )
    phot = phot[phot['mjd'] <= phot['cid'].map(window_end_mjd)]

    # 4. token type 'n' (not observed): expected survey bands without a measurement on a visit day
    survey_per_object = pq.read_table(objects_path, columns=['cid', 'field']).to_pandas()
    survey_per_object['survey'] = survey_per_object['field'].str.split('+').str[0]
    survey_band_pairs = pd.DataFrame(
        [(survey, band) for survey, bands in SURVEY_BANDS.items() for band in bands],
        columns=['survey', 'band'],
    )
    expected_visits = (
        phot[['cid', 'mjd']]
        .drop_duplicates()
        .merge(survey_per_object[['cid', 'survey']], on='cid')
        .merge(survey_band_pairs, on='survey')
        .drop(columns='survey')
    )
    phot = expected_visits.merge(phot, on=['cid', 'mjd', 'band'], how='left')
    phot['token_type'] = phot['token_type'].fillna('n')

    # 5. observational-scenario channel for the matched KN twin (carried, NOT a model input)
    phot['flux_true'] = 10.0 ** ((ZP_FLUXCAL - phot['sim_mag_obs']) / 2.5)

    return add_token_magnitudes(phot)


def add_token_magnitudes(phot_table, zero_point=ZP_FLUXCAL, snr_limit=SNR_MIN):
    """Per-token apparent magnitude and its error, by token type."""
    phot_table = phot_table.copy()
    magnitude = np.full(len(phot_table), np.nan)
    sigma_magnitude = np.full(len(phot_table), np.nan)

    token_type = phot_table['token_type'].to_numpy()
    is_detection = token_type == 'd'
    is_upper_limit = token_type == 'u'
    flux = phot_table['fluxcal'].to_numpy()
    flux_error = phot_table['fluxcal_err'].to_numpy()

    magnitude[is_detection] = zero_point - 2.5 * np.log10(flux[is_detection])
    sigma_magnitude[is_detection] = (2.5 / np.log(10)) * flux_error[is_detection] / flux[is_detection]
    magnitude[is_upper_limit] = zero_point - 2.5 * np.log10(snr_limit * flux_error[is_upper_limit])

    phot_table['mag'] = magnitude
    phot_table['sigma_mag'] = sigma_magnitude
    return phot_table


def sample_epoch_window(object_table, shift_probability=0.20, n_epochs=EPOCHS_PER_WINDOW,
                        random_generator=None):
    """Select the (up to) n_epochs VISIT days the model sees for one object, anchored at the
    first detection. With probability shift_probability (when shift-eligible) slide the window
    forward by one epoch. Returns (window_rows, anchor_mjd, was_shifted)."""
    if random_generator is None:
        random_generator = np.random.default_rng()

    visit_mjds = np.sort(object_table['mjd'].unique())
    detection_mjds = object_table.loc[object_table['token_type'] == 'd', 'mjd'].unique()

    is_shift_eligible = len(visit_mjds) >= 2 and (visit_mjds[1] in detection_mjds)
    was_shifted = is_shift_eligible and (random_generator.random() < shift_probability)

    start_index = 1 if was_shifted else 0
    window_mjds = visit_mjds[start_index:start_index + n_epochs]
    anchor_mjd = window_mjds[0]

    window_rows = object_table[object_table['mjd'].isin(window_mjds)].copy()
    return window_rows, anchor_mjd, was_shifted


def encode_window_to_tokens(window_rows, anchor_mjd, redshift, redshift_error=np.nan):
    """Convert one object's sampled visit window into the PyTorch tensors the transformer
    consumes. Reads the module-global MAG_MEAN/MAG_STD/SIGMA_MAG_MEAN/SIGMA_MAG_STD set by
    build_dataloaders on the train split."""
    window_rows = window_rows.assign(
        _band_order=window_rows['band'].map(BAND_TO_INDEX)
    ).sort_values(['mjd', '_band_order']).drop(columns='_band_order')

    delta_time = (window_rows['mjd'] - anchor_mjd).to_numpy(dtype=np.float32)
    band_index = window_rows['band'].map(BAND_TO_INDEX).to_numpy(dtype=np.int64)
    token_type_index = window_rows['token_type'].map(TOKEN_TYPE_TO_INDEX).to_numpy(dtype=np.int64)

    raw_magnitude = window_rows['mag'].to_numpy(dtype=np.float32)
    raw_sigma_magnitude = window_rows['sigma_mag'].to_numpy(dtype=np.float32)
    magnitude_mask = ~np.isnan(raw_magnitude)        # d and u carry a magnitude, n does not
    sigma_mask = ~np.isnan(raw_sigma_magnitude)      # only d carries an error

    flux_true = window_rows['flux_true'].to_numpy(dtype=np.float32)
    fluxcal_err = window_rows['fluxcal_err'].to_numpy(dtype=np.float32)
    zero_point = window_rows['zp'].to_numpy(dtype=np.float32)

    magnitude = np.where(magnitude_mask, (raw_magnitude - MAG_MEAN) / MAG_STD, 0.0).astype(np.float32)
    sigma_magnitude = np.where(
        sigma_mask, (raw_sigma_magnitude - SIGMA_MAG_MEAN) / SIGMA_MAG_STD, 0.0
    ).astype(np.float32)

    has_redshift = bool(np.isfinite(redshift))

    tokens = {
        'delta_time': torch.from_numpy(delta_time),
        'band_index': torch.from_numpy(band_index),
        'token_type_index': torch.from_numpy(token_type_index),
        'magnitude': torch.from_numpy(magnitude),
        'sigma_magnitude': torch.from_numpy(sigma_magnitude),
        'magnitude_mask': torch.from_numpy(magnitude_mask.astype(np.float32)),
        'sigma_mask': torch.from_numpy(sigma_mask.astype(np.float32)),
        'redshift': torch.tensor(redshift if has_redshift else 0.0, dtype=torch.float32),
        'redshift_error': torch.tensor(
            redshift_error if np.isfinite(redshift_error) else 0.0, dtype=torch.float32
        ),
        'has_redshift': torch.tensor(float(has_redshift), dtype=torch.float32),
        'flux_true': torch.from_numpy(flux_true),
        'fluxcal_err': torch.from_numpy(fluxcal_err),
        'zp': torch.from_numpy(zero_point),
    }
    return tokens


def split_cids_by_class(group_per_cid, fractions=(0.70, 0.15, 0.15), random_seed=42):
    """Stratified grouped split: shuffle each target class's cids independently and carve the
    train / val / test fractions inside the class. Returns three numpy arrays of cids."""
    random_generator = np.random.default_rng(random_seed)
    train_fraction, validation_fraction, _ = fractions

    train_cids = []
    validation_cids = []
    test_cids = []
    for group_name, class_group in group_per_cid.groupby(group_per_cid):
        cids = class_group.index.to_numpy().copy()
        random_generator.shuffle(cids)
        n_objects = len(cids)
        n_train = int(round(train_fraction * n_objects))
        n_validation = int(round(validation_fraction * n_objects))

        train_cids.extend(cids[:n_train])
        validation_cids.extend(cids[n_train:n_train + n_validation])
        test_cids.extend(cids[n_train + n_validation:])

    return np.array(train_cids), np.array(validation_cids), np.array(test_cids)


def truncate_to_visible_epochs(window_rows, random_generator, max_epochs=EPOCHS_PER_WINDOW):
    """Random prefix truncation for early classification: reveal only the first k visit-day
    epochs, k ~ Uniform{1, ..., max_epochs}, capped at the epochs the object actually has."""
    epoch_mjds = np.sort(window_rows['mjd'].unique())
    target_epochs = int(random_generator.integers(1, max_epochs + 1))
    visible_epochs = min(target_epochs, len(epoch_mjds))
    visible_mjds = epoch_mjds[:visible_epochs]
    return window_rows[window_rows['mjd'].isin(visible_mjds)]


class HourglassWindowDataset(Dataset):
    """One example = one object's sampled token window plus its integer class label and true
    redshift. data_aug=True (training) turns on window shift, prefix truncation to 1/2/3
    epochs, and redshift dropout; data_aug=False (val/test) feeds the full 3-epoch window
    with the true z. The true redshift is always kept in true_redshift."""

    def __init__(self, cids, photometry_by_cid, redshift_by_cid, label_by_cid,
                 data_aug=False, random_seed=None):
        self.cids = list(cids)
        self.photometry_by_cid = photometry_by_cid
        self.redshift_by_cid = redshift_by_cid
        self.label_by_cid = label_by_cid
        self.data_aug = data_aug
        self.random_generator = np.random.default_rng(random_seed)

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, index):
        cid = self.cids[index]
        shift_probability = SHIFT_PROBABILITY if self.data_aug else 0.0
        window_rows, anchor_mjd, _ = sample_epoch_window(
            self.photometry_by_cid[cid],
            shift_probability=shift_probability,
            random_generator=self.random_generator,
        )
        if self.data_aug:
            window_rows = truncate_to_visible_epochs(window_rows, self.random_generator)

        true_redshift = self.redshift_by_cid[cid]
        redshift_for_input = true_redshift
        if self.data_aug and self.random_generator.random() < REDSHIFT_DROPOUT_PROBABILITY:
            redshift_for_input = np.nan

        tokens = encode_window_to_tokens(
            window_rows, anchor_mjd=anchor_mjd, redshift=redshift_for_input,
        )
        tokens['label'] = torch.tensor(self.label_by_cid[cid], dtype=torch.long)
        tokens['cid'] = torch.tensor(int(cid), dtype=torch.long)
        tokens['true_redshift'] = torch.tensor(
            true_redshift if np.isfinite(true_redshift) else np.nan, dtype=torch.float32,
        )
        return tokens


def precompute_cid_arrays(phot):
    """Vectorize the per-object pandas work into plain numpy arrays, once. Each object's rows
    are pre-sorted by (mjd, band_index) — the same order encode_window_to_tokens produces — so
    FastHourglassWindowDataset.__getitem__ is pure array slicing with no pandas per call.
    Returns cid -> dict of arrays + the unique sorted visit days and shift-eligibility flag."""
    work = phot.copy()
    work['band_index'] = work['band'].map(BAND_TO_INDEX).astype(np.int64)
    work['token_type_index'] = work['token_type'].map(TOKEN_TYPE_TO_INDEX).astype(np.int64)
    work = work.sort_values(['cid', 'mjd', 'band_index'], kind='stable')

    cid = work['cid'].to_numpy()
    mjd = work['mjd'].to_numpy(dtype=np.float64)
    band_index = work['band_index'].to_numpy(dtype=np.int64)
    token_type_index = work['token_type_index'].to_numpy(dtype=np.int64)
    mag = work['mag'].to_numpy(dtype=np.float32)
    sigma_mag = work['sigma_mag'].to_numpy(dtype=np.float32)
    flux_true = work['flux_true'].to_numpy(dtype=np.float32)
    fluxcal_err = work['fluxcal_err'].to_numpy(dtype=np.float32)
    zp = work['zp'].to_numpy(dtype=np.float32)

    unique_cids, start_index = np.unique(cid, return_index=True)  # cid is sorted -> contiguous blocks
    end_index = np.concatenate([start_index[1:], [len(cid)]])

    arrays_by_cid = {}
    for object_cid, start, end in zip(unique_cids, start_index, end_index):
        object_mjd = mjd[start:end]
        visit_mjds = np.unique(object_mjd)  # sorted unique visit days
        is_detection = token_type_index[start:end] == TOKEN_TYPE_TO_INDEX['d']
        detection_mjds = np.unique(object_mjd[is_detection])
        is_shift_eligible = bool(len(visit_mjds) >= 2 and (visit_mjds[1] in detection_mjds))
        arrays_by_cid[int(object_cid)] = {
            'mjd': object_mjd,
            'band_index': band_index[start:end],
            'token_type_index': token_type_index[start:end],
            'mag': mag[start:end],
            'sigma_mag': sigma_mag[start:end],
            'flux_true': flux_true[start:end],
            'fluxcal_err': fluxcal_err[start:end],
            'zp': zp[start:end],
            'visit_mjds': visit_mjds,
            'is_shift_eligible': is_shift_eligible,
        }
    return arrays_by_cid


class FastHourglassWindowDataset(Dataset):
    """Numpy-only equivalent of HourglassWindowDataset — same augmentations, same RNG call
    sequence, identical output tensors — but ~20-40x faster per item (no pandas in
    __getitem__). Used by build_dataloaders so training is not dataloader-bound."""

    def __init__(self, cids, arrays_by_cid, redshift_by_cid, label_by_cid,
                 data_aug=False, random_seed=None, force_epochs=None, force_redshift=None):
        self.cids = [int(cid) for cid in cids]
        self.arrays_by_cid = arrays_by_cid
        self.redshift_by_cid = redshift_by_cid
        self.label_by_cid = label_by_cid
        self.data_aug = data_aug
        self.random_generator = np.random.default_rng(random_seed)
        # deterministic regime override (used for the balanced validation set; ignored in training).
        # force_epochs in {1,2,3} fixes how many visit-day epochs are revealed (capped at available);
        # force_redshift in {True, False} fixes whether z is shown (False -> [NO_Z] token).
        self.force_epochs = force_epochs
        self.force_redshift = force_redshift

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, index):
        cid = self.cids[index]
        arrays = self.arrays_by_cid[cid]
        visit_mjds = arrays['visit_mjds']
        random_generator = self.random_generator

        # shift augmentation: rng.random() drawn only when shift-eligible, exactly as the
        # short-circuit in sample_epoch_window (so the RNG stream matches the pandas path).
        shift_probability = SHIFT_PROBABILITY if self.data_aug else 0.0
        was_shifted = False
        if arrays['is_shift_eligible']:
            was_shifted = random_generator.random() < shift_probability

        start_epoch = 1 if was_shifted else 0
        window_mjds = visit_mjds[start_epoch:start_epoch + EPOCHS_PER_WINDOW]
        anchor_mjd = window_mjds[0]

        if self.force_epochs is not None:
            visible_epochs = min(self.force_epochs, len(window_mjds))
        elif self.data_aug:
            target_epochs = int(random_generator.integers(1, EPOCHS_PER_WINDOW + 1))
            visible_epochs = min(target_epochs, len(window_mjds))
        else:
            visible_epochs = len(window_mjds)
        last_mjd = window_mjds[visible_epochs - 1]

        object_mjd = arrays['mjd']
        left = np.searchsorted(object_mjd, anchor_mjd, side='left')
        right = np.searchsorted(object_mjd, last_mjd, side='right')

        delta_time = (object_mjd[left:right] - anchor_mjd).astype(np.float32)
        band_index = arrays['band_index'][left:right].copy()
        token_type_index = arrays['token_type_index'][left:right].copy()
        raw_magnitude = arrays['mag'][left:right]
        raw_sigma_magnitude = arrays['sigma_mag'][left:right]
        magnitude_mask = ~np.isnan(raw_magnitude)
        sigma_mask = ~np.isnan(raw_sigma_magnitude)

        magnitude = np.where(magnitude_mask, (raw_magnitude - MAG_MEAN) / MAG_STD, 0.0).astype(np.float32)
        sigma_magnitude = np.where(
            sigma_mask, (raw_sigma_magnitude - SIGMA_MAG_MEAN) / SIGMA_MAG_STD, 0.0
        ).astype(np.float32)

        true_redshift = self.redshift_by_cid[cid]
        redshift_for_input = true_redshift
        if self.force_redshift is not None:
            if not self.force_redshift:
                redshift_for_input = np.nan          # forced [NO_Z] regime
        elif self.data_aug and random_generator.random() < REDSHIFT_DROPOUT_PROBABILITY:
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
            'flux_true': torch.from_numpy(arrays['flux_true'][left:right].copy()),
            'fluxcal_err': torch.from_numpy(arrays['fluxcal_err'][left:right].copy()),
            'zp': torch.from_numpy(arrays['zp'][left:right].copy()),
            'label': torch.tensor(self.label_by_cid[cid], dtype=torch.long),
            'cid': torch.tensor(cid, dtype=torch.long),
            'true_redshift': torch.tensor(
                true_redshift if np.isfinite(true_redshift) else np.nan, dtype=torch.float32,
            ),
        }


def collate_token_windows(batch):
    """Pad a list of variable-length token dicts into rectangular batch tensors and build the
    padding mask (True = padded slot the transformer must ignore). Globals just stack."""
    batch_size = len(batch)
    max_tokens = max(item['delta_time'].shape[0] for item in batch)

    padded = {
        key: torch.zeros(batch_size, max_tokens, dtype=batch[0][key].dtype)
        for key in PER_TOKEN_KEYS + SCENARIO_KEYS
    }
    padding_mask = torch.ones(batch_size, max_tokens, dtype=torch.bool)

    for row_index, item in enumerate(batch):
        n_tokens = item['delta_time'].shape[0]
        for key in PER_TOKEN_KEYS + SCENARIO_KEYS:
            padded[key][row_index, :n_tokens] = item[key]
        padding_mask[row_index, :n_tokens] = False

    collated = dict(padded)
    collated['padding_mask'] = padding_mask
    for key in GLOBAL_KEYS:
        collated[key] = torch.stack([item[key] for item in batch])
    collated['label'] = torch.stack([item['label'] for item in batch])
    collated['cid'] = torch.stack([item['cid'] for item in batch])
    collated['true_redshift'] = torch.stack([item['true_redshift'] for item in batch])
    return collated


def build_dataloaders(objects_path, photometry_path, batch_size=64,
                      fractions=(0.70, 0.15, 0.15), split_seed=42, train_seed=0,
                      num_workers=0):
    """Load Hourglass, build the stratified grouped split, fit the magnitude normalization on
    the train split, and return the three DataLoaders plus metadata."""
    global MAG_MEAN, MAG_STD, SIGMA_MAG_MEAN, SIGMA_MAG_STD

    phot = load_photometry(objects_path, photometry_path)

    surviving_cids = phot['cid'].unique()
    objects_meta = pq.read_table(objects_path, columns=['cid', 'class', 'z_cmb']).to_pandas()
    objects_meta = objects_meta[objects_meta['cid'].isin(surviving_cids)]
    objects_meta = objects_meta[~objects_meta['class'].isin(DROP_CLASSES)].set_index('cid')
    objects_meta['group'] = objects_meta['class'].map(CLASS_TO_GROUP)

    kept_cids = objects_meta.index.to_numpy()
    group_per_cid = objects_meta['group']
    redshift_by_cid = objects_meta['z_cmb'].to_dict()
    label_by_cid = objects_meta['group'].map(GROUP_TO_LABEL).to_dict()

    train_cids, validation_cids, test_cids = split_cids_by_class(
        group_per_cid, fractions=fractions, random_seed=split_seed)

    assert len(set(train_cids) & set(validation_cids)) == 0
    assert len(set(train_cids) & set(test_cids)) == 0
    assert len(set(validation_cids) & set(test_cids)) == 0
    assert len(train_cids) + len(validation_cids) + len(test_cids) == len(kept_cids)

    # fit magnitude normalization on TRAIN detections only, then reuse for val/test
    train_detection_rows = phot[(phot['token_type'] == 'd') & (phot['cid'].isin(train_cids))]
    MAG_MEAN = float(train_detection_rows['mag'].mean())
    MAG_STD = float(train_detection_rows['mag'].std())
    SIGMA_MAG_MEAN = float(train_detection_rows['sigma_mag'].mean())
    SIGMA_MAG_STD = float(train_detection_rows['sigma_mag'].std())

    # keep only the kept (non-dropped-class) objects, then vectorize each object's rows into
    # numpy arrays once so __getitem__ never touches pandas (the dataloader bottleneck).
    phot = phot[phot['cid'].isin(kept_cids)]
    arrays_by_cid = precompute_cid_arrays(phot)

    train_dataset = FastHourglassWindowDataset(
        train_cids, arrays_by_cid, redshift_by_cid, label_by_cid,
        data_aug=True, random_seed=train_seed)
    validation_dataset = FastHourglassWindowDataset(
        validation_cids, arrays_by_cid, redshift_by_cid, label_by_cid, data_aug=False)
    test_dataset = FastHourglassWindowDataset(
        test_cids, arrays_by_cid, redshift_by_cid, label_by_cid, data_aug=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_token_windows, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                   collate_fn=collate_token_windows, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_token_windows, num_workers=num_workers)

    # balanced validation: the SAME validation objects evaluated under every regime
    # {1,2,3 epochs} x {with z, without z}. Deterministic (data_aug=False) -> stable metric per
    # epoch, while representing the early-classification and no-redshift cases the model deploys on.
    validation_regime_loaders = []
    for epochs in (1, 2, 3):
        for has_z in (True, False):
            regime_dataset = FastHourglassWindowDataset(
                validation_cids, arrays_by_cid, redshift_by_cid, label_by_cid,
                data_aug=False, force_epochs=epochs, force_redshift=has_z)
            regime_loader = DataLoader(regime_dataset, batch_size=batch_size, shuffle=False,
                                       collate_fn=collate_token_windows, num_workers=num_workers)
            name = f'{epochs}ep_{"z" if has_z else "noz"}'
            validation_regime_loaders.append({'name': name, 'epochs': epochs,
                                              'has_z': has_z, 'loader': regime_loader})

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
            'train': len(train_cids),
            'validation': len(validation_cids),
            'test': len(test_cids),
        },
    }
