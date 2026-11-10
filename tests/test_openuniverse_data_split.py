"""The leakage-aware split of `training/openuniverse_data.py`, on a synthetic `meta`.

`training/` is not a package, so the module is imported off its directory the way the training
scripts do it."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

torch = pytest.importorskip("torch")
from openuniverse_data import _leakage_aware_split  # noqa: E402


def _synthetic_meta(number_of_parents=400, copies_per_parent=3, number_of_kn_models=50, seed=0):
    """OpenUniverse contaminants (one object per parent, two classes), KN realizations of a few
    models, and izc re-renderings that carry their parent's group key."""
    rng = np.random.default_rng(seed)
    parent_keys = np.array([f"snana_{index % 7}_{index}" for index in range(number_of_parents)])
    parent_labels = np.where(rng.random(number_of_parents) < 0.5, "SNII", "SNIa")

    kn_groups = np.array([f"sim_{index % number_of_kn_models}" for index in range(4 * number_of_kn_models)])

    izc_parent = np.repeat(parent_keys, copies_per_parent)
    izc_labels = np.repeat(parent_labels, copies_per_parent)

    group_key = np.concatenate([parent_keys, kn_groups, izc_parent])
    orig_label = np.concatenate([parent_labels, np.full(len(kn_groups), "KN"), izc_labels])
    is_kn = np.concatenate(
        [np.zeros(number_of_parents, bool), np.ones(len(kn_groups), bool), np.zeros(len(izc_parent), bool)]
    )
    is_izc = np.concatenate(
        [np.zeros(number_of_parents, bool), np.zeros(len(kn_groups), bool), np.ones(len(izc_parent), bool)]
    )
    return {
        "group_key": group_key,
        "orig_label": orig_label,
        "is_kn": is_kn,
        "is_izc": is_izc,
        "redshift": np.zeros(len(group_key), np.float32),
    }


def test_validation_and_test_hold_no_izc():
    meta = _synthetic_meta()
    train, validation, test = _leakage_aware_split(meta, (0.8, 0.1, 0.1), random_seed=1)
    assert not meta["is_izc"][validation].any()
    assert not meta["is_izc"][test].any()
    assert meta["is_izc"][train].any()


def test_izc_with_a_held_out_parent_is_dropped_not_trained_on():
    meta = _synthetic_meta()
    train, validation, test = _leakage_aware_split(meta, (0.8, 0.1, 0.1), random_seed=1)
    held_out_groups = set(meta["group_key"][validation]) | set(meta["group_key"][test])
    trained_izc_groups = set(meta["group_key"][train][meta["is_izc"][train]])
    assert not (trained_izc_groups & held_out_groups)

    every_index = np.concatenate([train, validation, test])
    dropped = np.setdiff1d(np.arange(len(meta["is_kn"])), every_index)
    assert meta["is_izc"][dropped].all()
    assert set(meta["group_key"][dropped]) <= held_out_groups


def test_izc_of_a_training_parent_joins_train():
    meta = _synthetic_meta()
    train, validation, test = _leakage_aware_split(meta, (0.8, 0.1, 0.1), random_seed=1)
    is_izc = meta["is_izc"]
    training_parents = set(meta["group_key"][train][~is_izc[train] & ~meta["is_kn"][train]])
    izc_index = np.flatnonzero(is_izc)
    expected = izc_index[np.isin(meta["group_key"][izc_index], list(training_parents))]
    assert np.array_equal(np.sort(train[is_izc[train]]), np.sort(expected))


def test_the_catalogue_split_ignores_the_izc():
    """Adding izc objects must not move a single OpenUniverse object or KN across the split."""
    with_izc = _synthetic_meta()
    without_izc = {key: value[~with_izc["is_izc"]] for key, value in with_izc.items()}
    split_with = _leakage_aware_split(with_izc, (0.8, 0.1, 0.1), random_seed=3)
    split_without = _leakage_aware_split(without_izc, (0.8, 0.1, 0.1), random_seed=3)
    for side_with, side_without in zip(split_with, split_without, strict=True):
        catalogue_side = side_with[~with_izc["is_izc"][side_with]]
        assert np.array_equal(np.sort(catalogue_side), np.sort(side_without))


# ----------------------------------------------------------------------------- readers
pd = pytest.importorskip("pandas")
from openuniverse_data import (  # noqa: E402
    BRIGHT_LIMIT_MAG,
    _read_afterglow_parquet,
    _read_contaminants_parquet,
    token_cache_name,
)

WINDOW_BANDS = ["Z087", "Y106", "J129"]


def _write_windows(path, objects):
    """One tiny long-format parquet: `objects` maps object_id -> (label, brightest mag_true).
    Every object gets 2 epochs x 3 bands; the first row carries the brightest true magnitude."""
    rows = []
    for object_id, (label, brightest) in objects.items():
        for epoch in (1, 2):
            for band in WINDOW_BANDS:
                first = epoch == 1 and band == WINDOW_BANDS[0]
                rows.append(
                    {
                        "object_id": object_id,
                        "label": label,
                        "z_CMB": 0.3,
                        "days_since_detection": 5.0 * (epoch - 1),
                        "band": band,
                        "observed": True,
                        "detected": True,
                        "snr": 20.0,
                        "mag_observed": 22.0,
                        "mag_err": 0.05,
                        "mag_limit_5sigma": 25.0,
                        "mag_true": brightest if first else 22.0,
                    }
                )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_objects_brighter_than_the_saturation_limit_are_dropped(tmp_path):
    path = tmp_path / "windows.parquet"
    _write_windows(
        path,
        {
            "snana_1_10": ("SN Ia", 22.0),
            "snana_1_11": ("SLSN-I", BRIGHT_LIMIT_MAG - 0.5),  # saturated at one epoch and band
            "snana_1_12": ("SN II", BRIGHT_LIMIT_MAG),  # at the limit: kept
            "snana_1_13": ("SN Ib", np.inf),  # band with no flux: never saturated
        },
    )
    big, counts, meta = _read_contaminants_parquet(str(path), "deep")
    assert meta["number_saturated"] == 1
    assert list(meta["group_key"]) == ["snana_1_10", "snana_1_12", "snana_1_13"]
    assert list(meta["orig_label"]) == ["SN Ia", "SN II", "SN Ib"]
    assert len(counts) == 3 and counts.sum() == 3 * 6 == len(big["day"])


def test_afterglow_group_is_the_object_id_and_keeps_its_own_label(tmp_path):
    path = tmp_path / "afterglow_windows_deep.parquet"
    _write_windows(
        path, {"afterglow_0000007": ("GRB afterglow", 21.0), "afterglow_0000003": ("GRB afterglow", 23.0)}
    )
    _, _, meta = _read_afterglow_parquet(str(path), "deep")
    assert list(meta["group_key"]) == ["afterglow_0000003", "afterglow_0000007"]
    assert set(meta["orig_label"]) == {"GRB afterglow"}


def test_afterglows_are_split_like_any_other_contaminant_class():
    """They are not augmentation: every split has to hold some, and a group never straddles two."""
    meta = _synthetic_meta()
    afterglow_keys = np.array([f"afterglow_{index:07d}" for index in range(200)])
    meta = {
        "group_key": np.concatenate([meta["group_key"], afterglow_keys, afterglow_keys]),
        "orig_label": np.concatenate([meta["orig_label"], np.full(400, "GRB afterglow")]),
        "is_kn": np.concatenate([meta["is_kn"], np.zeros(400, bool)]),
        "is_izc": np.concatenate([meta["is_izc"], np.zeros(400, bool)]),
        "redshift": np.concatenate([meta["redshift"], np.zeros(400, np.float32)]),
    }
    train, validation, test = _leakage_aware_split(meta, (0.8, 0.1, 0.1), random_seed=1)
    is_afterglow = meta["orig_label"] == "GRB afterglow"
    for side in (train, validation, test):
        assert is_afterglow[side].any()
    for left, right in ((train, validation), (train, test), (validation, test)):
        assert not set(meta["group_key"][left]) & set(meta["group_key"][right])


def test_token_cache_name_is_one_file_per_mix():
    assert token_cache_name(False, False) == "openuniverse_tokens.npz"
    assert token_cache_name(True, False) == "openuniverse_tokens_izc.npz"
    assert token_cache_name(True, True) == "openuniverse_tokens_izc_afterglow.npz"
