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
