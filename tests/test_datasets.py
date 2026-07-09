import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kilonova.datasets import openuniverse  # noqa: E402


def test_load_early_windows_tags_tier(early_window_frame, tmp_path):
    deep = early_window_frame.copy()
    deep.loc[deep.index[:3], "band"] = "F184"  # an F band marks the file as deep
    deep_path = tmp_path / "early_windows_deep.parquet"
    deep.to_parquet(deep_path, index=False)
    wide_path = tmp_path / "early_windows_wide.parquet"
    early_window_frame.to_parquet(wide_path, index=False)

    combined = openuniverse.load_early_windows([deep_path, wide_path])
    assert set(combined["tier"]) == {"deep", "wide"}
    assert combined["object_id"].str.endswith(("_deep", "_wide")).all()
    # same input object stays unique across tiers
    assert combined["object_id"].nunique() == 2


def test_dataset_token_encoding_shapes(early_window_frame):
    normalization = openuniverse.compute_normalization(early_window_frame)
    assert np.isfinite(list(normalization.values())).all()

    dataset = openuniverse.EarlyWindowDataset(early_window_frame, normalization)
    assert len(dataset) == 1
    item = dataset[0]
    number_of_tokens = len(early_window_frame)  # one token per (epoch, band) row
    for key in openuniverse.PER_TOKEN_KEYS:
        assert len(item[key]) == number_of_tokens, key
    assert int(item["label"]) == openuniverse.GROUP_TO_INDEX["Ia"]

    # d / u / n token pattern: detections, upper limits, gaps
    type_counts = np.bincount(item["token_type_index"], minlength=3)
    detected = int(early_window_frame["detected"].sum())
    unobserved = int((~early_window_frame["observed"]).sum())
    assert type_counts[openuniverse.TOKEN_TYPE_TO_INDEX["d"]] == detected
    assert type_counts[openuniverse.TOKEN_TYPE_TO_INDEX["n"]] == unobserved


def test_collate_pads_and_masks(early_window_frame):
    import pandas as pd

    short = early_window_frame[early_window_frame["epoch"] <= 2].copy()
    short["object_id"] = "obj_2"
    long_df = pd.concat([early_window_frame, short], ignore_index=True)
    normalization = openuniverse.compute_normalization(long_df)
    dataset = openuniverse.EarlyWindowDataset(long_df, normalization)
    batch = openuniverse.collate_token_windows([dataset[0], dataset[1]])

    max_tokens = len(early_window_frame)
    assert batch["magnitude"].shape == (2, max_tokens)
    assert batch["padding_mask"].shape == (2, max_tokens)
    # padding_mask is True on PADDED slots
    assert int(batch["padding_mask"][1].sum()) == max_tokens - len(short)
    assert batch["label"].shape == (2,)
