"""Tiny synthetic fixtures mirroring the pipeline schemas — no real data files."""

import numpy as np
import pandas as pd
import pytest

BANDS_DEEP = ["Z087", "Y106", "J129", "H158", "F184"]


@pytest.fixture
def early_window_frame():
    """Hand-built long DataFrame with the early_windows parquet schema: one object,
    4 epochs x 3 bands, mixing detections, upper limits and unobserved gaps."""
    rows = []
    rng = np.random.default_rng(0)
    for epoch_index, days in enumerate([0.0, 5.0, 10.0, 15.0], start=1):
        for band_position, band in enumerate(["Z087", "Y106", "J129"]):
            observed = not (epoch_index == 2 and band == "J129")
            detected = observed and not (epoch_index == 4 and band == "Z087")
            magnitude = 24.0 + 0.3 * epoch_index + 0.1 * band_position
            rows.append(
                {
                    "object_id": "obj_1",
                    "gentype": 10,
                    "label": "SN Ia",
                    "z_CMB": 0.3,
                    "epoch": epoch_index,
                    "days_since_detection": days,
                    "band": band,
                    "observed": observed,
                    "mag_true": magnitude,
                    "mag_observed": magnitude + rng.normal(0, 0.05) if detected else np.nan,
                    "mag_err": 0.05 if detected else np.nan,
                    "snr": 20.0 if detected else 2.0,
                    "detected": detected,
                    "mag_limit_5sigma": 25.5 if observed else np.nan,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def field_catalog_parquet(tmp_path):
    """Object catalog parquet like snana_XXXXX.parquet (id, z_CMB, gentype), with one FIXMAG row."""
    catalog = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "z_CMB": [0.1, 0.4, 1.2, 0.2],
            "gentype": [10, 32, 42, 99],
        }
    )
    path = tmp_path / "snana_test.parquet"
    catalog.to_parquet(path, index=False)
    return path


@pytest.fixture
def tier_constants():
    """Plain per-tier constants dict like build_tier_constants returns, without touching galsim."""
    bands = BANDS_DEEP
    return {
        "tier": "deep",
        "exposure_time": {band: 300.0 for band in bands},
        "anchor_band": "Z087",
        "bands": bands,
        "zeropoint": {band: 26.0 for band in bands},
        "noise_floor_variance": {band: 2000.0 for band in bands},
        "field_name": "TEST",
    }
