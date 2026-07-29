import numpy as np
import pytest

from kilonova.simulation import early_windows

galsim = pytest.importorskip("galsim")  # source_flux_electrons needs the Roman collecting area


def bright_model(bands, magnitude=22.0):
    mjd = np.arange(0.0, 40.0, 2.5)
    return {band: (mjd, np.full(len(mjd), magnitude)) for band in bands}


def test_build_window_fixed_epoch_grid(tier_constants):
    window = early_windows.build_window_from_model(
        "123", bright_model(tier_constants["bands"]), tier_constants, redshift=0.2, gentype=10
    )
    assert window is not None
    assert window["epoch"].nunique() == early_windows.NUMBER_OF_EPOCHS
    # fixed grid: every epoch carries one row per tier band, observed or not
    assert len(window) == early_windows.NUMBER_OF_EPOCHS * len(tier_constants["bands"])
    assert set(window["band"]) == set(tier_constants["bands"])
    assert (window.loc[window["detected"], "mag_err"].dropna() > 0).all()
    # time axis anchored at the first detection
    assert window["days_since_detection"].min() == 0.0


def test_build_window_is_reproducible(tier_constants):
    first = early_windows.build_window_from_model(
        "123", bright_model(tier_constants["bands"]), tier_constants, redshift=0.2, gentype=10
    )
    second = early_windows.build_window_from_model(
        "123", bright_model(tier_constants["bands"]), tier_constants, redshift=0.2, gentype=10
    )
    assert first.equals(second)
    different_seed = early_windows.build_window_from_model(
        "123",
        bright_model(tier_constants["bands"]),
        tier_constants,
        redshift=0.2,
        gentype=10,
        noise_seed=99,
    )
    assert not first["mag_observed"].equals(different_seed["mag_observed"])


def test_band_without_flux_stays_observed_as_upper_limit(tier_constants):
    """A band the kilonova does not emit in (mag_true = +inf) is still observed by the cadence:
    it yields a noise-only non-detection, never an unobserved gap. Dropping it instead leaves the
    anchor band unobserved for kilonovas only -- a label-correlated artifact the model can latch
    onto, since the OpenUniverse contaminants never produce it."""
    bands = tier_constants["bands"]
    anchor_band = tier_constants["anchor_band"]
    model = bright_model(bands)
    anchor_mjd, _ = model[anchor_band]
    model[anchor_band] = (anchor_mjd, np.full(len(anchor_mjd), np.inf))

    window = early_windows.build_window_from_model("123", model, tier_constants, 0.2, 10)

    assert window is not None  # the other bands still detect the transient
    anchor_rows = window[window["band"] == anchor_band]
    assert anchor_rows["observed"].all()
    assert not anchor_rows["detected"].any()
    assert (anchor_rows["snr"] == 0.0).all()
    assert anchor_rows["mag_err"].isna().all()  # 1.0857/snr is meaningless with no flux
    assert np.isfinite(anchor_rows["mag_limit_5sigma"]).all()
    assert np.isfinite(anchor_rows["mag_observed"]).all()


def test_build_window_none_when_undetectable(tier_constants):
    faint = bright_model(tier_constants["bands"], magnitude=45.0)
    assert early_windows.build_window_from_model("123", faint, tier_constants, 0.2, 10) is None
    assert early_windows.build_window_from_model("123", {}, tier_constants, 0.2, 10) is None


def test_sample_kn_realizations_on_grid_covers_every_redshift():
    redshift_grid = np.geomspace(0.01, 1.0, 5)
    realizations = early_windows.sample_kn_realizations_on_grid(
        redshift_grid, realizations_per_redshift=3, simulation_pool=[7, 8], rng=np.random.default_rng(0)
    )
    assert len(realizations) == 15
    assert sorted(realizations) == [realization["noise_id"] for realization in realizations.values()]
    redshift_counts = {}
    for realization in realizations.values():
        assert realization["simulation_id"] in (7, 8)
        assert 0.0 <= realization["explosion_offset_days"] < early_windows.EXPLOSION_OFFSET_MAX_DAYS
        assert 0 <= realization["angle_index"] < early_windows.N_ANGLE_BINS
        redshift_counts[realization["redshift"]] = redshift_counts.get(realization["redshift"], 0) + 1
    assert redshift_counts == {float(redshift): 3 for redshift in redshift_grid}


def test_collect_object_records_drops_fixmag_and_limits(field_catalog_parquet):
    records = early_windows.collect_object_records(field_catalog_parquet)
    assert [record[0] for record in records] == [1, 2, 3]  # gentype 99 dropped
    assert early_windows.collect_object_records(field_catalog_parquet, limit=2) == records[:2]
