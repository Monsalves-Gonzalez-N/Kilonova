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


def test_cadence_parity_selects_the_band_pair_of_the_first_epoch(tier_constants):
    """The kilonova visit grid is rebuilt from the merger (offset + 5*arange), so its visit 0 is
    always the first post-merger visit. Since a kilonova is fast it is almost always detected right
    there, and without a parity draw the first epoch would always carry the same two non-anchor
    bands -- a label shortcut, because the contaminants sit on the survey's own grid and split ~50/50
    (measured 92% vs 29%: the observation mask alone classified at 80.7%)."""
    bands = tier_constants["bands"]
    base_epochs = np.arange(0.0, 40.0, 5.0)
    model = bright_model(bands)

    def first_epoch_bands(parity):
        window = early_windows.build_window_from_model(
            "123",
            model,
            tier_constants,
            0.2,
            early_windows.KN_GENTYPE,
            base_epochs=base_epochs,
            visit_index_offset=parity,
        )
        first = window[window["days_since_detection"] == 0.0]
        return set(first.loc[first["observed"], "band"])

    even, odd = first_epoch_bands(0), first_epoch_bands(1)
    assert even != odd
    assert tier_constants["anchor_band"] in even & odd  # the anchor is observed either way
    assert len(even) == len(odd) == 3
    assert even | odd == set(bands)  # complementary non-anchor pairs, together the whole tier


def test_kn_object_id_is_unique_and_keeps_the_simulation_id_first():
    """kn_object_id used to leave out noise_id, the only field unique to a realization, so two draws
    colliding on sim/angle/z with offsets rounding alike shared an id (~2 per million). The
    simulation_id has to stay the FIRST field: training/openuniverse_data.py reads it off there to
    keep one ejecta model inside a single split."""
    realizations = early_windows.sample_kn_realizations_on_grid(
        np.geomspace(0.01, 1.0, 5),
        realizations_per_redshift=200,
        simulation_pool=[7, 8],
        rng=np.random.default_rng(0),
    )
    ids = [early_windows.kn_object_id(realization) for realization in realizations.values()]
    assert len(set(ids)) == len(ids)
    for realization, object_id in zip(realizations.values(), ids, strict=True):
        assert object_id.split("_")[0] == str(realization["simulation_id"])


def test_sample_kn_realizations_on_grid_draws_both_cadence_parities():
    realizations = early_windows.sample_kn_realizations_on_grid(
        np.geomspace(0.01, 1.0, 5),
        realizations_per_redshift=400,
        simulation_pool=[7, 8],
        rng=np.random.default_rng(0),
    )
    parities = np.array([r["cadence_parity"] for r in realizations.values()])
    assert set(np.unique(parities)) == {0, 1}
    assert parities.mean() == pytest.approx(0.5, abs=0.05)
    # the parity must be independent of the sub-period delay, not a function of it
    offsets = np.array([r["explosion_offset_days"] for r in realizations.values()])
    assert abs(np.corrcoef(parities, offsets)[0, 1]) < 0.1


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
