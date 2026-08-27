"""The low-redshift contaminant generator: kilonova.simulation.intermediate_z_contaminants.

Two of these tests exist because the code was wrong in exactly that way during development, and
both failures were silent enough to reach a pilot run:

  * the source list was written by hand and contained names that are not in the sncosmo registry;
    `test_listed_sources_exist_and_cover_the_roman_bands` re-derives the coverage instead of
    trusting the list.
  * each band was allowed its own phase range, so one object in a hundred reached the window with
    two observed bands in the first epoch instead of three -- the configuration the training set
    never contains and that `training/diagnostics/anchor_band_ablation.py` shows the model cannot
    read. `test_first_epoch_always_has_three_observed_bands` pins it.
"""

import numpy as np
import pytest

from kilonova.simulation.intermediate_z_contaminants import (
    ALL_ROMAN_BANDS,
    CLASS_FRACTION,
    GENTYPE_BY_LABEL,
    IZC_GENTYPE_OFFSET,
    PEAK_ABSOLUTE_MAGNITUDE,
    SOURCES_BY_LABEL,
    build_izc_windows,
    draw_population,
    roman_light_curve,
    saturation_magnitude,
)

galsim = pytest.importorskip("galsim")


def test_class_fraction_is_a_distribution():
    assert sum(CLASS_FRACTION.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(fraction > 0 for fraction in CLASS_FRACTION.values())


def test_every_generated_class_is_fully_specified():
    """A class that can be drawn must have sources, a luminosity function and a gentype."""
    for label in CLASS_FRACTION:
        assert SOURCES_BY_LABEL.get(label), label
        assert label in PEAK_ABSOLUTE_MAGNITUDE, label
        assert label in GENTYPE_BY_LABEL, label


def test_izc_gentypes_stay_separable_from_the_openuniverse_ones():
    """An izc object has to be findable in the parquet without a join against anything."""
    for label, gentype in GENTYPE_BY_LABEL.items():
        assert gentype + IZC_GENTYPE_OFFSET > 100, label


@pytest.mark.parametrize("tier", ["deep", "wide"])
def test_saturation_is_far_brighter_than_the_survey_depth(tier):
    from kilonova.photometry.roman_noise import SNR_DETECTION, build_tier_constants

    constants = build_tier_constants(tier)
    for band, magnitude in saturation_magnitude(tier).items():
        assert 12.0 < magnitude < 22.0, (tier, band, magnitude)
        assert band in constants["bands"]
    assert SNR_DETECTION == 5.0  # the depth these limits are meant to sit far above


def test_listed_sources_exist_and_cover_the_roman_bands():
    """Every listed source is in the registry AND spans R062 to F184 at z = 0.02, unextrapolated.

    This is the check the hand-written list failed: `snana-2007od` looked plausible and does not
    exist, and the Vincenzi templates that do exist stop at 11000 A rest-frame, which covers Roman
    only above z = 0.91."""
    sncosmo = pytest.importorskip("sncosmo")
    from kilonova.photometry.roman_noise import roman_bandpasses

    bandpasses = roman_bandpasses()
    lowest_redshift = 1.02
    required_blue = bandpasses["R062"].blue_limit * 10 / lowest_redshift
    required_red = bandpasses["F184"].red_limit * 10 / lowest_redshift

    for label, source_names in SOURCES_BY_LABEL.items():
        for source_name in source_names:
            source = sncosmo.get_source(source_name)
            assert source.minwave() <= required_blue, (label, source_name, source.minwave())
            assert source.maxwave() >= required_red, (label, source_name, source.maxwave())


def test_light_curve_bands_share_one_phase_grid():
    """All six bands must be sampled on the same days, or a scheduled band goes unobserved."""
    pytest.importorskip("sncosmo")
    realization = {
        "index": 0,
        "label": "SN IIP",
        "source_name": SOURCES_BY_LABEL["SN IIP"][0],
        "peak_absolute_magnitude": -16.8,
        "redshift": 0.05,
        "cadence_parity": 0,
    }
    curves = roman_light_curve(realization)
    assert set(curves) == set(ALL_ROMAN_BANDS)
    days = curves[ALL_ROMAN_BANDS[0]][0]
    assert len(days) > 5
    for band, (band_days, magnitudes) in curves.items():
        np.testing.assert_array_equal(band_days, days, err_msg=band)
        assert np.isfinite(magnitudes).all(), band


def test_first_epoch_always_has_three_observed_bands():
    """The invariant of the training set: one visit observes the anchor plus two, never fewer."""
    pytest.importorskip("sncosmo")
    random_generator = np.random.default_rng(0)
    redshifts = random_generator.uniform(0.02, 0.20, 12)
    population = draw_population(12, redshifts, random_generator)
    windows, rejected = build_izc_windows(population, "deep")

    assert rejected["coverage"] == 0
    assert len(windows) > 0
    observed = windows[windows["observed"]]
    first_epoch = observed[observed["epoch"] == 1]
    bands_per_object = first_epoch.groupby("object_id")["band"].nunique()
    assert (bands_per_object == 3).all(), bands_per_object.value_counts().to_dict()


def test_windows_carry_the_izc_gentype_and_the_openuniverse_label():
    pytest.importorskip("sncosmo")
    random_generator = np.random.default_rng(1)
    redshifts = random_generator.uniform(0.05, 0.15, 6)
    population = draw_population(6, redshifts, random_generator)
    windows, _ = build_izc_windows(population, "deep")

    assert len(windows) > 0
    assert (windows["gentype"] > IZC_GENTYPE_OFFSET).all()
    assert windows["object_id"].str.startswith("izc_").all()
    assert (windows["z_CMB"] >= 0.05).all() and (windows["z_CMB"] <= 0.15).all()
