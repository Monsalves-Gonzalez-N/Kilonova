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
    MODEL_FLUX_FLOOR_MAGNITUDE,
    PREFERRED_IA_MINIMUM_REDSHIFT,
    PREFERRED_IA_SOURCE,
    SN_II_SUBTYPE_FRACTION,
    SOURCES_BY_LABEL,
    UNIFORM_CLASS_SHARE,
    _tde_templates,
    build_izc_windows,
    draw_population,
    iax_shape_parameters,
    iax_source,
    register_sources,
    roman_light_curve,
    sample_iax_absolute_magnitude,
    saturation_magnitude,
    tde_peak_absolute_magnitudes,
    tde_template_count,
)

galsim = pytest.importorskip("galsim")


def test_class_fraction_is_a_distribution():
    assert sum(CLASS_FRACTION.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(fraction > 0 for fraction in CLASS_FRACTION.values())


def test_class_fraction_is_uniform_over_the_openuniverse_classes():
    """Equal share per class the classifier sees, with SN II split by subtype inside its own share.

    Pinned because it is a deliberate departure from OpenUniverse's mix: anyone who "fixes" this
    back to volumetric fractions reintroduces the low-redshift majority class the sample exists to
    avoid handing the model."""
    share_by_class = {}
    for label, fraction in CLASS_FRACTION.items():
        share_by_class.setdefault("SN II" if label in SN_II_SUBTYPE_FRACTION else label, 0.0)
        share_by_class["SN II" if label in SN_II_SUBTYPE_FRACTION else label] += fraction
    assert len(share_by_class) == 6
    for label, share in share_by_class.items():
        assert share == pytest.approx(UNIFORM_CLASS_SHARE, abs=1e-9), (label, share)
    for subtype, fraction in SN_II_SUBTYPE_FRACTION.items():
        assert CLASS_FRACTION[subtype] == pytest.approx(UNIFORM_CLASS_SHARE * fraction, abs=1e-9)


def test_every_generated_class_is_fully_specified():
    """A class that can be drawn must have sources, a luminosity function and a gentype.

    The luminosity function is checked by drawing from it rather than by membership in
    PEAK_ABSOLUTE_MAGNITUDE: SN Iax does not appear there, because Jha & Dai's is a linear law with
    Gaussian rolloffs and no (median, sigma) exists to put in that table."""
    random_generator = np.random.default_rng(0)
    for label in CLASS_FRACTION:
        assert SOURCES_BY_LABEL.get(label), label
        assert label in GENTYPE_BY_LABEL, label
        population = draw_population(8, np.full(8, 0.1), random_generator)
    drawn = {
        realization["label"] for realization in draw_population(400, np.full(400, 0.1), random_generator)
    }
    assert drawn == set(CLASS_FRACTION), drawn.symmetric_difference(CLASS_FRACTION)
    for realization in population:
        assert np.isfinite(realization["peak_absolute_magnitude"]), realization["label"]


def test_iax_luminosity_function_matches_jha_and_dai():
    """The published law spans M_V = -13 to -18 with rolloffs, not a Gaussian.

    `Iax-model.ipynb` prints "fraction brighter than -17.5" for its own 1001-object draw; the
    distribution here has to reproduce that shape, not merely be finite."""
    random_generator = np.random.default_rng(4)
    magnitudes = sample_iax_absolute_magnitude(random_generator, 200000)
    assert -20.0 <= magnitudes.min() and magnitudes.max() <= -11.0
    # Iax-model.ipynb prints 0.2318 for this fraction from a 1001-object draw.
    assert 0.20 < np.mean(magnitudes < -17.5) < 0.27
    assert -17.0 < np.median(magnitudes) < -15.0
    assert np.mean(magnitudes > -14.0) > np.mean(magnitudes < -18.0)


# (rise time, dm15B, dm15R) asked for, and the dm15B/dm15R Iax-model.ipynb reports after warping.
# Row one is the unwarped SN 2005hk SED, whose values the notebook prints as its inputs; it pins
# the base SED and its z/y smoothing, not just the warp.
IAX_NOTEBOOK_ROUND_TRIP = [
    (15.0000, 1.61714, 0.118087, 1.61714, 0.118087),
    (14.3271, 1.74331, 0.903555, 1.74642, 0.903997),
    (8.6317, 2.37500, 0.791852, 2.37445, 0.791813),
    (7.0824, 1.78634, 1.057413, 1.78982, 1.057959),
    (10.8090, 1.62789, 0.916287, 1.63142, 0.916850),
    (18.1049, 1.15467, 0.370067, 1.15796, 0.370559),
]


@pytest.mark.parametrize(
    ("rise_time", "decline_b", "decline_r", "expected_b", "expected_r"), IAX_NOTEBOOK_ROUND_TRIP
)
def test_iax_warp_reproduces_the_published_model(rise_time, decline_b, decline_r, expected_b, expected_r):
    """The warped SED has to land where Jha & Dai's own notebook lands, not merely somewhere.

    This is what makes the izc SN Iax the same model as OpenUniverse's rather than a lookalike, and
    it is the check that would catch a regression in the interp2d replacement (see
    IAX_WARP_ANCHORS_AA) or in the base SED repacking."""
    sncosmo = pytest.importorskip("sncosmo")
    model = sncosmo.Model(source=iax_source(rise_time, decline_b, decline_r))
    measured_b = model.bandmag("bessellb", "vega", 15.0) - model.bandmag("bessellb", "vega", 0.0)
    measured_r = model.bandmag("bessellr", "vega", 15.0) - model.bandmag("bessellr", "vega", 0.0)
    assert measured_b == pytest.approx(expected_b, abs=1e-3)
    assert measured_r == pytest.approx(expected_r, abs=1e-3)


def test_iax_shape_parameters_follow_the_width_luminosity_relation():
    """A faint SN Iax rises faster and declines faster; that is the whole content of the model."""
    random_generator = np.random.default_rng(1)
    bright = np.array([iax_shape_parameters(-18.0, random_generator) for _ in range(400)])
    faint = np.array([iax_shape_parameters(-14.0, random_generator) for _ in range(400)])
    assert bright[:, 0].mean() > faint[:, 0].mean()  # rise time
    assert bright[:, 1].mean() < faint[:, 1].mean()  # dm15(B)
    assert bright[:, 2].mean() < faint[:, 2].mean()  # dm15(R)


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

    register_sources()  # IAX_SOURCE_NAME is registered on demand, not at import
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


def test_a_window_never_carries_a_band_the_sed_does_not_cover():
    """`salt2-extended` has no flux redward of Y106 before rest-frame phase -10.

    Its NIR magnitudes there are ~59, finite and therefore invisible to the isfinite check that
    used to be the only guard; 24 % of the SN Ia of a matched pilot reached the window with H158
    and F184 near 60, a "very red, no near-infrared" shape nothing in the sky produces."""
    pytest.importorskip("sncosmo")
    random_generator = np.random.default_rng(11)
    redshifts = random_generator.uniform(0.02, 0.1, 40)
    population = draw_population(40, redshifts, random_generator)
    windows, _ = build_izc_windows(population, "deep")
    assert len(windows)
    assert windows["mag_true"].max() < MODEL_FLUX_FLOOR_MAGNITUDE


def test_the_first_phase_of_a_type_ia_clears_the_salt2_near_infrared_gap():
    pytest.importorskip("sncosmo")
    realization = {
        "index": 0,
        "label": "SN Ia",
        "source_name": "salt2-extended",
        "peak_absolute_magnitude": -19.404,
        "redshift": 0.05,
        "cadence_parity": 0,
        "salt2_x1": 0.0,
        "salt2_c": 0.0,
    }
    curves = roman_light_curve(realization)
    first_rest_frame_day = curves["F184"][0][0] / 1.05
    assert first_rest_frame_day >= -11.0, first_rest_frame_day


def test_izc_windows_carry_a_resolvable_label():
    """IZC_GENTYPE_OFFSET pushes the gentype out of GENTYPE_LABEL, which used to leave every izc
    window labelled UNKNOWN and made any per-class diagnostic impossible."""
    pytest.importorskip("sncosmo")
    random_generator = np.random.default_rng(12)
    population = draw_population(30, np.full(30, 0.06), random_generator)
    windows, _ = build_izc_windows(population, "deep")
    assert "UNKNOWN" not in set(windows["label"])
    assert set(windows["label"]) <= {"SN Ia", "SN Iax", "SN Ib", "SN Ic", "SN II", "TDE"}
    assert set(windows["izc_subtype"]) <= set(CLASS_FRACTION)


def test_type_ia_use_salt3_nir_wherever_it_covers_f184():
    """`salt3-nir` is OpenUniverse's own SN Ia model and halves a colour offset, but reaches only
    20000 A rest-frame. The switch has to happen exactly where F184 stops fitting inside it, and
    every SN Ia below that redshift still has to be generated -- a missing class in the brightest
    bin would be a worse artefact than the model change."""
    sncosmo = pytest.importorskip("sncosmo")
    from kilonova.photometry.roman_noise import roman_bandpasses

    register_sources()
    red_edge = roman_bandpasses()["F184"].red_limit * 10
    preferred = sncosmo.get_source(PREFERRED_IA_SOURCE)
    assert red_edge / preferred.maxwave() - 1.0 == pytest.approx(PREFERRED_IA_MINIMUM_REDSHIFT, abs=1e-6)
    for fallback in SOURCES_BY_LABEL["SN Ia"]:
        assert sncosmo.get_source(fallback).maxwave() >= red_edge

    random_generator = np.random.default_rng(5)
    redshifts = np.array([0.02, 0.049, 0.05, 0.2])
    sources = {}
    for redshift in redshifts:
        population = draw_population(1, np.array([redshift]), random_generator)
        while population[0]["label"] != "SN Ia":
            population = draw_population(1, np.array([redshift]), random_generator)
        sources[redshift] = population[0]["source_name"]
    assert sources[0.02] in SOURCES_BY_LABEL["SN Ia"]
    assert sources[0.049] in SOURCES_BY_LABEL["SN Ia"]
    assert sources[0.05] == PREFERRED_IA_SOURCE
    assert sources[0.2] == PREFERRED_IA_SOURCE


def test_tde_templates_are_a_population_not_a_prior():
    """MOSFiT's priors are fitting priors: drawn blind they put the peak photosphere between 5e2
    and 1e6 K and the peak luminosity between 1e38 and 1e45 erg/s, most of which is not a TDE. The
    bank is cut to what is observed, and this pins that the cut survived the last rebuild."""
    from astropy import constants

    phase, temperature, radius = _tde_templates()
    peak = int(np.argmin(np.abs(phase)))
    assert len(temperature) > 100, len(temperature)
    assert 1.4e4 < temperature[:, peak].min() and temperature[:, peak].max() < 5.1e4
    luminosity = 4.0 * np.pi * radius[:, peak] ** 2 * constants.sigma_sb.cgs.value * temperature[:, peak] ** 4
    assert 9e42 < luminosity.min() and luminosity.max() < 1.1e45
    # A TDE photosphere is close to isothermal, which is what the observations show. Measured over
    # the phases where the model has a photosphere at all: four templates of the bank touch T = 0 at
    # a single phase, which MODEL_FLUX_FLOOR_MAGNITUDE and `_longest_run` drop before it is used.
    warm = np.where(temperature > 0.0, temperature, np.nan)
    assert np.nanmedian(np.nanmax(warm, axis=1) / np.nanmin(warm, axis=1)) < 2.0


def test_tde_brightness_comes_from_the_model_not_from_a_drawn_magnitude():
    """The bank carries its own luminosity, so `draw_population` must hand `roman_light_curve` the
    magnitude of the template it picked rather than a number from a luminosity function."""
    pytest.importorskip("sncosmo")
    random_generator = np.random.default_rng(3)
    magnitudes = tde_peak_absolute_magnitudes()
    assert len(magnitudes) == tde_template_count()
    assert -23.0 < magnitudes.min() and magnitudes.max() < -15.0
    for _ in range(40):
        population = draw_population(1, np.array([0.1]), random_generator)
        if population[0]["label"] != "TDE":
            continue
        index = population[0]["tde_template_index"]
        assert population[0]["peak_absolute_magnitude"] == pytest.approx(magnitudes[index])
        break
    else:
        pytest.fail("no se sorteo ningun TDE en 40 intentos")
