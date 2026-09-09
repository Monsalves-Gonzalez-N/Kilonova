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
    IA_BASE_SOURCE_NAME,
    IA_PAD_WAVELENGTH,
    IA_SOURCE_NAME,
    IAX_BANK_SIZE,
    IAX_HOST_AV_RANGE,
    IAX_HOST_RV,
    IZC_GENTYPE_OFFSET,
    MAXIMUM_COLOUR_RATE,
    MODEL_FLUX_FLOOR_MAGNITUDE,
    PEAK_ABSOLUTE_MAGNITUDE,
    REST_FRAME_PHASES,
    SN_II_SUBTYPE_FRACTION,
    SOURCES_BY_LABEL,
    UNIFORM_CLASS_SHARE,
    _iax_bank,
    _iax_base_sed,
    _sampling_grid,
    _tde_templates,
    build_izc_windows,
    build_model,
    defective_near_infrared_extension,
    draw_population,
    iax_phase_zero_offset,
    iax_source,
    iax_template,
    openuniverse_cosmology,
    register_sources,
    roman_light_curve,
    sample_iax_host_av,
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
    PEAK_ABSOLUTE_MAGNITUDE: neither SN Iax nor TDE appears there, because neither draws a
    magnitude -- one reads it off the template bank and the other off MOSFiT's physics."""
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


def test_iax_bank_replays_the_published_templates():
    """The bank is the notebook's own, not a resample of the same laws.

    `Iax-model.ipynb` seeds `np.random.seed(4)` and prints "fraction brighter than -17.5: 0.2318"
    for its 1001-object draw. Reproducing that number to the fourth decimal is what says the replay
    is on the notebook's random stream -- the luminosity function alone would only reproduce it to
    the sampling error of 1001 draws, which is 1.3 %."""
    absolute_v, rise_time, decline_b, decline_r = _iax_bank()
    assert len(absolute_v) == IAX_BANK_SIZE == 919
    # Over the notebook's full 1001 draws; the bank keeps the 919 OpenUniverse shipped.
    assert -20.0 <= absolute_v.min() and absolute_v.max() <= -11.0
    assert np.median(absolute_v) == pytest.approx(-15.9815, abs=1e-3)
    assert rise_time.min() > 0.0
    assert (decline_r >= 0.0).all()
    # The width-luminosity relations, which are the whole content of the model: a faint SN Iax
    # rises faster and declines faster.
    faint = absolute_v > -15.0
    bright = absolute_v < -17.0
    assert rise_time[bright].mean() > rise_time[faint].mean()
    assert decline_b[bright].mean() < decline_b[faint].mean()
    assert decline_r[bright].mean() < decline_r[faint].mean()


def test_iax_template_index_matches_openuniverse():
    """Row 0 of the replay is `template_index` 1 of the OpenUniverse catalogue.

    Verified against the catalogue itself rather than asserted: the dust-corrected peak absolute
    LSST-g magnitude of the 1923 OpenUniverse SNe Iax below z = 0.45 tracks the replayed M_V with
    slope +0.990 and correlation +0.983, where the same bank shuffled gives -0.022. These two rows
    are the endpoints of that check."""
    assert iax_template(0)[0] == pytest.approx(-12.8081, abs=1e-3)
    assert iax_template(918)[0] == pytest.approx(-13.7112, abs=1e-3)


def test_iax_is_normalised_at_phase_zero_not_at_peak():
    """The notebook sets V(phase 0) = M_V; sncosmo's own helper would set V(peak) = M_V.

    The two differ by 0.075 mag for the base SED and by up to 0.18 over the bank, gray in every
    band, and this module carried that offset until the convention was measured off cell 14."""
    sncosmo = pytest.importorskip("sncosmo")
    from astropy.cosmology import Planck18

    absolute_v, rise_time, decline_b, decline_r = _iax_bank()
    for row in (0, 400, 918):
        source = iax_source(rise_time[row], decline_b[row], decline_r[row])
        offset = iax_phase_zero_offset(source)
        assert -0.30 < offset < 0.0, row
        model = sncosmo.Model(source=source)
        model.set(z=0.1)
        model.set_source_peakabsmag(absolute_v[row] + offset, "bessellv", "vega", cosmo=Planck18)
        # What the convention claims: rest-frame V at phase zero is the bank's own M_V.
        phase_zero = model.source.bandmag("bessellv", "vega", 0.0) - Planck18.distmod(0.1).value
        assert phase_zero == pytest.approx(absolute_v[row], abs=1e-3), row


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


def test_type_ia_use_one_salt_source_at_every_redshift():
    """The class used to carry two spectral models split at z = 0.05, because `salt3-nir` stops
    1000 A short of the F184 red edge. Padding it removed the split; this pins that it stays
    removed, and that the pad is long enough to cover F184 at every redshift, z = 0 included."""
    sncosmo = pytest.importorskip("sncosmo")
    from kilonova.photometry.roman_noise import roman_bandpasses

    register_sources()
    red_edge = roman_bandpasses()["F184"].red_limit * 10
    assert IA_PAD_WAVELENGTH >= red_edge
    assert SOURCES_BY_LABEL["SN Ia"] == [IA_SOURCE_NAME]
    assert sncosmo.get_source(IA_SOURCE_NAME).maxwave() >= red_edge

    random_generator = np.random.default_rng(5)
    for redshift in (0.02, 0.049, 0.05, 0.2):
        population = draw_population(1, np.array([redshift]), random_generator)
        while population[0]["label"] != "SN Ia":
            population = draw_population(1, np.array([redshift]), random_generator)
        assert population[0]["source_name"] == IA_SOURCE_NAME, redshift


def test_padded_ia_source_matches_the_base_below_the_pad():
    """The pad is an extrapolation and has to stay confined to the 1000 A it was added for.

    Below 20000 A the padded source has to BE `salt3-nir` -- not approximately, exactly -- because
    that is what makes the pad a statement about the sliver of F184 nothing measures rather than a
    change to the SN Ia model. Above it, held flat at the last defined value."""
    sncosmo = pytest.importorskip("sncosmo")

    register_sources()
    base = sncosmo.get_source(IA_BASE_SOURCE_NAME)
    padded = sncosmo.get_source(IA_SOURCE_NAME)
    assert padded.maxwave() == IA_PAD_WAVELENGTH
    assert (padded.minwave(), padded.minphase(), padded.maxphase()) == (
        base.minwave(),
        base.minphase(),
        base.maxphase(),
    )

    phases = np.arange(base.minphase(), base.maxphase(), 3.0)
    inside = np.arange(base.minwave(), base.maxwave() + 1.0, 37.0)
    assert padded.flux(phases, inside) == pytest.approx(base.flux(phases, inside), rel=0.0, abs=0.0)

    edge = base.flux(phases, np.array([base.maxwave()]))
    for wavelength in (base.maxwave() + 250.0, IA_PAD_WAVELENGTH):
        assert padded.flux(phases, np.array([wavelength])) == pytest.approx(edge, rel=1e-12)


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


def test_the_drawn_cadence_parity_reaches_the_window():
    """`cadence_parity` was drawn, stored, passed and then discarded.

    `build_window_from_model` honours `visit_index_offset` only when it is handed the visit grid;
    the branch that derives the grid from the model's own range is the OpenUniverse one, where the
    parity is not free. Taking that branch pinned every izc object to an even first visit -- the
    first epoch carried (Z087, Y106, J129) 80 % of the time -- which is a cadence phase correlated
    with nothing but the class of the object, in the sample built to remove such correlations."""
    pytest.importorskip("sncosmo")
    random_generator = np.random.default_rng(2)
    # Enough objects that the threshold below is a statistic rather than a coin flip: at 24 the
    # exception described above is one object and the fraction moves by 0.04 per object.
    population = draw_population(96, random_generator.uniform(0.02, 0.3, 96), random_generator)
    band_sets = {}
    for parity in (0, 1):
        for realization in population:
            realization["cadence_parity"] = parity
        windows, _ = build_izc_windows(population, "deep")
        first_epoch = windows[(windows["epoch"] == 1) & windows["observed"]]
        band_sets[parity] = first_epoch.groupby("object_id")["band"].apply(lambda bands: tuple(sorted(bands)))
    # Not every object, and the exception is not a leak: the window starts at the FIRST DETECTION,
    # not at the first visit, so an object too faint to be detected at the visit its parity picked
    # starts a visit later -- which flips the parity back and lands it on the same band set under
    # both. That can only happen at the detection threshold, and the bug this guards against gave
    # zero flips out of every object rather than one exception out of a couple of dozen.
    flipped = (band_sets[0] != band_sets[1]).mean()
    assert flipped >= 0.9, flipped
    assert set(band_sets[0]) == set(band_sets[1])  # the same two sets, swapped, not new ones


def test_the_first_epoch_is_not_pinned_to_one_band_set():
    """The consequence of the fix above, measured the way the bias was measured."""
    pytest.importorskip("sncosmo")
    random_generator = np.random.default_rng(23)
    population = draw_population(120, random_generator.uniform(0.02, 0.4, 120), random_generator)
    windows, _ = build_izc_windows(population, "deep")
    first_epoch = windows[(windows["epoch"] == 1) & windows["observed"]]
    sets = first_epoch.groupby("object_id")["band"].apply(lambda bands: tuple(sorted(bands)))
    share = sets.value_counts(normalize=True)
    assert len(share) == 2, share.to_dict()
    assert share.max() < 0.65, share.to_dict()


def test_the_visit_grid_does_not_start_at_the_same_phase_every_time():
    """The other degree of freedom: the delay from the model's first phase to the first visit.

    In the sky the visit grid is fixed in absolute time and the explosion is not, so this is
    uniform over the 5-day interval between visits. Left at zero it would make the phase of the
    cadence a function of the redshift and of the template library, which is a function of class."""
    random_generator = np.random.default_rng(31)
    population = draw_population(200, np.full(200, 0.1), random_generator)
    offsets = np.array([realization["visit_phase_offset_days"] for realization in population])
    assert offsets.min() >= 0.0 and offsets.max() < 5.0
    assert np.percentile(offsets, 90) - np.percentile(offsets, 10) > 3.0


def test_the_luminosity_functions_carry_the_y106_calibration():
    """The medians are anchored to OpenUniverse's own M(Y106), not to a rest-frame B measurement.

    Normalising in rest-frame B fixes the brightness in B and leaves Y106 to each template's own
    B - Y colour, which is a property of the library: a class-correlated brightness offset inside
    the band the classifier reads. `scripts/calibrate_izc_brightness.py` measures it against
    OpenUniverse's own objects and the rule is that an offset below two standard errors of the
    median is noise and is not applied.

    The offsets below are the ones measured against OPENUNIVERSE'S OWN TEMPLATES, which is a
    different calibration from the one this test used to hold: with the SNANA/Nugent substitution
    the chain was -17.18 - 0.142 - 0.119 for SN Ib and -17.36 - 0.478 for SN Ic. Those templates
    are gone, so their calibration is gone with them, and what is left is one offset per class
    measured in one run.

    The comparison itself needs `data/openuniverse/early_windows_deep.parquet` and does not belong
    in a unit test; what is checked here is that its result survived. First the two offsets that
    were applied, which is what a well-meaning edit back to a literature luminosity function would
    silently undo, and then the brightness that comes out the far end -- generated at a fixed
    z = 0.1, where nothing is lost to the detection cut, so the number is the model's own and not a
    selection on it."""
    pytest.importorskip("sncosmo")

    # The distance modulus has to be the one the windows were built with, or the test measures the
    # difference between two cosmologies: 0.066 mag at z = 0.1 between Planck18 and OpenUniverse's.
    cosmology = openuniverse_cosmology()
    assert PEAK_ABSOLUTE_MAGNITUDE["SN Ib"][0] == pytest.approx(-17.441 + 0.275, abs=1e-6)
    assert PEAK_ABSOLUTE_MAGNITUDE["SN Ic"][0] == pytest.approx(-17.838 + 0.336, abs=1e-6)
    assert PEAK_ABSOLUTE_MAGNITUDE["SN IIP"][0] == pytest.approx(-16.872 + 0.084, abs=1e-6)
    assert PEAK_ABSOLUTE_MAGNITUDE["SN IIL"][0] == pytest.approx(-18.052 + 0.084, abs=1e-6)

    calibrated_median_y106 = {"SN Ic": -18.04, "SN Ib": -17.32, "SN IIP": -16.80}
    random_generator = np.random.default_rng(41)
    for position, (label, reference) in enumerate(calibrated_median_y106.items()):
        # The overwritten parameters get their OWN generator, so this median does not depend on how
        # many values `draw_population` happens to consume per object. It used to: adding one draw
        # inside it -- the SN Iax host AV -- moved the SN Ib median by three times the 0.079 mag
        # spread this statistic has over independent seeds, and failed a test about a calibration
        # that had not changed.
        parameter_generator = np.random.default_rng([41, position])
        population = draw_population(200, np.full(200, 0.1), random_generator)
        for index, realization in enumerate(population):
            realization["label"] = label
            realization["source_name"] = str(parameter_generator.choice(SOURCES_BY_LABEL[label]))
            realization["peak_absolute_magnitude"] = float(
                parameter_generator.normal(*PEAK_ABSOLUTE_MAGNITUDE[label])
            )
            for key in ("salt2_x1", "salt2_c", "tde_template_index", "host_av", "host_rv"):
                realization.pop(key, None)
            realization["index"] = index
        windows, _ = build_izc_windows(population, "deep")
        y106 = windows[(windows["band"] == "Y106") & np.isfinite(windows["mag_true"])]
        peak = y106.groupby(["object_id", "z_CMB"])["mag_true"].min().reset_index()
        absolute = peak["mag_true"].to_numpy() - cosmology.distmod(peak["z_CMB"].to_numpy()).value
        assert np.median(absolute) == pytest.approx(reference, abs=0.25), (label, np.median(absolute))


def test_the_iax_pre_explosion_region_is_suppressed_where_the_notebook_suppresses_it():
    """`flux[stretched < -rise_time] /= 2000` needs the notebook's grid to select anything.

    On the repacked file's own 81 phases from -15 the stretched grid starts at exactly -rise_time,
    the strict comparison fired on nothing, and the pre-explosion row reached the light curve as a
    plateau. On IAX_NOTEBOOK_PHASES it starts at -2 * rise_time and the suppression covers the real
    region between there and -rise_time, which is what cell 14 does."""
    pytest.importorskip("sncosmo")
    _, base_wavelength, base_flux = _iax_base_sed()
    for rise_time in (7.0, 15.0, 22.0):
        source = iax_source(rise_time, 1.6, 0.5)
        assert source.minphase() == pytest.approx(-2.0 * rise_time)
        suppressed = source.flux(-1.5 * rise_time, base_wavelength)
        kept = source.flux(-rise_time, base_wavelength)
        assert suppressed.max() < kept.max() / 100.0, rise_time


def test_no_window_carries_a_colour_the_model_cannot_produce():
    """The near-infrared shoulder MODEL_FLUX_FLOOR_MAGNITUDE cannot see.

    `salt2-extended` hands out rows like J129 = 24.38 next to Z087 = 17.59, then J129 = 21.88, then
    J129 = 17.93: every value is a legitimate magnitude and no absolute threshold separates them
    from a faint object. What no transient does is move a colour by 4 mag in a day."""
    pytest.importorskip("sncosmo")
    random_generator = np.random.default_rng(13)
    population = draw_population(40, random_generator.uniform(0.02, 0.4, 40), random_generator)
    for realization in population:
        curves = roman_light_curve(realization)
        if not curves:
            continue
        days = curves[ALL_ROMAN_BANDS[0]][0]
        magnitudes = np.array([curves[band][1] for band in ALL_ROMAN_BANDS])
        colours = magnitudes - magnitudes.min(axis=0)[None, :]
        rest_frame_step = np.diff(days) / (1.0 + realization["redshift"])
        rate = np.abs(np.diff(colours, axis=1)).max(axis=0) / rest_frame_step
        assert rate.max() <= MAXIMUM_COLOUR_RATE, (realization["label"], realization["source_name"])


def test_the_sampling_grid_reaches_the_red_edge():
    """`np.arange` is half open, and the coverage check is read off the sampled grid.

    One missing step at the red end made every band the model covers exactly to its own edge come
    back NaN. The unpadded `salt3-nir` was that case at z = 0.05."""
    grid = _sampling_grid(4000.0, 21000.0, 10.0)
    assert grid[0] == 4000.0 and grid[-1] == 21000.0
    ragged = _sampling_grid(4000.0, 21003.0, 10.0)
    assert ragged[-1] == 21003.0 and ragged[-2] == 21000.0


def test_a_type_ia_at_the_bottom_of_the_range_still_produces_a_curve():
    """The pad exists so that F184 survives at the reddest rest-frame the module ever samples.

    z = 0.02 is the bottom of the generated range and the case the unpadded source failed; 0.05 is
    where it used to become sufficient on its own and is kept as the other side of the old split."""
    pytest.importorskip("sncosmo")
    for redshift in (0.02, 0.05):
        realization = {
            "index": 0,
            "label": "SN Ia",
            "source_name": IA_SOURCE_NAME,
            "peak_absolute_magnitude": -19.404,
            "redshift": redshift,
            "cadence_parity": 0,
            "visit_phase_offset_days": 0.0,
            "salt2_x1": 0.0,
            "salt2_c": 0.0,
        }
        curves = roman_light_curve(realization)
        assert set(curves) == set(ALL_ROMAN_BANDS), (redshift, sorted(curves))


def test_phases_are_measured_from_maximum_not_from_the_source_phase_zero():
    """The Nugent templates put phase zero at the explosion and maximum 11 to 17 d later; the SNANA
    and SALT ones put zero at maximum. Sampled on the source's own phases, the classes that only
    have Nugent templates would carry 20 fewer days of light curve after maximum, for no reason but
    the convention of the file they were read from."""
    pytest.importorskip("sncosmo")
    realization = {
        "index": 0,
        "label": "SN IIL",
        "source_name": "nugent-sn2l",
        "peak_absolute_magnitude": -18.05,
        "redshift": 0.05,
        "cadence_parity": 0,
        "visit_phase_offset_days": 0.0,
    }
    curves = roman_light_curve(realization)
    days = curves["Y106"][0] / 1.05
    magnitudes = curves["Y106"][1]
    assert days.max() > REST_FRAME_PHASES.max() - 2.0, days.max()
    assert abs(days[magnitudes.argmin()]) < 12.0, days[magnitudes.argmin()]


def test_the_tde_bank_is_cached_whole():
    """A window of 24 over 227 uniformly drawn templates measured no hits at all."""
    from kilonova.simulation.intermediate_z_contaminants import tde_source

    assert tde_source.cache_parameters()["maxsize"] is None
    assert tde_template_count() == 227


def test_several_tiers_give_what_one_tier_at_a_time_gives():
    """Asking for both tiers at once must be an optimisation, not a change of result.

    The light curve is the same object in every Roman band and which of them a tier observes is
    decided afterwards, so one call serves both; calling once per tier does the expensive half
    twice, and the tiers overlap almost completely (717 863 of the 717 864 wide contaminants of
    OpenUniverse are deep ones too)."""
    pytest.importorskip("sncosmo")
    random_generator = np.random.default_rng(17)
    population = draw_population(25, random_generator.uniform(0.02, 0.4, 25), random_generator)
    together = build_izc_windows(population, ["deep", "wide"])
    for tier in ("deep", "wide"):
        alone_windows, alone_rejected = build_izc_windows(population, tier)
        shared_windows, shared_rejected = together[tier]
        assert shared_rejected == alone_rejected, tier
        assert len(shared_windows) == len(alone_windows), tier
        if len(alone_windows):
            assert np.allclose(shared_windows["mag_true"], alone_windows["mag_true"], equal_nan=True)


def test_iax_host_extinction_reproduces_openuniverse():
    """SN Iax is the one class OpenUniverse dusts by hand and the one class this module dusts.

    The percentiles are OpenUniverse's own, measured over the 115 645 SNe Iax of the 33 healpix
    catalogues. They are pinned here rather than recomputed because the catalogues are not in the
    repository, and they are what the three parameters of the AV law were fitted to: a drift in any
    of them means the law no longer describes the population it was taken from."""
    pytest.importorskip("sncosmo")

    openuniverse = {
        1: 0.009,
        5: 0.039,
        16: 0.126,
        25: 0.202,
        50: 0.440,
        75: 0.801,
        84: 1.027,
        95: 1.752,
        99: 2.608,
    }
    drawn = sample_iax_host_av(np.random.default_rng(0), 200_000)
    assert drawn.min() >= IAX_HOST_AV_RANGE[0] and drawn.max() <= IAX_HOST_AV_RANGE[1]
    for percentile, expected in openuniverse.items():
        assert np.percentile(drawn, percentile) == pytest.approx(expected, abs=0.02), percentile
    assert drawn.mean() == pytest.approx(0.5918, abs=0.01)

    # The screen has to DIM what leaves the model, which it only does if the drawn absolute
    # magnitude is set on the bare source; normalising through the dust would undo it exactly.
    register_sources()
    random_generator = np.random.default_rng(3)
    population = [
        one for one in draw_population(600, np.full(600, 0.05), random_generator) if one["label"] == "SN Iax"
    ][:5]
    assert population, "no SN Iax drawn"
    for realization in population:
        assert realization["host_rv"] == IAX_HOST_RV
        bare = {key: value for key, value in realization.items() if key not in ("host_av", "host_rv")}
        dimming = build_model(realization).bandmag("bessellv", "ab", 0.0) - build_model(bare).bandmag(
            "bessellv", "ab", 0.0
        )
        # Band-integrated rather than monochromatic at 5500 A, so a few per cent above AV itself.
        assert dimming == pytest.approx(realization["host_av"], rel=0.12)

    # And no other class gets one.
    for realization in draw_population(400, np.full(400, 0.1), np.random.default_rng(7)):
        assert ("host_av" in realization) == (realization["label"] == "SN Iax"), realization["label"]


def test_openuniverse_templates_have_no_zero_flux_gap():
    """The audit's SOUND test, over the templates OpenUniverse itself used.

    `defective_near_infrared_extension` runs two tests and they are not equally good. This pins
    which is which, and it is settled by measurement rather than by argument now that the templates
    are OpenUniverse's own rather than our reconstruction of them:

      * THE ZERO-FLUX GAP DISCRIMINATES. A partial run of exact zeros inside 9000-21000 A is not
        physics under any model. Every SNANA template carried one, at 42 to 84 of the 91 sampled
        phases; the `snsedextend` reconstruction carried 1559 of them across 35 sources, and its
        F184 - Y106 colour was 0.85 mag from OpenUniverse's and swung 2 mag across the sample's
        redshift range. These templates carry ZERO, and their colour matches OpenUniverse's to
        0.004 mag in SN II. That is the whole discrimination, and this test holds it.

      * F184 BRIGHTER THAN H158 DOES NOT. It was always documented as a heuristic -- it assumes a
        smooth declining continuum, and `salt3-nir-f184` violates it at 52 phases because a SN Ia
        HAS a secondary near-infrared maximum. It is now known to be worse than that: 35 of
        OpenUniverse's own 44 core-collapse templates violate it, and those templates are the
        reference this sample is measured against. A criterion the reference fails is not a defect
        criterion, so it is deliberately NOT asserted here.
    """
    pytest.importorskip("sncosmo")
    register_sources()
    audited = 0
    for label, source_names in SOURCES_BY_LABEL.items():
        if label in ("SN Iax", "TDE", "SN Ia"):
            # None is an extended library template: SN Iax is warped per object, TDE is a blackbody
            # over a MOSFiT photosphere, and SALT3 is parametric.
            continue
        for source_name in source_names:
            gaps = [
                reason
                for reason in defective_near_infrared_extension(source_name)
                if "zero-flux gap" in reason
            ]
            assert not gaps, (label, source_name, len(gaps), gaps[:3])
            audited += 1
    assert audited == 44


def test_openuniverse_drew_no_hypernova_and_no_sn_iin_or_iib():
    """What OpenUniverse actually drew, which is narrower than the library it drew from.

    Read off `template_index` across 979 557 core-collapse objects: exactly 44 templates appear,
    17 SN IIP, 7 SN IIL, 13 SN Ib and 7 SN Ic. Three consequences are pinned here because each one
    used to be an open decision or a documented worry:

      * NO SN IIn AND NO SN IIb. The V19 library carries templates of both. Generating either would
        put a class in the sample that the population it stands in for does not contain, and it is
        also what settles the question of adding SN IIb rather than leaving it to judgement.
      * NO SN 1998bw. The hypernova template is in the library and OpenUniverse did not draw it, so
        the cross-class contamination this module used to guard against -- SN 1998bw supplying 22 %
        of the SN Ib draws when it was listed under both classes -- cannot arise here at all.
      * The SN II subtype split is 17/24 and 7/24, MEASURED, where it used to be the 0.70/0.15/0.15
        of Li et al. (2011) standing in for exactly this.
    """
    assert set(SOURCES_BY_LABEL) == {"SN IIP", "SN IIL", "SN Ib", "SN Ic", "SN Ia", "SN Iax", "TDE"}
    assert len(SOURCES_BY_LABEL["SN IIP"]) == 17
    assert len(SOURCES_BY_LABEL["SN IIL"]) == 7
    assert len(SOURCES_BY_LABEL["SN Ib"]) == 13
    assert len(SOURCES_BY_LABEL["SN Ic"]) == 7
    every = sum(SOURCES_BY_LABEL.values(), [])
    assert not [name for name in every if "1998bw" in name]
    assert SN_II_SUBTYPE_FRACTION == {"SN IIP": 17.0 / 24.0, "SN IIL": 7.0 / 24.0}


def test_the_pruned_templates_are_still_defective():
    """The templates the audit removed fail it, so the cut is reproducible and not a preference.

    Named explicitly because the cost was severe -- every SNANA template goes, leaving four
    distinct SEDs for the whole core-collapse half of the sample --
    and a future reader is owed the ability to re-run the exact decision.
    """
    pytest.importorskip("sncosmo")
    register_sources()
    removed = [
        "snana-2004hx",
        "snana-2005gi",
        "snana-2006gq",
        "snana-2006iw",
        "snana-2006jl",
        "snana-2006kn",
        "snana-2006kv",
        "snana-2007iz",
        "snana-2007kw",
        "snana-2007ky",
        "snana-2007lb",
        "snana-2007ld",
        "snana-2007lj",
        "snana-2007ll",
        "snana-2007lx",
        "snana-2007lz",
        "snana-2007md",
        "snana-2007ms",
        "snana-2007nr",
        "snana-2007nv",
        "snana-2007nw",
        "snana-2007pg",
        "snana-2004gv",
        "snana-2004ib",
        "snana-2005hm",
        "snana-2006ep",
        "snana-2006jo",
        "snana-2007nc",
        "snana-2007y",
        "snana-04d1la",
        "snana-04d4jv",
        "snana-2004fe",
        "snana-2004gq",
        "snana-2006fo",
        "snana-2006lc",
        "snana-sdss004012",
        "snana-sdss014475",
    ]
    for source_name in removed:
        assert defective_near_infrared_extension(source_name), source_name
        assert all(source_name not in names for names in SOURCES_BY_LABEL.values()), source_name


def test_sn_iil_and_sn_iip_nugent_templates_are_the_same_sed():
    """Pins the finding that `nugent-sn2l` and `nugent-sn2p` are one SED under two names.

    Not a defect to fix -- it is what the library ships -- and it is why substituting the Nugent
    sources for the missing V19 extension was never satisfying: while they were the only SN IIP and
    SN IIL sources, NO colour whatsoever separated those two classes, which differed in their light
    curve and their luminosity function and in nothing else. Kept as the provenance of that, and
    neither template is drawn any more. Checked across phases rather than
    at maximum alone, because the light curves genuinely do differ -- at +80 d the SN IIL has
    declined 4.41 mag against the SN IIP's 2.15 -- and only the spectral shape is shared.
    """
    sncosmo = pytest.importorskip("sncosmo")
    import numpy

    iip = sncosmo.get_source("nugent-sn2p")
    iil = sncosmo.get_source("nugent-sn2l")
    wavelength = numpy.arange(3000.0, 25000.0, 20.0)
    for offset in [0.0, 10.0, 20.0, 40.0, 60.0]:
        iip_flux = iip.flux(iip.peakphase("bessellb") + offset, wavelength)
        iil_flux = iil.flux(iil.peakphase("bessellb") + offset, wavelength)
        ratio = iil_flux / numpy.where(iip_flux > 0, iip_flux, numpy.nan)
        assert numpy.nanstd(ratio) / numpy.nanmean(ratio) < 0.05, offset
