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
    IZC_GENTYPE_OFFSET,
    MAXIMUM_COLOUR_RATE,
    MODEL_FLUX_FLOOR_MAGNITUDE,
    REFERENCE_ABSOLUTE_MAGNITUDE,
    REST_FRAME_PHASES,
    SN_II_SUBTYPE_FRACTION,
    SOURCES_BY_LABEL,
    UNIFORM_CLASS_SHARE,
    _core_collapse_archive_order,
    _iax_bank,
    _iax_base_sed,
    _sampling_grid,
    _tde_templates,
    apply_brightness_offset,
    build_izc_windows,
    build_model,
    core_collapse_source_by_template_index,
    defective_near_infrared_extension,
    deficit_bin_edges,
    draw_class_population,
    draw_population_from_parents,
    draw_redshifts_from_deficit,
    iax_source,
    iax_template,
    measure_brightness_offset,
    openuniverse_cosmology,
    redshift_deficit,
    register_sources,
    rendered_peak_magnitudes,
    roman_light_curve,
    saturation_magnitude,
    tde_template_count,
)

galsim = pytest.importorskip("galsim")

# --- a parent catalogue with no data in it -------------------------------------------------------
# The generator re-renders OpenUniverse objects, so every test that needs a population needs a
# parent catalogue. The real one is 135 MB of gitignored parquet and its light curves are on an
# external volume, so what these tests build instead is a table with the SCHEMA
# `openuniverse_parents.read_parent_catalog` produces and none of its content: one parent per
# core-collapse template, plus one of each of the other three classes.
#
# The core-collapse template indices are ascending, which is the one property
# `core_collapse_source_by_template_index` reads, and each row carries the gentype the archive's own
# label implies, which is the property it checks.
TEST_BRIGHTNESS_OFFSET = 1.4  # -19.4 + 1.4 = -18.0, a plausible supernova and a detectable one


def synthetic_parent_catalog():
    import pandas as pd

    _, labels = _core_collapse_archive_order()
    rows = []
    for index, label in enumerate(labels):
        rows.append(
            {
                "gentype": GENTYPE_BY_LABEL[label],
                "label": None,  # gentype 32 has none; the template resolves the subtype
                "template_index": 701 + index,
            }
        )
    rows.append({"gentype": 10, "label": "SN Ia", "template_index": 0})
    rows.append({"gentype": 12, "label": "SN Iax", "template_index": 42})
    rows.append({"gentype": 42, "label": "TDE", "template_index": 1})
    for index, row in enumerate(rows):
        row["healpix"] = 10050
        row["id"] = 100000000 + index
        row["parent_key"] = f"snana_10050_{row['id']}"
        row["redshift"] = 0.8
        row["peak_mjd"] = 62000.0
        row["salt2_x1"] = 0.3
        row["salt2_c"] = -0.02
        row["salt2_mB"] = 24.0
        row["host_av"] = 0.4 if row["gentype"] == 12 else np.nan
        row["host_rv"] = 3.1 if row["gentype"] == 12 else np.nan
    return pd.DataFrame(rows)


def draw_population(number, redshifts, random_generator, catalog=None):
    """A population of re-rendered parents with a fixed brightness.

    The brightness is the one thing these tests cannot measure -- it is read off a light curve in a
    16 GB hdf5 -- so it is stamped on instead. `test_the_brightness_offset_round_trips` is where the
    measurement itself is tested."""
    assert number == len(redshifts)
    if catalog is None:
        catalog = synthetic_parent_catalog()
    population = draw_population_from_parents(catalog, redshifts, random_generator)
    for realization in population:
        apply_brightness_offset(realization, TEST_BRIGHTNESS_OFFSET, 0.0, len(ALL_ROMAN_BANDS))
    return population


def test_class_fraction_is_a_distribution():
    assert sum(CLASS_FRACTION.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(fraction > 0 for fraction in CLASS_FRACTION.values())


def test_class_fraction_is_uniform_over_the_openuniverse_classes():
    """Equal share per class the classifier sees, with SN II split by subtype inside its own share.

    Pinned because it is a deliberate departure from OpenUniverse's mix: anyone who "fixes" this
    back to volumetric fractions reintroduces the low-redshift majority class the sample exists to
    avoid handing the model.

    FOUR classes, not six. SN Iax and TDE are the two whose SED this side supplies rather than
    moves, and generating them would make the sample an addition to OpenUniverse's population
    instead of a redistribution of it -- see NO SUBSTITUTIONS in the module docstring."""
    share_by_class = {}
    for label, fraction in CLASS_FRACTION.items():
        share_by_class.setdefault("SN II" if label in SN_II_SUBTYPE_FRACTION else label, 0.0)
        share_by_class["SN II" if label in SN_II_SUBTYPE_FRACTION else label] += fraction
    assert len(share_by_class) == 4
    assert not {"SN Iax", "TDE"} & set(CLASS_FRACTION)
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


def test_the_brightness_offset_round_trips():
    """Measure a brightness off a light curve this module made itself, and get it back.

    The generator's central claim is that the reference magnitude cancels: render a parent's model
    at the parent's redshift with a KNOWN absolute magnitude, hand the peaks of that render to
    `measure_brightness_offset` as if they were the parent's own photometry, and the offset that
    comes back has to be the known magnitude minus the reference, with no band-to-band spread.

    It is the one test of the measurement that needs no external data, and it is what would catch a
    normalisation band, a cosmology or a phase convention leaking into the offset."""
    pytest.importorskip("sncosmo")
    register_sources()
    random_generator = np.random.default_rng(5)
    population = draw_population(12, np.full(12, 0.1), random_generator)
    for realization in population:
        for absolute_magnitude in (-17.0, -19.9):
            parent_peaks = rendered_peak_magnitudes(
                dict(realization, peak_absolute_magnitude=absolute_magnitude),
                realization["parent_redshift"],
            )
            offset, spread, bands = measure_brightness_offset(realization, parent_peaks)
            assert bands >= 1, realization["label"]
            assert offset == pytest.approx(
                absolute_magnitude - REFERENCE_ABSOLUTE_MAGNITUDE, abs=1e-6
            ), realization["label"]
            if bands > 1:
                assert spread == pytest.approx(0.0, abs=1e-6), realization["label"]


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


def test_the_tde_template_is_a_property_of_the_parent():
    """Every re-rendering of one TDE parent has to be the same TDE.

    The parent cannot supply the SED -- OpenUniverse's is AT2019qiz and MOSFiT's bank stands in --
    so the template is drawn, and it is drawn from the parent's own id rather than from the caller's
    generator. Otherwise a parent re-rendered twenty times would be twenty different objects sharing
    one measured brightness and one split group.

    Driven through `draw_class_population` rather than through the sample: TDE is retained but no
    longer in CLASS_FRACTION, so `draw_population` never reaches it and this test would pass
    vacuously. It guards the model, which is still here, for whoever re-enables the class."""
    pytest.importorskip("sncosmo")
    catalog = synthetic_parent_catalog()
    by_parent = {}
    for seed in range(6):
        population = draw_class_population(catalog, "TDE", np.full(40, 0.1), np.random.default_rng(seed))
        for realization in population:
            assert realization["label"] == "TDE"
            assert 0 <= realization["tde_template_index"] < tde_template_count()
            drawn = by_parent.setdefault(realization["parent_key"], realization["tde_template_index"])
            assert drawn == realization["tde_template_index"], realization["parent_key"]
    assert by_parent


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


def test_the_measured_brightness_is_the_only_brightness():
    """No class may carry a luminosity function any more, and the check is by draw and not by name.

    Every realization's `peak_absolute_magnitude` has to be REFERENCE_ABSOLUTE_MAGNITUDE plus the
    offset measured off its parent, for every class alike -- so a class whose brightness came back
    from a Gaussian, a template bank or MOSFiT's own physics would fail here. That is what
    `apply_brightness_offset` is for and what this pins: the number the model is normalised with
    comes from one place."""
    pytest.importorskip("sncosmo")
    random_generator = np.random.default_rng(11)
    population = draw_population(60, np.full(60, 0.1), random_generator)
    assert {one["label"] for one in population} == set(CLASS_FRACTION)
    for realization in population:
        assert realization["peak_absolute_magnitude"] == pytest.approx(
            REFERENCE_ABSOLUTE_MAGNITUDE + realization["brightness_offset"]
        )
        # And it reaches the model: normalising is the last thing `build_model` does.
        model = build_model(realization)
        peak = model.source_peakabsmag("bessellb", "ab", cosmo=openuniverse_cosmology())
        assert peak == pytest.approx(realization["peak_absolute_magnitude"], abs=1e-3)


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


def test_the_host_screen_is_the_parents_own_and_only_sn_iax_has_one():
    """The dust a re-rendered object carries is the dust its parent was given, or none.

    OpenUniverse dusts SN Iax by hand and nothing else -- SN Ia carry theirs inside SALT's `c` and
    the core-collapse classes carry none at all, which is a known bug of the simulation. The
    generator used to reproduce the SN Iax AV distribution by fitting it; now it copies the number,
    and what has to be pinned is that it copies it to the right class and does not invent one."""
    pytest.importorskip("sncosmo")
    catalog = synthetic_parent_catalog()
    for realization in draw_population(120, np.full(120, 0.1), np.random.default_rng(7), catalog):
        parent = catalog[catalog["parent_key"] == realization["parent_key"]].iloc[0]
        assert ("host_av" in realization) == (realization["label"] == "SN Iax"), realization["label"]
        if "host_av" in realization:
            assert realization["host_av"] == pytest.approx(parent["host_av"])
            assert realization["host_rv"] == pytest.approx(parent["host_rv"])
            model = build_model(realization)
            assert model.get("hostebv") == pytest.approx(parent["host_av"] / parent["host_rv"])


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


def test_the_deficit_is_the_hungriest_tier_and_not_the_sum():
    """Two tiers, one object: the sample fills the larger deficit, not their total.

    `build_izc_windows` renders a light curve once and lets each tier observe the bands it observes,
    so one generated object serves deep and wide alike -- and OpenUniverse's own tiers are nearly
    the same transients anyway. Summing the two deficits generated twice the sample that was
    needed, which is 800 000 objects of rendering."""
    edges = np.array([0.0, 0.1, 0.2, 0.3])
    kilonovae = {
        "deep": np.array([0.05, 0.05, 0.05, 0.15, 0.25]),
        "wide": np.array([0.05, 0.15, 0.15, 0.25]),
    }
    contaminants = {"deep": np.array([0.25, 0.25]), "wide": np.array([0.15])}
    _, deficit = redshift_deficit(kilonovae, contaminants, edges)
    # deep asks for (3, 1, 0) and wide for (1, 1, 1): the bin-by-bin maximum, never the sum.
    assert list(deficit) == [3, 1, 1]


def test_a_bin_the_survey_already_covers_asks_for_nothing():
    """The sample is one-directional: it never removes a contaminant and never adds one where
    OpenUniverse already has more than the kilonovae do."""
    edges = np.array([0.0, 0.5, 1.0])
    _, deficit = redshift_deficit({"deep": np.array([0.25])}, {"deep": np.array([0.25, 0.25, 0.75])}, edges)
    assert list(deficit) == [0, 0]


def test_drawn_redshifts_fill_their_own_bin():
    """One redshift per object the deficit asks for, inside the bin that asked for it.

    Uniform inside the bin rather than on the kilonova grid point: 50 spikes of contaminants would
    be a feature of the generation and the classifier reads the redshift."""
    edges = np.array([0.02, 0.1, 0.4])
    deficit = np.array([5, 3])
    redshifts = draw_redshifts_from_deficit(edges, deficit, np.random.default_rng(0))
    assert len(redshifts) == 8
    assert ((redshifts[:5] >= 0.02) & (redshifts[:5] < 0.1)).all()
    assert ((redshifts[5:] >= 0.1) & (redshifts[5:] < 0.4)).all()
    assert len(np.unique(redshifts)) == 8
    # And the scale is a knob on the count, not on the shape.
    assert len(draw_redshifts_from_deficit(edges, deficit, np.random.default_rng(0), scale=0.5)) == 4


def test_the_deficit_bins_are_the_kilonova_grid():
    """One bin per kilonova redshift, since that is what the histogram it is counted against has."""
    edges = deficit_bin_edges()
    grid = np.geomspace(0.01, 1.0, 50)
    assert len(edges) == len(grid) + 1
    for index, redshift in enumerate(grid):
        assert edges[index] < redshift < edges[index + 1], redshift


def test_the_object_id_carries_the_parent_the_split_groups_by():
    """`training/openuniverse_data.py` reads the split group off this string and nothing else.

    Its rule is "drop the izc_ prefix, keep three fields", which has to give back exactly the
    `object_id` the parent itself carries in the OpenUniverse windows -- `snana_{healpix}_{id}` --
    so that a parent and every re-rendering of it land on one side of the split. If the id format
    changes, this fails before a training run silently leaks."""
    pytest.importorskip("sncosmo")
    register_sources()
    population = draw_population(6, np.full(6, 0.08), np.random.default_rng(1))
    windows, _ = build_izc_windows(population, "deep")
    assert len(windows)
    for object_id, parent_key in zip(windows["object_id"], windows["parent_key"], strict=True):
        assert object_id.startswith("izc_")
        assert "_".join(object_id.split("_")[1:4]) == parent_key
        assert parent_key.startswith("snana_")


def test_the_parent_phase_window_is_the_generators_own():
    """Both sides of the brightness measurement span the same rest-frame phases.

    `openuniverse_parents` repeats the limits instead of importing them, so that the module a
    catalogue reader depends on stays free of the generator. This is the seam that keeps."""
    from kilonova.simulation.openuniverse_parents import PEAK_PHASE_LIMITS

    assert PEAK_PHASE_LIMITS == (REST_FRAME_PHASES[0], REST_FRAME_PHASES[-1])


def test_the_parent_peak_is_taken_over_the_phase_window_only():
    """A minimum over the whole light curve is not the same number as a minimum over the window.

    The synthetic object below is brightest 200 rest-frame days after maximum, which is outside
    every phase this module renders; reading that as its peak would make the object 3 magnitudes
    too bright at the redshift it is re-rendered at."""
    from kilonova.simulation.openuniverse_parents import parent_peak_magnitudes

    redshift, peak_mjd = 1.0, 60000.0
    rest_phase = np.array([-40.0, -10.0, 0.0, 30.0, 200.0])
    group = {
        "mjd": peak_mjd + rest_phase * (1.0 + redshift),
        "mag_Y": np.array([26.0, 24.0, 23.5, 24.5, 20.0]),
        "mag_F": np.array([26.0, np.nan, 99.0, 24.9, 20.0]),
    }
    peaks = parent_peak_magnitudes(group, redshift, peak_mjd, ["Y106", "F184"])
    assert peaks["Y106"] == pytest.approx(23.5)
    # 99 is SNANA's "no flux" and NaN is no model: neither is a magnitude, so F184's peak is the
    # only real value inside the window.
    assert peaks["F184"] == pytest.approx(24.9)


def test_the_template_mapping_is_checked_against_the_catalogue():
    """The archive records template NAMES and the catalogue records template INDICES; the mapping
    between them is recovered from the order both are in, and then verified object by object.

    A catalogue whose gentypes disagree with the archive's labels has to raise rather than quietly
    re-render SN Ib as SN Ic -- the order is the only thing that ties the two files together."""
    catalog = synthetic_parent_catalog()
    mapping = core_collapse_source_by_template_index(catalog)
    _, labels = _core_collapse_archive_order()
    assert len(mapping) == len(labels)
    for template_index, (source_name, label) in mapping.items():
        assert source_name in SOURCES_BY_LABEL[label], (template_index, source_name)

    core_collapse = catalog[catalog["gentype"].isin((21, 26, 32))]
    scrambled = catalog.copy()
    first = core_collapse.index[0]
    scrambled.loc[first, "gentype"] = 21 if catalog.loc[first, "gentype"] != 21 else 26
    with pytest.raises(ValueError, match="but its objects carry gentype"):
        core_collapse_source_by_template_index(scrambled)


def test_the_parent_catalogue_reads_what_the_re_render_inherits(tmp_path):
    """The reader over a catalogue with the release's own schema: parameters out of the two ragged
    columns, the group key, and the AV = -9 sentinel.

    That sentinel is the trap. OpenUniverse writes "no host screen" as AV = -9, which is nine
    magnitudes of dust if read literally, and every core-collapse object and every SN Ia in the
    release carries it."""
    import pandas as pd

    from kilonova.simulation import openuniverse_parents

    catalog = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            # 42 is a TDE, which PARENT_GENTYPES no longer lists: the reader has to drop it.
            "gentype": [21, 32, 10, 42],
            "z_CMB": [0.5, 0.8, 1.2, 0.3],
            "peak_mjd": [60000.0, 60100.0, 60200.0, 60300.0],
            "AV": [0.44, -9.0, -9.0, -9.0],
            "RV": [3.1, -9.0, -9.0, -9.0],
            "model_param_names": [
                ["template_index"],
                ["template_index"],
                ["template_index", "salt2_x0", "salt2_x1", "salt2_c", "salt2_mB"],
                ["template_index"],
            ],
            "model_param_values": [[7.0], [701.0], [0.0, 1e-5, 0.7, -0.03, 24.5], [1.0]],
        }
    )
    catalog.to_parquet(tmp_path / "snana_10050.parquet", index=False)

    parents = openuniverse_parents.read_parent_catalog(tmp_path)
    # The TDE is gone, and with it every class whose SED this side would have to supply.
    assert list(parents["parent_key"]) == ["snana_10050_1", "snana_10050_2", "snana_10050_3"]
    assert list(parents["template_index"]) == [7, 701, 0]
    # gentype 32 has no label of its own: OpenUniverse pools SN IIP and SN IIL into it and only the
    # template says which one an object is.
    assert list(parents["label"][[0, 2]]) == ["SN Ib", "SN Ia"]
    assert parents["label"].isna()[1]
    assert parents.loc[0, "host_av"] == pytest.approx(0.44)
    assert parents.loc[0, "host_rv"] == pytest.approx(3.1)
    assert not np.isfinite(parents.loc[1, "host_av"])
    assert not np.isfinite(parents.loc[2, "host_av"])
    assert parents.loc[2, "salt2_x1"] == pytest.approx(0.7)
    assert parents.loc[2, "salt2_mB"] == pytest.approx(24.5)
    assert not np.isfinite(parents.loc[0, "salt2_x1"])  # a core-collapse row has no SALT parameters
