import numpy as np
import pytest

from kilonova.photometry import roman_noise


def test_read_noise_decreases_with_exposure_toward_floor():
    exposures = np.array([30.0, 60.0, 150.0, 300.0, 1000.0, 10000.0])
    values = roman_noise.read_noise_electrons(exposures)
    assert np.all(np.diff(values) < 0)
    assert np.all(values > np.sqrt(roman_noise.READ_FLOOR_VARIANCE))
    assert roman_noise.read_noise_electrons(1e7) == pytest.approx(5.0, abs=0.01)


def test_bands_observed_at_visit_anchor_always_plus_two():
    bands = ["Z087", "Y106", "J129", "H158", "F184"]
    anchor = "Z087"
    for visit_index in range(6):
        observed = roman_noise.bands_observed_at_visit(visit_index, bands, anchor)
        assert observed[0] == anchor
        assert len(observed) == 3
    # each non-anchor band appears every other visit
    schedule = roman_noise.cadence_schedule(np.arange(0, 30, 5.0), bands, anchor)
    assert len(schedule[anchor]) == 6
    for band in bands[1:]:
        assert len(schedule[band]) == 3


@pytest.mark.parametrize(
    ("tier", "expected_sequences"),
    [("wide", {"RZY", "RJH"}), ("deep", {"ZYJ", "ZHF"})],
)
def test_bands_observed_at_visit_matches_hltds_sequences(tier, expected_sequences):
    """The published HLTDS cadence is one sequence of RZY or RJH (wide) / ZYJ or ZHF (deep) every
    5 days. The counts asserted above do not pin this down: pairing each non-anchor band with the
    next-but-one instead of its neighbour gives the same counts but RZJ/RYH and ZYH/ZJF, i.e. the
    wrong two filters per visit. Assert the sequences themselves."""
    bands = roman_noise.build_tier_constants(tier)["bands"]
    anchor = roman_noise.TIER_ANCHOR_BAND[tier]
    sequences = {
        "".join(band[0] for band in roman_noise.bands_observed_at_visit(visit_index, bands, anchor))
        for visit_index in range(6)
    }
    assert sequences == expected_sequences


def test_epochs_from_first_detection_takes_n_consecutive():
    epoch_times = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    selected = roman_noise.epochs_from_first_detection(epoch_times, 5.0, 4)
    assert selected.tolist() == [5.0, 10.0, 15.0, 20.0]
    # detection after the last epoch -> fewer epochs, never wraps
    assert roman_noise.epochs_from_first_detection(epoch_times, 22.0, 4).tolist() == [25.0]


def test_field_center_for_tier_is_seeded():
    first = roman_noise.field_center_for_tier("deep", np.random.default_rng(0))
    second = roman_noise.field_center_for_tier("deep", np.random.default_rng(0))
    assert first == second
    assert first[0] in roman_noise.HLTDS_FIELDS_BY_TIER["deep"]


def test_source_flux_and_limiting_magnitude_roundtrip():
    pytest.importorskip("galsim")
    zeropoint = roman_noise.roman_zeropoint()["H158"]
    exposure = 420.0
    # a source exactly at the 5-sigma limiting magnitude must have SNR = 5
    flux_error = 250.0
    limiting_magnitude = roman_noise.limiting_magnitude_5sigma(flux_error, exposure, zeropoint)
    flux_at_limit = roman_noise.source_flux_electrons(limiting_magnitude, exposure, zeropoint)
    assert flux_at_limit / flux_error == pytest.approx(5.0, rel=1e-9)


def test_build_tier_constants_structure():
    pytest.importorskip("galsim")
    constants = roman_noise.build_tier_constants("deep")
    assert constants["bands"] == ["Z087", "Y106", "J129", "H158", "F184"]
    assert constants["anchor_band"] == "Z087"
    for band in constants["bands"]:
        assert constants["noise_floor_variance"][band] > 0
        assert 10.0 < constants["zeropoint"][band] < 30.0  # galsim AB zeropoints (1 s, 1 cm2)
