import numpy as np
import pytest

extinction = pytest.importorskip(
    "kilonova.simulation.extinction", reason="needs pyphot/ray/dustmaps (sim extras)"
)


def test_build_redshift_grid_endpoints():
    linear = extinction.build_redshift_grid(0.01, 1.0, 50, "linear")
    assert len(linear) == 50
    assert linear[0] == pytest.approx(0.01)
    assert linear[-1] == pytest.approx(1.0)
    logarithmic = extinction.build_redshift_grid(0.01, 1.0, 50, "log")
    assert np.all(np.diff(np.diff(logarithmic)) > 0)  # log spacing widens


def test_extinction_pools_are_seeded_and_bounded():
    np.random.seed(42)
    av_first = extinction.sample_extinction_av(1000)
    np.random.seed(42)
    av_second = extinction.sample_extinction_av(1000)
    assert np.array_equal(av_first, av_second)
    assert (av_first >= 0).all() and (av_first <= 10.0).all()

    rv = extinction.sample_extinction_rv(1000)
    assert (rv >= 3.0).all() and (rv <= 3.2).all()


def test_wood_vasey_pdf_positive_and_decaying():
    av_grid = np.linspace(0.0, 10.0, 100)
    probability = extinction.wood_vasey_pdf(av_grid)
    assert (probability > 0).all()
    assert probability[-1] < probability[10]


def test_ab_magnitude_to_flux_lambda_monotonic():
    wavelength = np.array([10000.0, 10000.0])
    bright, faint = extinction.ab_magnitude_to_flux_lambda(wavelength, np.array([20.0, 25.0]))
    assert bright > faint > 0


def test_shift_spectrum_to_redshift_stretches_wavelength():
    rest_wavelength = np.linspace(3000.0, 9000.0, 50)
    rest_flux = np.ones(50)
    observed_wavelength, observed_flux = extinction.shift_spectrum_to_redshift(
        rest_wavelength, rest_flux, redshift=1.0
    )
    assert observed_wavelength[0] == pytest.approx(6000.0)
    assert observed_wavelength[-1] == pytest.approx(18000.0)


@pytest.mark.parametrize("redshift", [0.1, 0.5])
def test_observed_spectrum_conserves_bolometric_flux_without_dust(redshift):
    """Same invariant as tests/test_spectra.py for this module's own copy of the recipe: with the
    dust switched off, the observed bolometric flux must be L / (4 pi d_L^2). shift_spectrum_to
    only stretches the wavelength axis, so f_lambda needs an explicit 1/(1+z) on top of the
    geometric dimming."""
    intrinsic_distance_parsec = 10.0
    rest_wavelength = np.linspace(1000.0, 20000.0, 500)
    rest_flux = np.full_like(rest_wavelength, 1e-8)

    result = extinction.generate_observed_kilonova_spectrum(
        rest_wavelength,
        rest_flux,
        extinction_av_host=0.0,
        extinction_rv_host=3.1,
        redshift=redshift,
        ebv_milky_way=0.0,
        intrinsic_distance_parsec=intrinsic_distance_parsec,
    )

    distance_parsec = result["parameters"]["luminosity_distance_parsec"]
    observed_luminosity = (
        np.trapezoid(result["flux_observed"], result["wavelength_observed"]) * distance_parsec**2
    )
    rest_luminosity = np.trapezoid(rest_flux, rest_wavelength) * intrinsic_distance_parsec**2
    assert observed_luminosity == pytest.approx(rest_luminosity, rel=1e-6)
