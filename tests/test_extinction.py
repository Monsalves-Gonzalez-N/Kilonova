import numpy as np
import pytest

extinction = pytest.importorskip("kilonova.simulation.extinction",
                                 reason="needs pyphot/ray/dustmaps (sim extras)")


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
