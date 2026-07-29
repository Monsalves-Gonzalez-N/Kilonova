"""Synthetic photometry: the SED -> Roman AB magnitude recipe of kilonova.photometry.spectra.

The SEDs here are analytic, so the expected magnitudes are known in closed form and the tests
pin the physics rather than reproducing whatever the code currently returns.
"""

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck18

from kilonova.photometry.spectra import (
    INTRINSIC_DISTANCE_PARSEC,
    magnitudes_for_bands,
    redshift_and_dim_spectrum,
)

galsim = pytest.importorskip("galsim")  # the bandpass integration needs the Roman throughputs

SPEED_OF_LIGHT_AA_PER_S = const.c.to(u.AA / u.s).value


def luminosity_distance_parsec(redshift):
    if redshift <= 0:
        return INTRINSIC_DISTANCE_PARSEC
    return Planck18.luminosity_distance(redshift).to(u.pc).value


def flat_fnu_spectrum(absolute_magnitude_ab, wavelength_rest_aa):
    """Rest-frame f_lambda of a source that is flat in f_nu at `absolute_magnitude_ab` (10 pc).
    Flat f_nu makes the AB magnitude independent of the bandpass, so every band has the same
    analytic answer."""
    flux_nu = 10 ** (-0.4 * (absolute_magnitude_ab + 48.6))
    return flux_nu * SPEED_OF_LIGHT_AA_PER_S / wavelength_rest_aa**2


@pytest.mark.parametrize("redshift", [0.1, 0.3, 1.0])
def test_redshift_and_dim_conserves_bolometric_flux(redshift):
    """The observed bolometric flux must be L / (4 pi d_L^2). Integrating f_lambda over the
    observed axis stretches it by (1+z), so f_lambda itself has to carry a 1/(1+z): without it
    the integral comes out exactly (1+z) too large."""
    wavelength_rest_aa = np.linspace(1000.0, 20000.0, 500)
    flux_rest_lambda = np.full_like(wavelength_rest_aa, 1e-8)

    wavelength_observed_aa, flux_observed_lambda, distance_parsec = redshift_and_dim_spectrum(
        wavelength_rest_aa, flux_rest_lambda, redshift
    )

    observed_luminosity = np.trapezoid(flux_observed_lambda, wavelength_observed_aa) * distance_parsec**2
    rest_luminosity = np.trapezoid(flux_rest_lambda, wavelength_rest_aa) * INTRINSIC_DISTANCE_PARSEC**2
    assert observed_luminosity == pytest.approx(rest_luminosity, rel=1e-6)


@pytest.mark.parametrize("redshift", [0.0, 0.2, 0.5])
@pytest.mark.parametrize("band", ["R062", "H158"])
def test_flat_fnu_magnitude_matches_analytic_k_correction(redshift, band):
    """End-to-end check against the closed form. For a source flat in f_nu the observed AB
    magnitude is m = M + 5 log10(d_L/10pc) - 2.5 log10(1+z) in every band: the last term is the
    K-correction, and dropping the 1/(1+z) in f_lambda makes the source that much too bright."""
    absolute_magnitude_ab = -16.0
    wavelength_rest_aa = np.geomspace(1000.0, 130000.0, 4000)
    flux_rest_lambda = flat_fnu_spectrum(absolute_magnitude_ab, wavelength_rest_aa)

    magnitudes = magnitudes_for_bands(wavelength_rest_aa, flux_rest_lambda, redshift, bands=[band])

    distance_modulus = 5.0 * np.log10(luminosity_distance_parsec(redshift) / INTRINSIC_DISTANCE_PARSEC)
    expected = absolute_magnitude_ab + distance_modulus - 2.5 * np.log10(1.0 + redshift)
    assert magnitudes[band] == pytest.approx(expected, abs=0.005)


def lanthanide_curtain_spectrum(wavelength_rest_aa, blue_cutoff_aa=9000.0):
    """A red kilonova SED: exactly zero blue of `blue_cutoff_aa`, flat in f_lambda above it.
    Mimics the lanthanide curtain that leaves R062 with no flux at all."""
    return np.where(wavelength_rest_aa > blue_cutoff_aa, 1e-8, 0.0)


def test_band_without_flux_is_an_upper_limit_not_a_gap():
    """Zero flux in a band is a physical value, not missing data: the survey still observes the
    band and gets a non-detection. It must not come back as NaN, which is what makes the pipeline
    drop the band and flag the epoch unobserved."""
    wavelength_rest_aa = np.geomspace(1000.0, 130000.0, 4000)
    flux_rest_lambda = lanthanide_curtain_spectrum(wavelength_rest_aa)

    magnitudes = magnitudes_for_bands(
        wavelength_rest_aa, flux_rest_lambda, redshift=0.05, bands=["R062", "H158"]
    )

    assert np.isposinf(magnitudes["R062"])  # covered but dark: a non-detection, not missing data
    assert np.isfinite(magnitudes["H158"])  # the red band still carries the real photometry


@pytest.mark.parametrize("flux_scale", [1.0, 1e-20, 1e-40])
def test_covered_band_never_returns_nan_however_faint(flux_scale):
    """NaN has to keep meaning "band not covered" alone. Integrating a bandpass the source is dark
    in leaves the quadrature free to return an exact zero or a rounding-sized negative, and the
    magnitude of a negative flux is NaN -- which would silently drop the band as if it had never
    been observed. Scaling the source down walks it through both regimes."""
    wavelength_rest_aa = np.geomspace(1000.0, 130000.0, 4000)
    flux_rest_lambda = lanthanide_curtain_spectrum(wavelength_rest_aa) * flux_scale

    magnitudes = magnitudes_for_bands(
        wavelength_rest_aa, flux_rest_lambda, redshift=0.5, bands=["R062", "Z087", "H158"]
    )

    for band, magnitude in magnitudes.items():
        assert not np.isnan(magnitude), f"{band} came back NaN despite being covered"


def test_isolated_blue_bin_cannot_bridge_the_zero_gap():
    """A single Monte Carlo bin far outside the band must not change that band's magnitude.
    Masking the zeros away used to leave a gap that the linear interpolant spanned with a
    straight line right across the bandpass, inventing flux out of one stray bin."""
    wavelength_rest_aa = np.geomspace(1000.0, 130000.0, 4000)
    flux_rest_lambda = lanthanide_curtain_spectrum(wavelength_rest_aa)
    with_stray_bin = flux_rest_lambda.copy()
    with_stray_bin[np.argmin(np.abs(wavelength_rest_aa - 1360.0))] = 1e-12

    clean = magnitudes_for_bands(wavelength_rest_aa, flux_rest_lambda, 0.05, bands=["R062"])
    contaminated = magnitudes_for_bands(wavelength_rest_aa, with_stray_bin, 0.05, bands=["R062"])

    assert contaminated["R062"] == clean["R062"]
