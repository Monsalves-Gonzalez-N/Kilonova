"""Kilonova synthetic photometry — single source of truth for the SED -> Roman AB magnitude step.

OpenUniverse transients already carry true per-band magnitudes in the snana hdf5, so they need no
synthetic photometry. The LANL kilonova grid only stores rest-frame spectra (flux_rest), so the KN
path must redshift + dim the spectrum and integrate it through the galsim.roman bandpasses to get an
AB magnitude per band. That recipe — validated in kilonova_dataloader.ipynb — lives here so the
pipeline (generate_early_windows.py) and the dataloader import it instead of each keeping a copy.

The Roman bandpasses are reused from roman_photometry to keep one definition of the instrument.
"""

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18
from specutils import Spectrum

from kilonova.photometry.roman_noise import roman_bandpasses

FLUX_UNIT = u.Unit("erg / (s cm2 AA)")
INTRINSIC_DISTANCE_PARSEC = 10.0  # the LANL flux_rest is the absolute flux of the source at 10 pc
ALL_ROMAN_BANDS = ["R062", "Z087", "Y106", "J129", "H158", "F184"]  # union of deep + wide tiers


def redshift_and_dim_spectrum(
    wavelength_rest_aa,
    flux_rest_lambda,
    redshift,
    intrinsic_distance_parsec=INTRINSIC_DISTANCE_PARSEC,
    cosmology=Planck18,
):
    """Rest-frame spectrum at 10 pc -> observed-frame spectrum at the luminosity distance of `redshift`.
    No extinction: only the redshift (specutils) and the distance dimming (10 pc / d_L)^2. Returns
    (wavelength_observed_aa, flux_observed_lambda, luminosity_distance_parsec)."""
    spectrum_rest = Spectrum(spectral_axis=wavelength_rest_aa * u.AA, flux=flux_rest_lambda * FLUX_UNIT)
    spectrum_observed = Spectrum(spectral_axis=spectrum_rest.spectral_axis, flux=spectrum_rest.flux)
    spectrum_observed.shift_spectrum_to(redshift=redshift)
    wavelength_observed_aa = spectrum_observed.spectral_axis.to(u.AA).value
    flux_observed_lambda = spectrum_observed.flux.value

    if redshift > 0:
        luminosity_distance_parsec = cosmology.luminosity_distance(redshift).to(u.pc).value
    else:
        luminosity_distance_parsec = intrinsic_distance_parsec
    distance_dimming = (intrinsic_distance_parsec / luminosity_distance_parsec) ** 2
    return wavelength_observed_aa, flux_observed_lambda * distance_dimming, luminosity_distance_parsec


def spectrum_to_roman_magnitudes(wavelength_observed_aa, flux_observed_lambda, bands=ALL_ROMAN_BANDS):
    """AB magnitude per band integrating the (already observed-frame) spectrum through the galsim.roman
    bandpasses. NaN in a band the spectrum does not cover (e.g. the blue edge at high z). Expects the
    spectrum already masked to flux > 0 (see magnitudes_for_bands for the masking convenience)."""
    import galsim

    order = np.argsort(wavelength_observed_aa)
    spectral_energy_distribution = galsim.SED(
        galsim.LookupTable(wavelength_observed_aa[order], flux_observed_lambda[order], interpolant="linear"),
        wave_type="Ang",
        flux_type="flambda",
    )
    magnitudes = {}
    for band in bands:
        bandpass = roman_bandpasses()[band]
        covered = (
            wavelength_observed_aa.min() <= bandpass.blue_limit * 10
            and wavelength_observed_aa.max() >= bandpass.red_limit * 10
        )
        magnitudes[band] = (
            float(spectral_energy_distribution.calculateMagnitude(bandpass)) if covered else np.nan
        )
    return magnitudes


def magnitudes_for_bands(wavelength_rest_aa, flux_rest_lambda, redshift, bands=ALL_ROMAN_BANDS):
    """Full KN SED -> AB mag in one call: mask flux > 0, redshift + dim, integrate through bandpasses.
    The pipeline uses this; the dataloader instead splits it (redshift_and_dim_spectrum +
    spectrum_to_roman_magnitudes) to also keep d_L and plot the observed spectrum."""
    valid_mask = flux_rest_lambda > 0
    wavelength_observed_aa, flux_observed_lambda, _ = redshift_and_dim_spectrum(
        wavelength_rest_aa[valid_mask],
        flux_rest_lambda[valid_mask],
        redshift,
    )
    return spectrum_to_roman_magnitudes(wavelength_observed_aa, flux_observed_lambda, bands=bands)
