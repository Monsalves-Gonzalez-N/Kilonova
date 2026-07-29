"""Kilonova synthetic photometry — single source of truth for the SED -> Roman AB magnitude step.

OpenUniverse transients already carry true per-band magnitudes in the snana hdf5, so they need no
synthetic photometry. The LANL kilonova grid only stores rest-frame spectra (flux_rest), so the KN
path must redshift + dim the spectrum and integrate it through the galsim.roman bandpasses to get an
AB magnitude per band. That recipe lives here so the pipeline (generate_early_windows.py) and the
dataloader import it instead of each keeping a copy. It is pinned by tests/test_spectra.py against
closed-form SEDs: a bolometric-conservation check and a flat-f_nu source whose AB magnitude is
analytic in every band and at every redshift. (An earlier note claimed the recipe was validated in
kilonova_dataloader.ipynb; that comparison did not catch the missing 1/(1+z), hence the tests.)

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
    No extinction: only the redshift (specutils) and the dimming (10 pc / d_L)^2 / (1 + z). Returns
    (wavelength_observed_aa, flux_observed_lambda, luminosity_distance_parsec).

    The 1/(1+z) is not optional and is easy to lose: specutils only stretches the wavelength axis,
    it does not rescale the flux. Since dlambda_obs = (1+z) dlambda_rest, integrating f_lambda over
    the stretched axis picks up an extra (1+z), so f_lambda itself must carry the inverse for the
    observed bolometric flux to come out as L / (4 pi d_L^2). Photometrically this is the
    K-correction term: dropping it makes every source 2.5 log10(1+z) too bright."""
    spectrum_rest = Spectrum(spectral_axis=wavelength_rest_aa * u.AA, flux=flux_rest_lambda * FLUX_UNIT)
    spectrum_observed = Spectrum(spectral_axis=spectrum_rest.spectral_axis, flux=spectrum_rest.flux)
    spectrum_observed.shift_spectrum_to(redshift=redshift)
    wavelength_observed_aa = spectrum_observed.spectral_axis.to(u.AA).value
    flux_observed_lambda = spectrum_observed.flux.value

    if redshift > 0:
        luminosity_distance_parsec = cosmology.luminosity_distance(redshift).to(u.pc).value
    else:
        luminosity_distance_parsec = intrinsic_distance_parsec
    dimming = (intrinsic_distance_parsec / luminosity_distance_parsec) ** 2 / (1.0 + redshift)
    return wavelength_observed_aa, flux_observed_lambda * dimming, luminosity_distance_parsec


def spectrum_to_roman_magnitudes(wavelength_observed_aa, flux_observed_lambda, bands=ALL_ROMAN_BANDS):
    """AB magnitude per band integrating the (already observed-frame) spectrum through the galsim.roman
    bandpasses. Two distinct non-numbers come out of this, and the pipeline treats them differently:
    NaN means the spectrum does not cover the band (e.g. the blue edge at high z) and there is nothing
    to say about it; +inf means the band is covered but carries no flux, which is a real measurement —
    the survey observes it and gets a non-detection. Pass the spectrum with its zeros intact."""
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
        if not covered:
            magnitudes[band] = np.nan
            continue
        # calculateMagnitude is calculateFlux followed by this same log, but it turns a band with
        # no flux into a NaN indistinguishable from "not covered": the quadrature over an integrand
        # that is zero across the bandpass returns either an exact 0 or a rounding-sized negative
        # (order 1e-35), and log10 of those gives -inf and NaN respectively. Both mean the same
        # physical thing, no flux, so both must come out as +inf.
        band_flux = spectral_energy_distribution.calculateFlux(bandpass)
        magnitudes[band] = float(-2.5 * np.log10(band_flux) + bandpass.zeropoint) if band_flux > 0 else np.inf
    return magnitudes


def magnitudes_for_bands(wavelength_rest_aa, flux_rest_lambda, redshift, bands=ALL_ROMAN_BANDS):
    """Full KN SED -> AB mag in one call: redshift + dim, then integrate through the bandpasses.
    The pipeline uses this; the dataloader instead splits it (redshift_and_dim_spectrum +
    spectrum_to_roman_magnitudes) to also keep d_L and plot the observed spectrum.

    The zero-flux bins of the LANL Monte Carlo spectra are kept, not filtered out. Dropping them
    used to leave a hole between the surviving samples that the linear interpolant spanned with a
    straight line, so a single stray blue bin could invent flux right across a bandpass the
    kilonova is dark in — and when no bin survived, the band was lost to the coverage check
    instead of being reported as the non-detection it is."""
    flux_rest_lambda = np.clip(np.asarray(flux_rest_lambda, dtype=float), 0.0, None)
    wavelength_observed_aa, flux_observed_lambda, _ = redshift_and_dim_spectrum(
        wavelength_rest_aa,
        flux_rest_lambda,
        redshift,
    )
    return spectrum_to_roman_magnitudes(wavelength_observed_aa, flux_observed_lambda, bands=bands)
