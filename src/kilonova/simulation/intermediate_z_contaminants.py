"""Low-redshift contaminants ("intermediate redshift contaminants", izc) to break the brightness
shortcut of the classifier.

The OpenUniverse contaminants and the LANL kilonovae were generated over nearly disjoint redshift
ranges -- median z 0.08 against 1.26 -- and therefore over nearly disjoint apparent magnitudes.
"Bright implies kilonova" is an almost perfect rule inside that dataset and a false one in the sky,
and `training/diagnostics/brightness_shortcut.py` measures that the model learned it: shifting every
magnitude of a distant contaminant window four magnitudes brighter, with colours, signal-to-noise
and the source-to-limit relation untouched, turns 99.97 % of them into kilonovae.

This module generates the missing population: ordinary supernovae at the redshifts where the survey
has none. It is deliberately one-directional. "Too faint to be a kilonova" is a real limit -- Roman
will not find one at z = 3 -- so nothing here touches the faint end; only the unearned bright end.

Nothing in `training/` imports this, and this imports nothing from `training/`. It writes its own
parquet files with the window schema of `build_window_from_model`, so a generated sample can be
inspected and thrown away without the training path ever knowing it existed.

Two substitutions the sample carries, both declared rather than hidden:

  * OpenUniverse drew core-collapse SEDs from `NON1ASED.V19_CC+HostXT_WAVEEXT`. The Vincenzi et al.
    (2019) templates that sncosmo ships stop at 11000 A rest-frame, which covers Roman only for
    z >= 0.91 -- the opposite of the range needed here. The sources below are the SNANA NON1A and
    Nugent libraries, which sncosmo ships already extended to 25000 A. Different library for the
    low-redshift half of the contaminants than for the high-redshift half.
  * SN Iax (8.8 % of the OpenUniverse mix) has no wavelength-extended source available and is left
    out entirely.

And one limitation that no choice of library removes: at high redshift the Roman bands sample the
rest-frame optical, which these templates measure; at low redshift they sample the rest-frame
near-infrared, which for core-collapse supernovae is poorly observed and is extrapolation in every
library. The izc sample carries more model uncertainty in H158 and F184 than the OpenUniverse one.
"""

import numpy as np
import pandas as pd

from kilonova.photometry.roman_noise import (
    PSF_NEA_PIX,
    build_tier_constants,
    collecting_area_cm2,
)
from kilonova.photometry.spectra import ALL_ROMAN_BANDS, spectrum_to_roman_magnitudes
from kilonova.simulation.early_windows import CADENCE_PARITY_PERIOD, build_window_from_model

# gentype of the OpenUniverse class this stands in for, plus the offset, so an izc object is always
# separable from a real one in the parquet while `label` still maps it to the same class.
IZC_GENTYPE_OFFSET = 200

# Sources vetted to cover R062 through F184 with no extrapolation at z = 0.02, the bluest and
# reddest edges included. tests/test_intermediate_z_contaminants.py re-checks this against sncosmo
# rather than trusting the list.
SOURCES_BY_LABEL = {
    # SN II split by subtype rather than pooled. Pooling them forced one luminosity function over
    # IIP, IIL and IIn together, and its sigma of 1.61 mag had tails reaching M = -20.8 -- a
    # superluminous supernova wearing a IIP label. The sources are already separated by subtype, so
    # the split costs nothing and removes the tail by construction.
    "SN IIP": [
        "nugent-sn2p",
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
    ],
    "SN IIL": ["nugent-sn2l"],
    "SN IIn": ["nugent-sn2n"],
    "SN Ib": [
        "nugent-hyper",
        "nugent-sn1bc",
        "snana-2004gv",
        "snana-2004ib",
        "snana-2005hm",
        "snana-2006ep",
        "snana-2006jo",
        "snana-2007nc",
        "snana-2007y",
    ],
    "SN Ic": [
        "nugent-hyper",
        "nugent-sn1bc",
        "snana-04d1la",
        "snana-04d4jv",
        "snana-2004fe",
        "snana-2004gq",
        "snana-2006fo",
        "snana-2006lc",
        "snana-sdss004012",
        "snana-sdss014475",
    ],
    "SN Ia": ["salt2-extended"],
}

# Class mix of the OpenUniverse contaminants, renormalized over the classes this module can
# generate (SN Iax and the 0.4 % of TDE/SLSN/PISN drop out).
# Class mix of the OpenUniverse contaminants, renormalized over the classes this module can
# generate (SN Iax and the 0.4 % of TDE/SLSN/PISN drop out), with OpenUniverse's single "SN II"
# share divided among the subtypes by their volumetric fractions (Li et al. 2011).
SN_II_SUBTYPE_FRACTION = {"SN IIP": 0.70, "SN IIL": 0.15, "SN IIn": 0.15}
CLASS_FRACTION = {
    "SN IIP": 0.568 * 0.70,
    "SN IIL": 0.568 * 0.15,
    "SN IIn": 0.568 * 0.15,
    "SN Ia": 0.185,
    "SN Ic": 0.124,
    "SN Ib": 0.123,
}

# The subtypes all map back to OpenUniverse gentype 32, the class they stand in for.
GENTYPE_BY_LABEL = {"SN Ia": 10, "SN Ib": 21, "SN Ic": 26, "SN IIP": 32, "SN IIL": 32, "SN IIn": 32}

# Peak absolute magnitude, (median, sigma). SN Ia is exact: OpenUniverse records salt2_mB, whose
# definition is already the rest-frame B peak, so M_B = mB - mu. The core-collapse values come from
# the low-redshift tail of the same catalogue (peak_mag_g - mu below z = 0.3, where the
# K-correction is small), which keeps the izc luminosity function on OpenUniverse's own scale
# instead of importing one. They are a first pass and are meant to be replaced by the survey's own
# luminosity functions if those become available.
PEAK_ABSOLUTE_MAGNITUDE = {
    "SN Ia": (-19.404, 0.270),
    "SN Ib": (-17.18, 1.17),
    "SN Ic": (-17.36, 1.22),
    # The SN II subtypes come from Richardson et al. (2014), because OpenUniverse records a single
    # pooled "SN II" and its low-redshift tail cannot be split after the fact. Literature inputs,
    # flagged as such, and the first thing to replace with the survey's own numbers.
    "SN IIP": (-16.80, 0.97),
    "SN IIL": (-17.98, 0.90),
    "SN IIn": (-18.62, 1.48),
}

# SALT2 shape and colour, from the OpenUniverse SN Ia population (salt2_x1, salt2_c).
SALT2_X1 = (0.152, 0.914)
SALT2_C = (-0.017, 0.075)

# No host extinction is applied. OpenUniverse records AV = -9 for every class except Iax, because
# the V19 templates carry the host dust inside the model (the "+HostXT" of the model name) and
# SALT2 absorbs it into `c`. Adding an AV here would redden the izc sample relative to the
# OpenUniverse one, which is exactly the kind of class-correlated artefact this module exists to
# remove.
REST_FRAME_PHASES = np.arange(-20.0, 71.0, 1.0)
# Observed-frame grid the spectrum is sampled on. The limits bracket R062 to F184 with room to
# spare; the grid is intersected with each model's own validity range, because a source that does
# not reach a Roman band should fail the coverage check inside `spectrum_to_roman_magnitudes` and
# have its object dropped, not raise out of sncosmo.
OBSERVED_WAVELENGTH_LIMITS = (4000.0, 22000.0)
OBSERVED_WAVELENGTH_STEP = 10.0
PHOTOMETRY_FLOOR = 1e-30  # erg/s/cm2/A; below this a phase carries no usable flux

# Detector full well, electrons per pixel. NOT a galsim.roman constant -- galsim models
# non-linearity but no hard saturation -- so this is the nominal H4RG figure and an assumption of
# this module. It also makes the bright limit below a LOWER bound: Roman reads up the ramp, so a
# source past this is measured from fewer reads rather than lost.
FULL_WELL_ELECTRONS = 1.0e5


def _sncosmo():
    """sncosmo is only needed to generate, never to train or evaluate, so it stays a lazy import."""
    import sncosmo

    return sncosmo


def draw_population(number, redshifts, random_generator):
    """`number` contaminants: class, source, peak absolute magnitude, redshift, cadence parity.

    `redshifts` is the redshift of each object, drawn by the caller from whatever target
    distribution the sample is meant to fill -- the deficit against the kilonova histogram, in the
    intended use -- because the choice of that distribution is the whole point of the sample and
    does not belong buried in here."""
    labels = random_generator.choice(list(CLASS_FRACTION), size=number, p=list(CLASS_FRACTION.values()))
    population = []
    for index, (label, redshift) in enumerate(zip(labels, redshifts, strict=True)):
        median, sigma = PEAK_ABSOLUTE_MAGNITUDE[label]
        realization = {
            "index": index,
            "label": str(label),
            "source_name": str(random_generator.choice(SOURCES_BY_LABEL[label])),
            "peak_absolute_magnitude": float(random_generator.normal(median, sigma)),
            "redshift": float(redshift),
            "cadence_parity": int(random_generator.integers(CADENCE_PARITY_PERIOD)),
        }
        if label == "SN Ia":
            realization["salt2_x1"] = float(random_generator.normal(*SALT2_X1))
            realization["salt2_c"] = float(random_generator.normal(*SALT2_C))
        population.append(realization)
    return population


def roman_light_curve(realization, cosmology=None):
    """{band: (observer days from peak, AB mag)} for one drawn contaminant.

    sncosmo supplies the SED and does the redshift, the dimming and the time dilation; the band
    integration is `spectrum_to_roman_magnitudes`, the same call the kilonova path and the AT2017gfo
    note use, so the instrument has one definition across the whole project.

    A band the redshifted spectrum does not cover comes back NaN and would reach the window as a
    `not observed` token. In the training set the first epoch always has exactly three observed
    bands, so such a window is out of distribution and would inject the very pathology
    `training/diagnostics/anchor_band_ablation.py` measures. `build_izc_windows` drops those
    objects rather than emitting them."""
    sncosmo = _sncosmo()
    if cosmology is None:
        from astropy.cosmology import Planck18

        cosmology = Planck18

    model = sncosmo.Model(source=realization["source_name"])
    model.set(z=realization["redshift"], t0=0.0)
    if "salt2_x1" in realization:
        model.set(x1=realization["salt2_x1"], c=realization["salt2_c"])
    model.set_source_peakabsmag(realization["peak_absolute_magnitude"], "bessellb", "ab", cosmo=cosmology)

    blue = max(model.minwave(), OBSERVED_WAVELENGTH_LIMITS[0])
    red = min(model.maxwave(), OBSERVED_WAVELENGTH_LIMITS[1])
    if red <= blue:
        return {}
    wavelength = np.arange(blue, red, OBSERVED_WAVELENGTH_STEP)

    source = model.source
    phases = REST_FRAME_PHASES[
        (REST_FRAME_PHASES >= source.minphase()) & (REST_FRAME_PHASES <= source.maxphase())
    ]
    # Every band has to span the SAME phases. `build_window_from_model` marks a band unobserved
    # when the model does not cover that epoch, so bands with different time coverage would leave a
    # `not observed` token in a slot the cadence did schedule -- a window with fewer than three
    # observed bands in the first epoch, which is exactly what the training set never contains.
    phase_rows = []
    for phase in phases:
        observer_day = phase * (1.0 + realization["redshift"])
        flux = np.clip(model.flux(observer_day, wavelength), 0.0, None)
        if flux.max() < PHOTOMETRY_FLOOR:
            continue
        magnitudes = spectrum_to_roman_magnitudes(wavelength, flux)
        if not all(np.isfinite(magnitudes[band]) for band in ALL_ROMAN_BANDS):
            continue
        phase_rows.append((observer_day, magnitudes))
    if not phase_rows:
        return {}
    days = np.array([day for day, _ in phase_rows])
    return {
        band: (days, np.array([magnitudes[band] for _, magnitudes in phase_rows])) for band in ALL_ROMAN_BANDS
    }


def build_izc_windows(population, tier, cosmology=None):
    """Early windows for a drawn population, in the schema of `build_window_from_model`.

    Objects whose spectrum does not cover every band of the tier are dropped, as are the ones the
    survey never detects -- the same rule the OpenUniverse path applies, where an undetected
    transient simply produces no window."""
    constants = build_tier_constants(tier)
    tier_bands = set(constants["bands"])
    bright_limit = saturation_magnitude(tier)
    windows, rejected = [], {"coverage": 0, "undetected": 0, "saturated_kept": 0}
    for realization in population:
        curves = roman_light_curve(realization, cosmology=cosmology)
        if not tier_bands.issubset(curves):
            rejected["coverage"] += 1
            continue
        model = {band: curves[band] for band in constants["bands"]}
        object_id = (
            f"izc_{realization['index']:08d}_{realization['label'].replace(' ', '')}"
            f"_{realization['redshift']:.4f}"
        )
        window = build_window_from_model(
            object_id,
            model,
            constants,
            realization["redshift"],
            GENTYPE_BY_LABEL[realization["label"]] + IZC_GENTYPE_OFFSET,
            noise_seed=realization["index"],
            visit_index_offset=realization["cadence_parity"],
        )
        if window is None:
            rejected["undetected"] += 1
            continue
        window["tier"] = tier
        observed = window[window["observed"]]
        saturated = observed["mag_true"] < observed["band"].map(bright_limit)
        if saturated.any():
            rejected["saturated_kept"] += 1
        windows.append(window)
    if not windows:
        return pd.DataFrame(), rejected
    return pd.concat(windows, ignore_index=True), rejected


def saturation_magnitude(tier):
    """{band: AB magnitude at which the peak pixel fills the well} for one tier.

    Reported, never enforced. Any rule that drops saturated contaminants has to drop saturated
    kilonovae too: the existing training set keeps 0.14 % of its deep kilonovae above this limit,
    and removing only the contaminants would rebuild the very "bright implies kilonova" asymmetry
    this sample exists to break. The policy belongs to whoever assembles the training set, so this
    only supplies the number.

    The peak pixel is taken to receive 1/NEA of the source counts, which is what the noise-equivalent
    area means; that is an approximation to the true PSF peak, good enough to flag a population."""
    constants = build_tier_constants(tier)
    magnitudes = {}
    for band in constants["bands"]:
        exposure = constants["exposure_time"][band]
        counts_at_saturation = FULL_WELL_ELECTRONS * PSF_NEA_PIX[band]
        rate = counts_at_saturation / exposure / collecting_area_cm2()
        magnitudes[band] = float(constants["zeropoint"][band] - 2.5 * np.log10(rate))
    return magnitudes
