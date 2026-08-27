"""Low-redshift contaminants ("intermediate redshift contaminants", izc) to break the brightness
shortcut of the classifier.

The OpenUniverse contaminants and the LANL kilonovae were generated over nearly disjoint redshift
ranges -- median z 1.29 for the contaminants against 0.09 for the kilonovae, deep tier -- and
therefore over nearly disjoint apparent magnitudes.
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
  * SN Iax (8.8 % of the OpenUniverse mix) is NOT a substitution. OpenUniverse used "1000 SED
    templates from the original PLAsTiCC model, already defined to 25000 A" (OpenUniverse2024,
    sec. 4.2); that model is Jha & Dai's, released at github.com/RutgersSN/SNIax-PLAsTiCC
    (commit 908762c). This module regenerates it from the same SN 2005hk base SED and the same
    luminosity function and width-luminosity relations, so the izc Iax and the OpenUniverse Iax
    come from one model. Only the TDE, SLSN-I and PISN of the mix (0.39 %) are left out, because
    OpenUniverse took those from observed SEDs of Gaia16apd and AT2019qiz and from ELAsTiCC, none
    of which sncosmo ships.

And one limitation that no choice of library removes: at high redshift the Roman bands sample the
rest-frame optical, which these templates measure; at low redshift they sample the rest-frame
near-infrared, which for core-collapse supernovae is poorly observed and is extrapolation in every
library. The izc sample carries more model uncertainty in H158 and F184 than the OpenUniverse one.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from kilonova.photometry.roman_noise import (
    PSF_NEA_PIX,
    build_tier_constants,
    collecting_area_cm2,
)
from kilonova.photometry.spectra import ALL_ROMAN_BANDS, spectrum_to_roman_magnitudes
from kilonova.simulation.early_windows import (
    CADENCE_PARITY_PERIOD,
    GENTYPE_LABEL,
    build_window_from_model,
)

# gentype of the OpenUniverse class this stands in for, plus the offset, so an izc object is always
# separable from a real one in the parquet while `label` still maps it to the same class.
IZC_GENTYPE_OFFSET = 200

# --- SN Iax ------------------------------------------------------------------------------------
# Jha & Dai's model is a single base SED plus a width-luminosity relation, not a set of finished
# templates: the released repository carries SN 2005hk and the notebook that warps it into the 1000
# templates PLAsTiCC (and therefore OpenUniverse) drew from. `data/iax/sn2005hk_sed.npz` is that
# base SED, repacked from SN2005hk/sn2005hk.sed.dat of commit 908762c (81 rest-frame phases -15 to
# +65 d, 4801 wavelengths 1000-25000 A, float32). Everything below reproduces Iax-model.ipynb.
IAX_SOURCE_NAME = "iax-2005hk"
IAX_BASE_SED_PATH = Path(__file__).resolve().parents[3] / "data" / "iax" / "sn2005hk_sed.npz"

# Luminosity function: linear in M_V with a Gaussian rolloff at each end, fitted by Jha & Dai to
# the 51 SNe Iax of their Table 1. Deliberately NOT a Gaussian -- the class spans M_V = -13 to -18
# and a symmetric law would misplace both ends.
IAX_ABSOLUTE_MAGNITUDE_RANGE = (-20.0, -11.0)
IAX_ABSOLUTE_MAGNITUDE_STEP = 0.01
IAX_BRIGHT_ROLLOFF, IAX_BRIGHT_SIGMA = -18.0, 0.5
IAX_FAINT_ROLLOFF, IAX_FAINT_SIGMA = -13.0, 0.4

# Width-luminosity relations, all in terms of (M_V + 19). Rise time follows Magee et al. (2016);
# the two decline rates are Jha & Dai's own fits. The scatter is theirs too and is what keeps the
# class from collapsing onto one light-curve shape.
IAX_RISE_TIME_SCATTER = 2.0
IAX_DECLINE_B_SCATTER = 0.25
IAX_DECLINE_R_SCATTER = 0.2
IAX_BASE_RISE_TIME = 15.0  # hardcoded in the SN 2005hk SED, per the notebook

# The dm15(R) warp is anchored at five wavelengths and has to be spread over the full grid. The
# notebook used scipy.interpolate.interp2d, removed from SciPy in 1.14; np.interp reproduces it
# between the anchors and clamps beyond them instead of extrapolating the last slope. That choice
# only matters redward of 9000 A -- which for Roman is most of the bands -- where continuing the
# 7500-9000 A slope out to 25000 A would be an extrapolation nothing in the data supports.
IAX_WARP_ANCHORS_AA = (2350.0, 5200.0, 6000.0, 7500.0, 9000.0)

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
    # Not a registry name until `_sncosmo()` registers it; see IAX_SOURCE_NAME.
    "SN Iax": [IAX_SOURCE_NAME],
}

# UNIFORM BY CLASS, ON PURPOSE. This is not OpenUniverse's mix and is not meant to be: it is a
# training-set choice, the same kind of choice as the redshift distribution `draw_population` takes
# from its caller, and a reader who takes it for a rate-based population will misread the sample.
#
# The reasoning. OpenUniverse's own mix is a strong function of redshift -- measured over the deep
# early windows, TDE runs 7.9 % of the contaminants at z 0.02-0.1 and 0.2 % beyond z = 1, while
# SN II runs 44 % and 40 % -- so there is no single "expected rate" to inherit here anyway. More to
# the point, a rate-weighted low-redshift sample would hand the classifier a second shortcut to
# replace the one this module removes: at low redshift, guess the commonest class. Equal shares per
# class give every contaminant class the same footing in the range where the kilonovae live.
#
# Above this sample's redshift range nothing changes: those contaminants are OpenUniverse's own and
# keep their volumetric mix untouched.
#
# The SN II subtypes are split among themselves by volumetric fraction (Li et al. 2011) rather than
# uniformly -- that split is a property of the class, not a choice about class balance, and
# OpenUniverse pools all three into one label the classifier never sees separated.
UNIFORM_CLASS_SHARE = 1.0 / 5.0
SN_II_SUBTYPE_FRACTION = {"SN IIP": 0.70, "SN IIL": 0.15, "SN IIn": 0.15}
CLASS_FRACTION = {
    "SN IIP": UNIFORM_CLASS_SHARE * SN_II_SUBTYPE_FRACTION["SN IIP"],
    "SN IIL": UNIFORM_CLASS_SHARE * SN_II_SUBTYPE_FRACTION["SN IIL"],
    "SN IIn": UNIFORM_CLASS_SHARE * SN_II_SUBTYPE_FRACTION["SN IIn"],
    "SN Ia": UNIFORM_CLASS_SHARE,
    "SN Iax": UNIFORM_CLASS_SHARE,
    "SN Ic": UNIFORM_CLASS_SHARE,
    "SN Ib": UNIFORM_CLASS_SHARE,
}

# The subtypes all map back to OpenUniverse gentype 32, the class they stand in for.
GENTYPE_BY_LABEL = {
    "SN Ia": 10,
    "SN Iax": 12,
    "SN Ib": 21,
    "SN Ic": 26,
    "SN IIP": 32,
    "SN IIL": 32,
    "SN IIn": 32,
}

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

# The band and magnitude system each luminosity function is defined in. Everything here is
# rest-frame B on the AB system except SN Iax, whose luminosity function Jha & Dai published in
# rest-frame V on the Vega system; converting it would mean assuming a colour the model already
# carries, so the drawn magnitude is applied in the band it was measured in.
PEAK_ABSOLUTE_MAGNITUDE_BAND = {"SN Iax": ("bessellv", "vega")}

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

# A band whose synthesized magnitude is fainter than this does not mean a faint transient: it means
# the model has no flux there. `salt2-extended` carries essentially none redward of Y106 before
# rest-frame phase -10, where its NIR magnitudes jump from ~20 to ~59 with nothing in between, so
# any cut inside that gap separates the two cases. This is NOT a detection cut -- the survey depth
# is around 26-27 and detection belongs to `build_window_from_model`; it only removes epochs the
# SED does not actually cover, which would otherwise reach the training set as a spurious
# "very red, no near-infrared" signature in exactly the bands the classifier reads.
MODEL_FLUX_FLOOR_MAGNITUDE = 40.0

# Detector full well, electrons per pixel. NOT a galsim.roman constant -- galsim models
# non-linearity but no hard saturation -- so this is the nominal H4RG figure and an assumption of
# this module. It also makes the bright limit below a LOWER bound: Roman reads up the ramp, so a
# source past this is measured from fewer reads rather than lost.
FULL_WELL_ELECTRONS = 1.0e5


def _sncosmo():
    """sncosmo is only needed to generate, never to train or evaluate, so it stays a lazy import.

    Registering the SN Iax base source here rather than at import time keeps `IAX_SOURCE_NAME`
    resolvable through the ordinary `sncosmo.get_source` path -- including from the coverage test,
    which is the point: the Iax source has to be checked against R062-F184 like every other one."""
    import sncosmo

    if IAX_SOURCE_NAME not in _REGISTERED_SOURCES:
        phase, wavelength, flux = _iax_base_sed()
        sncosmo.register(sncosmo.TimeSeriesSource(phase, wavelength, flux), IAX_SOURCE_NAME, force=True)
        _REGISTERED_SOURCES.add(IAX_SOURCE_NAME)
    return sncosmo


_REGISTERED_SOURCES = set()
_IAX_BASE_SED = None


def register_sources():
    """Put every source name of SOURCES_BY_LABEL in the sncosmo registry.

    All but IAX_SOURCE_NAME are there already; this exists so a caller that reaches sncosmo on its
    own -- the coverage test does -- resolves the Iax source through the same path as the rest."""
    _sncosmo()


def _iax_base_sed():
    """(phase, wavelength, flux) of SN 2005hk, smoothed the way Iax-model.ipynb smooths it.

    The z/y smoothing is not cosmetic here: it runs over 8000-11000 A, which redshifts into Z087
    and Y106 for exactly the population this module generates."""
    global _IAX_BASE_SED
    if _IAX_BASE_SED is None:
        from scipy.ndimage import gaussian_filter

        with np.load(IAX_BASE_SED_PATH) as archive:
            phase = archive["phase"].astype(float)
            wavelength = archive["wavelength"].astype(float)
            flux = archive["flux"].astype(float)
        wiggly = (wavelength >= 8000.0) & (wavelength <= 11000.0)
        smoothed = gaussian_filter(flux, [4.0, 0.0])
        late = phase >= phase[10]
        flux[np.ix_(late, wiggly)] = smoothed[np.ix_(late, wiggly)]
        _IAX_BASE_SED = (phase, wavelength, flux)
    return tuple(array.copy() for array in _IAX_BASE_SED)


def sample_iax_absolute_magnitude(random_generator, size):
    """Draw M_V (Vega) from the Jha & Dai luminosity function by inverting its CDF."""
    grid = np.arange(*IAX_ABSOLUTE_MAGNITUDE_RANGE, IAX_ABSOLUTE_MAGNITUDE_STEP)
    density = -(grid + 13.0) / 6.0 + 1.0
    bright = grid < IAX_BRIGHT_ROLLOFF
    density[bright] *= np.exp(-((grid[bright] - IAX_BRIGHT_ROLLOFF) ** 2) / 2.0 / IAX_BRIGHT_SIGMA**2)
    faint = grid > IAX_FAINT_ROLLOFF
    density[faint] *= np.exp(-((grid[faint] - IAX_FAINT_ROLLOFF) ** 2) / 2.0 / IAX_FAINT_SIGMA**2)
    cumulative = np.cumsum(density) * IAX_ABSOLUTE_MAGNITUDE_STEP
    cumulative /= cumulative[-1]
    return np.interp(random_generator.random(size), cumulative, grid)


def iax_shape_parameters(absolute_magnitude_v, random_generator):
    """(rise time, dm15B, dm15R) for one SN Iax of a given peak M_V."""
    offset = absolute_magnitude_v + 19.0
    rise_time = (
        21.0 - offset * 10.0 / 3.0 + 0.22 * offset**2 + random_generator.normal() * IAX_RISE_TIME_SCATTER
    )
    decline_b = 1.2 + offset / 6.0 + random_generator.normal() * IAX_DECLINE_B_SCATTER
    decline_r = abs(0.5 + offset / 15.0 + random_generator.normal() * IAX_DECLINE_R_SCATTER)
    return float(rise_time), float(decline_b), float(decline_r)


def iax_source(rise_time, decline_b, decline_r):
    """The SN 2005hk SED warped to one (rise time, dm15B, dm15R), as in Iax-model.ipynb.

    Three warps in order: the pre-maximum phases are stretched to give the rise time, the post-
    maximum flux is scaled by a log-normal kernel centred on day 15 to give dm15(B), and a second
    kernel with a wavelength ramp gives dm15(R) while leaving dm15(B) where the first warp put it.
    Amplitude is left alone -- `roman_light_curve` sets it from the drawn M_V."""
    sncosmo = _sncosmo()
    phase, wavelength, flux = _iax_base_sed()

    stretched = phase.copy()
    pre_maximum = phase < 0.0
    stretched[pre_maximum] *= rise_time / IAX_BASE_RISE_TIME
    # Before the stretched explosion epoch the SED is not defined; the notebook suppresses it
    # rather than letting the first pre-max row leak in as a plateau.
    flux[stretched < -rise_time] /= 2000.0
    model = sncosmo.Model(source=sncosmo.TimeSeriesSource(stretched, wavelength, flux))

    def decline(band):
        return model.bandmag(band, "vega", 15.0) - model.bandmag(band, "vega", 0.0)

    post_maximum = stretched > 1.0
    kernel = np.zeros_like(stretched)
    kernel[post_maximum] = np.exp(-((np.log10(stretched[post_maximum]) - np.log10(15.0)) ** 2) / 2.0 / 0.2**2)

    scale_b = 10.0 ** (-0.4 * (decline_b - decline("bessellb")))
    flux = flux * (1.0 + (scale_b - 1.0) * kernel)[:, None]
    model = sncosmo.Model(source=sncosmo.TimeSeriesSource(stretched, wavelength, flux))

    scale_r = 10.0 ** (-0.4 * (decline_r - decline("bessellr")))
    multiplier_b = np.ones_like(stretched)
    multiplier_r = 1.0 + (scale_r - 1.0) * 1.095 * kernel
    anchored = np.array([multiplier_b, multiplier_b, multiplier_r, multiplier_r, multiplier_b])
    ramp = np.array(
        [np.interp(wavelength, IAX_WARP_ANCHORS_AA, anchored[:, index]) for index in range(len(stretched))]
    )
    return sncosmo.TimeSeriesSource(stretched, wavelength, flux * ramp)


def draw_population(number, redshifts, random_generator):
    """`number` contaminants: class, source, peak absolute magnitude, redshift, cadence parity.

    `redshifts` is the redshift of each object, drawn by the caller from whatever target
    distribution the sample is meant to fill -- the deficit against the kilonova histogram, in the
    intended use -- because the choice of that distribution is the whole point of the sample and
    does not belong buried in here."""
    labels = random_generator.choice(list(CLASS_FRACTION), size=number, p=list(CLASS_FRACTION.values()))
    population = []
    for index, (label, redshift) in enumerate(zip(labels, redshifts, strict=True)):
        if label == "SN Iax":
            # M_V (Vega) from Jha & Dai's own luminosity function, not a Gaussian; see
            # PEAK_ABSOLUTE_MAGNITUDE_BAND for why the band differs from every other class.
            peak_absolute_magnitude = float(sample_iax_absolute_magnitude(random_generator, 1)[0])
        else:
            median, sigma = PEAK_ABSOLUTE_MAGNITUDE[label]
            peak_absolute_magnitude = float(random_generator.normal(median, sigma))
        realization = {
            "index": index,
            "label": str(label),
            "source_name": str(random_generator.choice(SOURCES_BY_LABEL[label])),
            "peak_absolute_magnitude": peak_absolute_magnitude,
            "redshift": float(redshift),
            "cadence_parity": int(random_generator.integers(CADENCE_PARITY_PERIOD)),
        }
        if label == "SN Ia":
            realization["salt2_x1"] = float(random_generator.normal(*SALT2_X1))
            realization["salt2_c"] = float(random_generator.normal(*SALT2_C))
        if label == "SN Iax":
            rise_time, decline_b, decline_r = iax_shape_parameters(peak_absolute_magnitude, random_generator)
            realization["iax_rise_time"] = rise_time
            realization["iax_decline_b"] = decline_b
            realization["iax_decline_r"] = decline_r
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

    if realization["label"] == "SN Iax":
        source = iax_source(
            realization["iax_rise_time"], realization["iax_decline_b"], realization["iax_decline_r"]
        )
    else:
        source = realization["source_name"]
    model = sncosmo.Model(source=source)
    model.set(z=realization["redshift"], t0=0.0)
    if "salt2_x1" in realization:
        model.set(x1=realization["salt2_x1"], c=realization["salt2_c"])
    band, magnitude_system = PEAK_ABSOLUTE_MAGNITUDE_BAND.get(realization["label"], ("bessellb", "ab"))
    model.set_source_peakabsmag(
        realization["peak_absolute_magnitude"], band, magnitude_system, cosmo=cosmology
    )

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
            phase_rows.append(None)
            continue
        magnitudes = spectrum_to_roman_magnitudes(wavelength, flux)
        usable = all(
            np.isfinite(magnitudes[band]) and magnitudes[band] < MODEL_FLUX_FLOOR_MAGNITUDE
            for band in ALL_ROMAN_BANDS
        )
        phase_rows.append((observer_day, magnitudes) if usable else None)
    phase_rows = _longest_run(phase_rows)
    if not phase_rows:
        return {}
    days = np.array([day for day, _ in phase_rows])
    return {
        band: (days, np.array([magnitudes[band] for _, magnitudes in phase_rows])) for band in ALL_ROMAN_BANDS
    }


def _longest_run(rows):
    """The longest unbroken stretch of usable phases, `None` marking an unusable one.

    Dropping unusable phases one by one would leave a hole in the middle of a light curve, and a
    hole is worse than a shorter curve: `build_window_from_model` reads epochs off a continuous
    model and would interpolate straight across it. In practice the unusable phases sit at the
    start -- a model with no near-infrared flux before it has risen -- so this trims the head."""
    best_start, best_length, start = 0, 0, None
    for index, row in enumerate(rows + [None]):
        if row is None:
            if start is not None and index - start > best_length:
                best_start, best_length = start, index - start
            start = None
        elif start is None:
            start = index
    return rows[best_start : best_start + best_length]


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
        # `build_window_from_model` reads the label off the gentype, and IZC_GENTYPE_OFFSET puts it
        # outside GENTYPE_LABEL, so every izc window came out "UNKNOWN". The label is restored in
        # OpenUniverse's own vocabulary -- the class this object stands in for -- and the finer
        # subtype this module draws (IIP/IIL/IIn, which OpenUniverse pools into "SN II") is kept in
        # its own column instead of being smuggled into `label`.
        window["label"] = GENTYPE_LABEL[GENTYPE_BY_LABEL[realization["label"]]]
        window["izc_subtype"] = realization["label"]
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
