"""Published photometry of real transients -> Roman bands -> the kilonova classifier.

The spectroscopic route (`at2017gfo_roman_prediction.ipynb`) integrates an X-shooter spectrum
through the Roman bandpasses. This module is the photometric route: it starts from broadband
magnitudes that somebody already published, in whatever filters they happened to use, and asks the
same question of the same checkpoint. That buys what the spectroscopic route cannot give — real
measurement uncertainties, and nights the spectral series does not cover.

Two sources:

    * `load_smartt_photometry`  AT2017gfo, the compilation of S. J. Smartt used in Coughlin et al.
      2018, shipped inside the ENGRAVE X-shooter release (`AT2017gfo_phot_compiled_sjs.dat`).
    * `load_villar_photometry`  AT2017gfo, Villar et al. 2017 (ApJL 851 L21) table 3, the
      homogenized 18-paper compilation, pulled from VizieR and cached next to the other data.

Two independent ways of turning those filters into Roman bands, reported side by side so the
conversion bias is visible rather than assumed away:

    * `roman_magnitudes_by_sed`      interpolate the observed SED through the measured points and
      integrate it against the same `galsim.roman` bandpasses the training set was built with.
    * `roman_magnitudes_by_nearest`  substitute the closest filter by pivot wavelength, no colour
      correction at all.

Everything downstream of the conversion is the live pipeline: `build_window_from_model` supplies
the Roman noise recipe, `_read_long_parquet` + `OpenUniverseWindowDataset` the tokenization, and
`LitKilonova` the checkpoint. Nothing about the instrument or the tokens is re-implemented here.

All magnitudes are AB and observed-frame (reddened, at the host redshift), matching the convention
of the spectroscopic notebook.
"""

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from kilonova.photometry.roman_noise import BASE_CADENCE_DAYS
from kilonova.photometry.spectra import ALL_ROMAN_BANDS, spectrum_to_roman_magnitudes
from kilonova.simulation.early_windows import KN_GENTYPE, N_KN_VISITS, build_window_from_model

SPEED_OF_LIGHT_AA_PER_S = 2.99792458e18
AB_ZEROPOINT = 48.60  # m_AB = -2.5 log10(F_nu [erg/s/cm2/Hz]) - 48.60

# Published band name -> pyphot filter. Only filters that can plausibly constrain the Roman range
# are listed; anything else in a compilation (Swift UV, mid-infrared, unfiltered "white") is dropped
# on purpose, because extrapolating an SED from them into 0.5-2.0 um would be inventing colour.
FILTER_LIBRARY_NAME = {
    "u": "SDSS_u",
    "U": "GROUND_JOHNSON_U",
    "B": "GROUND_JOHNSON_B",
    "V": "GROUND_JOHNSON_V",
    "g": "PS1_g",
    "r": "PS1_r",
    "R": "GROUND_COUSINS_R",
    "i": "PS1_i",
    "I": "GROUND_COUSINS_I",
    "z": "PS1_z",
    "y": "PS1_y",
    "Y": "PS1_y",
    "J": "2MASS_J",
    "H": "2MASS_H",
    "K": "2MASS_Ks",
    "Ks": "2MASS_Ks",
    "F110W": "HST_WFC3_F110W",
    "F160W": "HST_WFC3_F160W",
    "F475W": "HST_WFC3_F475W",
    "F606W": "HST_WFC3_F606W",
    "F625W": "HST_WFC3_F625W",
    "F775W": "HST_WFC3_F775W",
    "F814W": "HST_WFC3_F814W",
    "F070W": "JWST_NIRCAM_F070W",
    "F115W": "JWST_NIRCAM_F115W",
    "F150W": "JWST_NIRCAM_F150W",
    "F200W": "JWST_NIRCAM_F200W",
    "F277W": "JWST_NIRCAM_F277W",
    "F356W": "JWST_NIRCAM_F356W",
    "F444W": "JWST_NIRCAM_F444W",
}

VILLAR_VIZIER_CATALOG = "J/ApJ/851/L21"
VILLAR_MERGER_MJD = 57982.529  # the zero point of the Phase column of table 3


# ----------------------------------------------------------------------------- filter metadata
_pivot_wavelength_cache = {}


def pivot_wavelength_aa(filter_name):
    """Pivot wavelength [AA] of a published filter, from the pyphot library.

    Cached because the library reload dominates the cost of a Monte Carlo that touches it once per
    realization."""
    if filter_name not in _pivot_wavelength_cache:
        import pyphot

        library_name = FILTER_LIBRARY_NAME[filter_name]
        _pivot_wavelength_cache[filter_name] = float(pyphot.get_library()[library_name].lpivot.to("AA").value)
    return _pivot_wavelength_cache[filter_name]


def known_filters(filter_names):
    """The subset of `filter_names` this module knows how to place on a wavelength axis."""
    return [name for name in filter_names if name in FILTER_LIBRARY_NAME]


# ----------------------------------------------------------------------------- photometry readers
def load_smartt_photometry(path):
    """The compilation of S. J. Smartt (Coughlin et al. 2018), as published: observed, uncorrected.

    The file is ragged — the telescope names sit between the optical and the near-infrared block —
    so the 14 fixed fields are parsed positionally and the near-infrared is taken from the last
    numeric run of the line, whose final 6 values are J, J_err, H, H_err, K, K_err. Sentinel 9999
    marks a magnitude published without an error; it becomes NaN."""

    def as_float(token):
        value = np.nan if token.upper() == "NAN" else float(token)
        return np.nan if value > 9000 else value

    optical_names = ["U", "U_err", "g", "g_err", "r", "r_err", "i", "i_err", "z", "z_err", "y", "y_err"]
    near_infrared_names = ["J", "J_err", "H", "H_err", "K", "K_err"]
    rows = []
    for line in path.read_text().splitlines():
        tokens = line.split()
        if len(tokens) < 14 or not tokens[0].startswith("579"):
            continue
        optical = [as_float(token) for token in tokens[2:14]]
        # The tail is [optical telescope] [NIR phase] J J_err H H_err K K_err [NIR telescope], and
        # either telescope name may or may not be there. Collecting the maximal runs of numeric
        # tokens and taking the last run of at least 6 is what survives that raggedness; scanning
        # left to right and resetting on every name instead throws the block away whenever the line
        # ends with a telescope, which is most of them.
        numeric_runs = [[]]
        for token in tokens[14:]:
            try:
                numeric_runs[-1].append(as_float(token))
            except ValueError:
                numeric_runs.append([])
        long_runs = [run for run in numeric_runs if len(run) >= 6]
        near_infrared = long_runs[-1][-6:] if len(long_runs) > 0 else [np.nan] * 6
        rows.append(
            {
                "phase": float(tokens[1]),
                **dict(zip(optical_names, optical, strict=True)),
                **dict(zip(near_infrared_names, near_infrared, strict=True)),
            }
        )
    return pd.DataFrame(rows)


def load_villar_photometry(cache_path, refresh=False):
    """Villar et al. 2017 table 3 from VizieR, as a long frame (phase, filter, magnitude, error).

    Only the points Villar's own modelling kept are used: detections (no upper-limit flag) that are
    not flagged as outliers or excluded. The cache is a plain CSV so the notebook runs offline once
    the table has been pulled."""
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)

    from astroquery.vizier import Vizier

    table = Vizier(columns=["**"], row_limit=-1).get_catalogs(VILLAR_VIZIER_CATALOG)[0]
    frame = pd.DataFrame(
        {
            "phase": np.asarray(table["Phase"], dtype=float),
            "filter": [str(value).strip() for value in table["Band"]],
            "magnitude": np.asarray(table["mag"], dtype=float),
            "magnitude_error": np.asarray(table["e_mag"], dtype=float),
            "is_upper_limit": np.asarray([str(value).strip() == ">" for value in table["l_mag"]]),
            "is_outlier": np.asarray([str(value).strip() == "O" for value in table["Out"]]),
            "is_excluded": np.asarray([str(value).strip() == "X" for value in table["Exc"]]),
            "instrument": [str(value).strip() for value in table["Inst"]],
            "reference": [str(value).strip() for value in table["Ref"]],
        }
    )
    frame = frame[~frame["is_upper_limit"] & ~frame["is_outlier"] & ~frame["is_excluded"]]
    frame = frame[np.isfinite(frame["magnitude"]) & np.isfinite(frame["magnitude_error"])]
    frame = frame.drop(columns=["is_upper_limit", "is_outlier", "is_excluded"]).reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def smartt_to_long(photometry):
    """The wide Smartt table -> the same long (phase, filter, magnitude, magnitude_error) schema."""
    rows = []
    for record in photometry.to_dict("records"):
        for filter_name in ["U", "g", "r", "i", "z", "y", "J", "H", "K"]:
            magnitude = record.get(filter_name, np.nan)
            error = record.get(f"{filter_name}_err", np.nan)
            if not np.isfinite(magnitude) or not np.isfinite(error):
                continue
            rows.append(
                {
                    "phase": record["phase"],
                    "filter": filter_name,
                    "magnitude": magnitude,
                    "magnitude_error": error,
                }
            )
    return pd.DataFrame(rows)


def epoch_photometry(long_photometry, phase_center, phase_half_width):
    """The measurements inside +-`phase_half_width` of `phase_center`, one entry per filter.

    A real night gives several telescopes in the same band; they are averaged with inverse-variance
    weights, which is what makes a single SED out of a night of follow-up. Returns
    {filter: (magnitude, magnitude_error)}."""
    window = long_photometry[np.abs(long_photometry["phase"] - phase_center) <= phase_half_width]
    magnitudes = {}
    for filter_name, block in window.groupby("filter"):
        if filter_name not in FILTER_LIBRARY_NAME:
            continue
        weight = 1.0 / np.square(block["magnitude_error"].to_numpy())
        magnitude = float(np.sum(weight * block["magnitude"].to_numpy()) / np.sum(weight))
        error = float(np.sqrt(1.0 / np.sum(weight)))
        magnitudes[filter_name] = (magnitude, error)
    return magnitudes


# Bibcode of Villar's `Ref` column -> the citation the reader needs. Villar et al. 2017 homogenized
# 18 papers into one table but the DATA belong to the papers below, and the table keeps the
# provenance per point, which is what makes a per-instrument breakdown possible at all.
VILLAR_REFERENCE_LABEL = {
    "2017Natur.551...64A": "Arcavi et al. 2017, Nature 551, 64",
    "2017Natur.551...67P": "Pian et al. 2017, Nature 551, 67",
    "2017Natur.551...71T": "Troja et al. 2017, Nature 551, 71",
    "2017Natur.551...75S": "Smartt et al. 2017, Nature 551, 75",
    "2017ApJ...848L..17C": "Cowperthwaite et al. 2017, ApJL 848, L17",
    "2017ApJ...848L..24V": "Valenti et al. 2017, ApJL 848, L24",
    "2017ApJ...848L..27T": "Tanvir et al. 2017, ApJL 848, L27",
    "2017ApJ...848L..29D": "Diaz et al. 2017, ApJL 848, L29",
    "2017ApJ...850L...1L": "Lipunov et al. 2017, ApJL 850, L1",
    "2017arXiv171005436K": "Kasliwal et al. 2017, Science 358, 1559",
    "2017arXiv171005437E": "Evans et al. 2017, Science 358, 1565",
    "2017arXiv171005443D": "Drout et al. 2017, Science 358, 1570",
    "2017arXiv171005452C": "Coulter et al. 2017, Science 358, 1556",
    "2017arXiv171005846A": "Andreoni et al. 2017, PASA 34, e069",
    "2017arXiv171005848U": "Utsumi et al. 2017, PASJ 69, 101",
    "this paper": "Villar et al. 2017, ApJL 851, L21",  # the points Villar's own table contributes
}


def published_datasets(long_photometry, phase_grid, phase_half_width, minimum_filters=2):
    """One entry per (instrument, reference, night) of a compilation that keeps its provenance.

    This is the per-paper view of the same photometry the rest of the module already handles. It
    exists because "the classifier recognizes AT2017gfo" is a much harder claim to argue from a
    compilation -- a reader has to trust that the homogenization did not manufacture the colours --
    than from a single night of a single instrument in the filters that instrument actually has, cited
    to the paper that published it. VISTA/VIRCAM measured Y, J and Ks; that is a real band set, and
    what it maps onto in Roman is a fact about the two filter systems, not about the compilation.

    Requires `long_photometry` to carry `instrument` and `reference` columns (Villar's table does;
    the Smartt compilation does not, which is why only the first goes through here). Filters unknown
    to the pyphot map are dropped by `epoch_photometry`, so `minimum_filters` counts usable ones.

    Returns [{instrument, reference, label, phase_days, magnitudes}], one per night with enough
    filters; picking one night per instrument is left to the caller."""
    datasets = []
    for (instrument, reference), block in long_photometry.groupby(["instrument", "reference"]):
        for phase_days in phase_grid:
            magnitudes = epoch_photometry(block, phase_days, phase_half_width)
            if len(magnitudes) < minimum_filters:
                continue
            datasets.append(
                {
                    "instrument": str(instrument),
                    "reference": str(reference),
                    "label": VILLAR_REFERENCE_LABEL.get(str(reference), str(reference)),
                    "phase_days": float(phase_days),
                    "magnitudes": magnitudes,
                }
            )
    return datasets


# ----------------------------------------------------------------------------- filter conversion
def _sorted_by_pivot(magnitudes):
    entries = [(pivot_wavelength_aa(name), name, value) for name, (value, _) in magnitudes.items()]
    entries.sort()
    return entries


def sampled_bands(magnitudes, bands=ALL_ROMAN_BANDS):
    """The Roman bands that the published filters actually SAMPLE, not merely bracket.

    A band counts as sampled when at least one input filter has its pivot wavelength inside the
    band's bandpass. Spanning is not enough and this is not a technicality: NIRCam has nothing
    between F070W (0.704 um) and F115W (1.154 um), so Z087 (0.720-1.025 um) sits entirely inside
    that gap. An interpolated SED runs straight through it and hands back a perfectly finite
    magnitude that no measurement supports — and once that magnitude reaches the noise recipe it
    decides, on nothing, whether the band becomes a detection or a 5 sigma upper limit. Both are
    claims about an observation that does not exist. The honest output is no band at all, which the
    window turns into a `not observed` token, exactly like a filter the cadence skipped."""
    from kilonova.photometry.roman_noise import roman_bandpasses

    pivots = [pivot_wavelength_aa(name) for name in magnitudes]
    sampled = []
    for band in bands:
        bandpass = roman_bandpasses()[band]
        if any(bandpass.blue_limit * 10 <= pivot <= bandpass.red_limit * 10 for pivot in pivots):
            sampled.append(band)
    return sampled


def roman_magnitudes_by_sed(magnitudes, bands=ALL_ROMAN_BANDS):
    """Interpolate the observed SED through the measured points, then integrate it as a spectrum.

    Each AB magnitude becomes an F_nu at its filter's pivot wavelength; the SED between points is a
    piecewise power law (a straight line in log F_nu vs log lambda), which is the mildest assumption
    that stays positive and reproduces a smooth continuum. NO extrapolation: the spectrum handed to
    `spectrum_to_roman_magnitudes` spans only the measured pivots, so any Roman band whose bandpass
    runs past the bluest or reddest filter comes back NaN from its own coverage check rather than
    from a guess. And no interpolation across an unsampled band either — see `sampled_bands`.

    The result therefore carries the same Roman bandpasses and the same integration as the training
    set, which is the point of going through a spectrum instead of matching filters by eye."""
    entries = _sorted_by_pivot(magnitudes)
    if len(entries) < 2:
        return dict.fromkeys(bands, np.nan)

    pivots = np.array([pivot for pivot, _, _ in entries])
    flux_nu = np.array([10.0 ** (-0.4 * (value + AB_ZEROPOINT)) for _, _, value in entries])

    wavelength_aa = np.geomspace(pivots.min(), pivots.max(), 4000)
    log_flux_nu = np.interp(np.log(wavelength_aa), np.log(pivots), np.log(flux_nu))
    flux_lambda = np.exp(log_flux_nu) * SPEED_OF_LIGHT_AA_PER_S / np.square(wavelength_aa)
    integrated = spectrum_to_roman_magnitudes(wavelength_aa, flux_lambda, bands=bands)
    supported = set(sampled_bands(magnitudes, bands))
    return {band: (value if band in supported else np.nan) for band, value in integrated.items()}


def roman_magnitudes_by_nearest(magnitudes, bands=ALL_ROMAN_BANDS):
    """Substitute the published filter whose pivot is closest to the Roman band's, uncorrected.

    The naive thing a reader would do by hand, kept as the control: it applies no colour term, so
    the difference against `roman_magnitudes_by_sed` is exactly the price of the shortcut. It is
    held to the same sampling rule (`sampled_bands`), so a band nobody measured is left NaN instead
    of borrowing a filter from outside it — the substitution is only ever between filters that
    overlap, never across a gap."""
    from kilonova.photometry.roman_noise import roman_bandpasses

    entries = _sorted_by_pivot(magnitudes)
    if len(entries) == 0:
        return dict.fromkeys(bands, np.nan)
    pivots = np.array([pivot for pivot, _, _ in entries])
    values = np.array([value for _, _, value in entries])
    supported = set(sampled_bands(magnitudes, bands))

    roman_magnitudes = {}
    for band in bands:
        if band not in supported:
            roman_magnitudes[band] = np.nan
            continue
        band_pivot = roman_bandpasses()[band].effective_wavelength * 10.0
        closest = int(np.argmin(np.maximum(pivots / band_pivot, band_pivot / pivots)))
        roman_magnitudes[band] = float(values[closest])
    return roman_magnitudes


CONVERSION_METHODS = {"sed": roman_magnitudes_by_sed, "nearest": roman_magnitudes_by_nearest}


def perturb_magnitudes(magnitudes, random_generator):
    """One draw of the published photometry from its quoted errors.

    Perturbing the input magnitudes and only then converting is what propagates the measurement
    error through the interpolation, which is not linear in magnitude."""
    return {
        name: (float(random_generator.normal(value, error)), error)
        for name, (value, error) in magnitudes.items()
    }


SPECTROPHOTOMETRIC_GREY_SIGMA = 0.05  # mag, achromatic term of the flux-scaling error
SPECTROPHOTOMETRIC_TILT_SIGMA = 0.06  # mag, half the R062 <-> F184 swing of the chromatic term


def _band_tilt_lever(bands=ALL_ROMAN_BANDS):
    """-1 at the bluest Roman band, +1 at the reddest, linear in log lambda_eff in between."""
    from kilonova.photometry.roman_noise import roman_bandpasses

    effective_wavelength = {band: roman_bandpasses()[band].effective_wavelength for band in bands}
    bluest = min(effective_wavelength.values())
    reddest = max(effective_wavelength.values())
    pivot = np.sqrt(bluest * reddest)
    half_range = np.log(reddest / bluest) / 2.0
    return {
        band: float(np.log(wavelength / pivot) / half_range)
        for band, wavelength in effective_wavelength.items()
    }


def perturb_synthetic_magnitudes(roman_magnitudes, random_generator):
    """One draw of the spectrophotometric calibration error of a synthetic magnitude.

    A magnitude integrated from a spectrum carries no measurement error of its own worth speaking
    of: propagating the per-pixel error column of the ENGRAVE spectra through the bandpass integral
    gives 0.0003-0.007 mag over the first five days, because thousands of pixels average down. What
    it does carry is the error of the flux calibration that put the spectrum on that scale, and the
    ENGRAVE release measures it for us: its README tabulates observed against synthetic magnitudes
    per epoch after the rescaling, and over the bands that were NOT used to constrain the fit
    (r, H, K) the residuals scatter by 0.08-0.17 mag -- consistent with the release's own claim of
    agreement "mostly better than 5 % (0.05 mag)". The same order shows up in the literature on
    synthetic photometry generally (Bessell & Murphy 2012; Gaia Collaboration, Montegriffo et al.
    2023, who find the nominal propagated uncertainties systematically underestimated).

    The draw is NOT six independent gaussians. Each ENGRAVE spectrum was corrected by a scaling of
    the form F -> a (1 + b lambda) F, so the error has one achromatic component shared by every band
    and one chromatic tilt across the wavelength range. A classifier that reads colours barely feels
    the first and is moved by the second, so collapsing both into per-band noise would misstate the
    result in either direction depending on the sign. Drawing (grey, tilt) once per realization
    reproduces the structure: the per-band sigma runs from SPECTROPHOTOMETRIC_GREY_SIGMA at the
    middle of the range to sqrt(grey^2 + tilt^2) = 0.08 mag at R062 and F184.
    """
    grey = random_generator.normal(0.0, SPECTROPHOTOMETRIC_GREY_SIGMA)
    tilt = random_generator.normal(0.0, SPECTROPHOTOMETRIC_TILT_SIGMA)
    lever = _band_tilt_lever()
    return {band: magnitude + grey + tilt * lever[band] for band, magnitude in roman_magnitudes.items()}


# ----------------------------------------------------------------------------- survey depth
def survey_depth_5sigma(tier_constants):
    """The 5 sigma magnitude of one visit per band, for a source faint enough to add no shot noise.

    Not the same quantity as the `mag_limit_5sigma` column of a window: that one is derived from the
    error of the observation it sits in, so for a bright source it is dominated by the source's own
    Poisson noise and comes out several magnitudes brighter than the survey depth. This is the depth
    proper, and it is what says whether Roman would have seen a given transient at all."""
    from kilonova.photometry.roman_noise import SNR_DETECTION, collecting_area_cm2

    depths = {}
    for band in tier_constants["bands"]:
        exposure = tier_constants["exposure_time"][band]
        zeropoint = tier_constants["zeropoint"][band]
        noise_floor = tier_constants["noise_floor_variance"][band]
        # flux / sqrt(flux + floor) = SNR  ->  flux = (SNR^2 + sqrt(SNR^4 + 4 SNR^2 floor)) / 2
        squared = SNR_DETECTION**2
        flux = (squared + np.sqrt(squared**2 + 4.0 * squared * noise_floor)) / 2.0
        depths[band] = float(zeropoint - 2.5 * np.log10(flux / exposure / collecting_area_cm2()))
    return depths


def expected_measurement(magnitude, band, tier_constants):
    """Lo que Roman mediria de una fuente de magnitud `magnitude`: (S/N, error de magnitud).

    Deterministico y sin jitter de zeropoint: es la esperanza de la medicion, no una realizacion.
    Sirve para tabular "esta fuente entra con este error" sin que el numero dependa de la semilla,
    que a S/N cercano al umbral mueve la magnitud observada varias decimas."""
    from kilonova.photometry.roman_noise import (
        flux_error_electrons,
        source_flux_electrons,
    )

    flux = source_flux_electrons(
        magnitude, tier_constants["exposure_time"][band], tier_constants["zeropoint"][band]
    )
    flux_error = flux_error_electrons(flux, tier_constants["noise_floor_variance"][band])
    signal_to_noise = flux / flux_error
    return float(signal_to_noise), float(1.0857 / signal_to_noise)


# ----------------------------------------------------------------------------- Roman window
def single_epoch_window(object_id, roman_magnitudes, redshift, tier_constants, noise_seed, parity=0):
    """One Roman visit of a source whose six true magnitudes are already known.

    The magnitudes are handed to `build_window_from_model` as a model that exists only in a narrow
    interval around the visit, so the following visits of the 5 d cadence fall outside it and come
    back unobserved. Everything the classifier reads about the instrument — which three of the five
    tier bands the visit covers, the zeropoint jitter, the flux realization, the S/N, the magnitude
    error and the 5 sigma limit — is produced by that function, exactly as for the training set.

    Returns None when the noise realization detects nothing, which is itself the answer for a source
    at the limit: Roman would not have a candidate to classify."""
    visit_phase = 0.0
    model = {}
    for band in tier_constants["bands"]:
        magnitude = roman_magnitudes.get(band, np.nan)
        if not np.isfinite(magnitude):
            continue
        model[band] = (np.array([visit_phase - 0.01, visit_phase + 0.01]), np.array([magnitude, magnitude]))
    if len(model) == 0:
        return None

    window = build_window_from_model(
        object_id,
        model,
        tier_constants,
        redshift,
        KN_GENTYPE,
        base_epochs=visit_phase + BASE_CADENCE_DAYS * np.arange(N_KN_VISITS),
        noise_seed=noise_seed,
        visit_index_offset=parity,
    )
    if window is None:
        return None
    window["tier"] = tier_constants["tier"]
    return window


# ----------------------------------------------------------------------------- classifier
def load_classifier(checkpoint_path, normalization_path, device):
    """The soup checkpoint plus the magnitude normalization fitted on the train split.

    The dataset reads the four constants off module globals, so they are injected the same way
    `run_evaluation_test_only.py` does it."""
    import openuniverse_data as openuniverse_data_module
    from openuniverse_data import GROUP_ORDER
    from train_lightning import LitKilonova

    with open(normalization_path) as normalization_file:
        normalization = json.load(normalization_file)
    openuniverse_data_module.MAG_MEAN = normalization["MAG_MEAN"]
    openuniverse_data_module.MAG_STD = normalization["MAG_STD"]
    openuniverse_data_module.SIGMA_MAG_MEAN = normalization["SIGMA_MAG_MEAN"]
    openuniverse_data_module.SIGMA_MAG_STD = normalization["SIGMA_MAG_STD"]
    classifier = LitKilonova.load_from_checkpoint(
        str(checkpoint_path), class_weights=torch.ones(len(GROUP_ORDER)), map_location=device
    )
    return classifier.to(device).eval(), normalization


def classify_windows(windows_frame, parquet_path, classifier, device, epochs, has_redshift, batch_size=4096):
    """P(KN) for every object in a long window frame, in one {epochs} x {redshift} regime.

    The frame goes through the live reader instead of a local tokenizer, so the tokens are the ones
    the model was trained on. Returns a frame of (object_id, kn_probability) — the caller owns the
    meaning of the id."""
    from openuniverse_data import OpenUniverseWindowDataset, _read_long_parquet, collate_token_windows
    from train_lightning import MODEL_INPUT_KEYS

    windows_frame.to_parquet(parquet_path, index=False)
    big, counts, meta = _read_long_parquet(parquet_path, lambda ids: np.asarray(ids, dtype=str))
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    meta["offsets"] = offsets
    object_ids = np.sort(windows_frame["object_id"].unique())  # the reader factorizes in sorted order

    loader = DataLoader(
        OpenUniverseWindowDataset(
            np.arange(len(counts)),
            big=big,
            meta=meta,
            label_by_index=np.ones(len(counts), dtype=np.int64),
            data_aug=False,
            force_epochs=epochs,
            force_redshift=has_redshift,
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_token_windows,
    )
    probabilities = []
    with torch.no_grad():
        for batch in loader:
            model_input = {key: value.to(device) for key, value in batch.items() if key in MODEL_INPUT_KEYS}
            probabilities.append(torch.softmax(classifier(model_input), dim=1)[:, 1].float().cpu().numpy())
    return pd.DataFrame(
        {"object_id": object_ids, "kn_probability": np.concatenate(probabilities).astype(float)}
    )
