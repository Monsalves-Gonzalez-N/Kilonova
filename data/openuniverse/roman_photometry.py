"""Roman HLTDS photometric-error recipe — single source of truth.

Holds the constants and primitives of the noise recipe validated in
``openuniverse_hdf5_lightcurve_error.ipynb`` so that the pipeline
(``generate_early_windows.py``) and the dataloaders import them instead of
redefining them. Keeping one copy avoids the three definitions drifting apart.

Scope: ONLY the photometric-error recipe (read noise, NEA, sky/thermal/dark
floor, zeropoint jitter, source flux, SNR, limiting magnitude). The cadence
scheduling differs per consumer (observed-MJD vs epoch-position) and the SED ->
AB-magnitude step live in their own modules/notebooks, so they stay out of here.

The CCD equation in electrons (Howell 1989, eq. 14), the same one that
reproduces Hourglass's fluxcal_err:

    sigma^2(F) = F_source + NEA * (B_sky + B_thermal + B_dark + sigma_read^2)

Everything is traceable to galsim.roman except the read noise (Rose et al. 2025,
eq. 9, up-the-ramp) and the PSF NEA (Rose et al. 2025, Table 3).
"""

import numpy as np
import galsim
import galsim.roman as roman

SNR_DETECTION = 5.0
ZP_JITTER_SIGMA = 0.15  # mag, FOV scatter of the zeropoint (Rose et al. 2025, eq. 8)
FIELD_SEED = 0          # reproducibility of the per-tier field choice

# --- Roman High Latitude Time Domain Survey (HLTDS): tiers Wide / Deep ---
# Total exposure time per EPOCH (s) from the HLTDS design (coadd/MA-table per epoch).
# HLTDS filter name -> galsim.roman band: F062=R062, F087=Z087, F106=Y106,
# F129=J129, F158=H158, F184=F184 (K213 unused in the HLTDS).
EXPOSURE_TIME_BY_TIER = {
    "wide": {"R062": 60.0, "Z087": 85.0, "Y106": 95.0, "J129": 152.0, "H158": 294.0},
    "deep": {"Z087": 193.0, "Y106": 294.0, "J129": 307.0, "H158": 420.0, "F184": 1636.0},
}
# Cadence (used by the consumers' scheduling): base visit every BASE_CADENCE_DAYS;
# the ANCHOR band (bluest of the tier) is observed every visit.
BASE_CADENCE_DAYS = 5.0
TIER_ANCHOR_BAND = {"wide": "R062", "deep": "Z087"}
ALL_BANDS_BY_WAVELENGTH = ["R062", "Z087", "Y106", "J129", "H158", "F184", "K213"]

# Real HLTDS field centers (RA/Dec deg). ELAIS-N1 does Wide and Deep; EDFS_a only Deep;
# EDFS_b only Wide. All near the ecliptic poles (low zodiacal). The center matters because
# getSkyLevel computes the zodiacal light from the field's ecliptic latitude.
HLTDS_FIELD_CENTER = {
    "ELAIS-N1": (242.50417, 54.51000),
    "EDFS_a":   (58.90000, -49.32000),
    "EDFS_b":   (63.60000, -47.60000),
}
HLTDS_FIELDS_BY_TIER = {"deep": ["ELAIS-N1", "EDFS_a"], "wide": ["ELAIS-N1", "EDFS_b"]}

# Exposure-dependent read noise (Rose et al. 2025, eq. 9): denominator n(n+1), NOT (n+1).
# Decreases with t_exp and saturates at the floor sqrt(25)=5 e- (up-the-ramp). Replaces the
# static roman.read_noise, which does not know the revised HLTDS exposure times.
READ_FRAME_TIME = 3.04             # s between non-destructive reads
READ_FLOOR_VARIANCE = 25.0         # e-^2 (floor sqrt = 5 e-)
READ_RAMP_VARIANCE = 12 * 16 ** 2  # = 3072 e-^2

# PSF NEA per band (pix), Hourglass (Rose et al. 2025) Table 3: median best/worst NEA across the
# FOV from the Roman WFI technical website PSF. Tabulated instead of roman.getPSF because the
# analytic Cycle-9 PSF of galsim overestimates the NEA ~15-28% in the blue bands (R/Z/Y) vs that
# PSF (H/F agree). Independent of tier and exposure.
PSF_NEA_PIX = {"R062": 5.575, "Z087": 6.695, "Y106": 7.895,
               "J129": 9.210, "H158": 11.140, "F184": 16.335}

ROMAN_BANDPASSES = roman.getBandpasses()
COLLECTING_AREA_CM2 = roman.collecting_area
ROMAN_ZEROPOINT = {band: ROMAN_BANDPASSES[band].zeropoint
                   for band in ALL_BANDS_BY_WAVELENGTH if band in ROMAN_BANDPASSES}


def read_noise_electrons(exposure_time):
    """sigma_read [e-/pix] of one epoch (Rose et al. 2025, eq. 9). Decreases with exposure
    toward the 5 e- floor (many reads); ~13 e- for the short WIDE exposures."""
    number_of_reads = exposure_time / READ_FRAME_TIME
    ramp_variance = READ_RAMP_VARIANCE * (number_of_reads - 1) / (number_of_reads * (number_of_reads + 1))
    return np.sqrt(READ_FLOOR_VARIANCE + ramp_variance)


def field_center_for_tier(tier, rng=None):
    """Pick at random an HLTDS field that observes this tier -> (name, RA, Dec) [deg]:
    deep -> {ELAIS-N1, EDFS_a}, wide -> {ELAIS-N1, EDFS_b}. `rng` fixes the choice."""
    if rng is None:
        rng = np.random.default_rng()
    field_name = str(rng.choice(HLTDS_FIELDS_BY_TIER[tier]))
    field_ra, field_dec = HLTDS_FIELD_CENTER[field_name]
    return field_name, field_ra, field_dec


def build_tier_constants(tier, field_seed=FIELD_SEED):
    """Per-tier constants of the error recipe derived from galsim.roman at the tier's exposure
    times: bands, AB zeropoint, NEA-weighted background noise floor, chosen field. The noise
    floor is the background term NEA*(sky + thermal*t + dark*t + read^2), fixed per band."""
    exposure_time = EXPOSURE_TIME_BY_TIER[tier]
    anchor_band = TIER_ANCHOR_BAND[tier]
    bands = [band for band in ALL_BANDS_BY_WAVELENGTH if band in exposure_time]
    field_name, field_ra, field_dec = field_center_for_tier(tier, np.random.default_rng(field_seed))
    field_world_position = galsim.CelestialCoord(field_ra * galsim.degrees, field_dec * galsim.degrees)

    zeropoint = {band: ROMAN_BANDPASSES[band].zeropoint for band in bands}
    noise_floor_variance = {}
    for band in bands:
        exposure = exposure_time[band]
        # getSkyLevel returns e-/arcsec^2 -> e-/pixel via pixel_scale^2 before summing per-pixel terms.
        sky_level = roman.getSkyLevel(ROMAN_BANDPASSES[band], world_pos=field_world_position,
                                      exptime=exposure) * roman.pixel_scale ** 2
        read_noise = read_noise_electrons(exposure)
        background_per_pixel = (sky_level + roman.thermal_backgrounds[band] * exposure
                                + roman.dark_current * exposure + read_noise ** 2)
        noise_floor_variance[band] = PSF_NEA_PIX[band] * background_per_pixel
    return {"tier": tier, "exposure_time": exposure_time, "anchor_band": anchor_band,
            "bands": bands, "zeropoint": zeropoint, "noise_floor_variance": noise_floor_variance,
            "field_name": field_name}


def source_flux_electrons(mag_true, exposure, zeropoint):
    """Source counts [e-] of an AB magnitude over `exposure` s. Scalar or array; the zeropoint
    may carry the per-epoch FOV jitter (Rose et al. 2025, eq. 8) that moves the SNR."""
    return exposure * COLLECTING_AREA_CM2 * 10 ** (-0.4 * (mag_true - zeropoint))


def flux_error_electrons(flux_electrons, noise_floor_variance):
    """sigma_flux [e-] = sqrt(source Poisson + NEA-weighted background floor). The flux term is
    the count itself (Poisson variance of N electrons is N); the floor is the per-band background."""
    return np.sqrt(flux_electrons + noise_floor_variance)


def limiting_magnitude_5sigma(flux_error, exposure, zeropoint, snr=SNR_DETECTION):
    """5-sigma limiting AB mag for a source whose error is `flux_error`, consistent with the
    (possibly jittered) zeropoint of that epoch."""
    limiting_rate = snr * flux_error / exposure / COLLECTING_AREA_CM2
    return zeropoint - 2.5 * np.log10(limiting_rate)


def epochs_from_first_detection(epoch_times, first_detection_time, number):
    """The first `number` visit times (sorted, unique) at or after `first_detection_time`. The single
    rule that defines the early window: from the first S/N>=SNR_DETECTION detection, take `number`
    consecutive epochs. A visit without any detection still counts. Used both on the regular visit
    grid (pipeline) and on the observed-epoch times (notebooks)."""
    epoch_times = np.sort(np.unique(np.asarray(epoch_times, dtype=float)))
    return epoch_times[epoch_times >= first_detection_time][:number]


def bands_observed_at_visit(visit_index, bands, anchor):
    """Filters observed at one visit, indexed 0,1,2,... from the first visit. The HLTDS cadence rule:
    the anchor band (bluest of the tier) is observed at every visit; the other bands (ordered by
    wavelength) split into two interleaved pairs by index parity, so band i is observed at visits whose
    index has parity i % 2 (0,2,4,... vs 1,3,5,...). => each visit observes anchor + 2 bands. This is
    the single source of the cadence; both the per-visit and the per-band views below derive from it."""
    others = [band for band in bands if band != anchor]
    observed = [anchor]
    for band_position, band in enumerate(others):
        if visit_index % 2 == band_position % 2:
            observed.append(band)
    return observed


def cadence_schedule(visit_times, bands, anchor):
    """{band: array of observed visit times} over the regular visit grid `visit_times` (one every
    BASE_CADENCE_DAYS), applying bands_observed_at_visit per visit index: anchor at all visits, each
    other band every other visit. Inverse view of the same rule, for the per-band sampling loops."""
    visit_times = np.asarray(visit_times, dtype=float)
    observed_times = {band: [] for band in bands}
    for visit_index, visit_time in enumerate(visit_times):
        for band in bands_observed_at_visit(visit_index, bands, anchor):
            observed_times[band].append(visit_time)
    return {band: np.asarray(times, dtype=float) for band, times in observed_times.items()}


def first_epochs_since_detection(photometry, number, time_column="mjd"):
    """Subset of `photometry` covering the first `number` epochs since the first S/N>=SNR_DETECTION
    detection in any band (epochs_from_first_detection on the observed epoch times). Empty frame if the
    object is never detected. `time_column` is the epoch axis: 'mjd' (observed-MJD path) or 'dt'
    (synthetic-cadence path). Pure selection — plotting stays in each notebook."""
    detected_times = photometry.loc[photometry["detected"], time_column]
    if len(detected_times) == 0:
        return photometry.iloc[0:0]
    epoch_times = epochs_from_first_detection(photometry[time_column].to_numpy(),
                                              detected_times.min(), number)
    return photometry[photometry[time_column].isin(epoch_times)].sort_values([time_column, "band"])
