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
inspected and thrown away without touching the training path. What the training path does know is
how to READ one: `training/openuniverse_data.py` takes the two `izc_windows_{tier}.parquet` as
optional inputs and reads the split group off the `object_id`, which is where the parent it was
re-rendered from is written down.

WHAT AN OBJECT OF THIS SAMPLE IS. Not a draw from a luminosity function: a real OpenUniverse
object, re-rendered at a redshift it did not have. `openuniverse_parents.read_parent_catalog`
supplies the parents -- 1 323 089 of them over the 33 healpix -- and a re-rendered object inherits
from its parent everything the release records about it:

    inherited                    from
    --------------------------   -------------------------------------------------------------
    class                        `gentype`, with SN IIP / SN IIL resolved from the template
    spectral model               `template_index`: into the 44 core-collapse templates
                                 OpenUniverse drew, or into the 919-row SN Iax bank it drew, or
                                 the SALT source it used for SN Ia
    shape and colour (SN Ia)     `salt2_x1`, `salt2_c`
    host dust screen             `AV`, `RV` -- present only where OpenUniverse put one
    brightness                   MEASURED off the parent's own light curve, see below

Three things are drawn here and nothing else: the redshift, the phase of the survey's visit grid,
and the parity of the first visit. That is the whole point of the sample. The object is the same
object; only the distance is one the survey does not have an object at.

THE SPECTRAL MODEL IS ALWAYS THE PARENT'S OWN, and the sample's scope is drawn to keep it that
way. Core-collapse (SN Ib, SN Ic, SN IIP, SN IIL) are OpenUniverse's own templates, read out of the
release; SN Ia is SALT, which is what OpenUniverse itself drew from. There is no class left whose
SED this side supplies -- see NO SUBSTITUTIONS below for the two that were dropped to get here.

HOW THE BRIGHTNESS IS MEASURED, which is what replaces every luminosity function this module used
to carry. For a parent at z_parent the generator renders the PARENT'S OWN model at z_parent,
normalised to an arbitrary reference absolute magnitude, and takes the peak magnitude of each Roman
band over the same rest-frame phase window the parent's light curve is read over. The median over
the bands both sides cover,

    offset = median_band [ m_parent(band) - m_rendered(band | M = REFERENCE_ABSOLUTE_MAGNITUDE) ]

is the parent's own absolute magnitude expressed in units of the reference, and rendering that same
model at the DRAWN redshift with M = REFERENCE_ABSOLUTE_MAGNITUDE + offset is the object at its new
distance. The reference cancels exactly -- it enters both renders identically and only their
difference survives -- so no photometric system, no normalisation band and no phase-zero convention
has to be argued about. The three per-class conventions that carried those arguments
(`PEAK_ABSOLUTE_MAGNITUDE_BAND`, `iax_phase_zero_offset`, `tde_peak_absolute_magnitudes`) are gone
with them, and so is the per-class brightness calibration against OpenUniverse's median M(Y106):
a per-object measurement cannot be off by a class-wide offset, which is what that calibration
existed to remove.

WHY A MEDIAN OVER BANDS AND NOT ONE BAND. Which bands are usable is a function of z_parent, so no
single band serves the catalogue: the median parent sits at z = 1.5, where the blue Roman bands
sample rest-frame ultraviolet the templates do not reach. F184 is covered at every redshift and
R062 only below z = 0.6. The median over the covered ones also puts a band-to-band disagreement
between the template and the parent into the SPREAD of the offset rather than into the offset, and
that spread is recorded per object (`brightness_residual`) instead of being averaged away.

WHAT THE MEASUREMENT DOES NOT FIX, and must not be read as fixing: the colour. One number per
object cannot move the shape of an SED. Where the spectral model is the parent's own the rendering
and the parent differ only by this pipeline, and `brightness_residual` measures that difference
object by object. That is now the whole of it, because the two classes whose colour was the
substitution's rather than the parent's are no longer generated. It is a diagnostic and adjusts
nothing.

THE PARENTS ARE THE GENERATED POPULATION, NOT THE DETECTED ONE, and that is a trap this module
used to fall into. The catalogues hold 1 323 089 objects and the early windows hold 880 000: the
difference is the survey's detection cut. Fitting a distribution to the detected half imports its
Malmquist selection into a sample generated at redshifts that have none -- the SN Ia shape and
colour of this module were once fitted that way, and came out as the blue, broad end of the
population, worth 0.03 mag of colour and 0.02 of shape in the largest class of the sample. Drawing
parents from the catalogue removes the question rather than answering it.

The redshift distribution is not chosen here at all: `draw_population_from_parents` takes it from
its caller, and `kn-izc-windows` supplies the deficit of `redshift_deficit`.

NO SUBSTITUTIONS, and that is a scope decision rather than an achievement. The sample exists to
REPOPULATE the low-redshift bins of a population OpenUniverse already defines. A class whose SED
this side has to supply is not being moved in redshift, it is being added, and the sample would
then have to be defended as a model of that class rather than as a redistribution of
OpenUniverse's. So three kinds of class are out:

  * TDE. OpenUniverse's is the observed SED of AT2019qiz, never published as a usable template.
    MOSFiT's `tde` stood in for it and reproduced OpenUniverse's own photometry of its own objects
    to 0.19 mag, against 0.01 mag for the classes that remain, with a clear colour trend.
  * SN Iax. OpenUniverse's own model, but REGENERATED here from the published Rutgers notebook
    rather than read from the release -- the one class where the release ships no template this
    side can read. The regeneration is faithful enough to replay the notebook's own numbers and
    still lands 0.05 mag off OpenUniverse's photometry, which is the reimplementation showing.
  * SLSN-I and PISN (0.39 % of the mix), and every gentype `openuniverse_parents.PARENT_GENTYPES`
    does not list, for the older reason that they have no usable template on this side at all.

The models for the first two are still in this file, with their provenance and their tests, and
nothing draws them: putting one back means adding its label to CLASS_FRACTION and its gentype to
PARENT_GENTYPES, and re-opening the argument above.

There used to be a substitution larger than either, the core-collapse SEDs. It is gone -- those
templates are now read out of OpenUniverse's own release, see OPENUNIVERSE_ARCHIVE_PATH -- and the
limitation it carried is gone with it. Where this module once put the SNANA and Nugent libraries
below z = 0.91 and OpenUniverse's V19+HostXT above, which is a difference aligned with redshift and
therefore the exact shape of the shortcut this sample exists to remove, both halves are now the
same 44 templates.

The near-infrared extrapolation those templates carry is OpenUniverse's own and is shared by both
populations rather than added by this one: `_WAVEEXT` is an extension by the methods of Pierel et
al. (2018), and every OpenUniverse contaminant above z = 0.91 already rests on it. What remains
worth measuring is whether these models reproduce real supernovae at the redshifts this sample
generates, and that is measured rather than declared: `scripts/compare_csp_lightcurves.py` puts
them, continuous, under the discrete uBgVriYJH photometry of the Carnegie Supernova Project. Every
class here now has a CSP counterpart, the TDE having been the one that did not.
"""

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from kilonova.photometry.roman_noise import (
    BASE_CADENCE_DAYS,
    PSF_NEA_PIX,
    build_tier_constants,
    collecting_area_cm2,
)
from kilonova.photometry.spectra import ALL_ROMAN_BANDS, spectrum_to_roman_magnitudes
from kilonova.simulation import openuniverse_parents
from kilonova.simulation.early_windows import (
    CADENCE_PARITY_PERIOD,
    GENTYPE_LABEL,
    build_window_from_model,
)

# --- Cosmology ----------------------------------------------------------------------------------
# OpenUniverse's, measured off its own catalogue rather than taken from its paper, which quotes
# {Om, w0, wa} = {0.315, -1, 0} for the survey but not the cosmology SNANA was run with.
#
# The 224 118 SNe Ia of the 33 healpix carry SALT3 parameters, and SNANA generates their brightness
# deterministically: mB + alpha*x1 - beta*c - gammaDM = M0 + mu(z), with alpha = 0.15 and
# beta = 3.1 for every one of them. Fitting that against a flat LambdaCDM distance modulus over
# z = 0.027 to 2.98 leaves a residual scatter of 0.0001 mag -- an exact recovery, not a fit --
# and gives Om0 = 0.2650, which is the Outer Rim cosmology the simulation is built on. H0 and M0
# are degenerate in that fit; H0 = 70 gives M0 = -19.3634, which is SNANA's canonical -19.36 to
# 0.003 mag, so that is the pair adopted here.
#
# Why it matters. mu_OU - mu_Planck18 runs from -0.072 mag at z = 0.02 to -0.039 at z = 0.50, so
# on Planck18 an izc object of a given absolute magnitude reached the window 0.04 to 0.07 mag
# fainter than an OpenUniverse object of the same absolute magnitude at the same redshift. For the
# CALIBRATED classes that is absorbed into their offset and only the offset was wrong; for the
# three classes that carry their own brightness -- SN Ia from SALT, SN Iax from the bank, TDE from
# MOSFiT -- nothing absorbed it and the apparent magnitudes themselves were wrong. That is half the
# sample, and apparent magnitude is the only thing the classifier ever sees.
#
# Where it does NOT show up, which is the trap: in the calibration offsets themselves. That
# statistic subtracts the same distance modulus from both sides, and the detection cut then moves
# both medians the same way, so switching cosmology moves the measured SN Iax offset by 0.002 mag.
# The offsets cannot be used to diagnose the cosmology and a converged offset does not mean the
# cosmology is right.
OPENUNIVERSE_H0 = 70.0
OPENUNIVERSE_OM0 = 0.2650


@cache
def openuniverse_cosmology():
    from astropy.cosmology import FlatLambdaCDM

    return FlatLambdaCDM(H0=OPENUNIVERSE_H0, Om0=OPENUNIVERSE_OM0)


# gentype of the OpenUniverse class this stands in for, plus the offset, so an izc object is always
# separable from a real one in the parquet while `label` still maps it to the same class.
IZC_GENTYPE_OFFSET = 200

# --- SN Iax (RETAINED, NOT GENERATED) ------------------------------------------------------------
# Not in CLASS_FRACTION and not in PARENT_GENTYPES: this is the one class where OpenUniverse ships
# no template this side can read, so its SED has to be regenerated here rather than moved, and the
# regeneration lands 0.05 mag off OpenUniverse's own photometry. See NO SUBSTITUTIONS in the module
# docstring. Everything below is kept, with its tests, so that re-enabling it is a two-line change
# and not a re-derivation.
# NOT a substitution: this is OpenUniverse's own model, regenerated. OpenUniverse used "1000 SED
# templates from the original PLAsTiCC model" (OpenUniverse2024, sec. 4.2), which is Jha & Dai's
# model released at github.com/RutgersSN/SNIax-PLAsTiCC (commit 908762c, BSD-3): a single SN 2005hk
# base SED plus a luminosity function and width-luminosity relations, warped into a bank of
# templates by the repository's `Iax-model.ipynb`. `data/iax/sn2005hk_sed.npz` is that base SED,
# repacked from `SN2005hk/sn2005hk.sed.dat`, and `data/iax/jha2017-table1.csv` is the repository's
# own copy of the Jha (2017) Table 1.
#
# THE BANK IS REPLAYED, NOT RESAMPLED. The notebook is deterministic -- `np.random.seed(4)`, then
# three random-consuming steps in a fixed order -- so `_iax_bank()` reproduces its exact 1001 rows
# of (M_V, t_rise, dm15B, dm15R) rather than drawing new ones from the same distributions. This
# module used to sample the luminosity function continuously and then draw the shape parameters
# around the drawn magnitude, which is the same distribution but not the same objects, and it made
# every izc SN Iax an object OpenUniverse never generated.
#
# `data/iax/sn2005hk_sed.npz` is byte-for-byte the notebook's own input: the notebook loads
# `sn2005hk.source.pickle`, and flux ratios of that pickle against this npz are 1.000000 at every
# phase and wavelength (1st to 99th percentile).
#
# ONLY THE FIRST 919 ROWS SHIPPED. The notebook draws `ssize = 1001`, but OpenUniverse's catalogue
# carries `template_index` 1 to 919, contiguous and with no gaps, over all 115 645 of its SNe Iax.
# The mapping is `template_index - 1` into the replayed bank, verified rather than assumed: against
# the dust-corrected peak absolute LSST-g magnitude of the 1923 OpenUniverse SNe Iax below z = 0.45
# the replayed M_V gives slope +0.990 and correlation +0.983, where the same bank shuffled gives
# -0.022. So a draw is a uniform index into 1..919, and OpenUniverse's own draws are uniform over
# the redshifts this module generates (chi2/dof = 1.18 below z = 0.45; the non-uniformity above
# z = 1.5 is Malmquist selection, not a weight).
IAX_SOURCE_NAME = "iax-2005hk"
IAX_BASE_SED_PATH = Path(__file__).resolve().parents[3] / "data" / "iax" / "sn2005hk_sed.npz"
IAX_JHA_TABLE_PATH = Path(__file__).resolve().parents[3] / "data" / "iax" / "jha2017-table1.csv"
IAX_BANK_SEED = 4
IAX_BANK_DRAWN = 1001  # what the notebook draws
IAX_BANK_SIZE = 919  # what OpenUniverse shipped, and what a draw indexes

# Luminosity function: linear in M_V with a Gaussian rolloff at each end, fitted by Jha & Dai to
# the 51 SNe Iax of their Table 1. Deliberately NOT a Gaussian -- the class spans M_V = -13 to -18
# and a symmetric law would misplace both ends. It is used to BUILD the bank, never to draw from.
IAX_ABSOLUTE_MAGNITUDE_RANGE = (-20.0, -11.0)
IAX_ABSOLUTE_MAGNITUDE_STEP = 0.01
IAX_BRIGHT_ROLLOFF, IAX_BRIGHT_SIGMA = -18.0, 0.5
IAX_FAINT_ROLLOFF, IAX_FAINT_SIGMA = -13.0, 0.4
# The notebook inverts the CDF onto this grid and then interpolates the draws through it, a double
# interpolation that has to be reproduced step for step or the replay drifts off the published bank.
IAX_INVERSE_CDF_STEP = 0.0001
# Cell 6 fits the observed sample by rejection against a magnitude limit. Its result is never used,
# but it consumes 51 rejection loops' worth of random numbers between the magnitudes and the shape
# parameters, so the replay has to run it to stay on the notebook's random stream.
IAX_REJECTION_MAGNITUDE_LIMIT = 20.3
IAX_REJECTION_COSMOLOGY = (73.0, 0.3)  # H0, Om0, as the notebook sets them

# Width-luminosity relations, all in terms of (M_V + 19). Rise time follows Magee et al. (2016);
# the two decline rates are Jha & Dai's own fits. The scatter is theirs too and is what keeps the
# class from collapsing onto one light-curve shape.
IAX_RISE_TIME_SCATTER = 2.0
IAX_DECLINE_B_SCATTER = 0.25
IAX_DECLINE_R_SCATTER = 0.2
IAX_BASE_RISE_TIME = 15.0  # hardcoded in the SN 2005hk SED, per the notebook

# The notebook's own grid, and this module now builds the base SED on it rather than on the
# repacked file's 81 phases from -15 to +65. That is not cosmetic: three of the notebook's four
# processing steps are written as INDICES into this grid, and on a shorter grid they land on
# different phases. `flux[10:]` is phase -20 here and was phase -5 on the old grid; `flux[70:]` is
# phase +40; and the pre-explosion suppression of `iax_source` is a strict `<` that selects a real
# 15-day region here and selected nothing at all when the grid began at -15.
#
# The grid runs to +200 d but the SED does not. sncosmo returns zero outside the base file's own
# -15 to +65, so everything past +65 is the smoothing of that edge and peaks at 4e-16 -- the late
# extension is a decay to zero, not data. It is reproduced because OpenUniverse's templates carry
# it, not because it holds any flux.
IAX_NOTEBOOK_PHASES = np.linspace(-30.0, 200.0, 231)
IAX_NOTEBOOK_WAVELENGTHS = np.linspace(1000.0, 25000.0, 2401)
IAX_SMOOTHING_SIGMA_PHASES = 4.0
IAX_SMOOTHED_WAVELENGTHS = (8000.0, 11000.0)  # z and y, "too wiggly" in the notebook's words
IAX_SMOOTHED_FROM_INDEX = 10  # phase -20
IAX_LATE_DECLINE_PHASE = 50.0
IAX_LATE_DECLINE_EFOLD = 72.4  # d; enforces 0.015 mag/day past IAX_LATE_DECLINE_PHASE
IAX_LATE_SMOOTHED_FROM_INDEX = 70  # phase +40

# The dm15(R) warp is anchored at five wavelengths and has to be spread over the full grid. The
# notebook used `scipy.interpolate.interp2d`, removed from SciPy in 1.14; `RectBivariateSpline`
# with kx = ky = 1 is SciPy's own documented replacement for it.
#
# What happens redward of the last anchor is the question, because for Roman that is most of the
# bands. This module carried a comment claiming interp2d extrapolated the 7500-9000 A slope out to
# 25000 A and that the replacement therefore changed the model. It does not: interp2d's
# out-of-domain behaviour was nearest-neighbour, which is clamping, and RectBivariateSpline clamps
# too. The two agree to 2e-16 on this module's own anchors, and so did the `np.interp` per-row
# version this replaced. The ramp is flat past 9000 A in the notebook, in OpenUniverse's templates
# and here, and no version of this module ever differed on it.
IAX_WARP_ANCHORS_AA = (2350.0, 5200.0, 6000.0, 7500.0, 9000.0)
IAX_WARP_KERNEL_CENTRE_DAYS = 15.0
IAX_WARP_KERNEL_SIGMA_DEX = 0.2
IAX_WARP_R_GAIN = 1.095  # the notebook's own factor, which keeps dm15(B) where the first warp put it
IAX_PRE_EXPLOSION_SUPPRESSION = 2000.0

# Host extinction: the PARENT'S OWN screen, `AV` and `RV` straight out of the catalogue. SN Iax is
# the one class OpenUniverse dusts by hand -- RV is 3.1 for all 115 645 of them and AV runs between
# SNANA's generation limits of 0.001 and 3.0 -- and this module used to reproduce that distribution
# by fitting it, which is now unnecessary: the object being re-rendered carries the screen it was
# given. See the host extinction block below for why no other class gets one.

# --- TDE (RETAINED, NOT GENERATED) ---------------------------------------------------------------
# Not in CLASS_FRACTION and not in PARENT_GENTYPES: MOSFiT standing in for an SED OpenUniverse never
# published is an added source rather than a moved one, and it measured 0.19 mag off with a colour
# trend. See NO SUBSTITUTIONS in the module docstring. Kept below, with its tests, for the same
# reason as SN Iax.
# OpenUniverse took its TDE from the observed SED of AT2019qiz, which is not published as a usable
# template, so this is a declared substitution: the MOSFiT `tde` model of Guillochon et al. (2018),
# the same model Hourglass uses. It is also the safest extrapolation in this module. Every other
# class reaches the rest-frame near-infrared through a template that was extended there by fiat;
# the MOSFiT TDE is a blackbody photosphere, so at these wavelengths it is the Rayleigh-Jeans tail
# of a body whose temperature the model tracks -- nothing is being extrapolated at all.
#
# `data/tde/mosfit_tde_photospheres.npz` holds only the photosphere history (phase, temperature,
# radius, luminosity for each of the 227 draws that survive the population cut below), not a sampled
# spectrum: the Planck function is built from it on demand, which turns hundreds of megabytes of
# SED cube into 113 kB. Regenerated by
# `scripts/build_tde_templates.py`, which has to run in a separate environment -- MOSFiT pins
# numpy <= 1.26.4 and cannot be installed next to the pipeline.
#
# The draws are NOT MOSFiT's priors as they come. Those are fitting priors, deliberately
# uninformative: drawn blind they put the peak photosphere temperature anywhere between 5e2 and
# 1e6 K, and only a third of them land where TDEs are actually observed. The physics is MOSFiT's;
# the population is cut to the observed peak temperature range of optically selected TDEs from
# van Velzen et al. (2021), which the build script applies and records.
TDE_SOURCE_NAME = "mosfit-tde"
TDE_TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "data" / "tde" / "mosfit_tde_photospheres.npz"
# A blackbody has no structure to resolve, so the grid only has to be fine enough for the band
# integration, and wide enough to cover R062 through F184 at the lowest redshift generated.
# 100 A rather than the 20 A this started at: measured against the 20 A grid the six Roman
# magnitudes move by less than 1e-7 mag, and the coarser grid is what makes caching the whole
# template bank affordable (see `tde_source`).
TDE_WAVELENGTH_LIMITS = (1000.0, 30000.0)
TDE_WAVELENGTH_STEP = 100.0

# --- SN Ia -------------------------------------------------------------------------------------
# One SALT source at every redshift, which took a measurement to get to.
#
# OpenUniverse's SN Ia model is SALT3 extended into the near-infrared -- Pierel et al. (2022), which
# sncosmo ships as `salt3-nir`. Using it matters: against the 68 SN Ia OpenUniverse has below
# z = 0.1, the median Z087-Y106 of a matched sample sits 0.180 mag blue of them with
# `salt2-extended` and 0.080 mag blue with `salt3-nir`, so the switch removes more than half of a
# systematic colour offset in the largest single class of the sample.
#
# `salt3-nir` stops at 20000 A rest-frame and the red edge of F184 is at 21000 A, so it covers the
# band only above z = 0.05 exactly. This module used to fall back to `salt2-extended` below that,
# which reaches 24990 A -- one class carrying two spectral models, split at a redshift.
#
# `scripts/compare_ia_salt_sources.py` measures what that fallback costs, by putting both sources
# under the same CSP-I photometry with the same drawn population: same x1, same colour, same drawn
# peak absolute magnitude, so the two models are the same supernova and differ only in the shape
# that carries it into the near-infrared. On the decline rate m(+15 d) - m(max) -- which no
# distance and no host dust can move -- `salt2-extended` is too slow by 0.217, 0.242 and 0.351 mag
# in Y, J and H, and `salt3-nir` by 0.077, 0.035 and 0.134. Every supernova improves: 14 of 14 in
# Y, 12 of 12 in J, 9 of 9 in H.
#
# So the fallback is removed and `salt3-nir` is held flat over the 1000 A it is missing. That is an
# extrapolation, and it is bounded: F184 is a filter, not a top hat, and its throughput is already
# below 1 % of peak past 2054 nm. The fraction of the F184 photon response redward of 20000 A
# rest-frame is 0.143 % at z = 0.02 -- the reddest rest-frame this module ever samples -- 0.013 %
# at z = 0.03 and zero above z = 0.05. Measured through this module's own path, deleting that
# sliver outright moves the peak F184 magnitude by 0.0013 mag at z = 0.02 and 0.0001 mag at
# z = 0.03. What goes in it matters even less: held flat against continued along the fall
# `salt2-extended` shows over those 1000 A -- about 12 % -- the two differ by 0.00001 mag. Flat,
# because flat is the one rule that cannot amplify a slope read off the edge of a spline.
#
# The extrapolation is confined to the SED. `spectrum_to_roman_magnitudes` still reads coverage off
# the nominal band edge and is untouched, and so is the coverage test that forbids extrapolation --
# the padded source passes it the way any other source does, by reaching 21000 A.
IA_BASE_SOURCE_NAME = "salt3-nir"
IA_SOURCE_NAME = "salt3-nir-f184"
IA_PAD_WAVELENGTH = 21000.0  # A rest-frame; the red edge of F184, reached at z = 0

# THE CORE-COLLAPSE LIBRARY, read out of OpenUniverse's own light-curve release rather than
# reconstructed. `scripts/build_openuniverse_cc_templates.py` builds the archive and its docstring
# carries the provenance; this block carries what a reader of the module needs.
#
# OpenUniverse drew its core-collapse SEDs from `NON1ASED.V19_CC+HostXT_WAVEEXT`. The base half of
# that name is public and the `_WAVEEXT` half -- the near-infrared extension -- is not, and 11000 A
# rest-frame covers Roman only above z = 0.91, the opposite of the range this sample fills. The
# previous answer was to redo the extension with `snsedextend`, the public implementation of the
# method OpenUniverse cites. Measured against OpenUniverse's own photometry that failed: it made a
# comb of one hump per photometric anchor separated by runs of zero flux, worth a factor 0 to 3.7
# against theirs over 11000-20600 A, and a colour that swung 2 mag across the sample's redshift
# range where OpenUniverse's is flat.
#
# None of that reconstruction was necessary, because the templates are IN the release. Each
# object's hdf5 group carries `flambda`, the model SED itself, on a fixed observer-frame grid of
# 1850-24450 A, and every object of one `template_index` is one rest-frame template at a different
# redshift -- so dividing by (1 + z) recovers it. Below z = 0.187 an object reaches 20600 A rest,
# which is F184's red edge at this sample's own z = 0.02 floor.
#
# THREE MEASUREMENTS SAY THIS IS SOUND, and they are why the archive is trusted rather than fitted:
#   * Objects of one template at z = 0.081, 0.991 and 1.501, de-redshifted and normalised, agree to
#     0.5 % median over 2500-8000 A -- across a factor 15 in (1 + z).
#   * Over the 44 templates, the 10 to 12 independent objects combined into each agree to between
#     0.04 % and 1.05 %. That is the check that the de-redshifting and the phase alignment held.
#   * The PUBLIC pycoco base reproduces the extracted template to 1.3-1.4 % over 3000-10000 A on
#     the templates whose phase convention lines up. It is the control on everything else: the
#     library identification, the flux convention and the de-redshifting are all confirmed by it,
#     and what was wrong was only ever our own extension.
#
# WHAT THE ARCHIVE HOLDS: 44 templates, 3000-20600 A rest-frame, and a phase grid PER TEMPLATE.
# The phase axis is not shared because `peak_mjd` is OpenUniverse's own per-object peak, and for a
# SN II that sits at the start of the plateau, essentially at explosion -- so there is almost no
# light curve before it. Every SN Ib reaches -12 d; SN IIP and SN IIL reach a median of -3 and some
# only 0. A window shared by all of them would have to start at 0 and throw away the rise for the
# classes that have one. THIS IS A LIMITATION OF THE SIMULATION, NOT OF THE PIPELINE, and it
# belongs in the paper: a SN II contaminant reaches the classifier without its rise because
# OpenUniverse's model does not have one.
OPENUNIVERSE_ARCHIVE_PATH = Path(__file__).resolve().parents[3] / "data" / "openuniverse" / "cc_templates.npz"
OPENUNIVERSE_SOURCE_PREFIX = "ou-"


# Sources vetted to cover R062 through F184 with no extrapolation at z = 0.02, the bluest and
# reddest edges included. tests/test_intermediate_z_contaminants.py re-checks this against sncosmo
# rather than trusting the list.
#
# The core-collapse entries are OpenUniverse's OWN templates, read out of its light-curve files by
# `scripts/build_openuniverse_cc_templates.py`; see the OPENUNIVERSE_ARCHIVE_PATH block above.
#
# THEY ARE 44, AND THAT IS EVERY ONE OPENUNIVERSE DREW -- the count is not a survival rate. Nothing
# was cut here: the release's own `template_index` says which of the library's templates each
# object used, and across 979 557 core-collapse objects exactly 44 appear, 17 SN IIP, 7 SN IIL,
# 13 SN Ib and 7 SN Ic. That is 24 II + 13 Ib + 7 Ic, the 24/13/7 the OpenUniverse paper reports.
#
# SN IIn IS GONE, AND SO IS SN IIb, and it is the same fact about the simulation rather than two
# decisions of ours: the V19 library carries templates of both SNTYPE 21 (IIn) and 23 (IIb), and
# OpenUniverse drew NEITHER. Keeping a SN IIn class would have meant generating a contaminant class
# the population this sample stands in for does not contain. Its only source had also been
# `nugent-sn2n`, whose structure measured against a 500 A running median is 0.000000 at every
# wavelength -- a featureless continuum standing in for a class defined by its narrow Balmer
# emission. This also settles, without a judgement call, the question of whether to add SN IIb.
SOURCES_BY_LABEL = {
    # SN II split by subtype rather than pooled, because the library separates them by its own
    # SNTYPE (20 against 22) and it is OpenUniverse's output catalogue that pools them into gentype
    # 32. The split is what lets a parent of gentype 32 be resolved into the subtype its template
    # says it is, which is the finer label the parquet carries in `izc_subtype`.
    "SN IIP": [
        "ou-ASASSN14jb",
        "ou-SN1987A",
        "ou-SN1999em",
        "ou-SN2004et",
        "ou-SN2005cs",
        "ou-SN2008bj",
        "ou-SN2008in",
        "ou-SN2009N",
        "ou-SN2009bw",
        "ou-SN2009ib",
        "ou-SN2012A",
        "ou-SN2012aw",
        "ou-SN2013ab",
        "ou-SN2013am",
        "ou-SN2013fs",
        "ou-SN2016X",
        "ou-SN2016bkv",
    ],
    "SN IIL": [
        "ou-ASASSN15oz",
        "ou-SN2007od",
        "ou-SN2009dd",
        "ou-SN2009kr",
        "ou-SN2013by",
        "ou-SN2013ej",
        "ou-SN2014G",
    ],
    "SN Ib": [
        "ou-SN1999dn",
        "ou-SN2004gq",
        "ou-SN2004gv",
        "ou-SN2005bf",
        "ou-SN2005hg",
        "ou-SN2006ep",
        "ou-SN2007Y",
        "ou-SN2007uy",
        "ou-SN2008D",
        "ou-SN2009iz",
        "ou-SN2009jf",
        "ou-SN2012au",
        "ou-iPTF13bvn",
    ],
    "SN Ic": [
        "ou-SN1994I",
        "ou-SN2004aw",
        "ou-SN2004fe",
        "ou-SN2004gt",
        "ou-SN2007gr",
        "ou-SN2011bm",
        "ou-SN2013ge",
    ],
    # Not a registry name until `_sncosmo()` registers it either; see IA_SOURCE_NAME.
    "SN Ia": [IA_SOURCE_NAME],
    # Not a registry name until `_sncosmo()` registers it; see IAX_SOURCE_NAME.
    "SN Iax": [IAX_SOURCE_NAME],
    # Likewise, and a stand-in for the whole bank: `build_model` builds the template the parent's
    # own id selects.
    "TDE": [TDE_SOURCE_NAME],
}

# UNIFORM BY CLASS, ON PURPOSE. This is not OpenUniverse's mix and is not meant to be: it is a
# training-set choice, the same kind of choice as the redshift distribution
# `draw_population_from_parents` takes from its caller, and a reader who takes it for a rate-based
# population will misread the sample.
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
UNIFORM_CLASS_SHARE = 1.0 / 4.0
# Measured off OpenUniverse's own template set rather than taken from Li et al. (2011): of the 24
# SN II templates it drew, 17 are SNTYPE 20 and 7 are SNTYPE 22. The volumetric split this used to
# carry was a stand-in for exactly this number, and the number is now readable.
SN_II_SUBTYPE_FRACTION = {"SN IIP": 17.0 / 24.0, "SN IIL": 7.0 / 24.0}
#
# SN Iax and TDE are NOT here, and that is the sample's scope rather than an omission: every class
# it generates has to be one whose spectral model is OpenUniverse's own, so that the sample is a
# redistribution of OpenUniverse's population in redshift and nothing else. See PARENT_GENTYPES in
# `openuniverse_parents` for what each of the two would have cost, and the SN Iax and TDE blocks
# above for the models themselves, which are kept but no longer drawn.
CLASS_FRACTION = {
    "SN IIP": UNIFORM_CLASS_SHARE * SN_II_SUBTYPE_FRACTION["SN IIP"],
    "SN IIL": UNIFORM_CLASS_SHARE * SN_II_SUBTYPE_FRACTION["SN IIL"],
    "SN Ia": UNIFORM_CLASS_SHARE,
    "SN Ic": UNIFORM_CLASS_SHARE,
    "SN Ib": UNIFORM_CLASS_SHARE,
}

# The subtypes all map back to OpenUniverse gentype 32, the class they stand in for. SN Iax (12)
# and TDE (42) are kept here: the models still exist and a caller that re-enables one needs its
# gentype, and nothing reads this map for a label CLASS_FRACTION does not carry.
GENTYPE_BY_LABEL = {
    "SN Ia": 10,
    "SN Iax": 12,
    "SN Ib": 21,
    "SN Ic": 26,
    "SN IIP": 32,
    "SN IIL": 32,
    "TDE": 42,
}

# The one number a re-rendered object needs that its parent's catalogue row does not carry: the
# brightness. It is measured per object -- see the module docstring -- and this is the reference the
# measurement is expressed against.
#
# ITS VALUE IS ARBITRARY AND CANCELS. It normalises the render at the parent's redshift, from which
# the offset is measured, and it normalises the render at the drawn redshift, to which the offset is
# added back; the two renders are the same model and the reference enters both identically. -19.4 is
# chosen only so that a printed `peak_absolute_magnitude` reads as a plausible absolute magnitude
# for a supernova rather than as an offset from nothing.
REFERENCE_ABSOLUTE_MAGNITUDE = -19.4
# Rest-frame B on AB, for every class alike. The old per-class normalisation band existed because a
# drawn luminosity function had to be applied in the band it was published in; nothing is drawn any
# more, and the band a reference is applied in cancels with the reference.
REFERENCE_MAGNITUDE_BAND = ("bessellb", "ab")

# What OpenUniverse's SN Ia brightness IS, which makes that class the one place the measurement can
# be checked against an exact answer rather than against another measurement. Over all 224 118 of
# its SNe Ia,
#
#     salt2_mB + 0.15 * salt2_x1 - 3.1 * salt2_c - mu(z) = -19.363447 +- 0.000103 mag
#
# with alpha = 0.15 and beta = 3.1 for every object and gammaDM identically zero. That residual is
# not a fit, it is an identity: OpenUniverse's SNe Ia carry NO intrinsic scatter at all, and the
# 0.269 mag spread of their absolute mB is entirely the spread of x1 and c through that relation.
# So a SN Ia parent's absolute magnitude is known in closed form, and the offset measured off its
# light curve can be compared against it. IT DOES NOT AGREE, and the disagreement is the reason to
# measure rather than to compute. Over 60 SNe Ia of one healpix, measured minus (salt2_mB - mu):
#
#     median  +0.19 mag   -- a system offset, and an expected one: mB is a rest-frame B magnitude
#                            on SALT's own system and the measurement is anchored in whatever
#                            rest-frame region the Roman bands covered at the parent's redshift
#     rms      0.21 mag   -- object to object, and NOT expected
#
# That scatter is uncorrelated with x1 (-0.13), with c (+0.07) and with redshift (+0.06), and a fit
# in x1 and c removes none of it. It is also ACHROMATIC: within one object the bands agree to
# 0.039 mag while between objects the offset moves by 0.21, so it is an amplitude and not a colour.
# The catalogue identity above says these objects have no intrinsic scatter; their LIGHT CURVES say
# they do. The likeliest reading is the intrinsic-scatter model SNANA applies to the flux and does
# not fold back into the reported mB -- that is what such a model looks like, a per-object grey
# offset of a tenth or two -- but nothing here proves it, and no other per-object quantity in the
# catalogue (lens_dmu, mw_EBV, v_pec) is non-zero to explain it.
#
# Either way the sample inherits it, because the brightness is read off the light curve. A
# generator that took the catalogue at its word would have produced SNe Ia with no scatter at all.
SALT2_ALPHA = 0.15
SALT2_BETA = 3.1
SALT2_M0 = -19.363447  # at OPENUNIVERSE_H0; degenerate with it, see the cosmology block

# Host extinction, class by class, and NOT because of what is physically right: the sample's job is
# to be indistinguishable from OpenUniverse's contaminants except in redshift, so any departure
# from what OpenUniverse did becomes a class-correlated feature the classifier can learn and the
# sky does not have. Audited against the 33 healpix catalogues, 1 352 231 objects:
#
#   * SN Ia (gentype 10) carry theirs inside SALT's `c`, so this module does too, through the
#     parent's own `salt2_c`. AV is recorded as -9 because there is no separate screen to record,
#     not because there is no dust.
#   * The core-collapse classes (21, 26, 32) carry NONE, and that is a bug of OpenUniverse's, not a
#     property of the templates. Its own section 3.2.4, "Known issues": "For the core collapse
#     models (SNII, SNIb, SNIc), the wavelength range was extended for the set of templates that
#     had been corrected for host-galaxy extinction. However, host extinction was not enabled in
#     the simulation." The "+HostXT" of `NON1ASED.V19_CC+HostXT_WAVEEXT` marks templates the host
#     dust was taken OUT of; it was never put back. The catalogue agrees: AV = -9 and
#     `template_index` is the only model parameter those rows carry. So no dust here either.
#   * TDE (42) records none, and the MOSFiT bank cut to van Velzen et al. (2021) carries whatever
#     the observed population carries. Consistent, nothing to add.
#   * SN Iax (12) is the one class OpenUniverse gives an explicit screen -- AV median 0.440, 84th
#     percentile 1.027, RV 3.1 -- and the re-rendered object is given the screen ITS OWN parent
#     was given. It is the only dust this module applies to anything.
#
# Milky Way extinction is applied by OpenUniverse in SkyCatalog rather than in SNANA, so the
# catalogues carry `mw_extinction_applied = False` and mag_true is free of it for every class
# alike. It creates no class structure and this module adds none.

# Rest-frame days from MAXIMUM, not from the source's own phase zero. The two are not the same
# thing across this library and the difference is class-correlated: SALT puts phase zero at B
# maximum, the SN Iax base SED puts it at maximum, the MOSFiT bank is centred on peak luminosity,
# and the core-collapse archive is on OpenUniverse's own `peak_mjd` axis, which for a SN II sits at
# the start of the plateau rather than at a maximum. `peak_phase` resolves each of them to the same
# axis, and it is the axis the parent's light curve is read on too, so both sides of the brightness
# measurement span the same rest-frame phases.
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

# Mag per rest-frame day, on a colour measured against the brightest band of the same phase. The
# absolute floor above only catches a template with no flux at all; this catches the shoulder just
# inside it, where the reported magnitudes are ordinary numbers arranged in an order no transient
# produces. See `_drop_colour_discontinuities` for the light curve that motivated it and for the
# measured rate of real colour evolution, which is an order of magnitude below this.
MAXIMUM_COLOUR_RATE = 1.0

# Detector full well, electrons per pixel. NOT a galsim.roman constant -- galsim models
# non-linearity but no hard saturation -- so this is the nominal H4RG figure and an assumption of
# this module. It also makes the bright limit below a LOWER bound: Roman reads up the ramp, so a
# source past this is measured from fewer reads rather than lost.
FULL_WELL_ELECTRONS = 1.0e5


@cache
def _registry_source(source_name):
    """One instance per registry name, kept for the life of the process.

    `sncosmo.Model(source="snana-2006jl")` goes back to the registry on every call and the registry
    re-reads and re-splines the template's ASCII file: profiling a mixed population put 79 % of the
    generation time inside `read_griddata_ascii`, three million calls to its comment stripper. The
    sources are read-only here -- every model sets its own parameters before use -- so one instance
    each is enough."""
    return _sncosmo().get_source(source_name)


def _padded_ia_source():
    """`salt3-nir` with its last defined flux held constant out to IA_PAD_WAVELENGTH.

    See the IA_SOURCE_NAME block for what the pad is worth and why it is flat. The mechanics: a
    SALT source carries its spectral surfaces as two bicubic interpolators over (phase, rest-frame
    wavelength) -- M0 and M1, the mean SED and the x1 derivative -- and `SALT2Source._flux` reads
    the wavelength range off those. The colour law is analytic and already extrapolates smoothly
    past 20000 A, so only the surfaces need extending. Each is re-evaluated on its own grid, its
    last wavelength column repeated across the pad, and rebuilt on the extended grid; below 20000 A
    the result is the original to machine precision, which
    `test_padded_ia_source_matches_the_base_below_the_pad` checks.

    `sncosmo.get_source` builds a new instance per call, so the object mutated here is this
    module's own and the registry entry for `salt3-nir` is untouched."""
    import sncosmo
    from sncosmo.salt2utils import BicubicInterpolator

    source = sncosmo.get_source(IA_BASE_SOURCE_NAME)
    step = float(source._wave[1] - source._wave[0])
    pad = np.arange(source._wave[-1] + step, IA_PAD_WAVELENGTH + 0.5 * step, step)
    wavelength = np.concatenate([source._wave, pad])
    for surface_name, surface in list(source._model.items()):
        values = surface(source._phase, source._wave)
        held = np.repeat(values[:, -1:], len(pad), axis=1)
        source._model[surface_name] = BicubicInterpolator(
            source._phase, wavelength, np.concatenate([values, held], axis=1)
        )
    source._wave = wavelength
    source.name = IA_SOURCE_NAME
    return source


def _openuniverse_templates():
    """{source name: (phase, wavelength, flux)} of the OpenUniverse archive, read once.

    The archive shares ONE wavelength axis across its templates and gives each its own phase grid,
    because the phases come from the original pycoco SEDs and differ template to template while the
    wavelengths come from snsedextend and do not.

    Nothing is repaired on the way in. An earlier version interpolated across the interior
    zero-flux gaps; it was removed, because reproducing OpenUniverse's method is the whole reason
    these templates are extended here and patching the method's output is not reproducing it. It
    also bought nothing measurable: filling moved F184 by 0.010 mag at z = 0.05 and by 0.000 in
    every band at z >= 0.2, and it did not make the sources pass the audit either, since the left
    edge of the gap is already 1.7e-18 against a row maximum of 4.8e-2 -- the fill drew an
    exponential ramp up from nothing and sncosmo's bicubic spline rang a few parts in 10 000 below
    zero across it."""
    global _OPENUNIVERSE_TEMPLATES
    if _OPENUNIVERSE_TEMPLATES is None:
        if not OPENUNIVERSE_ARCHIVE_PATH.exists():
            raise FileNotFoundError(
                f"{OPENUNIVERSE_ARCHIVE_PATH} is missing; "
                "build it with scripts/build_openuniverse_cc_templates.py"
            )
        templates = {}
        with np.load(OPENUNIVERSE_ARCHIVE_PATH) as archive:
            wavelength = archive["wavelength"].astype(float)
            for index, name in enumerate(archive["template_names"]):
                source_name = OPENUNIVERSE_SOURCE_PREFIX + str(name).replace("pycoco_", "")
                templates[source_name] = (
                    archive[f"phase_{index}"].astype(float),
                    wavelength,
                    archive[f"flux_{index}"].astype(float),
                )
        _OPENUNIVERSE_TEMPLATES = templates
    return _OPENUNIVERSE_TEMPLATES


def _sncosmo():
    """sncosmo is only needed to generate, never to train or evaluate, so it stays a lazy import.

    Registering the SN Iax base source here rather than at import time keeps `IAX_SOURCE_NAME`
    resolvable through the ordinary `sncosmo.get_source` path -- including from the coverage test,
    which is the point: the Iax source has to be checked against R062-F184 like every other one."""
    import sncosmo

    if IA_SOURCE_NAME not in _REGISTERED_SOURCES:
        sncosmo.register(_padded_ia_source(), IA_SOURCE_NAME, force=True)
        _REGISTERED_SOURCES.add(IA_SOURCE_NAME)
    if IAX_SOURCE_NAME not in _REGISTERED_SOURCES:
        phase, wavelength, flux = _iax_base_sed()
        sncosmo.register(sncosmo.TimeSeriesSource(phase, wavelength, flux), IAX_SOURCE_NAME, force=True)
        _REGISTERED_SOURCES.add(IAX_SOURCE_NAME)
    # Every OpenUniverse source at once rather than on demand. They come out of one archive, so the read is
    # shared, and `register_sources` promises the whole of SOURCES_BY_LABEL is resolvable.
    if OPENUNIVERSE_SOURCE_PREFIX not in _REGISTERED_SOURCES:
        _REGISTERED_SOURCES.add(OPENUNIVERSE_SOURCE_PREFIX)
        for source_name, (phase, wavelength, flux) in _openuniverse_templates().items():
            sncosmo.register(sncosmo.TimeSeriesSource(phase, wavelength, flux), source_name, force=True)
    if TDE_SOURCE_NAME not in _REGISTERED_SOURCES:
        # Marked registered before the source is built, because `tde_source` calls back into here
        # and would otherwise recurse. The registry entry is the first template of the bank, which
        # stands for all of them in the coverage check: every template shares one wavelength grid.
        _REGISTERED_SOURCES.add(TDE_SOURCE_NAME)
        sncosmo.register(tde_source(0), TDE_SOURCE_NAME, force=True)
    return sncosmo


_REGISTERED_SOURCES = set()
_OPENUNIVERSE_TEMPLATES = None
_IAX_BASE_SED = None
_IAX_BANK = None


def register_sources():
    """Put every source name of SOURCES_BY_LABEL in the sncosmo registry.

    Only SN Ia's `salt3-nir` is a stock sncosmo entry, and even that one is registered padded under
    a different name. The core-collapse sources come from the V19 archive, SN Iax and TDE are built
    here, so a caller that reaches sncosmo on its own -- the coverage test does -- has to come
    through this first."""
    _sncosmo()


# The audit that pruned SOURCES_BY_LABEL, kept as code rather than as a list so the test can
# re-derive the decision from sncosmo instead of trusting a list someone typed once.
NEAR_INFRARED_AUDIT_LIMITS = (9000.0, 21000.0)  # A rest-frame: where the extension takes over
# A SWEEP, not one redshift, and that is a correction: this was 0.1 alone, "mid-range for this
# sample", and at 0.1 the V19 sources show no inversion at all while at 0.2 and beyond they show a
# severe one. A single redshift tests one alignment of the rest-frame artefacts with the bands and
# reports it as if it were the template's whole behaviour.
NEAR_INFRARED_AUDIT_REDSHIFTS = (0.05, 0.1, 0.2, 0.3, 0.5)
NEAR_INFRARED_AUDIT_STEP = 50.0


def defective_near_infrared_extension(source_name):
    """Why a template's near-infrared extension is unusable; an empty list means it passes.

    Both tests are mechanical and both are aimed at the EXTENSION rather than at the spectrum the
    template was measured from. See the SOURCES_BY_LABEL block for what they found and what
    removing the failures cost.

    THE TWO TESTS ARE NOT EQUALLY SOUND, and the pruning of SOURCES_BY_LABEL rests on the first.
    A partial zero-flux hole in the middle of a bandpass is not physics under any model. The
    F184-brighter-than-H158 test is a heuristic instead: it assumes a smooth declining continuum,
    and a source with real near-infrared structure can violate it honestly. `salt3-nir-f184` does,
    at 52 of the sampled phases, because a SN Ia HAS a secondary near-infrared maximum -- and the
    same source's measured colours against OpenUniverse's own SALT3 agree to 0.017 mag in all four
    bands, which is what says the violation is the model and not a defect. Every SNANA template
    fails the hole test on its own, at 42 to 84 of the 91 sampled phases, so nothing in the cut
    depends on the heuristic.
    """
    sncosmo = _sncosmo()
    source = _registry_source(source_name)
    phase = float(source.peakphase("bessellb"))
    reasons = []

    blue = max(source.minwave(), NEAR_INFRARED_AUDIT_LIMITS[0])
    red = min(source.maxwave(), NEAR_INFRARED_AUDIT_LIMITS[1])
    rest_wavelength = np.arange(blue, red, NEAR_INFRARED_AUDIT_STEP)

    models = []
    for redshift in NEAR_INFRARED_AUDIT_REDSHIFTS:
        model = sncosmo.Model(source=source)
        model.set(z=redshift, t0=0.0)
        observed_blue = max(model.minwave(), OBSERVED_WAVELENGTH_LIMITS[0])
        observed_red = min(model.maxwave(), OBSERVED_WAVELENGTH_LIMITS[1])
        models.append(
            (redshift, model, _sampling_grid(observed_blue, observed_red, OBSERVED_WAVELENGTH_STEP))
        )

    # EVERY phase the module samples, not just maximum. Auditing at maximum alone is what the first
    # version of this did, and it was close to arbitrary with respect to the defect: it removed
    # templates broken at that one phase and kept templates equally broken twenty days later, so
    # the pruned sample's colour residuals moved in directions that meant nothing. The colour
    # statistic takes each band at ITS own maximum, which for the red bands is nowhere near B
    # maximum, so a template only has to be sound where it is read.
    for offset in REST_FRAME_PHASES:
        epoch = phase + offset
        if not (source.minphase() <= epoch <= source.maxphase()):
            continue

        # PARTIAL gaps only. A phase where the whole near-infrared is zero is the template before
        # explosion or after it has faded, which is a fact about the source and not a hole in it.
        if rest_wavelength.size:
            rest_flux = source.flux(epoch, rest_wavelength)
            if (rest_flux <= 0.0).any() and (rest_flux > 0.0).any():
                reasons.append(f"zero-flux gap at {offset:+.0f} d")

        for redshift, model, observed_wavelength in models:
            flux = np.clip(model.flux(epoch * (1.0 + redshift), observed_wavelength), 0.0, None)
            if flux.max() <= 0.0:
                continue
            magnitudes = spectrum_to_roman_magnitudes(observed_wavelength, flux)
            # On the Rayleigh-Jeans side of any declining continuum an AB magnitude gets fainter
            # towards the red, so a template whose F184 outshines its H158 is reporting the rising
            # ramp its extension drew across the water bands.
            if np.isfinite(magnitudes["F184"]) and np.isfinite(magnitudes["H158"]):
                if magnitudes["F184"] < magnitudes["H158"]:
                    reasons.append(f"F184 brighter than H158 at {offset:+.0f} d, z = {redshift:.2f}")
    return reasons


def _iax_base_sed():
    """(phase, wavelength, flux) of SN 2005hk, processed exactly as `Iax-model.ipynb` cell 13.

    Four steps in the notebook's order: resample the base SED onto IAX_NOTEBOOK_PHASES, smooth the
    z/y region in phase from index 10 on, impose the late-time decline past
    IAX_LATE_DECLINE_PHASE, and smooth everything from index 70 on. The z/y smoothing is the one
    that matters most here: it runs over 8000-11000 A, which redshifts into Z087 and Y106 for
    exactly the population this module generates."""
    global _IAX_BASE_SED
    if _IAX_BASE_SED is None:
        # sncosmo directly, not `_sncosmo()`: that helper registers this very source and would
        # recurse back into here.
        import sncosmo
        from scipy.ndimage import gaussian_filter

        with np.load(IAX_BASE_SED_PATH) as archive:
            base = sncosmo.TimeSeriesSource(
                archive["phase"].astype(float),
                archive["wavelength"].astype(float),
                archive["flux"].astype(float),
            )
        phase = IAX_NOTEBOOK_PHASES
        wavelength = IAX_NOTEBOOK_WAVELENGTHS
        # Outside the base file's own -15 to +65 this is zero, which is what the notebook gets too.
        original = sncosmo.Model(source=base).flux(phase, wavelength)
        flux = np.copy(original)

        wiggly = (wavelength >= IAX_SMOOTHED_WAVELENGTHS[0]) & (wavelength <= IAX_SMOOTHED_WAVELENGTHS[1])
        smoothed = gaussian_filter(original, [IAX_SMOOTHING_SIGMA_PHASES, 0.0])
        late = slice(IAX_SMOOTHED_FROM_INDEX, None)
        flux[late, wiggly] = smoothed[late, wiggly]

        declining = phase >= IAX_LATE_DECLINE_PHASE
        flux[declining, :] *= np.exp(
            -(phase[declining, None] - IAX_LATE_DECLINE_PHASE) / IAX_LATE_DECLINE_EFOLD
        )
        very_late = slice(IAX_LATE_SMOOTHED_FROM_INDEX, None)
        flux[very_late, :] = gaussian_filter(flux, [IAX_SMOOTHING_SIGMA_PHASES, 0.0])[very_late, :]

        _IAX_BASE_SED = (phase.copy(), wavelength.copy(), flux)
    return tuple(array.copy() for array in _IAX_BASE_SED)


def _iax_bank():
    """The notebook's own 1001 rows of (M_V, t_rise, dm15B, dm15R), replayed from `seed(4)`.

    Cells 1-12 of `Iax-model.ipynb`, in order, keeping every step that touches the random stream.
    `np.random.seed` selects the legacy MT19937 generator, whose stream numpy guarantees across
    versions, so this reproduces the published bank rather than a sample from the same laws.
    Verified by the fraction brighter than M_V = -17.5, which the notebook prints as 0.2318."""
    global _IAX_BANK
    if _IAX_BANK is None:
        from astropy import units
        from astropy.cosmology import FlatLambdaCDM

        table = pd.read_csv(IAX_JHA_TABLE_PATH)
        redshift = table["z"].to_numpy(dtype=float)

        state = np.random.RandomState(IAX_BANK_SEED)
        grid = np.arange(*IAX_ABSOLUTE_MAGNITUDE_RANGE, IAX_ABSOLUTE_MAGNITUDE_STEP)
        density = _iax_luminosity_function(grid)
        cumulative = np.cumsum(density) * IAX_ABSOLUTE_MAGNITUDE_STEP
        uniform_grid = np.arange(0.0, 1.0, IAX_INVERSE_CDF_STEP)
        inverse_cdf = np.interp(uniform_grid, cumulative, grid)

        absolute_v = np.interp(state.random_sample(IAX_BANK_DRAWN), uniform_grid, inverse_cdf)

        # Cell 6: result discarded, random draws not. See IAX_REJECTION_MAGNITUDE_LIMIT.
        hubble, matter_density = IAX_REJECTION_COSMOLOGY
        modulus = (
            5.0
            * np.log10(FlatLambdaCDM(H0=hubble, Om0=matter_density).luminosity_distance(redshift) / units.Mpc)
            + 25.0
        )
        for one_modulus in modulus:
            while True:
                trial = np.interp(state.random_sample(), uniform_grid, inverse_cdf)
                if trial + one_modulus < IAX_REJECTION_MAGNITUDE_LIMIT:
                    break

        offset = absolute_v + 19.0
        rise_time = (
            21.0
            - offset * 10.0 / 3.0
            + 0.22 * offset**2
            + state.normal(size=IAX_BANK_DRAWN) * IAX_RISE_TIME_SCATTER
        )
        decline_b = 1.2 + offset / 6.0 + state.normal(size=IAX_BANK_DRAWN) * IAX_DECLINE_B_SCATTER
        decline_r = np.abs(0.5 + offset / 15.0 + state.normal(size=IAX_BANK_DRAWN) * IAX_DECLINE_R_SCATTER)
        _IAX_BANK = tuple(array[:IAX_BANK_SIZE] for array in (absolute_v, rise_time, decline_b, decline_r))
    return _IAX_BANK


def _iax_luminosity_function(grid):
    """Jha & Dai's phi(M_V), normalised to unit area; see IAX_ABSOLUTE_MAGNITUDE_RANGE."""
    density = -(grid + 13.0) / 6.0 + 1.0
    bright = grid < IAX_BRIGHT_ROLLOFF
    density[bright] *= np.exp(-((grid[bright] - IAX_BRIGHT_ROLLOFF) ** 2) / 2.0 / IAX_BRIGHT_SIGMA**2)
    faint = grid > IAX_FAINT_ROLLOFF
    density[faint] *= np.exp(-((grid[faint] - IAX_FAINT_ROLLOFF) ** 2) / 2.0 / IAX_FAINT_SIGMA**2)
    return density / (np.sum(density) * IAX_ABSOLUTE_MAGNITUDE_STEP)


def iax_template(template_index):
    """(M_V, rise time, dm15B, dm15R) of one bank row, indexed as OpenUniverse indexes it."""
    bank = _iax_bank()
    return tuple(float(array[template_index]) for array in bank)


_TDE_TEMPLATES = None


def _tde_templates():
    """(phase, temperature, radius) of the MOSFiT photosphere bank, one row per drawn TDE."""
    global _TDE_TEMPLATES
    if _TDE_TEMPLATES is None:
        with np.load(TDE_TEMPLATE_PATH) as archive:
            _TDE_TEMPLATES = (
                archive["phase"].astype(float),
                archive["temperature"].astype(float),
                archive["radius"].astype(float),
            )
    return _TDE_TEMPLATES


def tde_template_count():
    return len(_tde_templates()[1])


@cache
def tde_source(template_index):
    """The blackbody spectrum of one drawn TDE photosphere, as an sncosmo source.

    Cached over the whole bank rather than over a window of it. The draws are uniform over 227
    templates, so the `lru_cache(maxsize=24)` this used to carry could not hold a working set and
    measured no hits at all; the bank at TDE_WAVELENGTH_STEP costs 0.35 MB a template, so keeping
    all of it is 79 MB and turns a 5 ms build into a lookup.

    Flux is the real thing -- pi B_lambda(T(t)) (R(t) / 10 pc)^2 -- rather than a shape to be
    rescaled later, so the source carries the brightness MOSFiT's physics implies. The sample does
    not use it: a re-rendered TDE is normalised to its parent's own brightness like every other
    class, and this bank supplies the colour and the shape of the light curve."""
    from astropy import units
    from astropy.modeling.models import BlackBody

    sncosmo = _sncosmo()
    phase, temperature, radius = _tde_templates()
    wavelength = np.arange(*TDE_WAVELENGTH_LIMITS, TDE_WAVELENGTH_STEP) * units.AA
    scale = 1.0 * units.erg / (units.cm**2 * units.AA * units.s * units.steradian)
    ten_parsec = (10.0 * units.pc).to(units.cm)

    # One BlackBody over the whole phase axis at once: per-phase objects cost 0.5 s a source, which
    # at a sixth of the sample would dominate the entire generation.
    # A handful of templates touch T = 0 at a single phase, where the model has no photosphere yet.
    # The Planck function divides by zero there, so the flux is set to what it physically is.
    one_temperature = temperature[template_index]
    warm = one_temperature > 0.0
    blackbody = BlackBody(temperature=np.where(warm, one_temperature, 1.0)[:, None] * units.K, scale=scale)
    radiance = np.where(warm[:, None], blackbody(wavelength[None, :]).value, 0.0)
    dilution = (radius[template_index] * units.cm / ten_parsec).decompose().value ** 2
    return sncosmo.TimeSeriesSource(phase, wavelength.value, np.pi * radiance * dilution[:, None])


def iax_source(rise_time, decline_b, decline_r):
    """The SN 2005hk SED warped to one (rise time, dm15B, dm15R), as `Iax-model.ipynb` cell 14.

    Three warps in order: the pre-maximum phases are stretched to give the rise time, the post-
    maximum flux is scaled by a log-normal kernel centred on day 15 to give dm15(B), and a second
    kernel with a wavelength ramp gives dm15(R) while leaving dm15(B) where the first warp put it.
    Amplitude is left alone -- `build_model` sets it from the template's own M_V."""
    from scipy.interpolate import RectBivariateSpline

    sncosmo = _sncosmo()
    phase, wavelength, flux = _iax_base_sed()

    stretched = phase.copy()
    pre_maximum = phase < 0.0
    stretched[pre_maximum] *= rise_time / IAX_BASE_RISE_TIME
    # Before the stretched explosion epoch the SED is suppressed rather than zeroed. Strict `<`,
    # the notebook's own comparison: the grid starts at -30 d, so this selects the real region
    # between -2 * rise_time and -rise_time and leaves the row at exactly -rise_time alone.
    flux[stretched < -rise_time] /= IAX_PRE_EXPLOSION_SUPPRESSION
    model = sncosmo.Model(source=sncosmo.TimeSeriesSource(stretched, wavelength, flux))

    def decline(band):
        return model.bandmag(band, "vega", 15.0) - model.bandmag(band, "vega", 0.0)

    post_maximum = stretched > 1.0
    kernel = np.zeros_like(stretched)
    kernel[post_maximum] = np.exp(
        -((np.log10(stretched[post_maximum]) - np.log10(IAX_WARP_KERNEL_CENTRE_DAYS)) ** 2)
        / 2.0
        / IAX_WARP_KERNEL_SIGMA_DEX**2
    )

    scale_b = 10.0 ** (-0.4 * (decline_b - decline("bessellb")))
    flux = flux * (1.0 + (scale_b - 1.0) * kernel)[:, None]
    model = sncosmo.Model(source=sncosmo.TimeSeriesSource(stretched, wavelength, flux))

    scale_r = 10.0 ** (-0.4 * (decline_r - decline("bessellr")))
    multiplier_b = np.ones_like(stretched)
    multiplier_r = 1.0 + (scale_r - 1.0) * IAX_WARP_R_GAIN * kernel
    anchored = np.array([multiplier_b, multiplier_b, multiplier_r, multiplier_r, multiplier_b])
    # Linear in wavelength between the anchors and linearly EXTRAPOLATED past 9000 A, which is what
    # `interp2d` did and what OpenUniverse's templates carry; see IAX_WARP_ANCHORS_AA.
    ramp = RectBivariateSpline(np.asarray(IAX_WARP_ANCHORS_AA), stretched, anchored, kx=1, ky=1)
    return sncosmo.TimeSeriesSource(stretched, wavelength, flux * ramp(wavelength, stretched).T)


# --- The redshift the sample is generated at ----------------------------------------------------
# The bins the deficit is counted in are the kilonova grid's own. `kn-kilonova-windows` puts its
# kilonovae on 50 logarithmic redshifts between 0.01 and 1.0, so a kilonova histogram is 50 spikes
# and any binning finer than the spacing between them measures the grid rather than the population.
# The edges are the geometric midpoints of that grid, which is the coarsest binning that keeps one
# grid point per bin.
DEFICIT_REDSHIFT_LIMITS = (0.01, 1.0)
DEFICIT_REDSHIFT_NODES = 50


def deficit_bin_edges(limits=DEFICIT_REDSHIFT_LIMITS, nodes=DEFICIT_REDSHIFT_NODES):
    """Bin edges around the kilonova redshift grid, one bin per grid point."""
    grid = np.geomspace(limits[0], limits[1], nodes)
    interior = np.sqrt(grid[1:] * grid[:-1])
    return np.concatenate([[grid[0] ** 2 / interior[0]], interior, [grid[-1] ** 2 / interior[-1]]])


def redshift_deficit(kilonova_redshifts_by_tier, contaminant_redshifts_by_tier, edges=None):
    """(edges, per-bin count) of the contaminants the training set does not have.

    THE STATISTIC. In each bin, how many contaminants a tier would need for the classifier to see
    as many of them as it sees kilonovae: max(0, N_kilonova - N_contaminant). Below z = 0.45 that
    number is essentially the whole kilonova histogram, because OpenUniverse's contaminants are
    almost all above it -- which is the shortcut this sample exists to remove, counted.

    THE MAXIMUM OVER TIERS, NOT THE SUM, and it changes the size of the sample by a factor of two.
    The tiers are not disjoint populations: 717 863 of the 717 864 wide contaminants of
    OpenUniverse are also deep ones, and one generated object serves both tiers because
    `build_izc_windows` renders the light curve once and lets each tier observe the bands it
    observes. So a bin needs as many objects as its hungriest tier asks for, not as many as both
    ask for together.

    Nothing is generated ABOVE the kilonova grid: there is no deficit there, and a contaminant at
    z = 2 is one OpenUniverse already has."""
    if edges is None:
        edges = deficit_bin_edges()
    deficits = []
    for tier, kilonova_redshifts in kilonova_redshifts_by_tier.items():
        kilonovae, _ = np.histogram(kilonova_redshifts, edges)
        contaminants, _ = np.histogram(contaminant_redshifts_by_tier[tier], edges)
        deficits.append(np.clip(kilonovae - contaminants, 0, None))
    return edges, np.max(deficits, axis=0)


def draw_redshifts_from_deficit(edges, deficit, random_generator, scale=1.0):
    """One redshift per object the deficit asks for, uniform inside its own bin.

    Uniform rather than at the bin's own grid point, because the kilonovae sit on a grid and the
    contaminants they have to be indistinguishable from do not: a spike of izc objects at each of
    50 redshifts would be a feature of the generation visible to the classifier in the redshift
    token, and in the apparent magnitude even without it."""
    counts = np.round(np.asarray(deficit, dtype=float) * scale).astype(int)
    redshifts = [
        random_generator.uniform(edges[index], edges[index + 1], count)
        for index, count in enumerate(counts)
        if count > 0
    ]
    return np.sort(np.concatenate(redshifts)) if redshifts else np.empty(0)


@cache
def _core_collapse_archive_order():
    """(template names, labels) of the archive, in the order it was written.

    The order is the whole content: `build_openuniverse_cc_templates.py` writes its templates in
    ascending `template_index`, so the n-th entry here is the n-th smallest template index
    OpenUniverse drew. That is what makes `core_collapse_source_by_template_index` a lookup rather
    than a table someone typed."""
    with np.load(OPENUNIVERSE_ARCHIVE_PATH) as archive:
        return tuple(str(name) for name in archive["template_names"]), tuple(
            str(label) for label in archive["labels"]
        )


def core_collapse_source_by_template_index(catalog):
    """{template_index: (source name, label)}, derived from the catalogue and CHECKED against it.

    The archive does not record which `template_index` each of its templates came from -- it
    records their names -- so the mapping is recovered from the one thing that fixes it: both the
    archive and the catalogue are in ascending template index. Nothing about that is assumed. The
    catalogue's own gentype for every object of a template has to agree with the label the archive
    carries for the template it lands on, over all 44 of them, or this raises: 21 is SN Ib, 26 is
    SN Ic and 32 is the pool SN IIP and SN IIL are drawn from."""
    core_collapse = catalog[catalog["gentype"].isin(openuniverse_parents.CORE_COLLAPSE_GENTYPES)]
    indices = sorted(core_collapse["template_index"].unique())
    names, labels = _core_collapse_archive_order()
    if len(indices) != len(names):
        raise ValueError(
            f"the catalogue drew {len(indices)} core-collapse templates and the archive holds "
            f"{len(names)}; the two cannot be matched by order"
        )
    gentypes = core_collapse.groupby("template_index")["gentype"].unique()
    mapping = {}
    for index, name, label in zip(indices, names, labels, strict=True):
        drawn = sorted(int(one) for one in gentypes.loc[index])
        if drawn != [GENTYPE_BY_LABEL[label]]:
            raise ValueError(
                f"template_index {index} maps to {name} ({label}, gentype "
                f"{GENTYPE_BY_LABEL[label]}) but its objects carry gentype {drawn}"
            )
        mapping[int(index)] = (OPENUNIVERSE_SOURCE_PREFIX + name.replace("pycoco_", ""), label)
    return mapping


def draw_population_from_parents(catalog, redshifts, random_generator, source_by_template_index=None):
    """One realization per entry of `redshifts`: a parent object, re-rendered at that redshift.

    `catalog` is `openuniverse_parents.read_parent_catalog`'s table. The class is drawn first, from
    CLASS_FRACTION, and the parent uniformly from the objects of that class -- WITH replacement,
    because the classes are not equally numerous in OpenUniverse (3769 TDE against 683 446 SN II)
    and equal shares here mean the rare ones are re-rendered many times over. Every copy of one
    parent carries its `parent_key`, which is what keeps them together on one side of the split.

    `redshifts` is drawn by the caller from whatever target distribution the sample is meant to
    fill -- the deficit against the kilonova histogram, in the intended use -- because the choice of
    that distribution is the whole point of the sample and does not belong buried in here.

    The brightness is NOT set here: it is measured from the parent's own light curve, which lives in
    a file this module never opens. `measure_brightness_offset` fills it in."""
    if source_by_template_index is None:
        source_by_template_index = core_collapse_source_by_template_index(catalog)

    # Indexed once per class rather than filtered per object: the catalogue is 1.3 million rows and
    # the sample draws from it a million times.
    by_label = {}
    for label in CLASS_FRACTION:
        # The core-collapse classes are selected by TEMPLATE and not by the catalogue's label. Two
        # of them have no label of their own -- OpenUniverse pools SN IIP and SN IIL into gentype
        # 32 -- and for the two that do, the template is the stricter statement: it says which SED
        # the object was drawn from, which is what is being re-rendered.
        wanted = [
            index
            for index, (_, template_label) in source_by_template_index.items()
            if template_label == label
        ]
        if wanted:
            block = catalog[catalog["template_index"].isin(wanted)]
        else:
            block = catalog[catalog["label"] == label]
        if block.empty:
            raise ValueError(f"the parent catalogue holds no {label}")
        by_label[label] = block.reset_index(drop=True)

    labels = random_generator.choice(
        list(CLASS_FRACTION), size=len(redshifts), p=list(CLASS_FRACTION.values())
    )
    population = []
    for index, (label, redshift) in enumerate(zip(labels, redshifts, strict=True)):
        block = by_label[str(label)]
        parent = block.iloc[int(random_generator.integers(len(block)))]
        population.append(
            realization_from_parent(
                parent, index, float(redshift), source_by_template_index, random_generator
            )
        )
    return population


def realization_from_parent(parent, index, redshift, source_by_template_index, random_generator):
    """One OpenUniverse object, ready to be rendered at `redshift`.

    `parent` is one row of `openuniverse_parents.read_parent_catalog`. Everything the release
    records about the object is carried over; what the random generator is for is the placement of
    the survey's visit grid, and, for a TDE, the one model a parent cannot supply."""
    gentype = int(parent["gentype"])
    core_collapse = gentype in openuniverse_parents.CORE_COLLAPSE_GENTYPES
    # The catalogue's label for a core-collapse object is the pooled one -- OpenUniverse has no
    # SN IIP and SN IIL, it has gentype 32 -- so the template it drew is what resolves the subtype.
    source_name, label = (
        source_by_template_index[int(parent["template_index"])]
        if core_collapse
        else (None, str(parent["label"]))
    )
    realization = {
        "index": int(index),
        "label": label,
        "redshift": float(redshift),
        "parent_key": str(parent["parent_key"]),
        "parent_healpix": int(parent["healpix"]),
        "parent_id": int(parent["id"]),
        "parent_redshift": float(parent["redshift"]),
        "parent_peak_mjd": float(parent["peak_mjd"]),
        # Filled by `measure_brightness_offset`, from the parent's own light curve.
        "peak_absolute_magnitude": np.nan,
        "brightness_offset": np.nan,
        "brightness_residual": np.nan,
        "brightness_bands": 0,
        # The two degrees of freedom of where the survey's visit grid falls on this transient, both
        # uniform because in the sky the grid is fixed in absolute time and the explosion is not:
        # the delay from the start of the model to the first visit, and the PARITY of that visit,
        # which decides whether it carries the two blue non-anchor bands or the two red ones.
        # Together they cover the full 10-day cycle of the band pattern. Without them every izc
        # object would be sampled from the same phase of the cadence, and since the model's own
        # start is set by the template, that phase would be a function of the class -- the exact
        # species of class-correlated artefact this module removes.
        "cadence_parity": int(random_generator.integers(CADENCE_PARITY_PERIOD)),
        "visit_phase_offset_days": float(random_generator.uniform(0.0, BASE_CADENCE_DAYS)),
    }
    if core_collapse:
        realization["source_name"] = source_name
    elif label == "SN Ia":
        realization["source_name"] = IA_SOURCE_NAME
        realization["salt2_x1"] = float(parent["salt2_x1"])
        realization["salt2_c"] = float(parent["salt2_c"])
        realization["salt2_mB"] = float(parent["salt2_mB"])
    elif label == "SN Iax":
        realization["source_name"] = IAX_SOURCE_NAME
        # OpenUniverse's `template_index` runs 1..919 over the bank the notebook drew and the bank
        # is indexed from zero. See IAX_BANK_SIZE for how that mapping was verified.
        iax_template_index = int(parent["template_index"]) - 1
        _, rise_time, decline_b, decline_r = iax_template(iax_template_index)
        realization["iax_template_index"] = iax_template_index
        realization["iax_rise_time"] = rise_time
        realization["iax_decline_b"] = decline_b
        realization["iax_decline_r"] = decline_r
    elif label == "TDE":
        realization["source_name"] = TDE_SOURCE_NAME
        # The one place a parent cannot supply the model: OpenUniverse's TDE is the observed SED of
        # AT2019qiz and its `template_index` indexes a bank that was never published, so the MOSFiT
        # template stands in. The parent still supplies the brightness and the group.
        #
        # Drawn from the PARENT'S OWN id rather than from the caller's generator, so that every
        # re-rendering of one parent is the same TDE. Otherwise a parent re-rendered twenty times
        # would be twenty different objects sharing one brightness, and the brightness measurement
        # -- which renders the model at the parent's redshift -- could not be shared between them.
        realization["tde_template_index"] = int(
            np.random.default_rng(int(parent["id"])).integers(tde_template_count())
        )
    else:
        raise ValueError(f"gentype {gentype} is not a class this module re-renders")
    if np.isfinite(parent["host_av"]):
        realization["host_av"] = float(parent["host_av"])
        realization["host_rv"] = float(parent["host_rv"])
    return realization


def draw_class_population(catalog, labels, redshifts, random_generator, source_by_template_index=None):
    """Realizations of ONE class, for the comparisons that fix the class and vary something else.

    `labels` is one label or several. Several is how a comparison against a survey that does not
    resolve the subtypes asks for a class: the Carnegie Supernova Project's type II release gives no
    SN IIP / SN IIL split, so it asks for both and the parents come out in the proportion
    OpenUniverse itself drew them, which is what the classifier sees.

    `draw_population_from_parents` picks the class itself, which is the right interface for
    generating a sample and the wrong one for a figure that puts the generator's SN Ib against an
    observed SN Ib. Everything else is drawn the way the sample draws it -- the parent, its
    template, its shape, its host screen.

    The brightness is left AT THE REFERENCE, because measuring it needs the parent's light curve out
    of a 16 GB hdf5 and the comparisons that use this difference it away: they compare a colour, a
    decline rate or two sources against each other on the same object. A caller that needs the real
    brightness runs `measure_population_brightness` over the result."""
    if source_by_template_index is None:
        source_by_template_index = core_collapse_source_by_template_index(catalog)
    labels = {labels} if isinstance(labels, str) else set(labels)
    wanted = [
        index for index, (_, template_label) in source_by_template_index.items() if template_label in labels
    ]
    block = (
        catalog[catalog["template_index"].isin(wanted)] if wanted else catalog[catalog["label"].isin(labels)]
    )
    if block.empty:
        raise ValueError(f"the parent catalogue holds no {sorted(labels)}")
    block = block.reset_index(drop=True)
    population = []
    for index, redshift in enumerate(redshifts):
        parent = block.iloc[int(random_generator.integers(len(block)))]
        realization = realization_from_parent(
            parent, index, float(redshift), source_by_template_index, random_generator
        )
        population.append(apply_brightness_offset(realization, 0.0, float("nan"), 0))
    return population


def rendered_band_curves(realization, redshift, cosmology=None):
    """{band: (phase, apparent AB magnitude)} of one realization's model at `redshift`.

    The phase axis is the TEMPLATE'S OWN, which for every class here is also OpenUniverse's: its
    `peak_mjd` is where the archive's phase grid was aligned, SALT puts phase zero at B maximum and
    so does SNANA's peak for a SN Ia, and the SN Iax base SED is published with maximum at zero. So
    this axis and `(mjd - peak_mjd) / (1 + z)` of the parent's own light curve are the same axis,
    which is what makes an overlay of the two meaningful. `roman_light_curve` measures from B
    maximum instead, because that is what the survey window needs.

    A band the redshifted spectrum does not cover is absent; a phase the model has no flux at is
    NaN. Unlike `roman_light_curve` this does not require every band at every phase, does not drop
    colour discontinuities and does not enforce the model's flux floor: all three of those guard
    against magnitudes that are spuriously FAINT, which is what the window cares about and what a
    peak is unaffected by."""
    model = build_model(dict(realization, redshift=float(redshift)), cosmology)

    blue = max(model.minwave(), OBSERVED_WAVELENGTH_LIMITS[0])
    red = min(model.maxwave(), OBSERVED_WAVELENGTH_LIMITS[1])
    if red <= blue:
        return {}
    wavelength = _sampling_grid(blue, red, OBSERVED_WAVELENGTH_STEP)

    source = model.source
    phases = peak_phase(realization) + REST_FRAME_PHASES
    phases = phases[(phases >= source.minphase()) & (phases <= source.maxphase())]
    rows = []
    for phase in phases:
        flux = np.clip(model.flux(phase * (1.0 + redshift), wavelength), 0.0, None)
        if flux.max() < PHOTOMETRY_FLOOR:
            rows.append(dict.fromkeys(ALL_ROMAN_BANDS, np.nan))
            continue
        rows.append(spectrum_to_roman_magnitudes(wavelength, flux))
    curves = {}
    for band in ALL_ROMAN_BANDS:
        magnitudes = np.array([row[band] for row in rows])
        if np.isfinite(magnitudes).any():
            curves[band] = (phases, magnitudes)
    return curves


def rendered_peak_magnitudes(realization, redshift, cosmology=None):
    """{band: brightest AB magnitude} of one realization's model at `redshift`, band by band.

    The counterpart of `openuniverse_parents.parent_peak_magnitudes`, over the same rest-frame phase
    window and in the same bands. A band the redshifted spectrum does not cover is absent, which is
    how the measurement handles a parent at z = 2.9 whose blue bands sample rest-frame ultraviolet
    no template reaches."""
    peaks = {}
    for band, (_, magnitudes) in rendered_band_curves(realization, redshift, cosmology).items():
        with np.errstate(invalid="ignore"):
            peak = np.nanmin(magnitudes)
        if np.isfinite(peak):
            peaks[band] = float(peak)
    return peaks


# WHERE THE BRIGHTNESS IS MEASURED, in the REST frame of the parent. A band is only used if its
# pivot wavelength, de-redshifted by the parent's own redshift, lands inside this window. Both
# edges are where a template stops being a measurement: below 3500 A the archive's own blue edge is
# at 3000 A and SALT3's ultraviolet is the least constrained part of it, and above 10000 A every
# core-collapse template is `_WAVEEXT`, the extension rather than the pycoco base.
#
# THIS IS MEASURED, NOT ASSUMED. Per band, per object, against the median of that object's own
# bands -- so a constant brightness error cancels and only the band-to-band disagreement is left --
# over 30 parents a class of one healpix, in magnitudes:
#
#     rest        SN Ia          SN Iax         TDE
#     ~ A       median  rms    median  rms    median  rms
#      3000     +0.000 0.358   -0.024 0.095   +0.289 0.180
#      4000     +0.004 0.156   -0.066 0.091   +0.245 0.102
#      5000     +0.000 0.083   +0.031 0.088   +0.105 0.112
#      6000     +0.000 0.150   +0.010 0.070   +0.059 0.086
#      8000     -0.005 0.043   +0.087 0.089   -0.030 0.073
#     10000     +0.017 0.019   +0.126 0.024   -0.076 0.022
#
# The SN Ia column is the reason for the blue edge: the median is zero at every wavelength -- there
# is no colour bias, both models are SALT3 -- but the per-object scatter is eight times larger in
# the ultraviolet than in the near-infrared, so a parent at z = 2 whose only usable bands are blue
# gets a noisy brightness for no reason but where its bands landed. The SN Iax and TDE columns are
# the other thing this cannot fix and does not try to: a monotonic colour trend, which is the
# reimplementation and the substitution showing, and which is what `brightness_residual` reports.
#
# The window is a preference and not a requirement. A parent with no band inside it falls back to
# every band both sides cover, because a noisy brightness is still a brightness and dropping the
# object would select on redshift.
BRIGHTNESS_REST_WAVELENGTH_LIMITS = (3500.0, 10000.0)


@cache
def _band_pivot_wavelengths():
    """{band: photon-weighted pivot wavelength in A}, the one number that says where a band sits."""
    from kilonova.photometry.roman_noise import roman_bandpasses

    pivots = {}
    for band, bandpass in roman_bandpasses().items():
        # galsim keeps its bandpasses in nm.
        wavelength = np.asarray(bandpass.wave_list, dtype=float) * 10.0
        throughput = np.array([bandpass(one) for one in bandpass.wave_list], dtype=float)
        pivots[band] = float(
            np.trapezoid(throughput * wavelength, wavelength) / np.trapezoid(throughput, wavelength)
        )
    return pivots


def measure_brightness_offset(realization, parent_peaks, cosmology=None):
    """The parent's own brightness, as an offset from REFERENCE_ABSOLUTE_MAGNITUDE.

    `parent_peaks` is `openuniverse_parents.parent_peak_magnitudes` for the same object. Returns
    (offset, band-to-band spread, number of bands); the offset is NaN when no band is common to
    both sides, which is a parent this sample cannot re-render.

    The spread adjusts nothing. It is the disagreement between the parent's own photometry and this
    pipeline's rendering of the parent's own model, band by band, which for the classes whose model
    is not a substitution is a measurement of the pipeline and for the others is the size of the
    substitution."""
    at_reference = dict(realization, peak_absolute_magnitude=REFERENCE_ABSOLUTE_MAGNITUDE)
    rendered = rendered_peak_magnitudes(at_reference, realization["parent_redshift"], cosmology)
    common = [band for band in ALL_ROMAN_BANDS if band in parent_peaks and band in rendered]
    pivots = _band_pivot_wavelengths()
    blue, red = BRIGHTNESS_REST_WAVELENGTH_LIMITS
    inside = [band for band in common if blue <= pivots[band] / (1.0 + realization["parent_redshift"]) <= red]
    differences = [parent_peaks[band] - rendered[band] for band in (inside or common)]
    if not differences:
        return float("nan"), float("nan"), 0
    spread = float(max(differences) - min(differences)) if len(differences) > 1 else float("nan")
    return float(np.median(differences)), spread, len(differences)


def apply_brightness_offset(realization, offset, spread, bands):
    """Write a measured brightness into a realization, in place, and return it."""
    realization["brightness_offset"] = float(offset)
    realization["brightness_residual"] = float(spread)
    realization["brightness_bands"] = int(bands)
    realization["peak_absolute_magnitude"] = REFERENCE_ABSOLUTE_MAGNITUDE + float(offset)
    return realization


def build_model(realization, cosmology=None, **model_keywords):
    """The sncosmo model of one drawn contaminant: source, redshift, shape and normalisation.

    Split out of `roman_light_curve` so that a caller which needs the same object in some other
    bandpass gets the same model rather than a second copy of these six lines -- the comparison
    against the Carnegie Supernova Project photometry (`scripts/compare_csp_lightcurves.py`)
    synthesizes it through the CSP natural system. `model_keywords` reaches `sncosmo.Model`
    untouched, which is how that script attaches a Milky Way dust screen; nothing in the generated
    sample uses it, and the only extinction the sample itself carries is the host screen its parent
    was given."""
    sncosmo = _sncosmo()
    if cosmology is None:
        cosmology = openuniverse_cosmology()

    if realization["label"] == "SN Iax":
        source = iax_source(
            realization["iax_rise_time"], realization["iax_decline_b"], realization["iax_decline_r"]
        )
    elif realization["label"] == "TDE":
        source = tde_source(realization["tde_template_index"])
    else:
        source = _registry_source(realization["source_name"])
    if "host_av" in realization:
        # Rest-frame, and appended to whatever the caller asked for rather than replacing it: the
        # Milky Way screen a caller attaches is a second, observer-frame effect on the same model.
        model_keywords = dict(model_keywords)
        model_keywords["effects"] = [*model_keywords.get("effects", []), sncosmo.CCM89Dust()]
        model_keywords["effect_names"] = [*model_keywords.get("effect_names", []), "host"]
        model_keywords["effect_frames"] = [*model_keywords.get("effect_frames", []), "rest"]
    model = sncosmo.Model(source=source, **model_keywords)
    model.set(z=realization["redshift"], t0=0.0)
    if "host_av" in realization:
        model.set(hostr_v=realization["host_rv"], hostebv=realization["host_av"] / realization["host_rv"])
    if "salt2_x1" in realization:
        model.set(x1=realization["salt2_x1"], c=realization["salt2_c"])
    band, magnitude_system = REFERENCE_MAGNITUDE_BAND
    magnitude = realization["peak_absolute_magnitude"]
    # Set on the bare source, so the measured absolute magnitude means the same thing whether or not
    # the caller attached an effect: a dust screen must dim what leaves the model, not be undone by
    # renormalising through it.
    model.set_source_peakabsmag(magnitude, band, magnitude_system, cosmo=cosmology)
    return model


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
    model = build_model(realization, cosmology)

    blue = max(model.minwave(), OBSERVED_WAVELENGTH_LIMITS[0])
    red = min(model.maxwave(), OBSERVED_WAVELENGTH_LIMITS[1])
    if red <= blue:
        return {}
    wavelength = _sampling_grid(blue, red, OBSERVED_WAVELENGTH_STEP)

    source = model.source
    phase_of_maximum = peak_phase(realization)
    phases = phase_of_maximum + REST_FRAME_PHASES
    phases = phases[(phases >= source.minphase()) & (phases <= source.maxphase())]
    # Every band has to span the SAME phases. `build_window_from_model` marks a band unobserved
    # when the model does not cover that epoch, so bands with different time coverage would leave a
    # `not observed` token in a slot the cadence did schedule -- a window with fewer than three
    # observed bands in the first epoch, which is exactly what the training set never contains.
    phase_rows = []
    for phase in phases:
        # The model's time axis is its own -- `t0` sits at the source's phase zero -- while the
        # axis this returns is days from maximum, so the two differ by `phase_of_maximum`.
        observer_day = (phase - phase_of_maximum) * (1.0 + realization["redshift"])
        flux = np.clip(model.flux(phase * (1.0 + realization["redshift"]), wavelength), 0.0, None)
        if flux.max() < PHOTOMETRY_FLOOR:
            phase_rows.append(None)
            continue
        magnitudes = spectrum_to_roman_magnitudes(wavelength, flux)
        usable = all(
            np.isfinite(magnitudes[band]) and magnitudes[band] < MODEL_FLUX_FLOOR_MAGNITUDE
            for band in ALL_ROMAN_BANDS
        )
        phase_rows.append((observer_day, magnitudes) if usable else None)
    phase_rows = _longest_run(_drop_colour_discontinuities(phase_rows, realization["redshift"]))
    if not phase_rows:
        return {}
    days = np.array([day for day, _ in phase_rows])
    return {
        band: (days, np.array([magnitudes[band] for _, magnitudes in phase_rows])) for band in ALL_ROMAN_BANDS
    }


def peak_phase(realization):
    """The source's own phase of B maximum, which REST_FRAME_PHASES is measured from.

    Zero for the two sources this module builds itself: the SN Iax base SED is published with
    maximum at phase 0 and the MOSFiT photosphere bank is stored on a phase axis centred on peak
    luminosity, so neither needs the scan. For the registry sources it is `Source.peakphase`,
    cached by name because it costs a band scan and the answer is a property of the template."""
    if realization["label"] in ("SN Iax", "TDE"):
        return 0.0
    return _registry_peak_phase(realization["source_name"])


@cache
def _registry_peak_phase(source_name):
    # Read off the source with its default parameters. For the SALT sources the peak moves by a few
    # hundredths of a day with x1, which is far below the 1 d phase step.
    return float(_registry_source(source_name).peakphase("bessellb"))


def _drop_colour_discontinuities(rows, redshift):
    """Mark unusable every phase whose colours jump faster than a transient's can.

    MODEL_FLUX_FLOOR_MAGNITUDE catches the extreme end of a template running out of flux -- the
    ~59 mag `salt2-extended` reports where it has none at all -- and cannot catch the shoulder just
    inside it, because there no absolute threshold exists: 24.4 mag is a perfectly ordinary
    magnitude for a faint object. What gives it away is not the value but the rate. A real
    `salt2-extended` SN Ia arrives at the window like this,

        day     R062   Z087   Y106   J129   H158   F184
        -13.3  16.93  17.59  20.17  24.38  23.08  23.36
        -12.2  16.64  17.52  19.74  21.88  24.46  23.66
        -11.2  16.43  16.95  17.49  17.93  18.49  18.68

    where J129 runs 24.38 -> 21.88 -> 17.93 while R062 moves by 0.5 mag over the same two days. The
    object is not varying; the model has no near-infrared flux yet and what it reports there is
    numerical dust. Reaching the training set, those two rows are a "very red, no near-infrared"
    signature in exactly the bands the classifier reads, and only for the classes whose template
    library does that -- a class-correlated artefact, which is the species this module exists to
    remove.

    The test is on colour rather than on brightness, so a genuine fast rise is not touched: every
    band is measured against the brightest band of its own phase, which is the one least likely to
    be the broken one, and a phase is dropped when any of those five colours moves by more than
    MAXIMUM_COLOUR_RATE per rest-frame day. Over a 300-object pilot the median rate is 0.05 mag/d
    and the 99th percentile 0.7, so the cut sits well outside ordinary colour evolution -- but the
    distribution has no gap, and this is a threshold, not a proof. What makes it safe is where the
    offenders sit: every pair above the cut in that pilot was in the first three phases of the
    curve, at 26-39 mag, and carried a row like H158 = 38.9 next to Y106 = 26.7, or R062 four
    magnitudes fainter than Y106. The earlier phase of the offending pair is the one dropped, so
    `_longest_run` trims a head rather than punching a hole."""
    days = [None if row is None else row[0] for row in rows]
    colours = [
        None if row is None else {band: row[1][band] - min(row[1].values()) for band in ALL_ROMAN_BANDS}
        for row in rows
    ]
    kept = list(rows)
    for index in range(len(rows) - 1):
        if colours[index] is None or colours[index + 1] is None:
            continue
        rest_frame_days = (days[index + 1] - days[index]) / (1.0 + redshift)
        if rest_frame_days <= 0.0:
            continue
        jump = max(abs(colours[index + 1][band] - colours[index][band]) for band in ALL_ROMAN_BANDS)
        if jump / rest_frame_days > MAXIMUM_COLOUR_RATE:
            kept[index] = None
    return kept


def _sampling_grid(blue, red, step):
    """`step`-spaced wavelengths from `blue` to `red`, the red endpoint included.

    `np.arange` is half open, so it stops one step short of `red` and the last sample lands at
    `red - step`. `spectrum_to_roman_magnitudes` reads coverage off the sampled grid rather than off
    the model, so that missing step made a band the model covers exactly to its red edge come back
    NaN, and the object be dropped for "no coverage". It bit exactly where the coverage is exact:
    the unpadded `salt3-nir` reached F184 at z = 0.05 and no further, so every SN Ia in
    z = [0.0500, 0.0505) -- about 480 objects of the intended run -- disappeared in silence."""
    count = int(np.floor((red - blue) / step)) + 1
    grid = blue + step * np.arange(count)
    if grid[-1] < red:
        grid = np.append(grid, red)
    return grid


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

    `tier` may be one tier or several. Several is the cheap way to ask: the light curve
    `roman_light_curve` returns is the same object seen in every Roman band, and which of them a
    tier observes is decided afterwards, so one call serves both tiers and calling this once per
    tier does the expensive half of the work twice. It matters because the tiers are not disjoint
    populations -- 717 863 of the 717 864 wide contaminants of OpenUniverse are also deep ones.

    Returns {tier: (windows, rejected)} when given several tiers, and the bare pair when given one.

    Objects whose spectrum does not cover every band of a tier are dropped from that tier, as are
    the ones the survey never detects -- the same rule the OpenUniverse path applies, where an
    undetected transient simply produces no window."""
    tiers = [tier] if isinstance(tier, str) else list(tier)
    per_tier = {
        one_tier: (
            build_tier_constants(one_tier),
            saturation_magnitude(one_tier),
            [],
            {"coverage": 0, "undetected": 0, "saturated_kept": 0},
        )
        for one_tier in tiers
    }
    for realization in population:
        curves = roman_light_curve(realization, cosmology=cosmology)
        for one_tier, (constants, bright_limit, windows, rejected) in per_tier.items():
            _add_one_window(realization, curves, constants, bright_limit, windows, rejected, one_tier)
    results = {
        one_tier: ((pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()), rejected)
        for one_tier, (_, _, windows, rejected) in per_tier.items()
    }
    return results[tiers[0]] if isinstance(tier, str) else results


def _add_one_window(realization, curves, constants, bright_limit, windows, rejected, tier):
    """One tier's window for one already-computed light curve, appended in place."""
    if not set(constants["bands"]).issubset(curves):
        rejected["coverage"] += 1
        return
    model = {band: curves[band] for band in constants["bands"]}
    # The visit grid is built here rather than left to `build_window_from_model`, because the
    # branch that derives it from the model's own MJD range ignores `visit_index_offset`: it is the
    # OpenUniverse branch, where the visits ARE the survey's and the parity is not free. Passing
    # the grid takes the same path the kilonovae take, which is where the drawn parity is honoured.
    # Until this call passed it, `cadence_parity` was drawn, stored and thrown away, and every izc
    # object started on an even visit -- 80/20 towards (Z087, Y106, J129) in the first epoch where
    # OpenUniverse runs 60/40 the other way.
    days = model[constants["bands"][0]][0]
    base_epochs = np.arange(
        days.min() + realization["visit_phase_offset_days"], days.max() + 1e-9, BASE_CADENCE_DAYS
    )
    # The parent comes FIRST and unmodified, because the split reads its group off this string:
    # `training/openuniverse_data.py` maps an izc id to `snana_{healpix}_{object}`, which is the
    # same group key the parent itself carries in the OpenUniverse windows. Every re-rendering of
    # one parent, and the parent, land on one side of the split. The redshift and the running index
    # follow to make the id unique -- one parent is re-rendered many times.
    object_id = f"izc_{realization['parent_key']}_{realization['redshift']:.4f}_{realization['index']:08d}"
    window = build_window_from_model(
        object_id,
        model,
        constants,
        realization["redshift"],
        GENTYPE_BY_LABEL[realization["label"]] + IZC_GENTYPE_OFFSET,
        base_epochs=base_epochs,
        noise_seed=realization["index"],
        visit_index_offset=realization["cadence_parity"],
    )
    if window is None:
        rejected["undetected"] += 1
        return
    window["tier"] = tier
    # `build_window_from_model` reads the label off the gentype, and IZC_GENTYPE_OFFSET puts it
    # outside GENTYPE_LABEL, so every izc window came out "UNKNOWN". The label is restored in
    # OpenUniverse's own vocabulary -- the class this object stands in for -- and the finer
    # subtype this module draws (IIP/IIL/IIn, which OpenUniverse pools into "SN II") is kept in
    # its own column instead of being smuggled into `label`.
    window["label"] = GENTYPE_LABEL[GENTYPE_BY_LABEL[realization["label"]]]
    window["izc_subtype"] = realization["label"]
    # What this object was re-rendered from, and how well the rendering reproduced it. None of the
    # four is read by the training path; they are what makes a generated window traceable back to
    # the OpenUniverse object it came from.
    window["parent_key"] = realization["parent_key"]
    window["parent_z_CMB"] = realization["parent_redshift"]
    window["brightness_offset"] = realization["brightness_offset"]
    window["brightness_residual"] = realization["brightness_residual"]
    observed = window[window["observed"]]
    saturated = observed["mag_true"] < observed["band"].map(bright_limit)
    if saturated.any():
        rejected["saturated_kept"] += 1
    windows.append(window)


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


# --- Running the sample -------------------------------------------------------------------------
# One task per healpix, because that is what the brightness measurement costs: it reads the parent's
# light curve out of a 16 GB hdf5, and grouping the population by the file its parents live in opens
# each of the 33 files once instead of once per object. The rendering does not care, so it rides
# along in the same task, and what crosses the process boundary is the path of a parquet shard
# rather than the windows themselves: 670 000 objects are 27 million rows over the two tiers, which
# is more than this machine can hold as DataFrames while it waits for the last worker.


def measure_population_brightness(population, hdf5_path, cosmology=None):
    """Fill in the brightness of every realization whose parent lives in one hdf5, in place.

    Returns the number of realizations left without one: a parent whose light curve shares no band
    with the rendering of its own model, which cannot be re-rendered and is dropped by the caller.

    The measurement is cached per parent. One parent is re-rendered many times -- the rare classes
    tens of times -- and every copy of it has the same model at the same parent redshift, so the
    render that costs the measurement is done once."""
    import h5py

    unmeasured = 0
    measured_by_parent = {}
    with h5py.File(hdf5_path, "r") as handle:
        for realization in population:
            parent_id = realization["parent_id"]
            if parent_id not in measured_by_parent:
                group = handle.get(str(parent_id))
                if group is None:
                    measured_by_parent[parent_id] = (float("nan"), float("nan"), 0)
                else:
                    peaks = openuniverse_parents.parent_peak_magnitudes(
                        group,
                        realization["parent_redshift"],
                        realization["parent_peak_mjd"],
                        ALL_ROMAN_BANDS,
                    )
                    measured_by_parent[parent_id] = measure_brightness_offset(realization, peaks, cosmology)
            offset, spread, bands = measured_by_parent[parent_id]
            if bands == 0:
                unmeasured += 1
                continue
            apply_brightness_offset(realization, offset, spread, bands)
    return unmeasured


def run_izc_healpix(healpix, population, source_directory, tiers, shard_directory=None, cosmology=None):
    """Measure, render and window every object of one healpix.

    Returns ({tier: parquet shard path}, summary) when `shard_directory` is given and
    ({tier: windows}, summary) when it is not. The shards are how the full run survives its own
    size: 670 000 objects are 27 million rows over the two tiers, and holding them as DataFrames
    until the end needs more memory than this machine has. Each task writes its own and the parent
    streams them together."""
    hdf5_path = Path(source_directory) / f"snana_{healpix}.hdf5"
    if hdf5_path.stat().st_size == 0:
        # A cloud-storage placeholder reads as an empty file rather than as an error, which would
        # silently produce a sample with no brightness at all. See docs/generate_datasets.md.
        raise SystemExit(f"{hdf5_path} is empty; mark it available offline before running")
    unmeasured = measure_population_brightness(population, hdf5_path, cosmology)
    measurable = [one for one in population if one["brightness_bands"] > 0]
    results = build_izc_windows(measurable, tiers, cosmology)

    summary = {"objects": len(population), "unmeasured": unmeasured}
    output = {}
    for tier in tiers:
        windows, rejected = results[tier]
        summary[tier] = dict(rejected, windows=int(windows["object_id"].nunique()) if len(windows) else 0)
        if shard_directory is None:
            output[tier] = windows
            continue
        if not len(windows):
            output[tier] = None
            continue
        shard = Path(shard_directory) / f"izc_windows_{tier}_{healpix}.parquet"
        windows.to_parquet(shard, index=False)
        output[tier] = shard
    return output, summary


_IZC_WORKER_STATE = {}


def _izc_worker_initializer(source_directory, tiers, shard_directory):
    _IZC_WORKER_STATE["source_directory"] = source_directory
    _IZC_WORKER_STATE["tiers"] = list(tiers)
    _IZC_WORKER_STATE["shard_directory"] = shard_directory
    register_sources()


def _izc_healpix_task(work_item):
    healpix, population = work_item
    shards, summary = run_izc_healpix(
        healpix,
        population,
        _IZC_WORKER_STATE["source_directory"],
        _IZC_WORKER_STATE["tiers"],
        _IZC_WORKER_STATE["shard_directory"],
    )
    return healpix, shards, summary


def run_izc_tiers(population, source_directory, tiers, output_paths, workers=1, shard_directory=None):
    """Generate the whole sample and write one parquet per tier. Returns (totals, per-tier summary).

    Ordered `imap` over the healpix in sorted order, not `imap_unordered`, so the row order of the
    parquet is reproducible between runs."""
    import logging
    import multiprocessing
    import os
    import shutil
    import time

    import pyarrow.parquet as pq

    logger = logging.getLogger(__name__)
    by_healpix = {}
    for realization in population:
        by_healpix.setdefault(realization["parent_healpix"], []).append(realization)
    work_items = sorted(by_healpix.items())

    if shard_directory is None:
        shard_directory = Path(next(iter(output_paths.values()))).parent / ".izc_shards"
    shard_directory = Path(shard_directory)
    shard_directory.mkdir(parents=True, exist_ok=True)

    shards = {tier: [] for tier in tiers}
    totals = {"objects": 0, "unmeasured": 0}
    per_tier = {tier: {"windows": 0, "coverage": 0, "undetected": 0, "saturated_kept": 0} for tier in tiers}
    start = time.time()

    if workers > 1:
        # Each worker single-threaded: 6 processes x N BLAS threads saturates the machine.
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(variable, "1")
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(
            workers,
            initializer=_izc_worker_initializer,
            initargs=(str(source_directory), tuple(tiers), str(shard_directory)),
        )
        stream = pool.imap(_izc_healpix_task, work_items, chunksize=1)
    else:
        _izc_worker_initializer(str(source_directory), tiers, str(shard_directory))
        pool = None
        stream = (_izc_healpix_task(item) for item in work_items)

    try:
        for counter, (healpix, written, summary) in enumerate(stream, start=1):
            totals["objects"] += summary["objects"]
            totals["unmeasured"] += summary["unmeasured"]
            for tier in tiers:
                if written[tier] is not None:
                    shards[tier].append(written[tier])
                for key, value in summary[tier].items():
                    per_tier[tier][key] += value
            elapsed = time.time() - start
            rate = totals["objects"] / elapsed if elapsed else 0.0
            logger.info(
                "[izc] healpix %d (%d/%d)  objects=%d  unmeasured=%d  %s  %.1f obj/s  ETA %.0f min",
                healpix,
                counter,
                len(work_items),
                totals["objects"],
                totals["unmeasured"],
                "  ".join(f"{tier}={per_tier[tier]['windows']}" for tier in tiers),
                rate,
                (len(population) - totals["objects"]) / rate / 60.0 if rate else 0.0,
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    for tier in tiers:
        if not shards[tier]:
            logger.warning("[%s] no izc object reached a detection; nothing written", tier)
            continue
        writer = None
        rows = 0
        for shard in shards[tier]:
            table = pq.read_table(shard)
            rows += table.num_rows
            if writer is None:
                writer = pq.ParquetWriter(output_paths[tier], table.schema)
            writer.write_table(table)
        writer.close()
        logger.info(
            "[%s] DONE windows=%d rows=%d dropped: no coverage=%d undetected=%d -> %s",
            tier,
            per_tier[tier]["windows"],
            rows,
            per_tier[tier]["coverage"],
            per_tier[tier]["undetected"],
            output_paths[tier],
        )
    shutil.rmtree(shard_directory, ignore_errors=True)
    return totals, per_tier
