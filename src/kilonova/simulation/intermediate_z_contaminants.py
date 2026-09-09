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

WHAT EACH CLASS IS MADE OF. Every choice below is argued where it is made; this is the index.

    class     spectral model                     luminosity function from     calibrated?
    -------   --------------------------------   --------------------------   -----------
    SN Ia     salt3-nir (Pierel et al. 2022),    a Gaussian fitted to           no
              held flat to 21000 A                OpenUniverse's own salt2_mB
    SN Iax    SN 2005hk warped by Jha & Dai's    the replayed template bank,    no
              model, the bank replayed            row by row
    SN Ib     SNANA NON1A + nugent-sn1bc         OpenUniverse's peak_mag_g      YES
    SN Ic     SNANA NON1A + nugent-hyper         OpenUniverse's peak_mag_g      YES
    SN IIP    nugent-sn2p + 23 SNANA NON1A       Richardson et al. (2014)       YES
    SN IIL    nugent-sn2l                        Richardson et al. (2014)       YES
    SN IIn    nugent-sn2n                        Richardson et al. (2014)       YES
    TDE       MOSFiT `tde` (Guillochon+ 2018)    MOSFiT's own physics           no

The third column is where each luminosity function STARTED, not what its median is now. Two
different quantities live there and confusing them wastes a session: OpenUniverse's `salt2_mB` and
`peak_mag_g` are catalogue columns read once to seed the luminosity function, while the CALIBRATION
is anchored on something else entirely -- M(Y106). For a calibrated class the median below no
longer equals its seed.

"Calibrated" means the median of the luminosity function carries a measured offset that makes the
class reproduce OpenUniverse's own median M(Y106): the brightest Y106 `mag_true` of the window
minus the distance modulus, deep tier, z < 0.45, both populations after the same detection cut.
`scripts/calibrate_izc_brightness.py` measures it and PEAK_ABSOLUTE_MAGNITUDE records it. One band
because the offset is one scalar per class -- more bands would measure the same number five times,
not fit five. Y106 is the anchor because it is mid-range, present in both tiers, and at these
redshifts still samples the rest-frame red optical, where these templates are observed rather than
extrapolated; anchoring on F184 would put the module's largest declared uncertainty inside the
anchor. What a scalar cannot move is the COLOUR, which the templates set, so the same script also
reports the per-band residual as a diagnostic that adjusts nothing. It is formed PER OBJECT --
median of M(band) - M(Y106) on objects detected in both, one population against the other -- which
differences away the distance and the brightness and leaves the shape of the SED. Measured on 2500
objects a class, in magnitudes, positive meaning the izc class is REDDER than OpenUniverse's:

    class     Z087     J129     H158     F184     reading
    -------   ------   ------   ------   ------   -------------------------------------------
    SN Ia     +0.001   -0.002   -0.014   -0.017   the control: both sides are SALT3
    SN Iax    -0.044   +0.031   +0.059   +0.069   same model on both sides, and it shows
    SN Ib     -0.112   -0.013   +0.079   +0.153   mixed
    SN II     +0.025   -0.118   -0.181   -0.271   monotonic in wavelength: OpenUniverse redder
    SN Ic     -0.026   -0.157   -0.034   +0.285   not monotonic, sign flips, unexplained
    TDE       +0.053   -0.043   -0.077   -0.100   small, and it is a model substitution anyway

SN Ia and SN Iax are the two classes whose spectral model is the same on both sides, and they are
the two that come out flat. That is the control passing: the statistic is measuring the library,
not the machinery. The core-collapse classes are where the libraries differ and they are where the
residual is, up to 0.27 mag in F184 for SN II -- in the direction the module predicted, since the
V19+HostXT SEDs OpenUniverse used are redder in the near-infrared than the SNANA and Nugent ones
here.

THAT ORDERING IS MISLEADING ON ITS OWN, and the width is what fixes it. A classifier does not see
a median, it sees an object against the spread of the class, so the residual that matters is the
one measured in units of the population's own colour scatter. Same measurement, divided by the izc
sigma the script now also records:

    class     Z087   J129   H158   F184     izc sigma in F184  vs OpenUniverse's
    -------   ----   ----   ----   ----     -----------------  -----------------
    SN Ia     0.02   0.03   0.14   0.15     0.114              0.157
    SN Iax    0.37   0.22   0.35   0.31     0.226              0.264
    SN Ic     0.04   0.28   0.05   0.44     0.648              0.213
    SN II     0.13   0.44   0.60   0.61     0.446              0.418
    SN Ib     0.71   0.04   0.21   0.26     0.582              0.248
    TDE       1.61   1.59   1.57   1.59     0.063              0.024

TDE is the worst class by this measure and it is not close, even though its raw residual is the
second smallest in the table. Its colour has almost no scatter on either side -- a MOSFiT
photosphere is a blackbody and AT2019qiz is one observed SED -- so a tenth of a magnitude is more
than one and a half population widths, where the same tenth of a magnitude inside SN II's 0.45 mag
spread is a fifth of one. Read this way the ordering of the sample's colour problems is TDE first
by a factor of two and a half, then SN Ib in Z087, then SN II in the near-infrared.

The last column carries a second finding, which is not about the median at all: SN Ic and SN Ib are
three and two times WIDER in colour than the OpenUniverse classes they stand in for, and SN Ia is
narrower. A distribution of the right centre and the wrong width is separable too, and for SN Ic --
0.648 against 0.213 in F184 -- the width is the larger discrepancy by far. Its likely cause is
visible in SOURCES_BY_LABEL: the class is a heterogeneous set of templates spanning `nugent-hyper`,
the broad-lined SN 1998bw, and each draw takes one whole template rather than interpolating, so the
template-to-template colour scatter enters the population directly. The SN Ia case is the opposite
sign and has its own named cause: SALT2_C is drawn as a Gaussian, and the distribution OpenUniverse
drew `c` from (Scolnic & Kessler 2016) is skewed, not Gaussian.

WHAT THIS MEANS AND WHAT IT DOES NOT. It is NOT a bug to fix by tuning: a scalar luminosity offset
cannot move a colour, and warping a template to match another template would replace a measured
disagreement with a fitted one. It IS a limitation with teeth, because the library split is aligned
with redshift -- V19 above, SNANA/Nugent below -- so a colour offset that depends on the library
looks to the classifier exactly like a colour offset that depends on redshift, which is the shape
of the shortcut this module exists to remove. The size above bounds it: 0.27 mag in the worst band
of the worst class against the four magnitudes of the brightness shortcut. It belongs in the paper
with that comparison, and the test that settles it is whether a classifier can separate izc from
OpenUniverse contaminants AT FIXED REDSHIFT in the range where both exist.

SN Ic is the one line here that is not understood: the residual is not monotonic in wavelength and
changes sign, which is structure rather than a colour temperature offset, and nothing in the choice
of library predicts it.

The three uncalibrated classes are uncalibrated on purpose: their brightness is not a free
parameter to fit, it comes from a published model, and moving it would make the class agree by
construction instead of by model.

Two further sources of population, both measured off OpenUniverse rather than assumed: the SN Ia
shape and colour (SALT2_X1, SALT2_C) and the SN Iax host dust screen (IAX_HOST_AV_RANGE), which is
the only extinction this module applies to anything. The redshift distribution is not chosen here
at all -- `draw_population` takes it from its caller.

Two substitutions the sample carries, both declared rather than hidden:

  * OpenUniverse drew core-collapse SEDs from `NON1ASED.V19_CC+HostXT_WAVEEXT`. Its BASE templates
    are public and were checked here rather than assumed: `NON1ASED.V19_CC+HostXT` ships in the
    SNANA SNDATA_ROOT distribution (Zenodo record 4015325), 67 `pycoco_*.SED.gz` files, and its
    `SIMGEN_INCLUDE_NON1A.INPUT` carries 17 SNTYPE 20 (IIP) + 7 SNTYPE 22 (IIL) = 24 II, 13 Ib and
    7 Ic -- exactly the 24/13/7 the OpenUniverse paper reports, so this is not a lookalike set but
    the same one, with the per-template rate weights and luminosity functions that simulation used.
    What is NOT public is the `_WAVEEXT` half of the name. Those templates run 1605 to 11000 A,
    measured off the files, and the string `WAVEEXT` appears nowhere in either the 14 216 entries
    of the 2024-07-04 release, contemporaneous with OpenUniverse2024, or the 14 396 of the
    2026-04-10 one: the six NON1ASED models shipped are J17_CC, K10_CC, P18_CC, S11_CC,
    V19_CC+HostXT and V19_CC_noHostXT. The extension to 25000 A by the methods of Pierel et al.
    (2018) is what is missing, and 11000 A covers Roman only for z >= 0.91 -- the opposite of the
    range needed here. So the sources below are the SNANA NON1A and Nugent libraries, which sncosmo
    ships already extended. Different library for the low-redshift half of the contaminants than
    for the high-redshift half.

    That is a substitution of convenience and there is a better path, not taken yet: `snsedextend`
    (Pierel, on PyPI) is the public implementation of the very method OpenUniverse cites, so the
    published 11000 A templates can be extended HERE, by that method, instead of being replaced by
    a different library. Same base templates as OpenUniverse and the same extension procedure,
    which is what the colour residual needs in order to mean anything.
  * OpenUniverse's TDE is the observed SED of AT2019qiz, which is not published as a usable
    template; MOSFiT's `tde` stands in for it. SLSN-I and PISN (0.39 % of the mix) are left out
    entirely for the same reason.

SN Iax is NOT a substitution and is the one class regenerated from OpenUniverse's own model rather
than a lookalike -- same base SED, same warps, same template bank, same index. See the SN Iax
block below for what that took.

And one limitation that no choice of library removes: at high redshift the Roman bands sample the
rest-frame optical, which these templates measure; at low redshift they sample the rest-frame
near-infrared, which for core-collapse supernovae is poorly observed and is extrapolation in every
library. The izc sample carries more model uncertainty in H158 and F184 than the OpenUniverse one.

That extrapolation is measured, not just declared: `scripts/compare_csp_lightcurves.py` puts these
same models, continuous, under the discrete uBgVriYJH photometry of the Carnegie Supernova Project
at the redshifts this module generates. Every class here has a CSP counterpart except the TDE.
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

# --- SN Iax ------------------------------------------------------------------------------------
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

# Host extinction, and the only class in this module that gets an explicit dust screen. See the
# host extinction block below for why the others do not: SN Ia carry theirs inside SALT's `c` and
# the OpenUniverse core-collapse classes carry none at all. SN Iax is the class OpenUniverse dusts
# by hand, and leaving it undusted here left the izc Iax half a magnitude brighter and bluer than
# the OpenUniverse Iax of the same class -- a class-correlated offset of exactly the kind the
# sample exists to remove.
#
# Measured off OpenUniverse's own 115 645 SNe Iax: RV is 3.1 for every one of them, and AV is
# bounded at 0.001 and 3.0, which are SNANA's generation limits and are reproduced here rather than
# smoothed over. The shape is the Wood-Vasey et al. (2007) eq. 2 form OpenUniverse names for its
# other dusted classes, exp(-AV/tau) + W exp(-AV^2/2 sigma^2).
#
# The three parameters are a FIT to OpenUniverse's realized AV values, not OpenUniverse's own
# declared inputs, which the catalogues do not carry. Read them as a parameterisation and not as a
# provenance: (sigma, W) are degenerate against each other and the pair that comes out is a broad
# second component rather than the narrow core the form is usually written for. What is checked is
# the distribution itself, and it is reproduced to a KS distance of 0.0015 -- every percentile from
# the 1st to the 99th within 0.015 mag, the median to 0.000 and the mean to 0.001.
IAX_HOST_AV_RANGE = (0.001, 3.0)
IAX_HOST_AV_TAU = 1.010
IAX_HOST_AV_SIGMA = 0.555
IAX_HOST_AV_WEIGHT = 2.40
IAX_HOST_AV_STEP = 0.0005
IAX_HOST_RV = 3.1

# --- TDE ---------------------------------------------------------------------------------------
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
    # SN II split by subtype rather than pooled. Pooling them forced one luminosity function over
    # the subtypes together, and its sigma of 1.61 mag had tails reaching M = -20.8 -- a
    # superluminous supernova wearing a IIP label. The library separates them by its own SNTYPE
    # (20 against 22); it is OpenUniverse's output catalogue that pools them into gentype 32.
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
    # Likewise, and a stand-in for the whole bank: `roman_light_curve` builds the drawn template.
    "TDE": [TDE_SOURCE_NAME],
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
UNIFORM_CLASS_SHARE = 1.0 / 6.0
# Measured off OpenUniverse's own template set rather than taken from Li et al. (2011): of the 24
# SN II templates it drew, 17 are SNTYPE 20 and 7 are SNTYPE 22. The volumetric split this used to
# carry was a stand-in for exactly this number, and the number is now readable.
SN_II_SUBTYPE_FRACTION = {"SN IIP": 17.0 / 24.0, "SN IIL": 7.0 / 24.0}
CLASS_FRACTION = {
    "SN IIP": UNIFORM_CLASS_SHARE * SN_II_SUBTYPE_FRACTION["SN IIP"],
    "SN IIL": UNIFORM_CLASS_SHARE * SN_II_SUBTYPE_FRACTION["SN IIL"],
    "SN Ia": UNIFORM_CLASS_SHARE,
    "SN Iax": UNIFORM_CLASS_SHARE,
    "SN Ic": UNIFORM_CLASS_SHARE,
    "SN Ib": UNIFORM_CLASS_SHARE,
    "TDE": UNIFORM_CLASS_SHARE,
}

# The subtypes all map back to OpenUniverse gentype 32, the class they stand in for.
GENTYPE_BY_LABEL = {
    "SN Ia": 10,
    "SN Iax": 12,
    "SN Ib": 21,
    "SN Ic": 26,
    "SN IIP": 32,
    "SN IIL": 32,
    "TDE": 42,
}

# Peak absolute magnitude, (median, sigma), in the normalisation band of
# PEAK_ABSOLUTE_MAGNITUDE_BAND. These are handles on a model, not measured B magnitudes, and the
# medians are anchored to the only thing that has to come out right: what Roman sees.
#
# The provenance, and why it stopped being enough. The starting values were OpenUniverse's own --
# salt2_mB - mu for SN Ia, peak_mag_g - mu below z = 0.3 for the core-collapse classes, Richardson
# et al. (2014) for the SN II subtypes OpenUniverse pools into one label. Normalising a template in
# rest-frame B fixes its brightness in B and leaves its brightness in Y106 to the template's own
# B - Y colour, and that colour is precisely what these libraries do not agree on: OpenUniverse's
# V19+HostXT core-collapse SEDs are redder in the near-infrared than the SNANA and Nugent templates
# used here. Forward-modelled at OpenUniverse's own low-redshift SN Ic redshifts, the izc SN Ic
# came out half a magnitude fainter in Y106 than OpenUniverse's -- a class-correlated brightness
# offset inside the very band the classifier reads, in the sample that exists to remove one.
#
# So the medians below carry a calibration offset, measured rather than assumed, and
# `scripts/calibrate_izc_brightness.py` is where it comes from. For each class, izc objects are
# generated at redshifts resampled from OpenUniverse's own low-redshift objects of that class
# (z < 0.45, deep tier), run through the same window builder and the same detection cut, and
# summarised by the same statistic: the brightest Y106 mag_true of the window minus the distance
# modulus. No K-correction is assumed anywhere, each template supplies its own colours, both sides
# use `openuniverse_cosmology()`, and the two populations are compared after the same selection.
#
# Measured on 2500 izc objects a class, standard error of the median in parentheses:
#
#     class     OpenUniverse M(Y106)      izc      offset      applied?
#     SN Ib           -17.612          -17.493   -0.119 (0.025)   yes
#     SN Ic           -18.123          -18.082   -0.041 (0.035)   no, 1.2 sigma
#     SN II           -17.246          -17.264   +0.018 (0.030)   no, 0.6 sigma
#     SN Ia           -18.785          -18.930   +0.145 (0.009)   no, see below
#     SN Iax          -16.476          -16.421   -0.055 (0.036)   no, see below
#     TDE             -17.560          -17.122   -0.438 (0.043)   no, see below
#
# An offset below two standard errors of the median is not applied: it is one draw's worth of
# noise, and tuning a class to it would make the sample agree with a sample rather than with a
# model. SN Ic and SN II are converged by that rule; SN Ib is not and its median carries -0.119.
#
# Three classes are never adjusted here, whatever they measure, because their brightness is not a
# free parameter -- it comes from a published model, and moving it would make the class agree by
# construction instead of by model:
#
#   * SN Ia, whose normalisation is OpenUniverse's own salt2_mB and whose model is OpenUniverse's
#     own SALT3. Its +0.145 (0.009) is the most significant residual in the table and it has a
#     known cause, which is not a luminosity function: OpenUniverse standardises, and this module
#     does not. See the SALT2_M0 block, where that is measured and where the fix is named.
#   * SN Iax, whose M_V is read straight off the template bank OpenUniverse drew from. This class
#     carried +0.587 mag in this module's comments, documented at length as an irreducible
#     disagreement between Jha & Dai's luminosity function and what PLAsTiCC realised. TWO things
#     turned out to be wrong with that, and they are separate.
#
#     First, the number. It is not reproducible. Run against the module as it stood at commit
#     a591116 -- its own code, its own Planck18 -- this script measures +0.073 (0.040), not +0.587.
#     The old figure came from an ad-hoc measurement that lived only in a comment, and whatever it
#     did differently is not recoverable. That is the reason this script exists.
#
#     Second, the model. Replaying `Iax-model.ipynb` against its own inputs found four real bugs in
#     the reproduction: the bank was resampled instead of replayed, the normalisation was at the
#     light-curve peak instead of at phase zero, the base SED was built on the repacked file's 81
#     phases instead of the notebook's 231 (which moved three index-addressed processing steps onto
#     the wrong phases), and the pre-explosion suppression selected nothing. Measured one at a time
#     they are worth +0.021, +0.025, +0.079 and, for the cosmology alongside them, +0.002 -- 0.128
#     mag together, which is exactly the distance from +0.073 to the -0.055 (0.036) the class sits
#     at now. The budget closes. Nothing was tuned to get there, and the class is now consistent
#     with OpenUniverse's own at the 1.5 sigma level.
#   * TDE, whose brightness is MOSFiT's physics. Its -0.438 (0.043) is the largest offset the
#     sample now carries and it is a real one: OpenUniverse's TDE are the observed SED of
#     AT2019qiz and this module's are a MOSFiT photosphere, so this is the size of the model
#     substitution declared in the module docstring, measured. It belongs in the paper as such.
#
# The sigmas are NOT calibrated, only the medians. Measured the same way the izc scatter runs 0.99
# to 1.70 against OpenUniverse's 0.33 to 1.19, and the two are not the same quantity: part of the
# observed spread is the template-to-template colour scatter, which is already in the sample.
PEAK_ABSOLUTE_MAGNITUDE = {
    "SN Ia": (-19.404, 0.270),
    "SN Ib": (-17.166, 1.17),
    "SN Ic": (-17.502, 1.22),
    "SN IIP": (-16.788, 0.97),
    "SN IIL": (-17.968, 0.90),
}

# The band and magnitude system each luminosity function is normalised in. Everything here is
# rest-frame B on the AB system except SN Iax, whose luminosity function Jha & Dai published in
# rest-frame V on the Vega system; converting it would mean assuming a colour the model already
# carries, so the drawn magnitude is applied in the band it was measured in.
#
# The B values are not on one photometric system either, and after the calibration above they do
# not need to be: SN Ia inherits SALT2's mB, which is BD+17, and the SN II subtypes came from
# Richardson et al. on Vega. Any such offset is inside the number that was anchored to M(Y106), and
# what leaves this module is a Roman magnitude, never a B one.
PEAK_ABSOLUTE_MAGNITUDE_BAND = {"SN Iax": ("bessellv", "vega")}

# Shape and colour, from the OpenUniverse SN Ia population. The names are OpenUniverse's own: its
# catalogue calls these columns `salt2_x1` and `salt2_c` even though its model is SALT3, which is
# SNANA's naming for the parameterisation rather than for the model, and keeping the names makes
# the two catalogues cross-referenceable.
#
# WHICH OpenUniverse population, which is the whole difficulty. These started as (0.152, 0.914) and
# (-0.017, 0.075). That is a DETECTED subsample: the mean c reproduces a peak_mag_F < 25 cut to the
# fourth decimal and the x1 falls between that cut and peak_mag_F < 26, and such a subsample has a
# median redshift of 0.62 to 1.38 depending on where the cut goes. Detection at those redshifts is
# Malmquist selection and it keeps the blue, broad end of the population. Applying it here imports
# that selection into a redshift range which does not have one: below z = 0.45 all 5622 of
# OpenUniverse's SNe Ia are detected, so generated and detected are the same population, and its
# mean c is -0.0067 rather than -0.017 and its mean x1 -0.014 rather than +0.152.
#
# Left as they were, the izc SN Ia came out bluer, brighter and BROADER than OpenUniverse's own SN
# Ia at the same redshift -- 0.031 mag of colour through beta = 3.1 and 0.025 mag of shape through
# alpha = 0.15, both in one direction, in the class that is the largest single share of the sample.
# Small, and exactly the species of class-correlated offset the module exists to remove.
#
# In `c` the dust is not separable and is not meant to be: SALT carries host extinction and
# intrinsic colour in the same number, drawn by OpenUniverse from Scolnic & Kessler (2016). That is
# why OpenUniverse records no AV for SN Ia (see the host extinction note below) and why attaching a
# dust screen here would count the reddening twice.
#
# WHAT IS STILL MISSING HERE, measured and not yet applied. OpenUniverse does not draw a SN Ia
# brightness independently of x1 and c the way this module does -- it standardises. Over all
# 224 118 of its SNe Ia,
#
#     salt2_mB + 0.15 * salt2_x1 - 3.1 * salt2_c - mu(z) = -19.363447 +- 0.000103 mag
#
# with alpha = 0.15 and beta = 3.1 for every object and gammaDM identically zero. That residual is
# not a fit, it is an identity: OpenUniverse's SN Ia carry NO intrinsic scatter at all, and the
# 0.269 mag spread of their absolute mB is entirely the spread of x1 and c through that relation.
# This module instead draws M_B from an independent Gaussian (PEAK_ABSOLUTE_MAGNITUDE["SN Ia"]),
# which reproduces the width of the distribution and not its structure: the izc SN Ia have no
# width-luminosity and no colour-luminosity relation, where OpenUniverse's have both exactly.
# Reproducing it means setting `x0` per object from that identity instead of calling
# `set_source_peakabsmag`, and it would replace the one remaining calibration this class needs.
SALT2_X1 = (-0.014, 0.909)
SALT2_C = (-0.0067, 0.0735)
SALT2_ALPHA = 0.15
SALT2_BETA = 3.1
SALT2_M0 = -19.363447  # at OPENUNIVERSE_H0; degenerate with it, see the cosmology block

# Host extinction, class by class, and NOT because of what is physically right: the sample's job is
# to be indistinguishable from OpenUniverse's contaminants except in redshift, so any departure
# from what OpenUniverse did becomes a class-correlated feature the classifier can learn and the
# sky does not have. Audited against the 33 healpix catalogues, 1 352 231 objects:
#
#   * SN Ia (gentype 10) carry theirs inside SALT's `c`, so this module does too, through SALT2_C.
#     AV is recorded as -9 because there is no separate screen to record, not because there is no
#     dust.
#   * The core-collapse classes (21, 26, 32) carry NONE, and that is a bug of OpenUniverse's, not a
#     property of the templates. Its own section 3.2.4, "Known issues": "For the core collapse
#     models (SNII, SNIb, SNIc), the wavelength range was extended for the set of templates that
#     had been corrected for host-galaxy extinction. However, host extinction was not enabled in
#     the simulation." The "+HostXT" of `NON1ASED.V19_CC+HostXT_WAVEEXT` marks templates the host
#     dust was taken OUT of; it was never put back. The catalogue agrees: AV = -9 and
#     `template_index` is the only model parameter those rows carry. So no dust here either.
#   * TDE (42) has it inside the luminosity function, which is also where the MOSFiT bank cut to
#     van Velzen et al. (2021) carries it. Consistent, nothing to add.
#   * SN Iax (12) is the one class OpenUniverse gives an explicit screen -- AV median 0.440, 84th
#     percentile 1.027, RV 3.1 -- and this module now reproduces it; see IAX_HOST_AV_RANGE. It is
#     the only dust this module applies to anything.
#
# Milky Way extinction is applied by OpenUniverse in SkyCatalog rather than in SNANA, so the
# catalogues carry `mw_extinction_applied = False` and mag_true is free of it for every class
# alike. It creates no class structure and this module adds none.
# Rest-frame days from MAXIMUM, not from the source's own phase zero. The two are not the same
# thing across this library and the difference is class-correlated: the SNANA and SALT sources put
# phase zero at B maximum, the Nugent ones -- the only sources SN IIL, SN IIn and part of Ib/Ic
# have -- put it at the explosion, with maximum 11 to 17 d later. Sampled on the source's own
# phases the Nugent classes would carry 20 fewer days of light curve after maximum than the others,
# for no reason but the convention of the file they were read from.
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
_TDE_PEAK_MAGNITUDES = None


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
    rescaled later, so the peak absolute magnitude of the source is the one MOSFiT's physics
    implies and no luminosity function has to be assumed on top; see
    `tde_peak_absolute_magnitudes`."""
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


def tde_peak_absolute_magnitudes():
    """Rest-frame B (AB) at peak for every template, from the model's own luminosity.

    Built once and cached. This is what stands in for a luminosity function: MOSFiT already fixes
    how bright each drawn TDE is, so drawing a magnitude on top would overwrite its physics with an
    assumption."""
    global _TDE_PEAK_MAGNITUDES
    if _TDE_PEAK_MAGNITUDES is None:
        magnitudes = []
        for index in range(tde_template_count()):
            # `Model.source_peakabsmag` divides by a distance modulus that is infinite at z = 0;
            # the source's own peak magnitude is already absolute, because the flux is built at
            # 10 pc.
            magnitudes.append(tde_source(index).peakmag("bessellb", "ab"))
        _TDE_PEAK_MAGNITUDES = np.array(magnitudes)
    return _TDE_PEAK_MAGNITUDES


def sample_iax_host_av(random_generator, size):
    """Draw the host AV of a SN Iax from OpenUniverse's distribution; see IAX_HOST_AV_RANGE."""
    grid = np.arange(IAX_HOST_AV_RANGE[0], IAX_HOST_AV_RANGE[1] + IAX_HOST_AV_STEP, IAX_HOST_AV_STEP)
    density = np.exp(-grid / IAX_HOST_AV_TAU) + IAX_HOST_AV_WEIGHT * np.exp(
        -(grid**2) / 2.0 / IAX_HOST_AV_SIGMA**2
    )
    cumulative = np.cumsum(density)
    cumulative /= cumulative[-1]
    return np.interp(random_generator.random(size), cumulative, grid)


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


def iax_phase_zero_offset(source):
    """V(peak) - V(phase 0) of a warped SN Iax, in magnitudes.

    The notebook normalises its templates so that rest-frame V at PHASE ZERO equals the drawn M_V
    (`amplitudes = 10**(-0.4 * (Msamp - sedVmag))` with `sedVmag` measured at phase 0), and that is
    the convention the SED files SNANA read were written in. `sncosmo.Model.set_source_peakabsmag`
    normalises at the light-curve peak instead, which for the base SED sits at phase +3.7 and is
    0.075 mag brighter -- a gray offset in every band, and one this module used to carry. Adding
    this offset to the drawn magnitude before calling `set_source_peakabsmag` reproduces the
    notebook's convention without reimplementing sncosmo's cosmology handling.

    It is per template, not a constant: the dm15 warps move the peak, and over the bank it runs
    from -0.07 to -0.27 mag."""
    sncosmo = _sncosmo()
    model = sncosmo.Model(source=source)
    phases = np.arange(-10.0, 20.0, 0.1)
    phases = phases[(phases >= source.minphase()) & (phases <= source.maxphase())]
    with np.errstate(divide="ignore", invalid="ignore"):
        # A fast riser explodes after -10 d, and V is undefined on the phases before that.
        magnitudes = model.bandmag("bessellv", "vega", phases)
        return float(np.nanmin(magnitudes)) - float(model.bandmag("bessellv", "vega", 0.0))


def draw_population(number, redshifts, random_generator):
    """`number` contaminants: class, source, peak absolute magnitude, redshift, cadence parity.

    `redshifts` is the redshift of each object, drawn by the caller from whatever target
    distribution the sample is meant to fill -- the deficit against the kilonova histogram, in the
    intended use -- because the choice of that distribution is the whole point of the sample and
    does not belong buried in here."""
    labels = random_generator.choice(list(CLASS_FRACTION), size=number, p=list(CLASS_FRACTION.values()))
    population = []
    for index, (label, redshift) in enumerate(zip(labels, redshifts, strict=True)):
        tde_template_index = None
        iax_template_index = None
        if label == "SN Iax":
            # No luminosity function is drawn either: a uniform index into the bank OpenUniverse
            # shipped, and that row's own four parameters. See IAX_BANK_SIZE, and
            # PEAK_ABSOLUTE_MAGNITUDE_BAND for why the band differs from every other class.
            iax_template_index = int(random_generator.integers(IAX_BANK_SIZE))
            peak_absolute_magnitude = iax_template(iax_template_index)[0]
        elif label == "TDE":
            # No luminosity function is drawn: MOSFiT already fixed how bright this TDE is, and the
            # template bank carries that. Drawing a magnitude on top would replace its physics.
            tde_template_index = int(random_generator.integers(tde_template_count()))
            peak_absolute_magnitude = float(tde_peak_absolute_magnitudes()[tde_template_index])
        else:
            median, sigma = PEAK_ABSOLUTE_MAGNITUDE[label]
            peak_absolute_magnitude = float(random_generator.normal(median, sigma))
        realization = {
            "index": index,
            "label": str(label),
            "source_name": str(random_generator.choice(SOURCES_BY_LABEL[label])),
            "peak_absolute_magnitude": peak_absolute_magnitude,
            "redshift": float(redshift),
            # The two degrees of freedom of where the survey's visit grid falls on this transient,
            # both uniform because in the sky the grid is fixed in absolute time and the explosion
            # is not: the delay from the start of the model to the first visit, and the PARITY of
            # that visit, which decides whether it carries the two blue non-anchor bands or the two
            # red ones. Together they cover the full 10-day cycle of the band pattern. Without
            # them every izc object would be sampled from the same phase of the cadence, and since
            # the model's own start is set by the template library, that phase would be a function
            # of the class -- the exact species of class-correlated artefact this module removes.
            "cadence_parity": int(random_generator.integers(CADENCE_PARITY_PERIOD)),
            "visit_phase_offset_days": float(random_generator.uniform(0.0, BASE_CADENCE_DAYS)),
        }
        if label == "SN Ia":
            realization["salt2_x1"] = float(random_generator.normal(*SALT2_X1))
            realization["salt2_c"] = float(random_generator.normal(*SALT2_C))
        if tde_template_index is not None:
            realization["tde_template_index"] = tde_template_index
        if iax_template_index is not None:
            _, rise_time, decline_b, decline_r = iax_template(iax_template_index)
            realization["iax_template_index"] = iax_template_index
            realization["iax_rise_time"] = rise_time
            realization["iax_decline_b"] = decline_b
            realization["iax_decline_r"] = decline_r
            realization["host_av"] = float(sample_iax_host_av(random_generator, 1)[0])
            realization["host_rv"] = IAX_HOST_RV
        population.append(realization)
    return population


def build_model(realization, cosmology=None, **model_keywords):
    """The sncosmo model of one drawn contaminant: source, redshift, shape and normalisation.

    Split out of `roman_light_curve` so that a caller which needs the same object in some other
    bandpass gets the same model rather than a second copy of these six lines -- the comparison
    against the Carnegie Supernova Project photometry (`scripts/compare_csp_lightcurves.py`)
    synthesizes it through the CSP natural system. `model_keywords` reaches `sncosmo.Model`
    untouched, which is how that script attaches a Milky Way dust screen; nothing in the generated
    sample uses it, and the module itself still applies no extinction of any kind."""
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
    band, magnitude_system = PEAK_ABSOLUTE_MAGNITUDE_BAND.get(realization["label"], ("bessellb", "ab"))
    magnitude = realization["peak_absolute_magnitude"]
    if realization["label"] == "SN Iax":
        # The bank's M_V is rest-frame V at PHASE ZERO, not at the light-curve peak; see
        # `iax_phase_zero_offset`.
        magnitude = magnitude + iax_phase_zero_offset(source)
    # Set on the bare source, so the drawn absolute magnitude means the same thing whether or not
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
