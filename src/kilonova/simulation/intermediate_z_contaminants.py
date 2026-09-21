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

NO SUBSTITUTIONS, AND NOW THERE ARE NONE. The sample exists to REPOPULATE the low-redshift bins of
a population OpenUniverse already defines: a class whose SED this side has to supply is not being
moved in redshift, it is being added, and the sample would then have to be defended as a model of
that class rather than as a redistribution of OpenUniverse's. That used to exclude classes. It no
longer excludes any that matter, because OpenUniverse published its models
(zenodo.org/records/14749318) and every SED here is read out of that release:

  * The core-collapse SEDs were recovered from OpenUniverse's light-curve files, and before that
    extended into the near-infrared here with `snsedextend`. Both are gone; the 44 templates are
    read from `NON1ASED.V19_CC+HostXT_WAVEEXT`.
  * SN Ia was sncosmo's `salt3-nir` held flat over the 1000 A it is missing. Now it is the
    release's own `SALT3.NIR_WAVEEXT`, which is the same model with 5000 A more of it.
  * TDE was MOSFiT's `tde` standing in for the observed SED of AT2019qiz, 0.19 mag off with a
    colour trend. Now it is `2019qiz.sed`: 0.017 mag, no trend.
  * SN Iax was a replay of the Rutgers notebook OpenUniverse cites, 0.19 mag off with a 0.28 mag
    colour trend. Now it is the 919 SEDs of `SIMSED.SNIax` that OpenUniverse actually drew.
  * SLSN-I was out because it had no template on this side at all. Now it is `2016apd.sed`:
    0.015 mag, and its own 10 pc calibration is OpenUniverse's exactly.

ONE CLASS IS STILL OUT, and for a reason about coverage rather than about provenance. PISN's SEDs
stop at 20000 A rest-frame while F184's red edge is 21000, so it cannot cover F184 without
extrapolating below z = 0.050 -- and this sample's redshift grid starts at 0.010. It cannot be
rendered in the bins the sample exists to fill. So are the OpenUniverse kilonovae (50), which are
the class the classifier is being taught to find, and gentype 99, which is not a transient at all:
27771 objects whose magnitude does not vary in time and is identical in all six bands, a
fixed-magnitude calibration source.

WHERE THE LUMINOSITY COMES FROM IS A SEPARATE QUESTION from where the SED does, and the answer is
not the same for every class. SN Iax, TDE and SLSN-I carry an absolute calibration their files
declare -- "erg/s/cm^2/A scaled to 10 pc" -- and it is OpenUniverse's own, verified to 0.15, 0.10
and 0.00 mag. The V19 core-collapse files declare nothing, and one of them is mis-normalised by
4 mag: `SN2011bm` reads M_B = -13.22 where its six SN Ic siblings read -16.5 to -17.8, and its
MAGOFF is -4.56 where theirs are -1.46 to 0.00. That MAGOFF is a repair, not a luminosity-function
adjustment, which is why core-collapse is the one class whose file normalisation is not trusted.
None of this reaches the generated sample today, which measures every brightness off the parent's
own light curve -- see REFERENCE_ABSOLUTE_MAGNITUDE -- but it is what a parametric route would use.

The near-infrared extension those templates carry is OpenUniverse's own and is shared by both
populations rather than added by this one: `_WAVEEXT` is an extension by the methods of Pierel et
al. (2018), and every OpenUniverse contaminant already rests on it. What remains
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

# --- SN Iax ---------------------------------------------------------------------------------------
# OpenUniverse's OWN SN Iax SEDs, read out of its published release.
#
# This used to be a REGENERATION. OpenUniverse used "1000 SED templates from the original PLAsTiCC
# model" (OpenUniverse2024, sec. 4.2) -- Jha & Dai's model at github.com/RutgersSN/SNIax-PLAsTiCC --
# and, believing the bank unpublished, this module replayed the repository's `Iax-model.ipynb` to
# rebuild it: a SN 2005hk base SED, a luminosity function, width-luminosity relations and a warp,
# all reproduced here deterministically. `MODELS-1_TRANSIENT_SED.tar` ships the bank itself, as
# `SIMSED.SNIax`, so the replay is gone with all 100 lines of constants it needed.
#
# WHAT IT COST AND WHAT THE RELEASE BUYS, both measured over the same 440 parents. The replay
# reproduced OpenUniverse's own photometry to 0.19 mag band-to-band, with a monotonic colour trend
# of 0.28 mag across R062 to F184 -- the size of a substitution, not of a pipeline, and the worst
# class in the sample by a factor of ten. The release's own SEDs give 0.012 mag and a colour trend
# of 0.05, all of it in R062. A factor of 16.
#
# THE RELEASE'S INDEX IS OFF BY ONE, and this is the one place that matters here. `NON1A.LIST` says
# `template_index` 1 is `SED-Iax-0001.dat`; OpenUniverse used `SED-Iax-0000.dat`. Measured over
# 4357 parents on 40 templates, the absolute magnitude OpenUniverse gave each object correlates with
# the one its file carries at r = 0.998 under that shift and r = 0.055 without it. The bank is a
# random draw per row, so neighbouring files are unrelated and the error hides perfectly as "the
# calibration was overwritten" -- which is what a first pass of this measurement concluded.
# `scripts/build_openuniverse_extra_templates.py` applies the shift, so the archive is in
# OpenUniverse's own `template_index` order and this module indexes it from zero.
#
# THE FLUX IS ABSOLUTE and is used as such: the files declare "erg/s/cm^2/A scaled to 10 pc" and
# the calibration is OpenUniverse's to 0.15 mag rms. Nothing is applied on top, unlike the
# core-collapse templates. Every SN Iax also carries a host dust screen -- AV != -9 for 100 % of
# them, median 0.440 -- which the parent supplies and `build_model` applies; leaving it out shows up
# as exactly the monotonic colour residual the replay had.
IAX_SOURCE_NAME_PREFIX = "ou-iax-"
IAX_TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "data" / "openuniverse" / "iax_templates.npz"
IAX_BANK_SIZE = 919  # what OpenUniverse shipped and what a `template_index` indexes

# --- SLSN-I ---------------------------------------------------------------------------------------
# OpenUniverse's own SLSN-I SED, a grey blackbody fit to Gaia16apd (Yan et al. 2017, Kanga et al.
# 2017). This class was never a substitution -- it simply had no template on this side at all, which
# is why it was excluded. It has one now.
#
# ONE TEMPLATE, which is what OpenUniverse drew: all 1122 of its SLSN-I carry `template_index` 1.
# Measured over all of them, the band-to-band residual against OpenUniverse's own photometry is
# 0.015 mag and its 10 pc calibration is OpenUniverse's EXACTLY: -21.75 against -21.75.
#
# PISN IS STILL OUT, and now for a reason that is not about how rare it is. Its SEDs stop at
# 20000 A rest-frame while F184's red edge is 21000, so PISN cannot cover F184 without
# extrapolating below z = 0.050 -- and this sample's grid starts at z = 0.010. It cannot be
# rendered in the bins the sample exists to fill. (Its residual is also 0.10-0.16 mag.)
SLSN_SOURCE_NAME = "ou-slsn-2016apd"
SLSN_TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "data" / "openuniverse" / "slsn_template.npz"

# --- TDE -----------------------------------------------------------------------------------------
# OpenUniverse's OWN TDE SED, read out of its published release.
#
# This was the module's largest declared substitution: OpenUniverse took its TDE from the observed
# SED of AT2019qiz, which was believed unpublished, so the MOSFiT `tde` model of Guillochon et al.
# (2018) stood in for it and measured 0.19 mag off with a colour trend. `MODELS-1_TRANSIENT_SED.tar`
# ships `NON1ASED.TDE-BBFIT/2019qiz.sed`, a blackbody fit to the Swift and ground-based photometry
# of Nicholl et al. (2020) and Hung et al. (2021). The substitution is gone.
#
# ONE TEMPLATE, and OpenUniverse drew exactly that: all 3769 of its TDEs carry `template_index` 1.
# So every TDE this module generates has the same SED, differing only in luminosity, redshift and
# where the cadence falls -- which is a property of the release and not a simplification here.
#
# THE FLUX IS ABSOLUTE and is used as such. The file's own header says
# "flux: erg/s/cm^2/A scaled to 10 pc", and measured against OpenUniverse's own photometry over 60
# parents the absolute magnitude it implies is 0.10 mag from the one OpenUniverse gave them. Unlike
# the core-collapse templates, nothing has to be applied on top.
TDE_SOURCE_NAME = "ou-tde-2019qiz"
TDE_TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "data" / "openuniverse" / "tde_template.npz"

# --- SN Ia -------------------------------------------------------------------------------------
# OpenUniverse's OWN SALT3 model, read out of its published release rather than approximated.
#
# OpenUniverse's SN Ia model is SALT3 extended into the near-infrared -- Pierel et al. (2022) --
# and `MODELS-1_TRANSIENT_SED.tar` ships it as `SALT3.NIR_WAVEEXT`, vendored here. sncosmo's stock
# `salt3-nir` IS that model over the range they share: measured surface against surface with
# x1 = c = 0, the two agree to a scale of 1.0000 and a median residual of 0.0000. What the release
# adds is the wavelength range, 2000-25000 A rest-frame against sncosmo's 2000-20000.
#
# THAT RANGE IS WHY THIS REPLACED A PAD. F184's red edge is at 21000 A and `salt3-nir` stops at
# 20000, so this module used to hold its last defined flux constant over the missing 1000 A. The
# release covers them with the model itself.
#
# WHAT THAT IS WORTH, measured against the pad it replaces: F184 alone, below z = 0.05 alone, and
# -0.022 mag at z = 0.01, -0.019 at z = 0.02, -0.011 at z = 0.03, -0.002 at z = 0.05, 0.000 above.
# Every other band is 0.0000 at every redshift. Small, but NOT the 0.00001 mag this block used to
# predict for it: that number compared the pad against continuing `salt2-extended`'s fall, and the
# real model over those 1000 A does neither.
#
# IT DOES NOT EXPLAIN THE SN Ia RESIDUAL, and the obvious guess was wrong. SN Ia is the worst class
# of the band-to-band residual against OpenUniverse's own photometry -- 0.039 mag against 0.006 for
# SN Ib and SN Ic -- and being the one class whose template was not read from the release made the
# pad the suspect. Measured over 400 SN Ia parents the residual is IDENTICAL to four decimals with
# either source, because that measurement renders at the PARENT's redshift, which is high, where
# F184 in the rest frame sits far inside sncosmo's ceiling. The pad only ever acted at the drawn
# redshift. Whatever makes SN Ia the worst class is still unexplained.
#
# `SALT3.INFO` also carries `SIGMA_INT: 0.106  # used in simulation`, which is NOT the 0.21 mag of
# grey scatter this module measures in OpenUniverse's SN Ia photometry (see REFERENCE_ABSOLUTE_
# MAGNITUDE). The two are not the same number and the difference is not understood; nothing here
# uses either, because the SN Ia brightness is measured off the parent light curve.
IA_SOURCE_NAME = "salt3-nir-ou"
IA_BASE_SOURCE_NAME = "salt3-nir"  # sncosmo's, kept only so the test can compare against it
IA_MODEL_PATH = Path(__file__).resolve().parents[3] / "data" / "openuniverse" / "salt3_nir_waveext"

# THE CORE-COLLAPSE LIBRARY, read out of OpenUniverse's own published models.
# `scripts/build_openuniverse_cc_templates.py` builds the archive and its docstring carries the
# provenance; this block carries what a reader of the module needs.
#
# OpenUniverse drew its core-collapse SEDs from `NON1ASED.V19_CC+HostXT_WAVEEXT` -- Vincenzi et al.
# (2019) corrected for host extinction, extended to 25000 A by the methods of Pierel et al. (2018)
# -- and publishes that library in full, at https://zenodo.org/records/14749318. The `.SED` files
# are read directly. The 3.77 GB tar they come in is NOT a dependency of the pipeline: the script
# is run by hand and this 23 MB archive is what gets versioned.
#
# TWO EARLIER ANSWERS ARE GONE, and what they cost is why the release is worth trusting over
# either. The first redid the near-infrared extension with `snsedextend`, the public implementation
# of the method OpenUniverse cites; measured against OpenUniverse's own photometry it made a comb
# of one hump per photometric anchor separated by runs of zero flux. The second recovered the
# templates out of the per-healpix light-curve files, where each object's `flambda` is the model
# SED itself at that object's redshift -- sound, and validated to 0.5 %, but it could only use
# objects below z = 0.187 (the observer-frame grid ends at 24450 A) and each template covered only
# the phases its objects' light curves sampled.
#
# THE MIGRATION WAS MEASURED, not assumed (docs/plan_templates_oficiales_ou.md):
#   * Against the recovered archive, allowing the phase offset and the achromatic scale that
#     `peak_phase` and `set_source_peakabsmag` absorb anyway, all 44 templates agree to a median of
#     0.3 % and a worst of 2.3 %. They are the same SEDs.
#   * Over 132 fixed parents the inferred brightness moves by 0.003 mag, and the band-to-band
#     residual against OpenUniverse's OWN photometry -- the one number that says whether this
#     pipeline reproduces the release -- improves from 0.0084 to 0.0079 mag.
#
# HOST EXTINCTION IS ZERO and that is deliberate. OpenUniverse's release note says, of its own
# simulation: "for the SNCC (II/Ib/Ic) models we mistakenly used the de-reddened SEDs and therefore
# did not model host extinction". These are those de-reddened SEDs. Reproducing the bug is the
# requirement; a sample that fixed it would differ from the release in a way correlated with class.
#
# WHAT THE ARCHIVE HOLDS: 44 templates on the files' OWN wavelength grid, 1605-25000 A rest-frame,
# and a phase grid PER TEMPLATE spanning -78 to +272 d, a median of 198 d each. Measured off the
# .npz on disk. An earlier version of this comment described the archive this one replaced --
# 3000-20600 A and a (-12, +35) d trim inherited from the recovered templates -- and kept saying so
# after the migration; the trim and the ceiling are gone, see the build script for why each went.
# The phase axis is not shared because the templates begin where their original spectroscopy did,
# which for a SN IIP is close to explosion and for a SN Ib is well before maximum.
OPENUNIVERSE_ARCHIVE_PATH = Path(__file__).resolve().parents[3] / "data" / "openuniverse" / "cc_templates.npz"
OPENUNIVERSE_SOURCE_PREFIX = "ou-"


# Sources vetted to cover R062 through F184 with no extrapolation at z = 0.02, the bluest and
# reddest edges included. tests/test_intermediate_z_contaminants.py re-checks this against sncosmo
# rather than trusting the list.
#
# The core-collapse entries are OpenUniverse's OWN templates, read out of its published model
# release by `scripts/build_openuniverse_cc_templates.py`; see the OPENUNIVERSE_ARCHIVE_PATH
# block above.
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
    # ONE NAME STANDS FOR THE 919-TEMPLATE BANK, and that is sound here because the coverage check
    # is about the grid: every SN Iax template shares one phase and one wavelength axis, verified
    # when the archive is built. `build_model` builds the one the parent's `template_index` names.
    "SN Iax": [IAX_SOURCE_NAME_PREFIX + "0"],
    # One template each, which is what OpenUniverse drew for both.
    "TDE": [TDE_SOURCE_NAME],
    "SLSN-I": [SLSN_SOURCE_NAME],
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
# ONE SHARE PER LABEL, and eight of them. This was 1/4 over CLASSES -- SN Ia, SN Ib, SN Ic and a
# SN II split into IIP and IIL by 17/24 and 7/24, the proportion of templates OpenUniverse drew --
# which gave core-collapse 75 % of the sample. Uniform over labels gives it 57 % and puts SN Iax,
# TDE and SLSN-I on the same footing as the rest.
#
# The 17/24-7/24 split is gone with it. It came from OpenUniverse's own template counts, which is
# the right thing to inherit when the goal is OpenUniverse's mix and irrelevant when it is not.
UNIFORM_CLASS_SHARE = 1.0 / 8.0
CLASS_FRACTION = {
    "SN Ia": UNIFORM_CLASS_SHARE,
    "SN Iax": UNIFORM_CLASS_SHARE,
    "SN Ib": UNIFORM_CLASS_SHARE,
    "SN Ic": UNIFORM_CLASS_SHARE,
    "SN IIP": UNIFORM_CLASS_SHARE,
    "SN IIL": UNIFORM_CLASS_SHARE,
    "SLSN-I": UNIFORM_CLASS_SHARE,
    "TDE": UNIFORM_CLASS_SHARE,
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
    "SLSN-I": 40,
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
REST_FRAME_PHASE_STEP = 1.0

# ESTA VENTANA YA NO ES LA DE LA CURVA DE LUZ. Solo queda en `rendered_band_curves`, que existe para
# comparar el brillo contra el del padre y tiene que medirse sobre las MISMAS fases que el padre
# (`openuniverse_parents.PEAK_PHASE_LIMITS`, apareadas por test). La curva de luz que se renderiza
# corre sobre toda la plantilla; ver `template_phase_grid`.


def template_phase_grid(source):
    """Toda la fase que la plantilla tiene, a `REST_FRAME_PHASE_STEP`, anclada en su primer punto.

    RECORTAR AQUI COSTABA PLANTILLAS ENTERAS. Esto era `peak_phase(...) + REST_FRAME_PHASES`, o sea
    una ventana de (-20, +70) d CENTRADA EN EL PICO. Una plantilla cuyo maximo llega tarde perdia su
    parte temprana aunque la tuviera: `pycoco_SN2005bf` empieza en -42.65 d y se renderizaba desde
    -22.5, `pycoco_SN1987A` empieza en -78.95 y se renderizaba desde -14. Medido sobre 986 objetos a
    z>0.5, la cobertura de esas plantillas era 36% y 35% -- y con la plantilla entera es 100% en
    ambas, 82% en `pycoco_SN2008D` y 89% en `pycoco_SN2011bm`. En las otras 40 no cambia nada: la
    ventana ya alcanzaba el inicio del archivo.

    El limite tardio tampoco hace falta. Ninguna epoca observada llega a +190 d, asi que el extremo
    rojo del eje no se lee; recortarlo solo ahorraba fases que igual salen NaN mas abajo.

    El paso se mantiene en 1 d, que es el de la plantilla en disco, para no interpolar de mas."""
    first, last = float(source.minphase()), float(source.maxphase())
    count = int(np.floor((last - first) / REST_FRAME_PHASE_STEP)) + 1
    return first + np.arange(count) * REST_FRAME_PHASE_STEP
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


def _official_ia_source():
    """OpenUniverse's own SALT3, read from IA_MODEL_PATH.

    `sncosmo.SALT3Source` wants the model directory uncompressed and wants all eight files, the
    variance and covariance surfaces included -- it reads them even though nothing here uses SALT's
    error model -- which is why the vendored copy is 17 MB rather than 3.

    `sncosmo.get_source` builds a new instance per call, so the object returned here is this
    module's own and the registry entry for `salt3-nir` is untouched."""
    import sncosmo

    if not IA_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{IA_MODEL_PATH} is missing; it is SALT3.NIR_WAVEEXT out of OpenUniverse's "
            "MODELS-1_TRANSIENT_SED.tar (zenodo.org/records/14749318), uncompressed"
        )
    return sncosmo.SALT3Source(modeldir=str(IA_MODEL_PATH), name=IA_SOURCE_NAME)


def _openuniverse_templates():
    """{source name: (phase, wavelength, flux)} of the OpenUniverse archive, read once.

    The archive shares ONE wavelength axis across its templates -- the build resamples every
    template onto it -- and gives each its own phase grid, because the templates begin where their
    original spectroscopy did and that differs template to template.

    Nothing is repaired on the way in, and nothing is audited either. The interior zero-flux gaps
    this used to interpolate across belonged to an extension made on this side with `snsedextend`;
    the release's own templates have none, measured. The audit that policed them,
    `defective_near_infrared_extension`, was removed on 2026-09-18 along with the tests that pinned
    it: its sound test found zero gaps in all 44 templates, and its own docstring called its second
    test unsound -- which is the one that still fired, on 37 of 44. It pruned nothing, because
    SOURCES_BY_LABEL holds every template OpenUniverse drew. Its premise was that the near-infrared
    extension is a degraded part of the model, and that is backwards: OpenUniverse generated the
    release's core-collapse photometry FROM `NON1ASED.V19_CC+HostXT_WAVEEXT`, so the extension is
    the model. See the BRIGHTNESS_REST_WAVELENGTH_LIMITS block."""
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

    Every class's source goes in, because `register_sources` promises the whole of
    SOURCES_BY_LABEL is resolvable through the ordinary `sncosmo.get_source` path -- which is what
    the coverage test needs: every source has to be checked against R062-F184 like every other."""
    import sncosmo

    if IA_SOURCE_NAME not in _REGISTERED_SOURCES:
        sncosmo.register(_official_ia_source(), IA_SOURCE_NAME, force=True)
        _REGISTERED_SOURCES.add(IA_SOURCE_NAME)
    if IAX_SOURCE_NAME_PREFIX not in _REGISTERED_SOURCES:
        # Marked registered before the source is built, because `iax_source` calls back into here.
        # ONE template stands for the bank in the registry, and in the coverage check with it: all
        # 919 share one phase and one wavelength grid, verified when the archive is built.
        _REGISTERED_SOURCES.add(IAX_SOURCE_NAME_PREFIX)
        sncosmo.register(iax_source(0), IAX_SOURCE_NAME_PREFIX + "0", force=True)
    if SLSN_SOURCE_NAME not in _REGISTERED_SOURCES:
        _REGISTERED_SOURCES.add(SLSN_SOURCE_NAME)
        sncosmo.register(slsn_source(), SLSN_SOURCE_NAME, force=True)
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
        sncosmo.register(tde_source(), TDE_SOURCE_NAME, force=True)
    return sncosmo


_REGISTERED_SOURCES = set()
_OPENUNIVERSE_TEMPLATES = None


def register_sources():
    """Put every source name of SOURCES_BY_LABEL in the sncosmo registry.

    Only SN Ia's `salt3-nir` is a stock sncosmo entry, and even that one is registered padded under
    a different name. The core-collapse sources come from the V19 archive, SN Iax and TDE are built
    here, so a caller that reaches sncosmo on its own -- the coverage test does -- has to come
    through this first."""
    _sncosmo()


@cache
def _iax_grid():
    """(phase, wavelength) shared by all 919 SN Iax templates; they are identical, verified."""
    with np.load(IAX_TEMPLATE_PATH) as archive:
        return archive["phase"].astype(float), archive["wavelength"].astype(float)


@cache
def iax_source(template_index):
    """One SN Iax template, as an sncosmo source, read from the archive on demand.

    `template_index` is zero-based: OpenUniverse's catalogue carries 1 to 919 and the archive is
    written in that order, so a caller subtracts one. Cached over the whole bank the way the TDE
    bank used to be -- the draws are uniform over 919 templates, so a window cannot hold a working
    set -- which at 0.45 MB a template is 415 MB if every one is drawn.

    The flux is absolute, erg/s/cm2/A at 10 pc, and `build_model` renormalises it."""
    sncosmo = _sncosmo()
    phase, wavelength = _iax_grid()
    with np.load(IAX_TEMPLATE_PATH) as archive:
        flux = archive[f"flux_{template_index}"].astype(float)
    return sncosmo.TimeSeriesSource(phase, wavelength, flux)


@cache
def slsn_source():
    """OpenUniverse's SLSN-I SED, as an sncosmo source, read once. Absolute, at 10 pc."""
    sncosmo = _sncosmo()
    with np.load(SLSN_TEMPLATE_PATH) as archive:
        return sncosmo.TimeSeriesSource(
            archive["phase"].astype(float),
            archive["wavelength"].astype(float),
            archive["flux_0"].astype(float),
        )


# --- The redshift the sample is generated at ----------------------------------------------------
# The bins the deficit is counted in are the kilonova grid's own. `kn-kilonova-windows` puts its
# kilonovae on 50 logarithmic redshifts between 0.01 and 1.0, so a kilonova histogram is 50 spikes
# and any binning finer than the spacing between them measures the grid rather than the population.
# The edges are the geometric midpoints of that grid, which is the coarsest binning that keeps one
# grid point per bin.
DEFICIT_REDSHIFT_LIMITS = (0.01, 1.0)
DEFICIT_REDSHIFT_NODES = 50


@cache
def tde_source():
    """OpenUniverse's TDE SED, as an sncosmo source, read once.

    Cached because it is read from disk and there is only ever one of it. The flux is absolute --
    erg/s/cm2/A at 10 pc -- so the source carries the brightness AT2019qiz had, and `build_model`
    renormalises it to whatever brightness the object is given."""
    sncosmo = _sncosmo()
    with np.load(TDE_TEMPLATE_PATH) as archive:
        return sncosmo.TimeSeriesSource(
            archive["phase"].astype(float),
            archive["wavelength"].astype(float),
            archive["flux_0"].astype(float),
        )


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
        # La plantilla que OpenUniverse le dio. Va aqui porque el anclaje contra la ventana
        # (`window_anchor`) busca su C por plantilla y solo tiene la realizacion en la mano.
        "parent_template_index": int(parent["template_index"]),
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
        # OpenUniverse's `template_index` runs 1..919 and the archive is written in that order, so
        # the index here is zero-based. The release's own NON1A.LIST is off by one against what
        # OpenUniverse actually used; the archive absorbs that, see the SN Iax block.
        iax_template_index = int(parent["template_index"]) - 1
        if not 0 <= iax_template_index < IAX_BANK_SIZE:
            raise ValueError(
                f"SN Iax template_index {parent['template_index']} is outside the 1..{IAX_BANK_SIZE} "
                "OpenUniverse shipped"
            )
        realization["iax_template_index"] = iax_template_index
        realization["source_name"] = IAX_SOURCE_NAME_PREFIX + str(iax_template_index)
    elif label == "TDE":
        # Nothing is drawn: OpenUniverse has ONE TDE template and every one of its TDEs carries
        # `template_index` 1. The MOSFiT bank this replaced needed a draw, and needed it to come
        # from the parent's own id so that a parent re-rendered twenty times stayed one object.
        realization["source_name"] = TDE_SOURCE_NAME
    elif label == "SLSN-I":
        # One template too: all 1122 of OpenUniverse's SLSN-I carry `template_index` 1.
        realization["source_name"] = SLSN_SOURCE_NAME
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

    The phase axis is the TEMPLATE'S OWN, and for the core-collapse sources it is NOT
    OpenUniverse's. That used to be true and stopped being true with the move to the published
    templates: the recovered archive was built by de-redshifting objects about `peak_mjd`, so its
    phase zero WAS OpenUniverse's, while a `.SED` file carries the native V19 axis. Measured
    against OpenUniverse's own stored SEDs over 43 templates, the two differ by a template-dependent
    offset -- exactly zero for 10 of them, a median of -2 d and as much as -20 and +16 -- and once
    that offset is allowed the SEDs agree to 1.0 % median. The offset is the convention, not an
    error. SALT still puts phase zero at B maximum and so does the SN Iax base SED.

    NOTHING HERE DEPENDS ON THAT, which is why the move was safe: the phases are measured from
    `peak_phase`, the source's own B maximum, and the only caller of this function discards the
    phase axis and keeps the magnitudes. An overlay of this against `(mjd - peak_mjd) / (1 + z)` of
    the parent's light curve would be misaligned by that offset and nothing does one.
    `roman_light_curve` also measures from B maximum, because that is what the survey window needs.

    A band the redshifted spectrum does not cover is absent; a phase the model has no flux at is
    NaN. Unlike `roman_light_curve` this does not require every band at every phase, does not drop
    colour discontinuities and does not enforce the model's flux floor: all three of those guard
    against magnitudes that are spuriously FAINT, which is what the window cares about and what a
    peak is unaffected by."""
    source = source_of(realization)
    phases = peak_phase(realization) + REST_FRAME_PHASES
    phases = phases[(phases >= source.minphase()) & (phases <= source.maxphase())]
    magnitudes_by_band = band_magnitudes_at_phases(realization, redshift, phases, cosmology)
    return {
        band: (phases, magnitudes)
        for band, magnitudes in magnitudes_by_band.items()
        if np.isfinite(magnitudes).any()
    }


def source_of(realization):
    """The sncosmo source a realization renders through, without the model around it.

    What `build_model` picks in its first six lines, needed on its own by the callers that only
    want the template's phase range -- the anchoring checks `minphase` before it renders anything.""" 
    if realization["label"] == "SN Iax":
        return iax_source(realization["iax_template_index"])
    if realization["label"] == "TDE":
        return tde_source()
    if realization["label"] == "SLSN-I":
        return slsn_source()
    return _registry_source(realization["source_name"])


def band_magnitudes_at_phases(realization, redshift, phases, cosmology=None):
    """{band: AB magnitudes} of one realization's model, sampled at `phases` and nowhere else.

    `phases` are the TEMPLATE'S OWN axis, the same one `rendered_band_curves` returns and the one
    `window_anchor` puts an observed epoch on. Split out of `rendered_band_curves` because the
    anchoring reads the model at four phases and nothing else: rendering the whole
    `REST_FRAME_PHASES` window to use 4 of its 91 points is 20 times the integrations for the same
    answer. A phase the model has no usable flux at comes back NaN in every band, and a band the
    redshifted spectrum does not reach comes back NaN at every phase -- neither is an error here,
    both are for the caller to check."""
    model = build_model(dict(realization, redshift=float(redshift)), cosmology)

    blue = max(model.minwave(), OBSERVED_WAVELENGTH_LIMITS[0])
    red = min(model.maxwave(), OBSERVED_WAVELENGTH_LIMITS[1])
    if red <= blue:
        return {}
    wavelength = _sampling_grid(blue, red, OBSERVED_WAVELENGTH_STEP)

    rows = []
    for phase in np.asarray(phases, dtype=float):
        flux = np.clip(model.flux(phase * (1.0 + redshift), wavelength), 0.0, None)
        if flux.max() < PHOTOMETRY_FLOOR:
            rows.append(dict.fromkeys(ALL_ROMAN_BANDS, np.nan))
            continue
        rows.append(spectrum_to_roman_magnitudes(wavelength, flux))
    return {band: np.array([row[band] for row in rows]) for band in ALL_ROMAN_BANDS}


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
# pivot wavelength, de-redshifted by the parent's own redshift, lands inside this window.
#
# THE WINDOW IS EMPIRICAL AND ITS CAUSE IS NOT ESTABLISHED. An earlier version of this comment
# justified both edges by the template ceasing to be "a measurement" there -- SALT3's ultraviolet
# being its least constrained part, and every core-collapse template above 10000 A being the
# `_WAVEEXT` extension rather than the pycoco base. That reasoning is wrong for this pipeline, and
# the red edge it produced is indefensible: OpenUniverse generated its own core-collapse light
# curves FROM `NON1ASED.V19_CC+HostXT_WAVEEXT`, so the extension is not a degraded version of the
# model, it IS the model, and there exists no un-extended variant that produced the release's
# photometry. The criterion here is reproducing OpenUniverse, not physical realism (see
# docs/ on the izc scope): where both sides integrate the same file, a badly constrained
# rest-frame region is constrained identically badly on both, and cancels.
#
# What survives is the measurement below, and it is a real effect with an unexplained cause. Since
# the SED is shared, the scatter cannot come from the SED: it has to come from the INTEGRATION --
# galsim.roman's bandpasses against the kcor SNANA used, the zeropoint convention, or the 10 A
# sampling grid. Until that is measured, the window stays as an empirical guard rather than as a
# statement about the templates, and it is a candidate for removal: dropping the blue bands costs
# exactly the objects whose parents sit at high redshift, which is where anchoring is hardest.
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

    source = source_of(realization)
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
    phases = template_phase_grid(source)
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


def run_izc_healpix(healpix, population, source_directory, tiers, shard_directory=None,
                    cosmology=None, windows_directory=None):
    """Measure, render and window every object of one healpix.

    Returns ({tier: parquet shard path}, summary) when `shard_directory` is given and
    ({tier: windows}, summary) when it is not. The shards are how the full run survives its own
    size: 670 000 objects are 27 million rows over the two tiers, and holding them as DataFrames
    until the end needs more memory than this machine has. Each task writes its own and the parent
    streams them together.

    WHERE THE BRIGHTNESS COMES FROM is `windows_directory`. Given one, it is read off the parent's
    own early window through `window_anchor` -- the analytic phase plus the frozen anchor table --
    and `source_directory` is not touched at all. Given none, it is read off the parent's full
    light curve in the release's hdf5, which is 16 GB per field on an external volume. The two
    agree to 0.002 mag in median over 1320 objects; the difference that matters is that the first
    one only knows the parents the survey detected."""
    if windows_directory is not None:
        from kilonova.simulation.window_anchor import measure_population_brightness_from_windows

        unmeasured = measure_population_brightness_from_windows(
            population, window_anchor_windows(windows_directory, healpix), cosmology=cosmology
        )
    else:
        hdf5_path = Path(source_directory) / f"snana_{healpix}.hdf5"
        if hdf5_path.stat().st_size == 0:
            # A cloud-storage placeholder reads as an empty file rather than as an error, which
            # would silently produce a sample with no brightness at all. See
            # docs/generate_datasets.md.
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


def window_anchor_windows(windows_directory, healpix):
    """Imported lazily and wrapped so that `window_anchor` can import this module, not the reverse."""
    from kilonova.simulation.window_anchor import windows_by_parent

    return windows_by_parent(windows_directory, healpix)


_IZC_WORKER_STATE = {}


def _izc_worker_initializer(source_directory, tiers, shard_directory, windows_directory=None):
    _IZC_WORKER_STATE["source_directory"] = source_directory
    _IZC_WORKER_STATE["tiers"] = list(tiers)
    _IZC_WORKER_STATE["shard_directory"] = shard_directory
    _IZC_WORKER_STATE["windows_directory"] = windows_directory
    register_sources()


def _izc_healpix_task(work_item):
    healpix, population = work_item
    shards, summary = run_izc_healpix(
        healpix,
        population,
        _IZC_WORKER_STATE["source_directory"],
        _IZC_WORKER_STATE["tiers"],
        _IZC_WORKER_STATE["shard_directory"],
        windows_directory=_IZC_WORKER_STATE.get("windows_directory"),
    )
    return healpix, shards, summary


def run_izc_tiers(population, source_directory, tiers, output_paths, workers=1, shard_directory=None,
                  windows_directory=None):
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
            initargs=(str(source_directory), tuple(tiers), str(shard_directory),
                      None if windows_directory is None else str(windows_directory)),
        )
        stream = pool.imap(_izc_healpix_task, work_items, chunksize=1)
    else:
        _izc_worker_initializer(str(source_directory), tiers, str(shard_directory),
                                None if windows_directory is None else str(windows_directory))
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
