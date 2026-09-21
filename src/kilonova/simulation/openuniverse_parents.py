"""The OpenUniverse objects the izc sample re-renders, and the brightness read off each one.

WHY A PARENT AT ALL. The izc sample used to be a population drawn from luminosity functions: a
class from `CLASS_FRACTION`, a peak absolute magnitude from a Gaussian fitted to OpenUniverse's own
catalogue columns, a shape from another Gaussian, and a per-class brightness offset measured
afterwards to make the median come out where OpenUniverse's is. Every one of those steps is a place
where the sample can disagree with the population it stands in for, and the disagreements had to be
measured one by one and argued about.

None of it is necessary. OpenUniverse's own objects are here, 1.3 million of them, each with the
template it was drawn from, the shape and colour it was drawn with, the dust screen it was given,
and its own light curve. Re-rendering one of them at a lower redshift keeps ALL of that and changes
exactly one thing, which is the one thing the sample exists to change. There is no luminosity
function left to get wrong: the brightness is the object's own, measured from its own light curve.

WHAT THIS MODULE SUPPLIES, and what it deliberately does not:

  * `read_parent_catalog` -- the 33 healpix catalogues as one table, one row per candidate parent,
    with everything the re-render inherits. Catalogue only: 135 MB of parquet, no light curves.
  * `parent_peak_magnitudes` -- the brightest magnitude of one object in each Roman band over a
    rest-frame phase window, read from its FULL light curve in the release's hdf5.

The rendering, the redshift draw and the window building live in `intermediate_z_contaminants`;
this module knows nothing about them and imports nothing from them.

WHY THE FULL LIGHT CURVE AND NOT A CATALOGUE COLUMN. The catalogue carries `peak_mag_g`,
`peak_mag_i` and `peak_mag_F`, and none of the three is the peak of a Roman band as this pipeline
measures one -- they are the simulation's own peaks in its own bands, and `peak_mag_F` in
particular is not F184. The early-window parquets are no use either: they hold four epochs from the
first detection, which is not the peak and for a fraction of the objects does not contain it. The
hdf5 group holds the model sampled over its whole span -- 406 epochs for a typical object -- in
mag_R through mag_F, which is what a peak has to be taken over.

WHY A PHASE WINDOW. The minimum has to be taken over the same rest-frame phases on both sides, or
it compares an object's peak against a template's plateau. `peak_mjd` is OpenUniverse's own per-
object peak, so (mjd - peak_mjd) / (1 + z) puts the object on the rest-frame days-from-maximum axis
the generator renders on, and the window is that axis's own limits.
"""

import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# The classes the izc sample re-renders: ONLY those whose spectral model is OpenUniverse's own.
# The sample exists to repopulate the low-redshift bins of a population OpenUniverse already
# defines, so a class whose SED this side has to supply would be adding a source to that population
# rather than moving one, and the sample would then have to be defended as a model of that class
# instead of as a redistribution of OpenUniverse's.
#
# SN Iax (12) AND TDE (42) WERE OUT AND ARE BACK, because the reason they were out has gone. Both
# were substitutions: OpenUniverse's TDE is the observed SED of AT2019qiz, believed unpublished, so
# MOSFiT's `tde` stood in and reproduced OpenUniverse's own photometry only to 0.19 mag; SN Iax was
# OpenUniverse's own model but REGENERATED here from the Rutgers notebook, landing 0.05 mag off.
# Both are now read from OpenUniverse's published model release -- see the SN Iax and TDE blocks in
# `intermediate_z_contaminants`.
#
# Still out, and for reasons of their own:
#   * PISN (57, 58) -- its SEDs stop at 20000 A rest-frame and F184's red edge is 21000, so it
#     cannot cover F184 without extrapolating below z = 0.050, and this sample's grid starts at
#     z = 0.010. It cannot be rendered in the bins the sample exists to fill. Its band-to-band
#     residual against OpenUniverse's own photometry is also 0.10-0.16 mag against 0.015 for the
#     classes that are in. SLSN-I (40) had been out with it and is now IN: one template, 1122
#     objects, 0.015 mag, and a 10 pc calibration that is OpenUniverse's exactly.
#   * gentype 99 -- NOT A TRANSIENT. 27771 objects whose magnitude does not vary in time at all
#     (0.000 mag over the light curve) and is identical in all six Roman bands, i.e. a flat AB
#     spectrum at fixed brightness: a calibration source. The `z_CMB` it carries means nothing.
#   * the OpenUniverse kilonovae (50) -- the class the classifier is being taught to find, which
#     cannot enter the contaminant sample.
PARENT_GENTYPES = (10, 12, 21, 26, 32, 40, 42)
CORE_COLLAPSE_GENTYPES = (21, 26, 32)

# gentype 32 is missing on purpose: OpenUniverse pools SN IIP and SN IIL into one label and only
# the template says which one an object is, so its label is resolved from `template_index`.
LABEL_BY_GENTYPE = {10: "SN Ia", 12: "SN Iax", 21: "SN Ib", 26: "SN Ic", 40: "SLSN-I", 42: "TDE"}

CATALOG_COLUMNS = [
    "id",
    "gentype",
    "z_CMB",
    "peak_mjd",
    "AV",
    "RV",
    "model_param_names",
    "model_param_values",
]

# OpenUniverse records "no host screen" as AV = -9 rather than as a missing value, so the sentinel
# has to be recognised: -9 means no screen, not nine magnitudes of extinction. See the host
# extinction block of `intermediate_z_contaminants` for which classes carry one and why.
NO_HOST_EXTINCTION_SENTINEL = -9.0

# Rest-frame days from maximum the peak magnitude is taken over, on both sides of the comparison.
# The generator's own `REST_FRAME_PHASES` limits, repeated here rather than imported so that this
# module does not depend on the one that uses it; the test pins them together.
PEAK_PHASE_LIMITS = (-20.0, 70.0)


def _model_parameter(names, values, wanted):
    for name, value in zip(names, values, strict=True):
        if name == wanted:
            return float(value)
    return np.nan


def read_parent_catalog(catalog_directory, gentypes=PARENT_GENTYPES):
    """One row per candidate parent over every snana_*.parquet of `catalog_directory`.

    Everything the re-render inherits and nothing else: which template the object was drawn from,
    the SALT shape and colour it was drawn with, the host screen it was given, its redshift and its
    peak epoch. The light curve is NOT read here -- it lives in the 16 GB hdf5 beside each
    catalogue and only the selected parents are ever read from it."""
    paths = sorted(glob.glob(os.path.join(str(catalog_directory), "snana_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no snana_*.parquet in {catalog_directory}")

    blocks = []
    for path in paths:
        healpix = int(os.path.basename(path).split("_")[1].split(".")[0])
        table = pq.read_table(path, columns=CATALOG_COLUMNS).to_pandas()
        table = table[table["gentype"].isin(gentypes)].copy()
        table["healpix"] = healpix
        blocks.append(table)
    catalog = pd.concat(blocks, ignore_index=True)

    catalog["template_index"] = [
        int(_model_parameter(names, values, "template_index"))
        for names, values in zip(catalog["model_param_names"], catalog["model_param_values"], strict=True)
    ]
    for parameter in ("salt2_x1", "salt2_c", "salt2_mB"):
        catalog[parameter] = [
            _model_parameter(names, values, parameter)
            for names, values in zip(catalog["model_param_names"], catalog["model_param_values"], strict=True)
        ]
    no_screen = np.isclose(catalog["AV"], NO_HOST_EXTINCTION_SENTINEL)
    catalog["host_av"] = np.where(no_screen, np.nan, catalog["AV"])
    catalog["host_rv"] = np.where(no_screen, np.nan, catalog["RV"])
    catalog["label"] = catalog["gentype"].map(LABEL_BY_GENTYPE)
    catalog["redshift"] = catalog["z_CMB"].astype(float)
    # The group a re-rendered object belongs to for the leakage-aware split, in the same vocabulary
    # `training/openuniverse_data.py` already uses for the OpenUniverse contaminants: every copy of
    # this parent, and the parent itself, are one group.
    catalog["parent_key"] = "snana_" + catalog["healpix"].astype(str) + "_" + catalog["id"].astype(str)
    return catalog[
        [
            "parent_key",
            "healpix",
            "id",
            "gentype",
            "label",
            "redshift",
            "peak_mjd",
            "template_index",
            "salt2_x1",
            "salt2_c",
            "salt2_mB",
            "host_av",
            "host_rv",
        ]
    ]


def parent_peak_magnitudes(group, redshift, peak_mjd, bands, phase_limits=PEAK_PHASE_LIMITS):
    """{band: brightest AB magnitude} of one OpenUniverse object over `phase_limits`.

    `group` is the object's hdf5 group, which carries the model sampled over its whole span. A band
    with no valid magnitude inside the window is absent from the result rather than NaN, so a
    caller can count what it got."""
    rest_phase = (np.asarray(group["mjd"][:], dtype=float) - float(peak_mjd)) / (1.0 + float(redshift))
    inside = (rest_phase >= phase_limits[0]) & (rest_phase <= phase_limits[1])
    if not inside.any():
        return {}
    peaks = {}
    for band in bands:
        magnitude = np.asarray(group["mag_" + band[0]][:], dtype=float)
        usable = inside & np.isfinite(magnitude) & (magnitude < 99.0)
        if usable.any():
            peaks[band] = float(magnitude[usable].min())
    return peaks
