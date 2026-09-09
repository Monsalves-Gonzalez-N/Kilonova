"""Read OpenUniverse's own core-collapse SED templates out of its light-curve files.

WHY THIS REPLACES `build_v19_extended_templates.py`. OpenUniverse drew its core-collapse SEDs from
`NON1ASED.V19_CC+HostXT_WAVEEXT`. The base half of that name is public and the `_WAVEEXT` half --
the extension into the near-infrared -- is not, and 11000 A rest-frame covers Roman only above
z = 0.91, the opposite of the range the izc sample needs. The previous answer was to redo that
extension with `snsedextend`, the public implementation of the method OpenUniverse cites. Measured
against OpenUniverse's own photometry that failed: our extension came out as a comb of one hump per
photometric anchor separated by runs of zero flux, worth a factor 0 to 3.3 against theirs across
11000-21000 A, while theirs is a smooth monotonic decline.

None of that reconstruction is necessary, because the templates are IN the release. Each object's
hdf5 group carries `flambda (n_mjd x 227)`, the model SED itself rather than a summary of it, on a
FIXED observer-frame grid of 1850-24450 A. Every object of one `template_index` is that one
rest-frame template at a different redshift, so dividing the grid by (1 + z) recovers it directly.

WHAT MAKES THAT SOUND, all measured rather than assumed:

  * Objects of one template at z = 0.081, 0.991 and 1.501, de-redshifted and normalised, agree to
    0.5 % median and 2.8 % worst over 2500-8000 A -- across a factor 15 in (1 + z).
  * The PUBLIC pycoco base agrees with the same de-redshifted SED to 2.3 % over 3000-10000 A. That
    is the control on everything else here: the library identification, the phase convention, the
    de-redshifting and the flux convention are all right, and the only thing that was ever wrong
    was our own extension.
  * `AV = -9` (OpenUniverse applies no host dust to core-collapse) and `mw_extinction_applied` is
    False with `mw_EBV = 0`, so `flambda` carries no extinction of any kind to undo.

WHICH OBJECTS. The observer-frame grid ends at 24450 A, so an object reaches 20600 A rest -- F184's
red edge at the sample's own z = 0.02 floor -- only below z = 0.187. All 44 templates OpenUniverse
actually drew have such objects, 12 to 70 each. The lowest-redshift ones are preferred because they
reach reddest, and several are averaged per template because they should be identical and a
disagreement is then visible rather than silent.

NETWORK. The files are 16 GB each and only a few objects are read from each, so they are opened
over HTTP range requests through fsspec rather than downloaded; h5py resolves a group by name
without listing, which is what makes this seconds rather than 33 x 16 GB. Point `--local-directory`
at a directory of already-downloaded files to skip the network.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_URL = (
    "https://nasa-irsa-simulations.s3.amazonaws.com/openuniverse2024/roman/full/"
    "roman_rubin_cats_v1.1.2_faint"
)
CORE_COLLAPSE_GENTYPES = (21, 26, 32)

# The observer-frame grid ends here, and it is what sets the redshift cut below.
OBSERVER_FRAME_RED_EDGE = 24450.0
# F184's red edge at z = 0.02, the sample's own floor. A template that does not reach this cannot
# supply F184 for the faintest-redshift object the generator draws.
REQUIRED_REST_RED_EDGE = 20600.0
MAXIMUM_REDSHIFT = OBSERVER_FRAME_RED_EDGE / REQUIRED_REST_RED_EDGE - 1.0

OBJECTS_PER_TEMPLATE = 8
# Same trim as the archive this replaces, and for the same reason: a kilonova is detected on four
# epochs of a five-day cadence, so phases outside this window never reach a generated object.
PHASE_LIMITS = (-12.0, 35.0)
PHASE_STEP = 1.0
WAVELENGTH_LIMITS = (3000.0, REQUIRED_REST_RED_EDGE)
WAVELENGTH_STEP = 5.0
# Where each object's SED is normalised before averaging. Well inside every object's rest coverage
# and away from both the blue cutoff and the extension.
NORMALISATION_WAVELENGTH = 8000.0
# And the phase it is measured at. Every object must reach it, which 94 % of them do; it is what
# puts objects of one template on a common scale before they are combined.
NORMALISATION_PHASE = 0.0

LABEL_BY_SNTYPE = {
    20: "SN IIP",
    22: "SN IIL",
    21: "SN IIn",
    23: "SN IIb",
    32: "SN Ib",
    33: "SN Ic",
    35: "SN Ic",
}


def read_catalogs(catalog_directory):
    """Every core-collapse object with its template index, redshift and peak epoch."""
    rows = []
    for path in sorted(glob.glob(os.path.join(catalog_directory, "snana_*.parquet"))):
        healpix = int(os.path.basename(path).split("_")[1].split(".")[0])
        table = pq.read_table(
            path, columns=["id", "gentype", "z_CMB", "peak_mjd", "model_param_values"]
        ).to_pandas()
        table = table[table.gentype.isin(CORE_COLLAPSE_GENTYPES)]
        table["template_index"] = [int(v[0]) for v in table.model_param_values]
        table["healpix"] = healpix
        rows.append(table[["id", "healpix", "template_index", "z_CMB", "peak_mjd"]])
    return pd.concat(rows, ignore_index=True)


def read_template_types(model_directory):
    """{template index: SNTYPE} joined across the library's own two index files."""
    file_by_index, types = {}, {}
    for line in (Path(model_directory) / "NON1A.LIST").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0] == "NON1A:":
            file_by_index[int(fields[1])] = fields[3]
    for line in (Path(model_directory) / "SIMGEN_INCLUDE_NON1A.INPUT").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[0] == "NON1A:":
            types[int(fields[1])] = int(fields[5])
    return file_by_index, types


def open_healpix(healpix, local_directory):
    import h5py

    if local_directory:
        path = Path(local_directory) / f"snana_{healpix}.hdf5"
        if path.exists():
            return h5py.File(path, "r")
    import fsspec

    return h5py.File(fsspec.open(f"{BASE_URL}/snana_{healpix}.hdf5", block_size=2**20).open(), "r")


def object_rest_frame_cube(group, redshift, peak_mjd, phases, wavelengths):
    """One object's SED on the shared rest-frame grid, normalised, NaN where it has no phase.

    Rest phase is (mjd - peak_mjd) / (1 + z) and rest wavelength is lambda / (1 + z); the flux is
    normalised rather than corrected for distance, because the shape is what a template carries and
    `set_source_peakabsmag` sets the scale downstream.

    NO OBJECT IS REQUIRED TO SPAN THE WHOLE PHASE WINDOW, and that is not a relaxation of rigour
    but a fact about the data. `peak_mjd` is OpenUniverse's own per-object peak and the model grid
    starts where the model starts, so the first rest phase runs from -25 d to well after maximum
    across objects, with a median of -3.6: demanding -12 to +35 from a single object keeps a
    quarter of them. Every object of one template is the SAME rest-frame SED, so they can be
    combined cell by cell instead, each contributing the phases it has. What every object IS
    required to cover is the wavelength window, since a partial one would bias a cell towards
    whichever objects reach reddest, and the normalisation phase, so the scales are commensurable.
    """
    rest_wavelength = group["lambda"][:] / (1.0 + redshift)
    if rest_wavelength.max() < wavelengths[-1] or rest_wavelength.min() > wavelengths[0]:
        return None
    rest_phase = (group["mjd"][:] - peak_mjd) / (1.0 + redshift)
    if rest_phase.min() > NORMALISATION_PHASE or rest_phase.max() < NORMALISATION_PHASE:
        return None

    flux = group["flambda"][:].astype(float)
    # Wavelength first, then phase: both axes are regular enough for linear interpolation, and the
    # phase axis is the one that differs object to object because the mjd grid is observer-frame.
    on_wavelength = np.array([np.interp(wavelengths, rest_wavelength, row) for row in flux])
    cube = np.array([np.interp(phases, rest_phase, on_wavelength[:, k]) for k in range(len(wavelengths))]).T
    # np.interp holds the end value flat outside the sampled range, which would invent a spectrum
    # for phases the object never had. Those cells are NaN and are simply absent from the median.
    outside = (phases < rest_phase.min()) | (phases > rest_phase.max())
    cube[outside, :] = np.nan

    anchor_row = np.array(
        [np.interp(NORMALISATION_PHASE, rest_phase, on_wavelength[:, k]) for k in range(len(wavelengths))]
    )
    anchor = float(np.interp(NORMALISATION_WAVELENGTH, wavelengths, anchor_row))
    if not np.isfinite(anchor) or anchor <= 0:
        return None
    return cube / anchor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogs", default="data/openuniverse/snana_catalogs")
    parser.add_argument(
        "--model-directory",
        required=True,
        help="the unpacked NON1ASED.V19_CC+HostXT model, for NON1A.LIST and SIMGEN_INCLUDE",
    )
    parser.add_argument(
        "--local-directory",
        default=None,
        help="directory of already-downloaded snana_<healpix>.hdf5; " "the rest are read over HTTP",
    )
    parser.add_argument("--output", type=Path, default=Path("data/openuniverse/cc_templates.npz"))
    parser.add_argument("--objects-per-template", type=int, default=OBJECTS_PER_TEMPLATE)
    arguments = parser.parse_args()

    file_by_index, sntype_by_index = read_template_types(arguments.model_directory)
    catalog = read_catalogs(arguments.catalogs)
    usable = catalog[catalog.z_CMB <= MAXIMUM_REDSHIFT].sort_values("z_CMB")
    print(
        f"{len(catalog)} core-collapse objects, {len(usable)} below z = {MAXIMUM_REDSHIFT:.3f} "
        f"over {usable.template_index.nunique()} templates"
    )

    phases = np.arange(PHASE_LIMITS[0], PHASE_LIMITS[1] + 0.5 * PHASE_STEP, PHASE_STEP)
    wavelengths = np.arange(
        WAVELENGTH_LIMITS[0], WAVELENGTH_LIMITS[1] + 0.5 * WAVELENGTH_STEP, WAVELENGTH_STEP
    )

    # Grouped by healpix so each 16 GB file is opened once.
    wanted = usable.groupby("template_index", group_keys=False).head(arguments.objects_per_template)
    print(f"reading {len(wanted)} objects from {wanted.healpix.nunique()} healpix files")

    cubes = {index: [] for index in sorted(wanted.template_index.unique())}
    for healpix, block in wanted.groupby("healpix"):
        handle = open_healpix(healpix, arguments.local_directory)
        kept = 0
        for row in block.itertuples():
            group = handle.get(str(row.id))
            if group is None:
                continue
            cube = object_rest_frame_cube(group, float(row.z_CMB), float(row.peak_mjd), phases, wavelengths)
            if cube is not None:
                cubes[row.template_index].append(cube)
                kept += 1
        handle.close()
        print(f"  healpix {healpix}: {kept}/{len(block)} objects usable")

    archive = {"wavelength": wavelengths.astype(np.float32)}
    names, labels, spreads = [], [], []
    for index in sorted(cubes):
        stack = cubes[index]
        if not stack:
            print(f"  template_index {index}: NO usable object, dropped")
            continue
        stack = np.array(stack)
        with np.errstate(invalid="ignore"):
            median = np.nanmedian(stack, axis=0)
        coverage = np.isfinite(stack).sum(axis=0)
        if (coverage == 0).any():
            thin = phases[(coverage == 0).any(axis=1)]
            print(f"    phases with NO object: {thin.min():+.0f} to {thin.max():+.0f} d")
        # Objects of one template are the same rest-frame SED, so their spread is a check and not a
        # measurement: it reports whether the de-redshifting and the phase alignment held.
        if len(stack) > 1:
            usable = np.isfinite(stack) & (median > 0)
            ratio = np.where(usable, stack / np.where(median > 0, median, np.nan), np.nan)
            spread = float(np.nanmedian(np.abs(ratio - 1.0)))
        else:
            spread = float("nan")
        # PHASE RANGE PER TEMPLATE, not one window for all, and the reason is physics rather than
        # data volume. `peak_mjd` is OpenUniverse's own peak, and for a SN II that sits at the
        # start of the plateau, essentially at explosion -- so there is almost no light curve
        # before it: SN IIP and SN IIL reach a median of -3 d and some only 0, while every SN Ib
        # reaches -12. A window shared by all of them would start at 0 and throw away the rise for
        # the classes that have one. Each template is trimmed to its own longest run of covered
        # phases instead, which is the format the archive it replaces already used.
        covered = coverage.min(axis=1) > 0
        if not covered.any():
            print(f"  template_index {index}: no phase covered by every wavelength, dropped")
            continue
        first, last = np.argmax(covered), len(covered) - 1 - np.argmax(covered[::-1])
        if not covered[first : last + 1].all():
            print(f"  template_index {index}: phase coverage is not contiguous, dropped")
            continue
        archive[f"phase_{len(names)}"] = phases[first : last + 1].astype(np.float32)
        archive[f"flux_{len(names)}"] = median[first : last + 1].astype(np.float32)
        archive[f"coverage_{len(names)}"] = coverage[first : last + 1].astype(np.int16)
        names.append(file_by_index[index].replace(".SED", ""))
        labels.append(LABEL_BY_SNTYPE[sntype_by_index[index]])
        spreads.append(spread)
        print(
            f"  template_index {index:3d}  {names[-1]:<26s} {labels[-1]:<7s} "
            f"{len(stack)} objects, spread {spread:.4f}, "
            f"phases {archive[f'phase_{len(names) - 1}'][0]:+.0f} to "
            f"{archive[f'phase_{len(names) - 1}'][-1]:+.0f} d"
        )

    archive["template_names"] = np.array(names)
    archive["labels"] = np.array(labels)
    archive["object_spread"] = np.array(spreads, dtype=np.float32)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, **archive)
    size = arguments.output.stat().st_size / 1e6
    print(
        f"\nwrote {arguments.output} ({size:.1f} MB, {len(names)} templates, "
        f"{len(wavelengths)} wavelengths, phase grid per template)"
    )
    for label in sorted(set(labels)):
        print(f"  {label:<8s} {labels.count(label)}")


if __name__ == "__main__":
    main()
