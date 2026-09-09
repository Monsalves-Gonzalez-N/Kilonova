"""Extend OpenUniverse's own core-collapse SED templates into the near-infrared, the way it did.

WHY THIS EXISTS. OpenUniverse drew its core-collapse SEDs from `NON1ASED.V19_CC+HostXT_WAVEEXT`.
The base half of that name is public -- `NON1ASED.V19_CC+HostXT` ships in the SNANA SNDATA_ROOT
distribution (Zenodo record 4015325), 67 `pycoco_*.SED.gz` files covering 1605 to 11000 A -- and its
`SIMGEN_INCLUDE_NON1A.INPUT` carries 17 SNTYPE 20 (IIP) + 7 SNTYPE 22 (IIL) = 24 II, 13 Ib and 7 Ic,
which is exactly the 24/13/7 the OpenUniverse paper reports. So this is not a lookalike library, it
is the one that simulation used.

The `_WAVEEXT` half is NOT public. The string appears nowhere in the 14 216 entries of the
2024-07-04 release, contemporaneous with OpenUniverse2024, nor in the 14 396 of the 2026-04-10 one.
What is missing is the extension to the near-infrared by the methods of Pierel et al. (2018), and
11000 A rest-frame covers Roman only above z = 0.91 -- the opposite of the range the izc sample
needs.

The module used to answer that by substituting a different library entirely, the SNANA NON1A and
Nugent sources sncosmo ships already extended. That put one library above z = 0.91 and another
below, which is a class-correlated difference aligned with redshift -- the shape of the very
shortcut the izc sample exists to remove. This script answers it the other way: `snsedextend`
(Pierel, PyPI) is the public implementation of the method OpenUniverse cites, so the published
templates are extended HERE, by that method. Same base templates, same procedure, and a colour
residual against OpenUniverse that means something.

WHAT IS NOT REPRODUCED, and it is the reason this is still an approximation rather than a copy.
Pierel et al. fit colour curves (U-B, r-J, r-H, r-K against phase) to a training sample and then
extend each SED to match them. This script does NOT refit those curves: it uses the ones snsedextend
ships in `data/default/type{II,Ib,Ic}/*Curve.pick`, which are the authors' own fits. Whether
OpenUniverse refit them against its own sample is not knowable from the outside. Not refitting is
also what keeps this runnable, since the fitting path is the one that needs pymc3 and theano.

ENVIRONMENT. snsedextend cannot live with the pipeline: `extendCC` reaches
`astropy.modeling.blackbody`, removed in astropy 4.3, and astropy 4.0 needs `numpy.asscalar`,
removed in numpy 1.23. So a separate env, the same pattern `scripts/build_tde_templates.py` uses
for MOSFiT:

    conda create -n snsed_gen python=3.8 -y
    conda install -n snsed_gen -c conda-forge "numpy=1.22" "astropy=4.2" scipy matplotlib pandas \\
        sncosmo extinction -y
    <env>/bin/pip install --no-deps snsedextend==0.3.7 pyParz

and pymc3, theano and seaborn stubbed out in site-packages. They are imported by `snsedextend.BIC`,
which is reached only from `fitColorCurve` -- the path this script does not take -- and pinning them
for real would drag theano into the environment. Make each stub raise ImportError from
`__getattr__`, so that taking that path fails loudly instead of silently doing something else.

Run it as:

    SNDATA_ROOT=<a directory> <env>/bin/python scripts/build_v19_extended_templates.py \\
        --sndata-tarball SNDATA_ROOT_2026-04-10.tar.gz --output data/v19/v19_extended.npz
"""

import argparse
import gzip
import os
import pickle
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

# The SNANA model directory inside the tarball, and the two files of it that are not SEDs.
MODEL_PATH = "models/NON1ASED/NON1ASED.V19_CC+HostXT"
TEMPLATE_LIST = "NON1A.LIST"
SIMGEN_INCLUDE = "SIMGEN_INCLUDE_NON1A.INPUT"

# SNANA SNTYPE -> the colour curve snsedextend ships. It ships three, for II, Ib and Ic, and
# OpenUniverse's set carries six types, so three of them are assigned a curve that was not fitted
# for them. That is a choice of ours and OpenUniverse may have made a different one:
#   * IIn (21) and IIb (23) take the SN II curve. IIb is the arguable one -- spectroscopically it
#     ends up closer to Ib -- but its light curve and its early spectrum are hydrogen-rich, and the
#     colour curve is a function of phase, not of the late classification.
#   * Ic-BL (35) takes the SN Ic curve, which is the same class with broader lines.
CURVE_BY_SNTYPE = {20: "II", 22: "II", 21: "II", 23: "II", 32: "Ib", 33: "Ic", 35: "Ic"}
# And the izc class each SNTYPE belongs to, which is what the generator draws by.
LABEL_BY_SNTYPE = {
    20: "SN IIP",
    22: "SN IIL",
    21: "SN IIn",
    23: "SN IIb",
    32: "SN Ib",
    33: "SN Ic",
    35: "SN Ic",
}

# Trimming, and it is the difference between a 206 MB artefact and one small enough to version.
# Both limits are set by what the sample is FOR: breaking the brightness shortcut, not reproducing
# OpenUniverse's population. A kilonova is detected on four epochs of a five-day cadence, so about
# twenty observer-frame days, and phases outside this window never reach a generated object.
PHASE_LIMITS = (-12.0, 35.0)
# Rest-frame wavelength. R062's blue edge at z = 0.02 is 4589 A and F184's red edge is 20588 A; the
# margin covers the whole redshift range the sample generates with room at both ends.
WAVELENGTH_LIMITS = (3000.0, 25000.0)


def read_template_types(model_directory):
    """{SED filename: SNTYPE} from the model's own two index files.

    NON1A.LIST maps the template INDEX to its SED file and SIMGEN_INCLUDE_NON1A.INPUT maps the same
    INDEX to its SNTYPE, its rate weight and its luminosity function. Joining them on the index is
    how a file learns what it is; neither file says it alone.
    """
    file_by_index = {}
    for line in (model_directory / TEMPLATE_LIST).read_text().splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0] == "NON1A:":
            file_by_index[int(fields[1])] = fields[3]

    types = {}
    for line in (model_directory / SIMGEN_INCLUDE).read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[0] == "NON1A:":
            index = int(fields[1])
            if index in file_by_index:
                types[file_by_index[index]] = int(fields[5])
    return types


def extract_model(tarball, destination):
    """Unpack the V19 model directory and gunzip its SEDs into `$SNDATA_ROOT/snsed/NON1A`.

    snsedextend resolves every SED path against that directory, reads $SNDATA_ROOT once at import,
    and reads the SEDs with `sncosmo.read_griddata_ascii`, which does not decompress. So the layout
    is not a preference; it is the interface.
    """
    model_directory = destination / "model"
    sed_directory = destination / "snsed" / "NON1A"
    model_directory.mkdir(parents=True, exist_ok=True)
    sed_directory.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tarball, "r:gz") as archive:
        members = [one for one in archive.getmembers() if MODEL_PATH in one.name and one.isfile()]
        if not members:
            raise RuntimeError(f"{tarball} carries no {MODEL_PATH}")
        for member in members:
            name = Path(member.name).name
            source = archive.extractfile(member)
            if name.endswith(".SED.gz"):
                with gzip.open(source, "rb") as compressed:
                    (sed_directory / name[: -len(".gz")]).write_bytes(compressed.read())
            else:
                (model_directory / name).write_bytes(source.read())
    return model_directory, sed_directory


def load_colour_curves(curve_name):
    """(colour table, colour curve dict) that snsedextend ships for one type.

    The curves are the authors' own fits, pickled under Python 2, so they need latin1. Reading them
    rather than refitting is what lets this run without pymc3; see the module docstring.
    """
    from astropy.table import Table
    from snsedextend.utils import __dir__ as package_directory

    directory = Path(package_directory) / "data" / "default" / (f"type{curve_name}")
    colour_table = Table.read(str(directory / (f"{curve_name}Colors.dat")), format="ascii")
    with open(directory / (f"{curve_name}Curve.pick"), "rb") as handle:
        colour_curves = pickle.load(handle, encoding="latin1")
    return colour_table, colour_curves


def extend_templates(sed_directory, template_types, output_directory, verbose=True, resume=False):
    """Run snsedextend over every template, one call per colour curve. Returns the files written."""
    import snsedextend

    output_directory.mkdir(parents=True, exist_ok=True)
    written = {}
    skipped = []
    already = complete_extensions(output_directory) if resume else set()
    if resume and verbose:
        print(f"resuming: {len(already)} extended templates already on disk")
    for curve_name in sorted(set(CURVE_BY_SNTYPE.values())):
        names = sorted(
            name
            for name, sntype in template_types.items()
            if CURVE_BY_SNTYPE.get(sntype) == curve_name and (sed_directory / name).exists()
        )
        if not names:
            continue
        colour_table, colour_curves = load_colour_curves(curve_name)
        if verbose:
            print(f"extending {len(names)} templates with the type {curve_name} colour curve")
        # One call per template rather than one call for the list. snsedextend raises out of the
        # whole batch when a single template defeats it -- `createSNSED` builds a spline over the
        # result and some templates come back with a degenerate grid -- and losing forty templates
        # to one bad one is not a trade worth making. The sample's job is to put contaminants at
        # low redshift, not to carry every template, so a failure is recorded and skipped.
        for name in names:
            if name in already:
                written[name] = curve_name
                continue
            try:
                snsedextend.extendCC(
                    colour_table,
                    colour_curves,
                    snType=curve_name,
                    outFileLoc=str(output_directory),
                    sedlist=[name],
                    verbose=False,
                    showplots=False,
                )
            except Exception as failure:
                if (output_directory / name).exists():
                    # Written before it raised: `createSNSED` only builds the return value, so the
                    # extended SED on disk is complete and usable.
                    written[name] = curve_name
                    print(
                        f"  {name}: kept, snsedextend raised only on its "
                        f"return value ({type(failure).__name__})"
                    )
                else:
                    skipped.append((name, curve_name, f"{type(failure).__name__}: {failure}"))
                continue
            if (output_directory / name).exists():
                written[name] = curve_name
            else:
                skipped.append((name, curve_name, "no output written"))
    if skipped:
        print(f"\nSKIPPED {len(skipped)} templates:")
        for name, curve_name, reason in skipped:
            print(f"  {name:<28s} ({curve_name})  {reason[:90]}")
    return written


def template_defect(path):
    """(kind, reason) for a written SED, or None if it is sound. Kind is "partial" or "ragged".

    Two different failures wear the same filename, and telling them apart is what decides whether
    running snsedextend again is worth anything.

    "partial" is a run killed by hand. Parsing does NOT catch it: snsedextend writes the grid in an
    order that leaves the killed file rectangular and readable, just short in wavelength -- the two
    measured cases came back as 2580 wavelengths over 1605-14500 A instead of 10 761 over
    1200-55000 A, and `read_griddata_ascii` accepted both. The test is the wavelength span, which
    is the whole point of the extension. Extending again fixes it.

    "ragged" is snsedextend itself and extending again reproduces it exactly. `_extrapolatesed`
    accumulates one flux array per phase and then writes them with `array(finalF)` against `wnew`,
    the wavelength grid of the LAST phase alone. When one phase extrapolates onto a different grid
    the list is ragged, numpy builds an object array, and the file comes out with one phase block
    of a different length -- SN 2008D's phase +4.0 d carries 15 320 rows, a 4559-row grid running
    1200-23990 A followed by the full 10 761-row one. It is also what raises the IndexError that
    `extend_templates` reports as harmless: `createSNSED` splines the ragged result. 23 of the 67
    templates hit it, and they are dropped rather than repaired -- see `pack`.
    """
    import sncosmo

    try:
        _, wavelength, flux = sncosmo.read_griddata_ascii(str(path))
    except Exception as failure:
        return "partial", f"unreadable ({type(failure).__name__})"
    if flux.ndim != 2:
        return "ragged", "phases do not share one wavelength grid"
    if wavelength.min() > WAVELENGTH_LIMITS[0] or wavelength.max() < WAVELENGTH_LIMITS[1]:
        return "partial", f"truncated at {wavelength.min():.0f}-{wavelength.max():.0f} A"
    return None


def complete_extensions(output_directory):
    """Names in `output_directory` that snsedextend need not be run over again.

    Both the sound templates and the ragged ones: rerunning a ragged template reproduces it byte
    for byte, so retrying costs two minutes and changes nothing. `pack` is where they are dropped.
    """
    keep = set()
    for path in sorted(output_directory.glob("*.SED")):
        defect = template_defect(path)
        if defect is None or defect[0] == "ragged":
            keep.add(path.name)
        else:
            print(f"  {path.name}: {defect[1]}, will be extended again")
    return keep


def pack(extended_directory, written, template_types, output):
    """Trim every extended SED to PHASE_LIMITS x WAVELENGTH_LIMITS and save one compressed archive.

    One wavelength grid for all of them, because they come off snsedextend on the same grid and a
    single shared axis is what makes the archive small enough to version. A template whose grid does
    not match is interpolated onto the shared one rather than silently dropped.
    """
    import sncosmo

    names = sorted(written)
    dropped = []
    phases, wavelengths, fluxes, labels, sources = [], None, [], [], []
    for name in names:
        # Dropped rather than repaired. The repair would be to keep the last full-grid block of a
        # ragged phase and discard the short one, and there is no way from outside snsedextend to
        # know whether that block is this colour iteration's flux or the previous one's -- a guess
        # that silently produces a template rather than an error. What is left after dropping is
        # 10 IIP, 5 IIL, 3 IIn, 12 Ib and 5 Ic, a real library per class where the module had one
        # or two Nugent templates, and the sample exists to break a brightness shortcut rather than
        # to carry every template OpenUniverse had.
        defect = template_defect(extended_directory / name)
        if defect is not None:
            dropped.append((name, defect[1]))
            continue
        phase, wavelength, flux = sncosmo.read_griddata_ascii(str(extended_directory / name))
        phase_mask = (phase >= PHASE_LIMITS[0]) & (phase <= PHASE_LIMITS[1])
        wavelength_mask = (wavelength >= WAVELENGTH_LIMITS[0]) & (wavelength <= WAVELENGTH_LIMITS[1])
        phase, wavelength = phase[phase_mask], wavelength[wavelength_mask]
        flux = flux[np.ix_(phase_mask, wavelength_mask)]
        # Refuse a grid that does not cover the window rather than interpolate over the gap. This
        # is where the truncated file of an interrupted run would have entered silently: np.interp
        # clamps, so a template extended only to 14500 A would have been packed with its reddest
        # measured flux held flat across the whole near-infrared and nothing would have said so.
        if wavelength.min() > WAVELENGTH_LIMITS[0] or wavelength.max() < WAVELENGTH_LIMITS[1]:
            raise RuntimeError(
                f"{name} spans only {wavelength.min():.0f}-{wavelength.max():.0f} A, short of "
                f"the {WAVELENGTH_LIMITS[0]:.0f}-{WAVELENGTH_LIMITS[1]:.0f} A window"
            )
        if wavelengths is None:
            wavelengths = wavelength
        elif not np.array_equal(wavelength, wavelengths):
            flux = np.array([np.interp(wavelengths, wavelength, row) for row in flux])
        phases.append(phase.astype(np.float32))
        fluxes.append(flux.astype(np.float32))
        labels.append(LABEL_BY_SNTYPE[template_types[name]])
        sources.append(name)

    archive = {
        "wavelength": wavelengths.astype(np.float32),
        "template_names": np.array(sources),
        "labels": np.array(labels),
    }
    for index in range(len(sources)):
        archive[f"phase_{index}"] = phases[index]
        archive[f"flux_{index}"] = fluxes[index]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **archive)
    if dropped:
        print(f"\nDROPPED {len(dropped)} templates:")
        for name, reason in dropped:
            print(f"  {name:<28s} {reason}")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sndata-tarball", type=Path, required=True, help="SNDATA_ROOT tar.gz from Zenodo record 4015325"
    )
    parser.add_argument("--output", type=Path, default=Path("data/v19/v19_extended.npz"))
    parser.add_argument("--work-directory", type=Path, default=None, help="defaults to $SNDATA_ROOT")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep the extended templates already on disk and extend only the rest",
    )
    arguments = parser.parse_args()

    if "SNDATA_ROOT" not in os.environ:
        sys.exit("SNDATA_ROOT must be set: snsedextend reads it at import and resolves SED paths against it")
    work_directory = arguments.work_directory or Path(os.environ["SNDATA_ROOT"])

    model_directory = work_directory / "model"
    sed_directory = work_directory / "snsed" / "NON1A"
    unpacked = (model_directory / SIMGEN_INCLUDE).exists() and (model_directory / TEMPLATE_LIST).exists()
    if arguments.resume and unpacked and any(sed_directory.glob("*.SED")):
        # Scanning the 2 GB tarball again costs minutes and produces the same bytes.
        print(f"resuming: {work_directory} is already unpacked")
    else:
        model_directory, sed_directory = extract_model(arguments.sndata_tarball, work_directory)
    template_types = read_template_types(model_directory)
    found = len(list(sed_directory.glob("*.SED")))
    print(f"{found} templates in the model, {len(template_types)} with a type")

    extended_directory = work_directory / "extended"
    if extended_directory.exists() and not arguments.resume:
        shutil.rmtree(extended_directory)
    written = extend_templates(
        sed_directory,
        template_types,
        extended_directory,
        verbose=not arguments.quiet,
        resume=arguments.resume,
    )
    print(f"extended {len(written)} templates")

    archive = pack(extended_directory, written, template_types, arguments.output)
    size_mb = arguments.output.stat().st_size / 1e6
    print(
        f"wrote {arguments.output} ({size_mb:.1f} MB, "
        f"{len(archive['template_names'])} templates, {len(archive['wavelength'])} wavelengths)"
    )
    for label in sorted(set(archive["labels"].tolist())):
        print(f"  {label:<8s} {(archive['labels'] == label).sum()}")


if __name__ == "__main__":
    main()
