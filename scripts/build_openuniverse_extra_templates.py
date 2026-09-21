"""Read OpenUniverse's SN Iax and TDE SED templates out of its published model release.

The counterpart of `build_openuniverse_cc_templates.py` for the two classes that used to be
SUBSTITUTIONS rather than OpenUniverse's own models, and that are substitutions no longer:

  * SN Iax was regenerated here from the Rutgers notebook OpenUniverse cites, and landed 0.05 mag
    off OpenUniverse's own photometry.
  * TDE was the MOSFiT `tde` model standing in for the observed SED of AT2019qiz, 0.19 mag off with
    a colour trend.

SLSN-I is here too, which was never a substitution -- it simply had no template on this side at
all. Its `2016apd.sed` reproduces OpenUniverse's own photometry to 0.015 mag and its 10 pc
calibration is OpenUniverse's exactly, -21.75 against -21.75.

PISN IS DELIBERATELY ABSENT. Its SEDs stop at 20000 A rest-frame and F184's red edge is at 21000,
so it cannot cover F184 without extrapolating below z = 0.050 -- and this sample's redshift grid
starts at 0.010. It cannot be rendered in the bins the sample exists to fill. Its band-to-band
residual against OpenUniverse's photometry is also 0.10-0.16 mag against 0.015 for the rest.

All of these are in `MODELS-1_TRANSIENT_SED.tar` (zenodo.org/records/14749318).

NOTHING IS RESAMPLED. Unlike the core-collapse archive, which puts 44 templates on one shared grid,
these are written on the grids the files come with: SN Iax is 94 phases by 1200 wavelengths and all
919 templates share it exactly, TDE is 84 by 2301. Storing them as they are is what makes the claim
"the SED is OpenUniverse's own, unmodified" literally true for these two classes.

THE SN Iax INDEX IS OFF BY ONE, and the release's own list is what is wrong. `NON1A.LIST` says
`template_index` 1 is `SED-Iax-0001.dat`; OpenUniverse used `SED-Iax-0000.dat`. Measured over 4357
parents on 40 templates, the absolute magnitude OpenUniverse gave each object correlates with the
one its file carries at r = 0.998 under that shift and at r = 0.055 without it -- and the bank is a
random draw per row, so neighbouring files are unrelated and the error hides perfectly as "the
calibration was overwritten". The directory holds 1001 files (0000 to 1000) while the list indexes
0001 to 0999, which is the visible symptom. This script writes templates in OpenUniverse's own
`template_index` order, 1 to 919, so downstream indexes 0 to 918 with no shift of its own.

THE FLUX IS ABSOLUTE and is kept that way. Both files declare it: "flux: erg/s/cm^2/A scaled to
10 pc". That is not true of the V19 core-collapse files, which declare nothing and one of which is
mis-normalised by 4 mag -- see the core-collapse script. Here the calibration is OpenUniverse's own,
verified against its photometry to 0.15 mag rms for SN Iax and 0.10 mag for TDE.
"""

import argparse
import gzip
from pathlib import Path

import numpy as np

IAX_SHIPPED = 919  # what OpenUniverse's catalogue draws: template_index 1 to 919
IAX_INDEX_SHIFT = -1  # see the module docstring


def read_sed(path):
    """(phase, wavelength, flux[phase, wavelength]) of one `phase wavelength flux` file.

    Lines that are not three numbers are headers -- `2019qiz.sed` carries an un-commented
    `phase wavelength flux` line -- and are skipped rather than trusted to start with '#'."""
    rows = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 3:
                continue
            try:
                rows.append([float(one) for one in fields])
            except ValueError:
                continue
    raw = np.array(rows)
    block = int(np.flatnonzero(np.diff(raw[:, 0]) != 0)[0]) + 1
    if len(raw) % block:
        raise ValueError(f"{path}: {len(raw)} rows is not a whole number of {block}-row blocks")
    return raw[::block, 0], raw[:block, 1], raw[:, 2].reshape(-1, block)


def build_iax(model_directory, output):
    directory = Path(model_directory)
    phase, wavelength, _ = read_sed(directory / "SED-Iax-0000.dat.gz")
    archive = {"phase": phase.astype(np.float32), "wavelength": wavelength.astype(np.float32)}
    for index in range(1, IAX_SHIPPED + 1):
        path = directory / f"SED-Iax-{index + IAX_INDEX_SHIFT:04d}.dat.gz"
        one_phase, one_wavelength, flux = read_sed(path)
        if not np.array_equal(one_phase, phase) or not np.array_equal(one_wavelength, wavelength):
            raise ValueError(f"{path}: grid differs from SED-Iax-0000; they were all identical")
        archive[f"flux_{index - 1}"] = flux.astype(np.float32)
        if index % 100 == 0:
            print(f"  {index}/{IAX_SHIPPED}")
    archive["source_files"] = np.array(
        [f"SED-Iax-{i + IAX_INDEX_SHIFT:04d}.dat" for i in range(1, IAX_SHIPPED + 1)]
    )
    write(archive, output, f"{IAX_SHIPPED} SN Iax templates")


def build_single(model_directory, filename, output, what):
    """One class whose whole model is a single SED file: TDE and SLSN-I."""
    phase, wavelength, flux = read_sed(Path(model_directory) / filename)
    write(
        {
            "phase": phase.astype(np.float32),
            "wavelength": wavelength.astype(np.float32),
            "flux_0": flux.astype(np.float32),
            "source_files": np.array([filename.replace(".gz", "")]),
        },
        output,
        what,
    )


def build_tde(model_directory, output):
    build_single(model_directory, "2019qiz.sed.gz", output, "1 TDE template")


def write(archive, output, what):
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **archive)
    print(
        f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB, {what}, "
        f"{len(archive['phase'])} phases x {len(archive['wavelength'])} wavelengths)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iax-directory", help="the unpacked SIMSED.SNIax of MODELS-1_TRANSIENT_SED.tar")
    parser.add_argument(
        "--tde-directory", help="the unpacked NON1ASED.TDE-BBFIT of MODELS-1_TRANSIENT_SED.tar"
    )
    parser.add_argument(
        "--slsn-directory", help="the unpacked NON1ASED.SLSN-I-BBFIT of MODELS-1_TRANSIENT_SED.tar"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/openuniverse"))
    arguments = parser.parse_args()

    if arguments.iax_directory:
        build_iax(arguments.iax_directory, arguments.output_dir / "iax_templates.npz")
    if arguments.tde_directory:
        build_tde(arguments.tde_directory, arguments.output_dir / "tde_template.npz")
    if arguments.slsn_directory:
        build_single(
            arguments.slsn_directory,
            "2016apd.sed.gz",
            arguments.output_dir / "slsn_template.npz",
            "1 SLSN-I template",
        )


if __name__ == "__main__":
    main()
