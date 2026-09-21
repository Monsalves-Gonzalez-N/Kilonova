"""Read OpenUniverse's own core-collapse SED templates out of its published model release.

WHERE THESE COME FROM. OpenUniverse drew its core-collapse SEDs from
`NON1ASED.V19_CC+HostXT_WAVEEXT` -- Vincenzi et al. (2019) corrected for host extinction, extended
to 25000 A by the methods of Pierel et al. (2018). That library is published in full:

    https://zenodo.org/records/14749318  ->  MODELS-1_TRANSIENT_SED.tar (3.77 GB)

so this script reads the `.SED` files directly. It used to recover the same templates out of the
16 GB per-healpix light-curve files instead, because the release note the library belongs to was
read as saying that the `_WAVEEXT` half of that name was not public. It is, and has been since
2025-01-27; `docs/plan_templates_oficiales_ou.md` records the migration.

WHAT THE RECOVERY COST, and what reading the files gives back:

  * The recovery read `flambda` on a FIXED observer-frame grid ending at 24450 A, so an object
    reached 20600 A rest -- F184's red edge at the sample's own z = 0.02 floor -- only below
    z = 0.187. That cut is gone; a file has no redshift.
  * It averaged 8 objects per template and each object covered only the phases its own light curve
    sampled, so several templates began at -4 d or later. The files carry the native phase grid,
    which runs from about -18 d out past +120.
  * The 3.77 GB tar is NOT a dependency of the pipeline. This script is run by hand and its 24 MB
    output is what `intermediate_z_contaminants` reads, which is why the archive format below is
    unchanged from the recovery it replaces.

THE TEMPLATE INDEX IS EXPLICIT AND IS NOT INFERRED. The library's own `NON1A.LIST` gives
`template_index -> file` and its `SIMGEN_INCLUDE_NON1A.INPUT` gives `template_index -> SNTYPE`;
both are read by `read_template_types`. The catalogue is still read, for one reason: to check that
the templates the library indexes are the templates OpenUniverse actually drew, and that their
gentypes agree. The library indexes 44 and OpenUniverse drew 44 -- 17 SN IIP, 7 SN IIL, 13 SN Ib
and 7 SN Ic -- but that is a fact to verify on every run, not to assume.

HOST EXTINCTION IS ZERO, and reproducing that is the point. OpenUniverse's own release note says
it: "for the SNCC (II/Ib/Ic) models we mistakenly used the de-reddened SEDs and therefore did not
model host extinction". These are those de-reddened SEDs. The sample must carry the same bug.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CORE_COLLAPSE_GENTYPES = (21, 26, 32)

# Where the flux must be positive for a phase to be kept: R062 through F184 over every
# redshift the generator draws.
COMPLETE_WAVELENGTH_LIMITS = (3000.0, 21000.0)
PHASE_STEP = 1.0  # see the block comment: the files' own phase axis breaks sncosmo's spline

# THE WAVELENGTH GRID IS THE FILE'S OWN AND IS NOT TOUCHED. All 44 files share it EXACTLY --
# 1605 to 25000 A in 5 A steps, 4680 points, verified on every one -- so there is nothing to
# resample and the archive covers F184 in the rest frame down to z = 0. The old archive stopped at
# 20600 A, which is F184's red edge at z = 0.02, so it covered F184 only above z = 0.019 while this
# sample's grid starts at 0.010: objects in the lowest bins were being dropped for want of coverage
# the model has.
#
# THE PHASE AXIS IS RESAMPLED ONTO A REGULAR 1 d GRID, and that is not a preference. The files'
# phases are the epochs the original spectroscopy had, so they are wildly irregular -- consecutive
# nodes 0.04 d apart next to gaps of 2 d -- and sncosmo's 2D spline is ill-conditioned on that:
# measured, `SN2016X` comes out NEGATIVE across the whole of 9000-21000 A at 8 d before its own B
# maximum, where the file itself has no non-positive cell anywhere in that range. On a regular 1 d
# grid the overshoot is gone. Linear in phase, over the run the template actually covers.
#
# THE PHASES ARE ALSO TRIMMED, for a reason of the model rather than of the interpolation. Some
# templates have runs of ZERO near-infrared flux at their extreme phases -- the V19 extension has
# nothing to extend where the original spectroscopy stops -- and the worst is 4750 A wide. A phase
# whose optical is healthy and whose near-infrared is zero reaches the window as "very red, no
# near-infrared", a class-correlated artefact and the exact species this sample exists to remove.
# So each template keeps the longest run of contiguous phases, CONTAINING ITS B MAXIMUM, over
# which the flux is positive everywhere in 3000-21000 A rest-frame -- the range the generator
# reads. 36 of the 44 lose nothing at all; the median template keeps 199 d where the old fixed
# window gave 47.
#
# The run has to contain B maximum rather than merely be the longest: `SN1994I` has a zero-flux
# region in the near-ultraviolet between +32 and +96 d, and its longest clean run is everything
# after it, which would discard the peak.
#
# WHAT THE OLD (-12, +35) d WINDOW COST, beyond the 47 d it left: it decided where `peak_phase`
# finds B maximum, and for the slow SNe II it put it inside the cut. SN1987A and SN2008bj peak
# after +35 d, so their "maximum" was an artefact of the window. The median template now keeps
# 199 d.

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
    """Every core-collapse object with its gentype and template index."""
    rows = []
    for path in sorted(glob.glob(os.path.join(catalog_directory, "snana_*.parquet"))):
        healpix = int(os.path.basename(path).split("_")[1].split(".")[0])
        table = pq.read_table(
            path, columns=["id", "gentype", "z_CMB", "peak_mjd", "model_param_values"]
        ).to_pandas()
        table = table[table.gentype.isin(CORE_COLLAPSE_GENTYPES)]
        table["template_index"] = [int(v[0]) for v in table.model_param_values]
        table["healpix"] = healpix
        rows.append(table[["id", "healpix", "gentype", "template_index", "z_CMB", "peak_mjd"]])
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


def source_name_from_file(filename):
    """`pycoco_ASASSN14jb_extended.SED` -> `pycoco_ASASSN14jb`, the name the archive carries.

    The `_extended` suffix is the library's mark for the near-infrared extension and is not part of
    the object's name. Stripping it is what keeps the archive's `template_names` -- and so the
    `ou-` source names hardcoded in `SOURCES_BY_LABEL` -- identical to the recovery's."""
    return filename.replace(".SED.gz", "").replace(".SED", "").replace("_extended", "")


def read_sed_file(path):
    """(phase, wavelength, flux[phase, wavelength]) of one `.SED`, exactly as the file has them.

    A NON1ASED `.SED` is `phase wavelength flux` per line, one full wavelength block per phase, on
    a regular wavelength grid and an IRREGULAR phase grid -- the phases are the epochs the original
    spectroscopy had, so they come at 1.8 d here and 5 d there. Neither axis is touched."""
    raw = np.loadtxt(path)
    block = int(np.flatnonzero(np.diff(raw[:, 0]) != 0)[0]) + 1
    if len(raw) % block:
        raise ValueError(f"{path}: {len(raw)} rows is not a whole number of {block}-row blocks")
    wavelength = raw[:block, 1]
    if not np.array_equal(raw[:, 1].reshape(-1, block), np.tile(wavelength, (len(raw) // block, 1))):
        raise ValueError(f"{path}: the wavelength grid is not the same at every phase")
    return raw[::block, 0], wavelength, raw[:, 2].reshape(-1, block)


def regular_phase_grid(first, last):
    """Grilla de PHASE_STEP anclada EXACTAMENTE en `first`, sin pasarse de `last`.

    EL ANCLA ES EL BORDE DEL ARCHIVO Y NO UN ENTERO, y eso no es cosmetico. Esto decia
    `np.arange(np.ceil(first), np.floor(last) + ...)`: redondear el borde de entrada hacia adelante
    tira entre 0 y 1 dia de SED que el archivo si tiene -- medido, 0.605 d de mediana y hasta 0.98 d,
    en 40 de las 44 plantillas (`pycoco_SN2007od` empieza en -7.98 y se guardaba desde -7.0).

    Esa rebanada esta lejos de ser inofensiva: OpenUniverse detecta sus objetos justo cuando
    aparecen, o sea pegados al inicio de la plantilla, asi que es donde se acumulan. Medido sobre 986
    objetos a z>0.5, el 25% cae a una fase anterior al primer punto guardado -- y en las 19
    plantillas donde eso pasa, lo que el objeto se pasa es SIEMPRE menor o igual a lo que el ceil
    habia tirado. Sin la SED de esa rebanada no se les puede anclar la magnitud.

    El paso sigue siendo de 1 d por la razon de siempre (el eje irregular del archivo rompe el spline
    de sncosmo, ver el comentario de arriba). Lo que cambia es donde empieza la regla, no su
    espaciado. El ultimo punto se queda en o antes de `last` para no extrapolar; el borde de salida
    pierde hasta 1 d y ahi no importa, porque ninguna epoca observada cae a +190 d."""
    count = int(np.floor((last - first) / PHASE_STEP)) + 1
    return first + np.arange(count) * PHASE_STEP


def complete_phase_run(phase, wavelength, flux):
    """(first, last) of the longest contiguous phase run containing B maximum and no flux hole.

    "No hole" is positive flux everywhere in COMPLETE_WAVELENGTH_LIMITS; see the block comment at
    the top for why a partial near-infrared hole must never reach the generator."""
    import sncosmo

    inside = (wavelength >= COMPLETE_WAVELENGTH_LIMITS[0]) & (wavelength <= COMPLETE_WAVELENGTH_LIMITS[1])
    complete = (flux[:, inside] > 0).all(axis=1)
    if not complete.any():
        raise ValueError("no phase of this template has positive flux across the whole band range")
    peak = float(sncosmo.TimeSeriesSource(phase, wavelength, flux).peakphase("bessellb"))
    at_peak = int(np.argmin(np.abs(phase - peak)))
    if not complete[at_peak]:
        raise ValueError(f"the phase of B maximum ({peak:+.1f} d) has a flux hole")
    first = at_peak
    while first > 0 and complete[first - 1]:
        first -= 1
    last = at_peak
    while last < len(complete) - 1 and complete[last + 1]:
        last += 1
    return first, last


def check_against_catalog(catalog, sntype_by_index):
    """The template indices OpenUniverse drew, checked against the ones the library indexes.

    Raises unless the two agree object for object: a template the catalogue drew and the library
    does not index cannot be rendered, and a gentype that disagrees with the library's SNTYPE means
    the index is being read as something it is not. 21 is SN Ib, 26 is SN Ic and 32 is the pool
    SN IIP and SN IIL are drawn from, which is why the check is one-way on the II."""
    gentype_by_label = {"SN IIP": 32, "SN IIL": 32, "SN Ib": 21, "SN Ic": 26}
    drawn = sorted(int(one) for one in catalog.template_index.unique())
    indexed = sorted(sntype_by_index)
    if drawn != indexed:
        raise ValueError(
            f"the catalogue drew {len(drawn)} templates and the library indexes {len(indexed)}; "
            f"only in the catalogue: {sorted(set(drawn) - set(indexed))}, "
            f"only in the library: {sorted(set(indexed) - set(drawn))}"
        )
    gentypes = catalog.groupby("template_index")["gentype"].unique()
    for index in drawn:
        label = LABEL_BY_SNTYPE[sntype_by_index[index]]
        carried = sorted(int(one) for one in gentypes.loc[index])
        if carried != [gentype_by_label[label]]:
            raise ValueError(
                f"template_index {index} is SNTYPE {sntype_by_index[index]} ({label}, gentype "
                f"{gentype_by_label[label]}) but its objects carry gentype {carried}"
            )
    return drawn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogs", default="data/openuniverse/snana_catalogs")
    parser.add_argument(
        "--model-directory",
        required=True,
        help="the unpacked NON1ASED.V19_CC+HostXT_WAVEEXT of MODELS-1_TRANSIENT_SED.tar",
    )
    parser.add_argument("--output", type=Path, default=Path("data/openuniverse/cc_templates.npz"))
    arguments = parser.parse_args()

    model_directory = Path(arguments.model_directory)
    file_by_index, sntype_by_index = read_template_types(model_directory)
    catalog = read_catalogs(arguments.catalogs)
    indices = check_against_catalog(catalog, sntype_by_index)
    print(
        f"{len(catalog)} core-collapse objects over {len(indices)} templates; "
        f"the library indexes the same {len(indices)} and every gentype agrees"
    )

    archive, names, labels, wavelength = {}, [], [], None
    for index in indices:
        filename = file_by_index[index]
        path = model_directory / filename
        if not path.exists():
            path = model_directory / f"{filename}.gz"
        phase, one_wavelength, flux = read_sed_file(path)
        if wavelength is None:
            wavelength = one_wavelength
            archive["wavelength"] = wavelength.astype(np.float32)
        elif not np.array_equal(one_wavelength, wavelength):
            raise ValueError(f"{path}: wavelength grid differs from the first template's")
        first, last = complete_phase_run(phase, one_wavelength, flux)
        regular = regular_phase_grid(phase[first], phase[last])
        resampled = np.array(
            [
                np.interp(regular, phase[first : last + 1], flux[first : last + 1, k])
                for k in range(flux.shape[1])
            ]
        ).T
        archive[f"phase_{len(names)}"] = regular.astype(np.float32)
        archive[f"flux_{len(names)}"] = resampled.astype(np.float32)
        names.append(source_name_from_file(filename))
        labels.append(LABEL_BY_SNTYPE[sntype_by_index[index]])
        print(
            f"  template_index {index:3d}  {names[-1]:<26s} {labels[-1]:<7s} "
            f"{len(regular):3d} fases de 1 d, {regular[0]:+.0f} a {regular[-1]:+.0f} d "
            f"(el archivo cubre {phase[0]:+.1f}..{phase[-1]:+.1f})"
        )

    archive["template_names"] = np.array(names)
    archive["labels"] = np.array(labels)
    archive["template_indices"] = np.array(indices, dtype=np.int32)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, **archive)
    size = arguments.output.stat().st_size / 1e6
    print(
        f"\nwrote {arguments.output} ({size:.1f} MB, {len(names)} templates, "
        f"{len(archive['wavelength'])} wavelengths, native phase grid per template)"
    )
    for label in sorted(set(labels)):
        print(f"  {label:<8s} {labels.count(label)}")


if __name__ == "__main__":
    main()
