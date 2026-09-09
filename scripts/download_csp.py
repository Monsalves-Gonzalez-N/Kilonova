"""Fetch the Carnegie Supernova Project I photometry the izc comparison is validated against.

Everything this writes under `data/csp` is downloaded, so none of it is in the repository; run this
once before `scripts/compare_csp_lightcurves.py`. Four things are fetched:

  * the DR3 tarball (Krisciunas et al. 2017), 134 white dwarf explosions in uBgVriYJH;
  * the stripped-envelope tables (Stritzinger et al. 2018), 34 SNe Ib/Ic/IIb in the same system;
  * the VizieR copy of each paper's Table 1, for the subtypes, the redshifts and the epochs of B
    maximum that the released tarballs leave out;
  * the type II photometry (Anderson et al. 2024), 94 SNe II in uBgVri and YJH, from VizieR
    J/A+A/692/A95;
  * the distance moduli, explosion epochs and Milky Way reddenings of the CSP-I SNe II, from
    Table 1 of Martinez et al. (2022). That table is in neither the journal's machine-readable set
    nor VizieR, so it is read out of the arXiv source of the paper, which ships it as the LaTeX
    file `sn_info.tab`, and written here as a plain CSV;
  * the Milky Way reddening at each supernova, from IRSA, cached as `mw_ebv.csv`.

TDE remains the one izc class with no CSP counterpart.
"""

import argparse
import io
import re
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen

CSP_TARBALLS = (
    ("https://csp.obs.carnegiescience.edu/data/CSP_Photometry_DR3.tgz", "CSP_Photometry_DR3.tgz"),
    ("https://csp.obs.carnegiescience.edu/data/CSPI_SE_SN_photometry.tar", "CSPI_SE_SN_photometry.tar"),
)
VIZIER_TABLES = (
    ("J/AJ/154/211", "catalog", "dr3_metadata.ecsv"),
    ("J/A+A/609/A134/table1", "table1", "se_metadata.ecsv"),
    ("J/A+A/692/A95", "sn", "snii_positions.ecsv"),
    ("J/A+A/692/A95", "photopt", "snii_optical.ecsv"),
    ("J/A+A/692/A95", "photnir", "snii_nir.ecsv"),
)

# Table 1 of Martinez et al. (2022), A&A 660, A40, as its arXiv source ships it: one LaTeX row per
# supernova, `value(error)` for the distance modulus and the explosion epoch. The three SNe II
# whose explosion epoch the paper leaves as `---` come through with an empty field.
MARTINEZ_EPRINT_URL = "https://arxiv.org/e-print/2111.06519"
MARTINEZ_TABLE_MEMBER = "sn_info.tab"
MARTINEZ_ROW = re.compile(
    r"^(?P<sn>\d{4}[A-Za-z]{1,2})\s*&(?P<host>[^&]*)&\s*(?P<modulus>[\d.]+)\((?P<modulus_error>[\d.]+)\)"
    r"\s*&\s*(?:(?P<explosion>[\d.]+)\((?P<explosion_error>[\d.]+)\)|---)"
    r"\s*&\s*(?P<reddening>[\d.]+)"
)


def download(url, path):
    if path.exists():
        print(f"  {path.name} ya está")
        return
    print(f"  {url}")
    with urlopen(url) as response:
        path.write_bytes(response.read())


def martinez_table(path):
    """Distance modulus, explosion epoch and E(B-V)_MW of the 74 CSP-I SNe II, as a CSV."""
    if path.exists():
        print(f"  {path.name} ya está")
        return
    print(f"  {MARTINEZ_EPRINT_URL}")
    request = Request(MARTINEZ_EPRINT_URL, headers={"User-Agent": "kilonova-csp-validation"})
    with urlopen(request) as response:
        payload = response.read()
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        source = archive.extractfile(MARTINEZ_TABLE_MEMBER).read().decode()

    rows = []
    for line in source.splitlines():
        match = MARTINEZ_ROW.match(line.strip())
        if match is None:
            continue
        fields = match.groupdict()
        rows.append(
            {
                "sn": fields["sn"],
                "host": fields["host"].strip(),
                "distance_modulus": float(fields["modulus"]),
                "distance_modulus_error": float(fields["modulus_error"]),
                "explosion_mjd": float(fields["explosion"]) if fields["explosion"] else float("nan"),
                # The paper prints the uncertainty as its last digit, so 53240.5(2) is +/- 0.2 d.
                "explosion_error_days": (
                    float(fields["explosion_error"]) / 10.0 if fields["explosion_error"] else float("nan")
                ),
                "ebv_mw_published": float(fields["reddening"]),
            }
        )
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {len(rows)} supernovas")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--directory", type=Path, default=Path("data/csp"))
    arguments = parser.parse_args()
    arguments.directory.mkdir(parents=True, exist_ok=True)

    print("fotometría CSP:")
    for url, name in CSP_TARBALLS:
        path = arguments.directory / name
        download(url, path)
        with tarfile.open(path) as archive:
            archive.extractall(arguments.directory, filter="data")

    print("tablas VizieR:")
    from astroquery.vizier import Vizier

    for catalog, member, name in VIZIER_TABLES:
        path = arguments.directory / name
        if path.exists():
            print(f"  {name} ya está")
            continue
        print(f"  {catalog} / {member}")
        tables = Vizier(columns=["**"], row_limit=-1).get_catalogs(catalog)
        wanted = [one for one in tables if str(one.meta["name"]).endswith(member)]
        wanted[0].write(path, overwrite=True)

    print("tabla 1 de Martinez et al. (2022):")
    martinez_table(arguments.directory / "snii_martinez2022.csv")

    print("extinción galáctica (IRSA):")
    from kilonova.validation import csp

    reddening = csp.milky_way_reddening(csp.load_metadata(arguments.directory), arguments.directory)
    print(f"  {len(reddening)} supernovas, E(B-V) mediana {reddening.median():.3f}")


if __name__ == "__main__":
    main()
