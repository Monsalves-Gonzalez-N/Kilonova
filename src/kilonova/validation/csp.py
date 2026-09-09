"""The Carnegie Supernova Project I photometry, as read from the released ASCII tables.

CSP-I is the one low-redshift sample that measures, in the same natural system and for the same
supernovae, both the rest-frame optical and the rest-frame near-infrared: uBgVri from the Swope and
YJH from RetroCam and WIRC. That is what makes it the test the izc generator needs. Roman's bands
at z = 0.02-0.1 sample the rest-frame near-infrared, which is where every template in
`kilonova.simulation.intermediate_z_contaminants` is an extrapolation, and CSP-I YJH is the
observed light curve of that extrapolation.

Two releases are read here:

  * DR3, Krisciunas et al. (2017): 134 white dwarf explosions, one `SN*_snpy.txt` per object, times
    as MJD - 53000. Subtypes, T(Bmax) and dm15(B) come from the VizieR copy of its Table 1,
    J/AJ/154/211, because the tarball's `tab1.dat` carries neither the subtype nor the epoch of
    maximum.
  * The stripped-envelope release, Stritzinger et al. (2018): 34 SNe Ib/Ic/IIb, photometry split
    across five tables by telescope and by camera generation. Redshift, spectral type and T(Bmax)
    come from VizieR J/A+A/609/A134/table1.

  * the type II release, Anderson et al. (2024): uBgVri and YJH for 94 SNe II, CSP-I and CSP-II
    together, from VizieR J/A+A/692/A95. It carries neither redshift nor epoch of explosion, so the
    distance modulus, the explosion epoch and the Milky Way reddening of the CSP-I half come from
    Table 1 of Martinez et al. (2022), read out of that paper's arXiv source.

Everything is returned in the CSP natural system, with each measurement carrying the sncosmo
bandpass name of the exact filter it was taken through -- which for CSP is not one name per letter:
V changed twice in 2006 and the near-infrared cameras differ between the Swope and the du Pont.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "csp"

# The three CSP V filters, which are three different pieces of glass and not three calibrations of
# one. The boundaries are the dates given in the DR3 README; the released DR3 files already carry
# the distinction in the filter code, the stripped-envelope tables do not and are split by date.
V_LC3009_LAST_MJD = 53749.0  # 2006 Jan 14
V_LC3014_LAST_MJD = 53760.0  # 2006 Jan 25

# DR3 filter code -> sncosmo bandpass. `Jrc2` is the RetroCam J after the January 2009 camera
# rebuild and sncosmo ships no separate transmission for it; it is mapped onto the Swope J, which
# is the closest thing that exists in the registry.
DR3_BANDPASS = {
    "u": "cspu",
    "g": "cspg",
    "r": "cspr",
    "i": "cspi",
    "B": "cspb",
    "V": "cspv9844",
    "V0": "cspv3009",
    "V1": "cspv3014",
    "Y": "cspys",
    "J": "cspjs",
    "Jrc2": "cspjs",
    "H": "csphs",
    "Ydw": "cspyd",
    "Jdw": "cspjd",
    "Hdw": "csphd",
}

# Effective wavelength order, for plotting and for reading a table blue to red.
BANDPASS_ORDER = (
    "cspu",
    "cspb",
    "cspg",
    "cspv9844",
    "cspv3009",
    "cspv3014",
    "cspr",
    "cspi",
    "cspys",
    "cspyd",
    "cspjs",
    "cspjd",
    "csphs",
    "csphd",
)

# The letter each bandpass measures, so that Swope and du Pont photometry of one supernova can be
# summarised together without pretending the two transmissions are identical.
BANDPASS_LETTER = {
    "cspu": "u",
    "cspb": "B",
    "cspg": "g",
    "cspv9844": "V",
    "cspv3009": "V",
    "cspv3014": "V",
    "cspr": "r",
    "cspi": "i",
    "cspys": "Y",
    "cspyd": "Y",
    "cspjs": "J",
    "cspjd": "J",
    "csphs": "H",
    "csphd": "H",
}

DR3_TIME_OFFSET_MJD = 53000.0
JULIAN_DATE_TO_MJD = -2400000.5

# Stripped-envelope tables, by (file, [column letters], sncosmo bandpass per letter). Table 5, the
# du Pont optical, is not read: its natural system is the du Pont's and sncosmo ships only the
# Swope optical transmissions, so those measurements have no bandpass to be synthesized through.
SE_PHOTOMETRY_TABLES = (
    ("Tab4.ascii.dat", ("u", "g", "r", "i", "B", "V")),
    ("Tab6.ascii.dat", ("Y", "J", "H")),
    ("Tab7.ascii.dat", ("Y", "J", "H")),
    ("Tab8.ascii.dat", ("Ydw", "Jdw", "Hdw")),
)
SE_SWOPE_NIR_BANDPASS = {"Y": "cspys", "J": "cspjs", "H": "csphs"}
SE_DUPONT_NIR_BANDPASS = {"Ydw": "cspyd", "Jdw": "cspjd", "Hdw": "csphd"}

# Spectral type as Stritzinger et al. print it -> the izc class it belongs to. Ic-BL is folded into
# SN Ic, which is what the generator does too: `nugent-hyper`, the SN 1998bw template, is one of
# the SN Ic sources.
SE_TYPE_TO_LABEL = {"Ib": "SN Ib", "Ic": "SN Ic", "Ic-BL": "SN Ic", "IIb": None}

# The type II release gives no subtype, and the izc generator splits SN II into IIP, IIL and IIn.
# The comparison is therefore against the pooled class, which is also the only thing the classifier
# ever sees: all three izc subtypes carry OpenUniverse gentype 32 and the single label "SN II".
TYPE_II_LABEL = "SN II"

# Telescope + instrument, as Anderson et al. print it, -> the natural system it was taken in. The
# release spans CSP-I and CSP-II and sncosmo ships transmissions only for the CSP-I combinations;
# the CSP-II ones (the e2v CCD on the Swope, RetroCam moved to the du Pont) are a different natural
# system with no bandpass to synthesize through, so those measurements are dropped rather than
# silently attributed to the CSP-I filters. `duPont+TEK5` is dropped for the same reason.
TYPE_II_OPTICAL_SYSTEMS = ("Swope+Site3",)
TYPE_II_NIR_BANDPASS = {
    "Swope+RetroCam": {"Y": "cspys", "J": "cspjs", "H": "csphs"},
    "duPont+WIRC": {"Y": "cspyd", "J": "cspjd", "H": "csphd"},
}

# DR3 subtype -> izc class. Only the two the generator actually produces are mapped; the peculiar
# subclasses (91bg, 91T, 86G, Ia-CSM, super-Chandrasekhar, 06bt, 06gz) are read and kept, but they
# have no counterpart in the izc mix and are not compared against one.
DR3_SUBTYPE_TO_LABEL = {"normal": "SN Ia", "Iax": "SN Iax"}


def _supernova_name(name):
    """One spelling for a supernova across three releases that do not share one.

    DR3 and the stripped-envelope tables print `2005el`; the type II release prints `SN2005el` for
    the IAU-designated objects and the survey name, `ASASS14gm` or `LSQ12fui`, for the rest."""
    name = str(name).strip()
    if name[:4].isdigit():
        return "SN" + name
    return name


def _photometry_frame(rows):
    frame = pd.DataFrame(rows, columns=["sn", "bandpass", "mjd", "mag", "mag_err"])
    frame = frame[np.isfinite(frame["mag"]) & np.isfinite(frame["mjd"])]
    return frame.sort_values(["sn", "bandpass", "mjd"], ignore_index=True)


def load_dr3_photometry(root=None):
    """Every DR3 measurement as one long frame: sn, bandpass, mjd, mag, mag_err."""
    directory = (root or DATA_ROOT) / "DR3"
    rows = []
    for path in sorted(directory.glob("SN*_snpy.txt")):
        with path.open() as handle:
            name = handle.readline().split()[0]
            bandpass = None
            for line in handle:
                fields = line.split()
                if not fields:
                    continue
                if fields[0] == "filter":
                    bandpass = DR3_BANDPASS[fields[1]]
                    continue
                rows.append(
                    (
                        name,
                        bandpass,
                        float(fields[0]) + DR3_TIME_OFFSET_MJD,
                        float(fields[1]),
                        float(fields[2]),
                    )
                )
    return _photometry_frame(rows)


def _se_bandpass(letter, mjd):
    if letter in SE_DUPONT_NIR_BANDPASS:
        return SE_DUPONT_NIR_BANDPASS[letter]
    if letter in SE_SWOPE_NIR_BANDPASS:
        return SE_SWOPE_NIR_BANDPASS[letter]
    if letter != "V":
        return DR3_BANDPASS[letter]
    if mjd <= V_LC3009_LAST_MJD:
        return "cspv3009"
    if mjd <= V_LC3014_LAST_MJD:
        return "cspv3014"
    return "cspv9844"


def load_stripped_envelope_photometry(root=None):
    """The Stritzinger et al. (2018) tables as one long frame, same schema as DR3.

    Each table is a run of per-supernova blocks: a line carrying only the name, then one row per
    epoch with a Julian date and a (magnitude, uncertainty) pair per filter, `INDEF` where the
    filter was not measured. The three header lines are skipped by the same rule that separates
    names from data -- a data row starts with a number, a name row does not."""
    directory = (root or DATA_ROOT) / "ASCII_tables"
    rows = []
    for filename, letters in SE_PHOTOMETRY_TABLES:
        name = None
        for line in (directory / filename).read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            try:
                julian_date = float(fields[0])
            except ValueError:
                if len(fields) == 1:
                    name = "SN" + fields[0]
                continue
            mjd = julian_date + JULIAN_DATE_TO_MJD
            for position, letter in enumerate(letters):
                magnitude, uncertainty = fields[1 + 2 * position : 3 + 2 * position]
                if magnitude == "INDEF" or uncertainty == "INDEF":
                    continue
                rows.append((name, _se_bandpass(letter, mjd), mjd, float(magnitude), float(uncertainty)))
    return _photometry_frame(rows)


def load_metadata(root=None):
    """One row per supernova: redshift, subtype, reference epoch, izc class, coordinates.

    The three releases are concatenated with a `release` column and the columns are named the same
    way for all of them, so a caller loops over supernovae and never over releases.

    `epoch_mjd` is the time origin the light curve is compared on and `epoch_kind` says what it is.
    For the two thermonuclear releases and the stripped-envelope one it is B maximum, which those
    papers fit; for the type II release it is the epoch of explosion, because a SN II has no
    maximum worth aligning on -- the plateau is flat and the B peak is a few days of fast rise that
    most of these light curves start after."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table

    def degrees(table):
        # Both VizieR copies print the coordinates sexagesimally, in separate columns.
        coordinates = SkyCoord(
            table["RAJ2000"].astype(str), table["DEJ2000"].astype(str), unit=(u.hourangle, u.deg)
        )
        return coordinates.ra.deg, coordinates.dec.deg

    directory = root or DATA_ROOT
    dr3 = Table.read(directory / "dr3_metadata.ecsv").to_pandas()
    stripped = Table.read(directory / "se_metadata.ecsv").to_pandas()

    dr3_rows = pd.DataFrame(
        {
            "sn": dr3["SN"].map(_supernova_name),
            "release": "DR3",
            "redshift": dr3["z"].astype(float),
            "subtype": dr3["Subtype1"].astype(str).str.strip(),
            # Table 1 prints two epochs of maximum and gives T(Bmax) only for the 73 supernovae
            # whose B light curve is sampled through it. Tpeak, the epoch of maximum of the
            # SNooPy template fit, is given for all 134 and agrees with T(Bmax) to well under a
            # day where both exist, so it fills in rather than leaving half the sample without a
            # time axis to compare on.
            "epoch_mjd": dr3["T(Bmax)"].astype(float).fillna(dr3["Tpeak"].astype(float)),
            "epoch_fitted": dr3["T(Bmax)"].astype(float).isna(),
            "epoch_kind": "bmax",
            "distance_modulus": np.nan,
            "right_ascension": degrees(dr3)[0],
            "declination": degrees(dr3)[1],
            "dm15_b": dr3["Dm15(B)"].astype(float),
        }
    )
    dr3_rows["label"] = dr3_rows["subtype"].map(DR3_SUBTYPE_TO_LABEL)

    stripped_rows = pd.DataFrame(
        {
            "sn": stripped["SN"].map(_supernova_name),
            "release": "SE",
            "redshift": stripped["z"].astype(float),
            "subtype": stripped["Type"].astype(str).str.strip(),
            # Table 1 gives the epoch as JD - 2450000.
            "epoch_mjd": stripped["T(B)max"].astype(float) + 2450000.0 + JULIAN_DATE_TO_MJD,
            "epoch_kind": "bmax",
            "distance_modulus": np.nan,
            "right_ascension": degrees(stripped)[0],
            "declination": degrees(stripped)[1],
            "dm15_b": np.nan,
            "epoch_fitted": False,
        }
    )
    stripped_rows["label"] = stripped_rows["subtype"].map(SE_TYPE_TO_LABEL)

    columns = list(dr3_rows.columns)
    type_ii = load_type_ii_metadata(root)[columns]
    return pd.concat([dr3_rows, stripped_rows[columns], type_ii], ignore_index=True)


def load_type_ii_photometry(root=None):
    """The Anderson et al. (2024) SN II light curves, same long schema as the other two releases."""
    from astropy.table import Table

    directory = root or DATA_ROOT
    rows = []
    for name, letters in (("snii_optical.ecsv", "ugriBV"), ("snii_nir.ecsv", "YJH")):
        table = Table.read(directory / name).to_pandas()
        for _, epoch in table.iterrows():
            mjd = float(epoch["JD"]) + JULIAN_DATE_TO_MJD
            system = str(epoch["Tel"]).strip()
            for letter in letters:
                magnitude, uncertainty = epoch[f"{letter}mag"], epoch[f"e_{letter}mag"]
                if not np.isfinite(magnitude) or not np.isfinite(uncertainty):
                    continue
                if letter in "YJH":
                    if system not in TYPE_II_NIR_BANDPASS:
                        continue
                    bandpass = TYPE_II_NIR_BANDPASS[system][letter]
                elif system not in TYPE_II_OPTICAL_SYSTEMS:
                    continue
                else:
                    bandpass = _se_bandpass(letter, mjd)
                rows.append(
                    (
                        _supernova_name(epoch["SN"]),
                        bandpass,
                        mjd,
                        float(magnitude),
                        float(uncertainty),
                    )
                )
    return _photometry_frame(rows)


def load_type_ii_metadata(root=None):
    """One row per CSP-I SN II: distance modulus, explosion epoch, coordinates, and a redshift.

    The redshift here is NOT a distance and is not used as one. Brightness comes from the published
    distance modulus, which for objects this nearby is the only defensible choice -- half of them
    sit below z = 0.01, where the peculiar velocity of the host is a larger term than anything this
    comparison measures. The redshift is only what shifts and dilates the SED, so it is inverted
    from that same distance modulus through the Hubble flow: at these distances an error of
    300 km/s in it moves the spectrum by a tenth of a percent in wavelength."""
    from astropy.table import Table

    directory = root or DATA_ROOT
    positions = Table.read(directory / "snii_positions.ecsv").to_pandas()
    published = pd.read_csv(directory / "snii_martinez2022.csv")

    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.cosmology import Planck18, z_at_value

    coordinates = SkyCoord(
        positions["RAJ2000"].astype(str), positions["DEJ2000"].astype(str), unit=(u.hourangle, u.deg)
    )
    positions = pd.DataFrame(
        {
            "sn": positions["SN"].map(_supernova_name),
            "right_ascension": coordinates.ra.deg,
            "declination": coordinates.dec.deg,
        }
    )
    published["sn"] = published["sn"].map(_supernova_name)
    metadata = positions.merge(published, on="sn", how="inner").dropna(subset=["explosion_mjd"])
    metadata["redshift"] = [
        float(z_at_value(Planck18.distmod, modulus * u.mag, zmax=0.5))
        for modulus in metadata["distance_modulus"]
    ]
    metadata["release"] = "SNII"
    metadata["subtype"] = "II"
    metadata["label"] = TYPE_II_LABEL
    metadata["dm15_b"] = np.nan
    metadata["epoch_mjd"] = metadata["explosion_mjd"]
    metadata["epoch_kind"] = "explosion"
    metadata["epoch_fitted"] = False
    return metadata


def load_photometry(root=None):
    return pd.concat(
        [
            load_dr3_photometry(root),
            load_stripped_envelope_photometry(root),
            load_type_ii_photometry(root),
        ],
        ignore_index=True,
    )


# CMB dipole apex and amplitude, Planck Collaboration (2020) -- needed because at these redshifts
# the heliocentric redshift is not a distance. At z = 0.005 the dipole alone is a 7 % change in the
# distance modulus, and the residual peculiar velocity of the host is a comparable term that no
# correction removes; `PECULIAR_VELOCITY_KM_S` is carried alongside as its size.
CMB_APEX_GALACTIC_LONGITUDE = 264.021
CMB_APEX_GALACTIC_LATITUDE = 48.253
CMB_DIPOLE_VELOCITY_KM_S = 369.82
PECULIAR_VELOCITY_KM_S = 300.0
SPEED_OF_LIGHT_KM_S = 299792.458


def cmb_frame_redshift(right_ascension, declination, heliocentric_redshift):
    """The heliocentric redshifts of the catalogue, corrected to the CMB frame."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    coordinates = SkyCoord(right_ascension, declination, unit="deg").galactic
    apex = SkyCoord(CMB_APEX_GALACTIC_LONGITUDE, CMB_APEX_GALACTIC_LATITUDE, unit="deg", frame="galactic")
    projection = np.cos(coordinates.separation(apex).to(u.rad).value)
    dipole_redshift = CMB_DIPOLE_VELOCITY_KM_S * projection / SPEED_OF_LIGHT_KM_S
    return (1.0 + np.asarray(heliocentric_redshift)) * (1.0 + dipole_redshift) - 1.0


def milky_way_reddening(metadata, root=None):
    """E(B-V) of the Milky Way at each supernova, Schlafly & Finkbeiner (2011), cached on disk.

    Queried from IRSA once and written to `mw_ebv.csv`; the file is what the comparison reads
    afterwards, so the figure can be regenerated without the network."""
    directory = root or DATA_ROOT
    cache_path = directory / "mw_ebv.csv"
    cached = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["sn", "ebv"])

    missing = metadata[~metadata["sn"].isin(cached["sn"])]
    if len(missing):
        from astropy.coordinates import SkyCoord
        from astroquery.ipac.irsa.irsa_dust import IrsaDust

        rows = []
        for _, supernova in missing.iterrows():
            table = IrsaDust.get_query_table(
                SkyCoord(supernova["right_ascension"], supernova["declination"], unit="deg"),
                section="ebv",
            )
            rows.append({"sn": supernova["sn"], "ebv": float(table["ext SandF mean"][0])})
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached.to_csv(cache_path, index=False)
    return cached.set_index("sn")["ebv"]


def milky_way_extinction_ratios(bandpasses, reddening_law_rv=3.1):
    """A_band / E(B-V) for each CSP bandpass, integrated through the filter.

    A single ratio per band rather than a per-object one: the colour term of doing it properly is a
    few thousandths of a magnitude over the reddening range of this sample, and the alternative
    would make the correction depend on the spectrum the comparison is trying to test."""
    import sncosmo
    from extinction import fitzpatrick99

    # The source spectrum the ratio is averaged over. Flat in f_lambda is the conventional choice
    # and the one CSP itself uses for its own A_band table.
    wavelength = np.arange(2500.0, 25000.0, 5.0)
    flux = np.ones_like(wavelength)
    ratios = {}
    for name in bandpasses:
        bandpass = sncosmo.get_bandpass(name)
        inside = (wavelength >= bandpass.minwave()) & (wavelength <= bandpass.maxwave())
        transmission = bandpass(wavelength[inside])
        extinguished = flux[inside] * 10.0 ** (
            -0.4 * fitzpatrick99(wavelength[inside], reddening_law_rv, reddening_law_rv)
        )
        clean_flux = np.trapezoid(flux[inside] * transmission * wavelength[inside], wavelength[inside])
        dusty_flux = np.trapezoid(extinguished * transmission * wavelength[inside], wavelength[inside])
        ratios[name] = 2.5 * np.log10(clean_flux / dusty_flux)
    return ratios
