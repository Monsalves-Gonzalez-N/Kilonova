"""The CSP reader, checked on the released tables.

The comparison it feeds is read by eye, which is exactly why the parsing has to be checked by
something else: a filter mapped to the wrong transmission, a Julian date read as an MJD or a V
measurement attributed to the wrong piece of glass would all produce a figure that still looks like
a light curve. The data are downloaded rather than committed, so every test here skips when
`scripts/download_csp.py` has not been run.
"""

import numpy as np
import pytest

from kilonova.validation import csp

pytest.importorskip("sncosmo")

if not (csp.DATA_ROOT / "DR3").exists() or not (csp.DATA_ROOT / "dr3_metadata.ecsv").exists():
    pytest.skip("CSP photometry not downloaded; run scripts/download_csp.py", allow_module_level=True)


@pytest.fixture(scope="module")
def photometry():
    return csp.load_photometry()


@pytest.fixture(scope="module")
def metadata():
    return csp.load_metadata()


def test_every_released_filter_code_is_mapped():
    """A filter code the reader does not know would be dropped silently by `DR3_BANDPASS[...]`."""
    codes = set()
    for path in sorted((csp.DATA_ROOT / "DR3").glob("SN*_snpy.txt")):
        for line in path.read_text().splitlines():
            if line.startswith("filter"):
                codes.add(line.split()[1])
    assert codes <= set(csp.DR3_BANDPASS)


def test_every_bandpass_resolves_and_is_ordered_by_wavelength():
    import sncosmo

    wavelengths = [sncosmo.get_bandpass(name).wave_eff for name in csp.BANDPASS_ORDER]
    # Not strictly sorted: the Swope and du Pont versions of one letter differ by a few angstroms
    # in either direction, which is the point of keeping them apart.
    letters = [csp.BANDPASS_LETTER[name] for name in csp.BANDPASS_ORDER]
    by_letter = {}
    for letter, wavelength in zip(letters, wavelengths, strict=True):
        by_letter.setdefault(letter, []).append(wavelength)
    means = [np.mean(by_letter[letter]) for letter in dict.fromkeys(letters)]
    assert means == sorted(means)


def test_photometry_is_read_in_the_right_units(photometry):
    assert len(photometry) > 30000
    # MJD of the CSP campaigns, 2004-2015, not a Julian date and not a DR3 offset day. CSP-I ends
    # in 2009; the type II release reaches into CSP-II.
    assert photometry["mjd"].between(53100.0, 57200.0).all()
    assert photometry["mag"].between(10.0, 25.0).all()
    assert (photometry["mag_err"] > 0.0).all()


def test_the_three_v_filters_are_separated_by_date(photometry):
    """CSP replaced the Swope V twice in January 2006 and the transmissions differ."""
    for bandpass, limits in (
        ("cspv3009", (0.0, csp.V_LC3009_LAST_MJD)),
        ("cspv3014", (csp.V_LC3009_LAST_MJD, csp.V_LC3014_LAST_MJD)),
        ("cspv9844", (csp.V_LC3014_LAST_MJD, np.inf)),
    ):
        stripped_envelope = photometry[
            (photometry["bandpass"] == bandpass) & photometry["sn"].str.match(r"SN\d{4}[a-z]{0,2}$")
        ]
        if not len(stripped_envelope):
            continue
        assert stripped_envelope["mjd"].between(*limits).all()


def test_every_supernova_with_metadata_has_photometry(photometry, metadata):
    """The reverse does not hold: the type II release carries CSP-II objects that Martinez et al.
    never modelled, so they reach the photometry frame with no distance and no explosion epoch and
    are skipped by the comparison."""
    assert set(metadata["sn"]) <= set(photometry["sn"])
    # SN 2008bk, at mu = 27.7, is the nearest; SN 2009ca at z = 0.096 the most distant.
    assert metadata["redshift"].between(0.0005, 0.12).all()
    assert metadata["sn"].is_unique


def test_the_izc_classes_present_are_the_ones_the_generator_makes(metadata):
    labelled = metadata.dropna(subset=["label"])
    assert set(labelled["label"]) == {"SN Ia", "SN Iax", "SN Ib", "SN Ic", "SN II"}
    # The type II release gives no subtype, so the comparison is against the pooled class; a row
    # claiming one of the three izc subtypes would mean a subtype was invented somewhere. TDE has
    # no CSP counterpart at all.
    assert not metadata["label"].isin(["SN IIP", "SN IIL", "SN IIn", "TDE"]).any()


def test_the_time_origin_falls_where_the_light_curve_is(metadata, photometry):
    """The three releases give their epochs on three time systems -- MJD, JD - 2450000 and MJD
    again -- and a botched conversion shows up here as an origin nowhere near the data. An
    explosion precedes the first point and a maximum sits inside the light curve, so the two kinds
    are checked against different bounds."""
    with_origin = metadata.dropna(subset=["epoch_mjd"])
    assert len(with_origin) > 200
    for _, supernova in with_origin.iterrows():
        observed = photometry.loc[photometry["sn"] == supernova["sn"], "mjd"]
        if supernova["epoch_kind"] == "explosion":
            # SN 2005af, found on its plateau, is the latest start at 92 days after explosion.
            assert observed.min() - 100.0 <= supernova["epoch_mjd"] <= observed.min() + 5.0
        else:
            assert observed.min() - 25.0 <= supernova["epoch_mjd"] <= observed.max() + 25.0


def test_type_ii_photometry_only_uses_systems_sncosmo_has(photometry, metadata):
    """The type II release spans CSP-I and CSP-II, and only the CSP-I natural systems are in the
    registry. A CSP-II measurement quietly attributed to a CSP-I filter is the failure this
    forbids, and it would be invisible in the figure."""
    type_ii = metadata[metadata["release"] == "SNII"]["sn"]
    of_type_ii = photometry[photometry["sn"].isin(type_ii)]
    assert len(of_type_ii) > 5000
    assert set(of_type_ii["bandpass"]) <= set(csp.BANDPASS_ORDER)
    # Swope+e2v is 317 of the 1951 optical epochs and has no transmission; if it had leaked in, the
    # count would rise by roughly that fraction.
    from astropy.table import Table

    released = Table.read(csp.DATA_ROOT / "snii_optical.ecsv").to_pandas()
    kept = released[released["Tel"].isin(csp.TYPE_II_OPTICAL_SYSTEMS)]
    assert len(kept) < len(released)


def test_type_ii_redshift_is_the_inverse_of_the_published_distance():
    """The brightness of a type II model comes from the published distance modulus, and it gets
    there by the redshift being inverted from it. If the two ever stop agreeing, every type II
    model magnitude is wrong by the difference."""
    import astropy.units as u
    from astropy.cosmology import Planck18

    metadata = csp.load_type_ii_metadata()
    predicted = Planck18.distmod(metadata["redshift"].to_numpy()).to_value(u.mag)
    assert np.allclose(predicted, metadata["distance_modulus"].to_numpy(), atol=1e-3)


def test_extinction_ratios_fall_with_wavelength():
    import sncosmo

    ratios = csp.milky_way_extinction_ratios(csp.BANDPASS_ORDER)
    # One bandpass per letter: the three V filters differ in width more than in effective
    # wavelength, so among themselves the ratio does not follow wave_eff and is not meant to.
    one_per_letter = {}
    for name in csp.BANDPASS_ORDER:
        one_per_letter.setdefault(csp.BANDPASS_LETTER[name], name)
    ordered = sorted(one_per_letter.values(), key=lambda name: sncosmo.get_bandpass(name).wave_eff)
    values = [ratios[name] for name in ordered]
    assert values == sorted(values, reverse=True)
    # Anchors against the CSP's own A_band / E(B-V) table, which is the same law through the same
    # filters on a slightly different assumed spectrum.
    assert ratios["cspb"] == pytest.approx(4.0, abs=0.15)
    assert ratios["csphs"] == pytest.approx(0.55, abs=0.10)


def test_cmb_correction_has_the_right_sign_at_both_poles():
    towards = csp.cmb_frame_redshift(
        *_galactic_to_equatorial(csp.CMB_APEX_GALACTIC_LONGITUDE, csp.CMB_APEX_GALACTIC_LATITUDE), 0.02
    )
    away = csp.cmb_frame_redshift(
        *_galactic_to_equatorial(csp.CMB_APEX_GALACTIC_LONGITUDE + 180.0, -csp.CMB_APEX_GALACTIC_LATITUDE),
        0.02,
    )
    dipole = csp.CMB_DIPOLE_VELOCITY_KM_S / csp.SPEED_OF_LIGHT_KM_S
    assert towards == pytest.approx(0.02 + 1.02 * dipole, rel=1e-3)
    assert away == pytest.approx(0.02 - 1.02 * dipole, rel=1e-3)


def _galactic_to_equatorial(longitude, latitude):
    from astropy.coordinates import SkyCoord

    coordinates = SkyCoord(longitude, latitude, unit="deg", frame="galactic").icrs
    return coordinates.ra.deg, coordinates.dec.deg
