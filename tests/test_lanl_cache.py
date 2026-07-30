"""The LANL spectra are per angular bin; the cache stores the isotropic equivalent.

The unit test pins the conversion itself. The integration test pins it against the AB magnitudes
LANL publishes next to the spectra, and is skipped when the raw grid is not mounted -- it is the
only check that catches a wrong *value* of the factor rather than a wrong shape.
"""

import numpy as np
import pytest

from kilonova.photometry.spectra import magnitudes_for_bands

# lanl_cache pulls in ray, and through simulation.extinction also pyphot and dustmaps, none of which
# the slim CI env installs -- import the module the same way test_extinction.py does so a missing sim
# extra skips this file instead of failing collection.
lanl_cache = pytest.importorskip(
    "kilonova.simulation.lanl_cache", reason="needs pyphot/ray/dustmaps (sim extras)"
)

pytest.importorskip("galsim")

CM_TO_ANG = 1e8
# LSST z (8700 A) and Roman Z087 (8720 A) are near enough to compare directly; 2MASS J/H against
# Roman J129/H158 are looser, which is why the tolerance below is 0.5 mag and not 0.05.
LANL_TO_ROMAN = [("z-band", "Z087"), ("J-band", "J129"), ("H-band", "H158")]


def test_isotropic_equivalent_scales_by_the_number_of_angular_bins():
    flux = np.ones((3, 5, 54), dtype=np.float32)
    converted = lanl_cache.isotropic_equivalent_flux(flux)
    assert converted.shape == flux.shape
    assert np.allclose(converted, 54.0)


def test_isotropic_equivalent_reads_the_bin_count_from_the_data():
    """The factor is 4 pi / dOmega_bin, so it has to follow the file's own binning, not a
    hardcoded 54 -- a grid with a different number of angles must scale by that number."""
    assert np.allclose(lanl_cache.isotropic_equivalent_flux(np.ones((2, 4, 18))), 18.0)


def test_schema_records_the_flux_convention():
    """The cached parquet is not a verbatim copy of the .dat column, so it must say so: a reader
    that applied the factor a second time would be 4.33 mag out."""
    metadata = lanl_cache.build_schema(np.linspace(1000.0, 128000.0, 1024)).metadata
    assert b"isotropic-equivalent" in metadata[b"flux_convention"]
    assert b"n_angles" in metadata[b"flux_angular_scaling"]


def _read_mag_blocks(path):
    blocks, label, rows = {}, None, []
    for line in open(path):
        stripped = line.strip()
        if stripped.startswith("#"):
            if rows:
                blocks[label] = np.array(rows)
                rows = []
            label = stripped.lstrip("# ").split("(")[0].strip()
        elif stripped:
            rows.append([float(value) for value in stripped.split()])
    if rows:
        blocks[label] = np.array(rows)
    return blocks


def _read_spectra(path):
    times = np.array([float(line.split("time[d]=")[1]) for line in open(path) if "# it=" in line])
    raw = np.loadtxt(path, comments="#", dtype=np.float64)
    n_wavelengths = raw.shape[0] // times.size
    raw = raw.reshape(times.size, n_wavelengths, -1)
    wavelength_aa = 0.5 * (raw[0, :, 0] + raw[0, :, 1]) * CM_TO_ANG
    return times, wavelength_aa, raw[:, :, 2:]


def test_cached_flux_reproduces_the_published_lanl_magnitudes(lanl_grid_dir):
    """End-to-end: our synthetic photometry over the converted spectrum must land on the AB
    magnitudes LANL publishes for the same model, phase and viewing angle. Without the factor
    every magnitude comes out 4.33 mag too faint."""
    stem = "Run_TP_dyn_all_lanth_wind1_all_md0.1_vd0.05_mw0.1_vw0.05"
    spectra_path = lanl_grid_dir / f"{stem}_spec_2020-03-31.dat"
    magnitudes_path = lanl_grid_dir / f"{stem}_mags_2020-03-31.dat"
    if not spectra_path.exists():
        pytest.skip(f"{spectra_path.name} not in the mounted grid")

    times, wavelength_aa, flux_per_bin = _read_spectra(spectra_path)
    flux = lanl_cache.isotropic_equivalent_flux(flux_per_bin)
    published = _read_mag_blocks(magnitudes_path)

    residuals = []
    for lanl_band, roman_band in LANL_TO_ROMAN:
        block = published[lanl_band]
        for time_position in (25, 40):
            for angle in (0, 27):
                nearest = int(np.argmin(np.abs(block[:, 1] - times[time_position])))
                expected = float(block[nearest, 2 + angle])
                ours = magnitudes_for_bands(
                    wavelength_aa, flux[time_position, :, angle], 0.0, bands=[roman_band]
                )[roman_band]
                assert np.isfinite(ours)
                residuals.append(ours - expected)

    residuals = np.array(residuals)
    # centred on zero: what is left is the LSST/2MASS vs Roman bandpass mismatch, not a scale error
    assert abs(np.median(residuals)) < 0.25, f"residuals {residuals}"
    assert np.abs(residuals).max() < 0.6, f"residuals {residuals}"
