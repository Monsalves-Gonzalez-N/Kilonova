# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This directory contains `romansims`, a library for projecting redshift completeness of Type Ia SNe and host galaxies from the Roman space telescope. The workflow builds a SNANA-compatible HOSTLIB (galaxy library) from real 3DHST photometric data combined with EAZYpy SED fitting.

## Environment

This code depends on `astropy`, `sncosmo`, `eazypy`, `seaborn`, and `scipy`. Use the conda environment defined in the parent repo (`environment.yml`).

## Data layout (relative to `romansims/`)

| Path | Contents |
|------|----------|
| `data/3dhst/` | 3DHST master catalog, photometric catalog, GALFIT morphology files (per CANDELS field), and Whitaker+2014 mass corrections |
| `data/eazypy/` | Per-field EAZYpy coefficient FITS files and the 13 spectral templates |
| `data/roman_filters/` | Roman WFI bandpass transmission curves (`.dat`), registered into `sncosmo` |

## Code architecture

### `romansims/catalogs.py` — core pipeline

Three main classes form a processing chain:

1. **`Catalog3DHST`** — loads and prepares the galaxy simulation catalog:
   - `prepare_simulation_catalog()` runs all steps in order (slow: >30 min due to SED photometry)
   - Steps: `append_galfit_params()` → `select_clean_galaxies()` → `append_photcat_params()` → `append_eazy_coefficients()` → `append_unique_sim_ids()` → `append_eazy_magnitudes()` → `apply_mass_corrections()` → `trim_simulation_catalog()`
   - State flags (`mass_corrected`, `subset_selected`, etc.) guard against re-running steps
   - `id_sim` = `id + ifield * 1e5` to ensure global uniqueness across CANDELS fields

2. **`SNANAHostLib`** — reads/writes SNANA HOSTLIB text files:
   - Can ingest a FITS table or an existing HOSTLIB
   - `mk_wgtmap_block()` creates the SN rate weight map using `ssnr_ah18_piecewise` (default) or `ssnr_ah18_smooth` from Andersen & Hjorth 2018
   - `write_hostlib()` and `write_wgtmap()` produce the final output files

3. **`CatalogBasedRedshiftSim`** — projects spectroscopic redshift completeness onto a host catalog:
   - `assign_snhost_prob()` → `pick_host_galaxies()` → `apply_specz_completeness_map()`

### `romansims/eazyseds.py` — SED simulation

- `EazySpecSim` reconstructs galaxy SEDs from 13 EAZYpy basis coefficients
- `register_roman_filters()` registers Roman bandpasses into the `sncosmo` registry (must be called before any magnitude integration)
- Paths default to `../data/` relative to the module; adjust if calling from a different working directory

## Typical usage pattern

```python
from romansims import catalogs

# Build catalog from scratch (slow)
cat = catalogs.Catalog3DHST()
cat.prepare_simulation_catalog()
cat.write_simulation_catalog_as_fits('roman_hostlib.fits')
cat.write_simulation_catalog_as_hostlib('roman_hostlib.txt')

# Or load a previously saved catalog
cat = catalogs.Catalog3DHST(load_simulation_catalog='roman_hostlib.fits')

# Redshift completeness projection
sim = catalogs.CatalogBasedRedshiftSim()
sim.read_galaxy_catalog('roman_hostlib.txt')
sim.assign_snhost_prob(snr_model='AH18PW')
sim.pick_host_galaxies(nsn=10000)
```

## Notebooks

The `romansims/romansims/` directory contains exploratory notebooks:
- `akari_snana_simcat_creation_from_3dhst.ipynb` — end-to-end HOSTLIB creation
- `roman_snIa_hostgal_specz_efficiencies.ipynb` — redshift completeness projections
- `akari_roman_wfi_tiling.ipynb` / `akari_imsim_diffim_example.ipynb` — tiling and image sims
