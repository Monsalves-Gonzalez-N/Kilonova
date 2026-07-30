# CLAUDE.md — Kilonova

## Mission

Classify kilonovae among transients observed by the Roman Space Telescope (HLTDS): a supervised
transformer over early-window light-curve tokens, recall-first against a follow-up budget.
Training data is **OpenUniverse only** (contaminants) plus LANL-grid kilonovae injected with the
same noise recipe. Hourglass (Rose et al. 2025) is NOT a training stage anymore — it is only the
source of the photometric noise recipe. Full science plan: `docs/plan.md` (its two-stage
Hourglass sections are marked superseded). Model design and training happen on another machine;
this repository owns data simulation and datasets.

## Layout

- `src/kilonova/` — installable package (`pip install -e .`, conda env `kilonova`).
  - `photometry/roman_noise.py` — the noise recipe, single source of truth (CCD equation,
    Rose et al. 2025 eqs 8–10 + Table 3 NEA). galsim imported lazily.
  - `photometry/spectra.py` — LANL SED → Roman AB mags (redshift + dimming, no extinction).
    Dimming is `(10pc/d_L)^2 / (1+z)`: specutils only stretches the wavelength axis, so f_lambda
    carries the K-correction term itself. Magnitudes distinguish **NaN = band outside the spectral
    coverage** from **+inf = band covered but with no flux**, which is a real non-detection the
    survey observes (token `u`), never a gap — do not collapse the two. Pinned by
    `tests/test_spectra.py` against closed-form SEDs.
  - `simulation/` — `lanl_cache` (grid → parquet), `extinction` (host+MW extinction photometry,
    STOP rules), `early_windows` (OpenUniverse contaminants + KN injection windows).
    The raw `.dat` spectra hold the flux **per angular bin**; what an observer in that direction
    measures is the isotropic equivalent, `4pi/dOmega_bin` = `n_angles` = **×54, i.e. 4.331 mag**.
    `lanl_cache.isotropic_equivalent_flux` applies it when building the cache, so
    `lanl_spectra.parquet` already stores observable flux (schema metadata `flux_convention`) —
    **never apply it a second time downstream.** Pinned by `tests/test_lanl_cache.py` against the
    AB magnitudes LANL ships in the `*_mags_*` files. Omitting it made every kilonova 4.33 mag too
    faint and cut the redshift reach from z~1 to z~0.24.
  - `datasets/openuniverse.py` — long DataFrame → transformer tokens (d/u/n types, [Z] regime).
  - `cli/` — `kn-cache-lanl`, `kn-extinguish`, `kn-early-windows`, `kn-run-openuniverse`.
- `configs/paths.yaml` — ALL data locations; override with `KN_<FIELD>` env vars or CLI flags.
  Never hardcode paths in code or notebooks; use `kilonova.config.load_paths()`.
- `notebooks/` — import from the package, never `sys.path` hacks.
- Big data: DVC (`dvc pull`; remote on `/Volumes/Elements/dvc-kilonova`). OpenUniverse source
  hdf5s live on `/Volumes/T7/openuniverse2025`.
- `data/openuniverse/README.md` says which file of that directory is the live one. The rule there:
  **the clean name is current; any suffix after the extension** (`.stale-cadence`,
  `.buggy-photometry`, `.stale-<date>`) **is a superseded generation kept only for comparison** —
  never feed one to a training run. Rename rather than delete when superseding, and add the row.
- Compute split: the full `kn-run-openuniverse` (deep+wide, 33 fields) runs on the training
  laptop (too expensive here); its `early_windows_{deep,wide}.parquet` come back via DVC.
  `kn-cache-lanl` is historical — `lanl_spectra.parquet` was built once from the OLD raw grid
  (`kn_sim_cube_v1`); the cached parquet is the working source of truth. Since 2026-07-30 the local
  copy of that raw grid is **gone except one model** (the `md0.1_vd0.05_mw0.1_vw0.05` `_spec_`/
  `_mags_` pair `tests/test_lanl_cache.py` pins the flux normalization against, 54 MB of the 46 GB).
  Rebuilding the cache, or pinning any other model, needs the Elements drive mounted.

## Noise recipe (provenance)

Validated against Hourglass `fluxcal_err` row by row (`docs/hourglass_noise_recipe.md`,
`notebooks/validation/validate_noise_recipe.ipynb`):

```
sigma^2(F) = F_source + NEA * (B_sky + B_thermal + B_dark + sigma_read^2)      [electrons]
```

- Read noise: Rose et al. 2025 eq. 9, denominator n(n+1), floor 5 e- (up-the-ramp).
- PSF NEA from Hourglass Table 3 (galsim's analytic PSF overestimates blue bands 15–28%).
- Zeropoint jitter sigma = 0.15 mag (eq. 8); 5-sigma limits computed per visit from the actual
  flux error, never a fixed survey depth.

## Working rules

- `pytest` + `ruff check` must pass before committing; CI enforces both.
- Smoke-test pipeline changes with `kn-early-windows --limit-ou 5 --output-dir /tmp/smoke`.
- New pipeline outputs: add as DVC stage outs (dvc.yaml) or `dvc add` artifacts — never git.
- `data/openuniverse/download_openuniverse_snana.sh` documents how the snana inputs were fetched.
