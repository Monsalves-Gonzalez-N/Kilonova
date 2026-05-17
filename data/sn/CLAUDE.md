# Kilonova classification — mission context

## Goal
Build a contrastive learning pipeline that classifies KN vs nonKN on Roman Space Telescope NIR light curves. Train on synthetic KN injected into the Roman survey + Hourglass SN photometry.

## Current phase: validating that synthetic KN curves are comparable to Hourglass

Before any contrastive work, we need to confirm we can reproduce Hourglass-style observations from a model magnitude using only the noise recipe (no SNANA in the loop). The bar is: same SNANA noise model → same `fluxcal_err` per row, modulo host.

## Key data sources
- `/Volumes/Elements/kn_sn_bundle/sn/hourglass_photometry.parquet` — Roman SN photometry (Rose et al. 2025). Columns: `cid, mjd, band, phot_flag, fluxcal, fluxcal_err, psf_nea, sky_sig, read_noise, zp, zp_err, sim_mag_obs`. NO host or A_V columns exposed.
- `/Volumes/Elements/kn_sn_bundle/sn/hourglass_objects.parquet` — object-level (cid, class, z_cmb, peak_mjd, ...).
- `/Volumes/Elements/kn_sn_bundle/kn/kn_grid_v1/` — pre-computed KN apparent magnitudes from Bulla 2019 SED grid.
- `Hourglass_simulation.pdf` (Rose et al. 2025, ApJ 988:65) — paper. Sec 2.3, eqs 8-10, Table 3 = full noise recipe.

## Code organization
- `data/sn/hourglass_snana_zenodo/funciones.py` — `ContrastiveLCDataset` (SN side, reads Hourglass parquet) + encoder + NT-Xent.
- `data/kn/kn_dataset.py` — `KilonovaContrastiveDataset` (KN side, simulates from grid + Roman noise).
- `data/kn/kilonova_syntetic.py` — Bulla model parsing, AB photometry, `apply_roman_noise` (the noise recipe under validation).

## Noise recipe (matches Hourglass, Rose et al. eqs 8-10 + Table 3)

Aplicamos la identidad SNANA `flux_calibrated(m) = 10^((27.5 − m)/2.5)` a dos magnitudes (`sim_mag_obs` y `zp`):

```
F_sim                = fluxcal · 10^(0.4·(mag − sim_mag_obs))      [fcal]
flux_calibrated(zp)  = 10^((27.5 − zp)/2.5)                         [fcal/e⁻]

σ²_pred = flux_calibrated(zp) · F_sim                               ← Poisson de la fuente
        + nea · (sky_sig² + read_noise²) · flux_calibrated(zp)²     ← fondo (zodi+thermal+lectura)
```

- `F_sim`: flujo verdadero (sin ruido) en `fluxcal`, vía la diferencia `mag − sim_mag_obs`.
- `flux_calibrated(zp)`: el `fluxcal` que correspondería a una fuente de magnitud igual al `zp` de la fila. Puente entre cuentas físicas (paper) y `fluxcal` (parquet, ZP=27.5).

Hourglass `fluxcal_err` además incluye un término "galaxy" (host SB Poisson) sumado dentro, no desagregable.

### Origen de la fórmula
Es la **ecuación CCD estándar** (Howell 1989, eq. 14) en espacio de cuentas:
```
σ²(F_phys) = F_phys + n_pix · (σ²_sky + σ²_read)        [e⁻²]
```
con `n_pix = NEA` para fotometría PSF de fuente puntual (King 1983; Naylor 1998), trasladada a `[fcal²]` por propagación lineal.

**Asimetría de los dos términos** (uno solo `flux_calibrated(zp)`, el otro al cuadrado):
- Fuente: el `α` que multiplica a `F_sim` es el **factor de Fano** del proceso Poisson calibrado a fcal — `[fcal]`. No es cambio de unidades, es la varianza-por-evento del compound Poisson.
- Fondo: `sky²` y `read²` ya son varianzas en `[e⁻²/pix]` desde el principio (eqs. 9-10). El `α²` es propagación lineal pura `[e⁻²] → [fcal²]`.

### Referencias
- **Howell (1989)**, *PASP* 101, 616 — ecuación CCD estándar (eq. 14).
- **Howell**, *Handbook of CCD Astronomy* (Cambridge, 2ª ed. 2006) — caps. 4-5, derivación pedagógica.
- **King (1983)**, *PASP* 95, 163 — definición formal de NEA.
- **Naylor (1998)**, *MNRAS* 296, 339 — extracción óptima PSF.
- **Fano (1947)**, *Phys. Rev.* 72, 26 — factor de Fano.
- **Kessler et al. (2009)**, *PASP* 121, 1028 — SNANA, implementación del modelo de ruido.
- **Hounsell et al. (2018)**, *ApJ* 867, 23 — primer uso del modelo SNANA con valores Roman.

## Hourglass paper notes (Rose et al. 2025, ApJ 988:65)

### Survey design (Sec 2.2)
- Two tiers, 5-day cadence, 2-yr survey.
- **Wide tier**: 19.04 deg², bands `R, Z, Y, J`, exp times `160/100/100/100 s` (Table 2).
- **Deep tier**: 4.20 deg², bands `Y, J, H, F`, exp times `300/300/300/900 s` (Table 2).
- Each band belongs to ONE tier except Y/J (in both). The simlib row's `(zp, sky_sig, ...)` tells you which.
- Detection cut: ≥2 observations with S/N > 5 (object-level). Per-row data includes sub-detections.
- Slew + settle = 70 s assumed.
- Host galaxies from 3DHST (110,798 entries); used for SB Poisson noise + correlations. Not in parquet.
- **Three independent extinction/contamination effects** (don't conflate):
  1. **Milky Way extinction** — Fitzpatrick 1999 law, R_V=3.1, Schlafly & Finkbeiner 2011 dust map. Per-object E(B-V) **IS exposed** as `mw_ebv` column in `hourglass_objects.parquet`. Applies to ALL transients including SN Ia.
  2. **Host A_V (internal host extinction)** — Wood-Vasey+2007 eq. 2: Gaussian core σ=0.6 mag + exponential tail τ=1.7 mag. Despite the paper calling it "galactic line-of-sight", this is INSIDE the host galaxy, not the MW. SN Ia skip this (already in SALT2/population model). **NOT in parquet.**
  3. **Host SB Poisson noise** — galaxy surface brightness at the SN position contributes Poisson noise; sumado dentro de `fluxcal_err`. **NOT in parquet.**
- SNANA pipeline order: `SED → ×host A_V → redshift → ×MW A_λ → filter → sim_mag_obs → +noise(source+sky+read+host_SB)`.

### Noise model (Sec 2.3)
- **ZP per row**: `ZP_AVG = ZP + 2.5·log₁₀(t_exp)` + Gaussian scatter σ=0.15 mag (FOV-dependent). Both effects already absorbed into the parquet's `zp` column (eq 8).
- **Read noise**: `σ_read = √(σ²_floor + σ²·(n−1)/(n+1))`, n=t_exp/3.04, σ²_floor=25, σ²=12·16²=3072 (eq 9).
- **Sky noise**: `σ_sky = √(t_exp·(σ²_zodi + σ²_thermal))` — combines zodi + thermal (eq 10). Dark current ignored.
- **NEA**: per (band, tier), median of best/worst FOV (Table 3).
- Table 3 reference values (DEEP: F → NEA=16.335, σ_sky=17.723, ZP=33.299, σ_read=5.942) match `ROMAN_DEEP/WIDE` in `kilonova_syntetic.py` exactly.

### Transients simulated (Sec 2.1, Table 1)
SN Ia, SNIa-91bg, SN Iax, CCSN, SLSN-I, TDE, ILOT, **KN (Bulla 2019, z=0.03–1.5, 550 templates)**, PISN, AGN. Hourglass detected only ~3 KN total at S/N>5 — KN are very rare in Hourglass even with rate boosted ×5.

### KN-specific facts
- KN rate inflated 5× over R. Abbott+2021 to overcome Poisson noise (paper Sec 2.1.8).
- KN templates use Bulla 2019 model (rest-frame 100–99,900 Å, phase −2 to 20 d).
- KN at z=0.3 in Fig 5 is barely detectable: <1 epoch with measurement, fast decay, NIR-only.

### Data release format (Appendix, Table 6)
`hourglass_photometry.parquet`: `cid, mjd, band, phot_flag, fluxcal, fluxcal_err, psf_nea, sky_sig, read_noise, zp, zp_err, sim_mag_obs`. `fluxcal_err` is "Poisson uncertainty on fluxcal, sky+galaxy+source" — host is sumado, no desagregable.
- `mag = 27.5 − 2.5·log₁₀(fluxcal)` (ZP_SNANA convention).

## Decision: what we model vs omit for KN
- **MW extinction**: SHOULD be applied to KN if we want apples-to-apples (it's just a line-of-sight effect, agnostic to transient type). `mw_ebv` is available per object in Hourglass; for synthetic KN we'd need to sample from the same dust map distribution or assign a representative value. *(Currently not applied — TODO.)*
- **Host A_V (internal)**: NOT modeled for KN. Justification: KN host demographics are poorly characterized (only AT2017gfo as anchor); adopting Wood-Vasey σ=0.6/τ=1.7 from SN populations would be unphysical assumption.
- **Host SB Poisson noise**: NOT modeled for KN. Not desagregable from `fluxcal_err`, and same uncertainty about KN hosts applies.
- Argument for the paper: we evaluate KN vs SN both *without* host contamination/A_V (use only MW extinction), and note this as a known systematic. Synthetic KN errors will be slightly optimistic vs Hourglass SN; quantified in the validation tests below.

## Validation tests (incremental)

### Test 1 — Recipe reproduces Hourglass `fluxcal_err` per row
Notebook: `data/sn/hourglass_snana_zenodo/validate_noise_recipe.ipynb`.
- Apply recipe per-row to Hourglass photometry using its own `psf_nea, sky_sig, read_noise, zp`.
- Compare `σ_pred` vs `fluxcal_err`. Expected median ratio ≈ 0.986; p05 lower in F/H DEEP (host-dominated).
- Status: notebook created, **not yet run**.

### Test 2 — *(pending)*

## Working preferences
- Spanish in conversation, English in code.
- Short focused responses; do not commit until explicitly asked.
- When user describes a test goal, design the test first; write code only after user confirms approach.
- Read paths/files when given explicitly; do not explore the repo broadly.
