# Reproducción de las figuras de detectabilidad Roman/kilonova — plan de test

Imágenes de referencia (paper externo, no en el repo): `~/Downloads/kn_widefield/*.png`, 14
figuras. Fuente identificada: **Chase et al. 2021, "Kilonova Detectability with Wide-Field
Instruments"**, ApJ, arXiv:[2105.12268](https://arxiv.org/abs/2105.12268) — Fig. 3/5/7 + Apéndice
A. Objetivo: reconstruir el mismo tipo de figura con nuestra propia grilla LANL
(`lanl_spectra.parquet`) y compararla visual/cuantitativamente contra el original, siguiendo la
metodología **tal como está descrita en el paper** (Sección "Metodología del paper", más abajo),
no supuestos propios. Este documento es el plan — no hay código todavía.

**Bloqueo práctico a resolver antes de implementar**: reproducir el paper 1:1 requiere las 48 600
realizaciones (900 simulaciones × 54 ángulos) de la grilla LANL completa. Según
`CLAUDE.md`/memoria (`project_lanl_angular_factor.md`), el `.dat` crudo local está **borrado salvo
un modelo** (`md0.1_vd0.05_mw0.1_vw0.05`, pinneado por `tests/test_lanl_cache.py`) — el resto
necesita el disco Elements montado para reconstruir `lanl_spectra.parquet`. Confirmar que
`lanl_spectra.parquet` ya cacheado localmente cubre la grilla completa (900×54) antes de asumir que
no hace falta el disco; si no la cubre, este trabajo no puede arrancar sin montar Elements.
También relevante: el fix del **factor angular ×54 (isotropic equivalent, 4.331 mag)** en
`lanl_cache.py` (2026-07-30) es exactamente la física que describe el paper — *"All 900
multi-dimensional simulations are rendered in 54 viewing angles, each subtending an equal solid
angle of 4π/54 sr"* — buena señal de que la convención de flujo del parquet cacheado ya está
alineada con la que usa el paper.

## Qué muestra cada figura

Todas comparten estructura: mapa de calor **Fracción Detectable** (0–1) en el plano
**Tiempo observador (d, eje x, log)** × **Redshift (eje y)**, eje derecho secundario en
distancia de luminosidad [Gpc]. Tres curvas de contorno grises (percentiles 0.05 / 0.5 / 0.95 de
la distribución de fracción-detectable) marcan el borde de detectabilidad. Algunas incluyen una
curva magenta sólida `AT 2017gfo` — la traza observada del evento real, banda por banda.

| Archivo | Banda | Corte de grilla |
|---|---|---|
| `RomanR/Z/Y/J.png` | R062/Z087/Y106/J129 | grilla completa (marginalizada), con overlay AT2017gfo |
| `RomanH.png` | H158 | grilla completa, eje z recortado a 0.5 (versión "texto principal") |
| `RomanH_forappendix.png` | H158 | igual que arriba pero eje z a 1.0 y leyenda completa (versión "apéndice") |
| `RomanF.png` | F184 | grilla completa, overlay AT2017gfo |
| `RomanH_mw_0.1/0.01/0.001.png` | H158 | **m_wind fijo** (0.1/0.01/0.001 M☉), resto de la grilla marginalizado |
| `RomanH_md_0.1/0.01/0.001.png` | H158 | **m_dyn fijo** (0.1/0.01/0.001 M☉), resto marginalizado |
| `RomanH_mw_0.1_md_0.1.png` | H158 | masa eyecta total fija = 0.2 M☉ (mw=md=0.1, split equal) |
| `RomanH_mw_0.001_md_0.001.png` | H158 | masa eyecta total fija = 0.002 M☉ (mw=md=0.001) |

`mw`/`md` en los nombres de archivo = `m_wind` / `m_dyn`, **no** extinción (confirmado por el
título dentro de cada imagen, ej. `Roman/H-band: m_wind = 0.1 M_sun`). Mapean directo a las
columnas `mass_wind` / `mass_dynamical` de `lanl_spectra.parquet`
(`src/kilonova/simulation/lanl_cache.py:37-39`).

## Piezas del repo que ya cubren esto

- Grilla LANL cacheada: `lanl_spectra.parquet` — columnas `mass_dynamical`, `velocity_dynamical`,
  `mass_wind`, `velocity_wind`, `angle_index`, `time_days`, espectro rest-frame, ya en convención
  de flujo observable (`lanl_cache.py`, factor angular ×54 aplicado — ver bloqueo práctico arriba).
- Redshift + dimming + magnitudes AB por banda: `spectra.py` /
  `extinction.py::generate_observed_kilonova_spectrum`,
  `compute_roman_ab_magnitudes` — **usar la versión ya corregida** (factor `1/(1+z)`, flujo cero
  no enmascarado; ver `docs/plan_kn_r062_sanity.md`, fix aplicado en `5e11827`/`fa06e59`). Esto
  cubre la Ec. 1 del paper (K-correction + dimming por distancia de luminosidad); **no** usar
  `roman_noise.py`/STOP-rules para el criterio de detección de esta reproducción — ver punto 1/2
  abajo, el paper compara directo contra una magnitud límite tabulada, no contra un S/N derivado
  de nuestra receta de ruido.
- Redshift grid builder: `extinction.py::build_redshift_grid`.
- Bandas Roman disponibles: `ALL_BANDS_BY_WAVELENGTH` en `roman_noise.py` — R062/Z087/Y106/J129/
  H158/F184/K213 (todas menos K213 aparecen en las figuras de referencia).

## Metodología del paper (Chase et al. 2021, Sec. 3–4, Tabla 1/2, Fig. 3/5/7)

Reemplaza la sección "Supuestos a confirmar" de la versión anterior de este documento — son hechos
del paper, no decisiones nuestras. Las imágenes de referencia son su Fig. 3 (LSST/r y Roman/H),
Fig. 5 (Roman/H para masa dyn/wind fija) y Fig. 7 (Roman/H para masa eyecta total fija — la
carpeta trae más bandas que las que el paper muestra en el cuerpo del texto; el resto están en su
Apéndice A, *"Similar figures for all filters ... are available in Appendix A"*).

1. **Definición de "detectable" en un punto (t, z)**: instantánea, sin ventana de cadencia ni
   S/N — *"We define a kilonova as detectable in a given filter if it outshines the limiting
   magnitude of the filter, as listed in Table 1"*. Es decir: `mag_AB(t_obs, z, banda) < m_lim(banda)`
   punto a punto. Confirma la opción que yo había recomendado, pero el criterio real es **una
   comparación contra una magnitud límite fija tabulada**, no un cálculo de S/N con nuestra receta
   de ruido.
2. **Profundidad/exposición — diverge de lo que yo proponía.** El paper **no** deriva la
   profundidad de un modelo de ruido tipo `roman_noise.py`; usa una **magnitud límite 5σ fija por
   banda**, tomada de la literatura (Tabla 1, Roman: Hounsell et al. 2018 + Scolnic et al. 2018;
   exposición nominal 67 s, FoV 0.28 deg², cadencia HLTDS wide implícita en esa referencia):

   | Banda Roman | λ_eff (Å) | m_lim (AB, 5σ) |
   |---|---|---|
   | R | 6160 | 26.2 |
   | Z | 8720 | 25.7 |
   | Y | 10600 | 25.6 |
   | J | 12900 | 25.5 |
   | H | 15800 | 25.4 |
   | F | 18400 | 24.9 |

   Para reproducir el paper 1:1 hay que usar esta tabla fija, no recalcular con nuestra receta de
   ruido. **Recomendación**: correr las dos versiones — (a) con esta tabla fija (comparación
   directa contra las imágenes), y (b) sustituyendo por el 5σ que da `roman_noise.py` en el tier
   wide (chequeo independiente de si nuestra receta de ruido, validada contra Hourglass, concuerda
   con estos límites de la literatura — si no, es una comparación interesante por sí misma, no un
   bug).
3. **Marginalización de la grilla LANL — confirmado uniforme, sin pesos.** Grilla: 900
   simulaciones (5 masas dyn × 5 masas wind × 3 vel dyn × 3 vel wind × 2 morfologías de wind
   {esférica, "peanut"} × 2 composiciones de wind {Y_e=0.27, 0.37}; dyn siempre morfología
   toroidal, Y_e=0.04 fijo — Tabla 2) × 54 ángulos de vista equisólidos = 48 600 realizaciones. La
   "fracción detectable" es el promedio **uniforme** (sin ponderar) sobre esas 48 600
   combinaciones (o el subconjunto que aplique en cada panel). Paneles de masa fija (Fig. 5): fijan
   **una sola** masa (dyn o wind) a un valor, dejan el resto de la grilla libre → 180 simulaciones
   × 54 ángulos = 9720 realizaciones por panel — coincide exactamente con `RomanH_md_*`/`RomanH_mw_*`.
   Paneles de masa eyecta total fija (Fig. 7): fijan dyn+wind = total, dejan el resto libre → sólo
   hay **un** par (dyn, wind) de la grilla discreta que suma cada total (p.ej. 0.001+0.001=0.002,
   0.1+0.1=0.2) → 36 simulaciones × 54 ángulos = 1944 realizaciones — coincide con
   `RomanH_mw_0.001_md_0.001` (total 0.002) y `RomanH_mw_0.1_md_0.1` (total 0.2).
4. **Extinción: confirmado que NO se incluye.** El paper no aplica dust ni de host ni de Vía
   Láctea — la Ec. 1 del paper es sólo la integral de bandpass con K-correction cosmológica
   (Hogg et al. 2002; Blanton & Roberts 2003; Oke & Sandage 1968), sobre el espectro rest-frame
   LANL ya redshifteado y con dimming por distancia de luminosidad. Confirma la recomendación
   anterior: **sin extinción** en la reproducción.
5. **Curva AT2017gfo: son datos espectroscópicos reales del evento, no un punto de la grilla
   LANL.** *"Figure 3 includes detectability constraints for AT2017gfo-like kilonovae, computed
   from spectroscopic data (Chornock+17; Cowperthwaite+17; Nicholl+17; Pian+17; Shappee+17;
   Smartt+17)"*, compilados en **kilonova.space** (Guillochon et al. 2017). Se redshiftean estos
   espectros observados con la misma Ec. 1 y se aplica el mismo criterio de detectabilidad. **No
   hay espectros dentro de las primeras 12 h post-merger** — por eso todas las curvas magenta
   arrancan alrededor de t~0.7–1 d, nunca antes. Nuestra grilla LANL no puede sustituir esto: si se
   quiere el overlay hay que conseguir la fotometría/espectroscopía real de AT2017gfo
   (kilonova.space u otra fuente), no es un derivado de `lanl_spectra.parquet`.
6. **Contornos 0.05/0.5/0.95: confirmado**, son percentiles de la misma distribución marginalizada
   del punto 3 (Fig. 3 caption: *"contours indicate the fraction of 48,600 simulated kilonovae ...
   brighter than the limiting magnitude"*).
7. **Cosmología (dato nuevo, no estaba en la versión anterior):** ΛCDM plana, `H0=67.4`,
   `Ωm=0.315`, `ΩΛ=0.685` (Planck Collaboration 2020) — usar exactamente estos parámetros en
   `astropy.cosmology.FlatLambdaCDM`, no los defaults de otra convención.
8. **Rango temporal rest-frame (dato nuevo):** los espectros LANL no cubren antes de **0.125 d
   (3 h) post-merger** en rest-frame — sesga las bandas UV/azules a tiempos observador cortos y
   redshift bajo (mencionado explícitamente como limitación del paper, no un bug nuestro si
   aparece igual en la reproducción).

## Pipeline propuesto

1. Notebook/script en `notebooks/validation/`, reusando `spectra.py`/`extinction.py` para
   redshift+dimming+magnitud, sin duplicar esa lógica y sin pasar por `roman_noise.py` en la
   corrida principal (punto 2: el paper usa `m_lim` tabulado, no S/N).
2. Grilla de tiempo observador: log-spaced, mismo rango que las figuras (~0.25–40 d).
3. Grilla de redshift: `build_redshift_grid`, rango 0–1 (0–0.5 para el recorte "texto
   principal" de `RomanH.png`).
4. Para cada banda × t_obs × z: tomar las filas de grilla LANL que apliquen al panel (completo,
   una masa fija, o el único par dyn/wind que da la masa total fija — ver punto 3 de la
   metodología), interpolar a `time_days` rest-frame = t_obs/(1+z), generar magnitud AB observada
   (K-correction + dimming, sin extinción), marcar detectado si `mag < m_lim(banda)` (tabla del
   punto 2), promediar uniforme sobre simulación × ángulo → fracción detectable.
5. Plot: `pcolormesh` t_obs (log) × z, colormap `magma`/`inferno` (igual paleta que el original),
   contornos 0.05/0.5/0.95, eje derecho de distancia de luminosidad con
   `astropy.cosmology.FlatLambdaCDM(H0=67.4, Om0=0.315)` (punto 7).
6. Overlay AT2017gfo (opcional, requiere conseguir la fotometría real del evento — punto 5; si no
   se consigue, omitir esa curva y comparar solo el mapa de calor + contornos).
7. Repetir por banda y por corte de grilla (9 paneles H-band + 5 paneles multi-banda = 14 figuras,
   igual a la carpeta de referencia). Correr además la variante con `roman_noise.py`/5σ en vez de
   la tabla fija (punto 2b) como chequeo cruzado, no como sustituto de la comparación principal.

## Comparación / criterio de éxito

- **Cualitativo (primer filtro)**: forma general del contorno 0.5 (pico en t~1-3 d, caída hacia z
  alto y hacia t grande) y ordenamiento entre bandas (H alcanza z más alto que R, consistente con
  K-correction/NIR) deben coincidir a ojo.
- **Cuantitativo**: el paper da el número exacto a igualar — `z50%(Roman/H) = 0.22`,
  `z50%(LSST/r) = 0.12` (Tabla 1/Fig. 3 del paper); también `z95%=0.10`, `z5%=0.79` para Roman/H.
  Tolerancia inicial ±0.02 en z (más estricta que la anterior porque ahora el objetivo es un valor
  publicado, no una lectura visual).
- **Paneles de masa fija (Fig. 5/7)**: el paper da los valores exactos — dyn=0.001M☉ → z50%=0.16;
  dyn=0.1M☉ → z50%=0.31 (pico a 3 d); wind=0.1M☉ → z50%=0.37; total=0.002M☉ vs 0.2M☉ con
  variabilidad `v=1.6` (Ec. 2 del paper) entre ambos extremos. Usar estos números, no solo el
  ordenamiento monotónico, ya que el paper los publica.
- Si el z_max del contorno 0.05 queda sistemáticamente desplazado en todas las bandas: sospechar
  primero la tabla de `m_lim` (punto 2) o la cosmología (punto 7) antes que un bug de física en
  `spectra.py`/`extinction.py`.

## Próximo paso

Confirmar que `lanl_spectra.parquet` local cubre la grilla 900×54 completa (bloqueo práctico,
arriba). Si sí, escribir el notebook siguiendo el pipeline. Si no, decidir si se monta Elements
para reconstruirlo o si la reproducción se limita a los cortes de grilla que sí estén cacheados.
