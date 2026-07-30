# Reproducción de las figuras de detectabilidad Roman/kilonova — plan de test

Imágenes de referencia (paper externo, no en el repo): `~/Downloads/kn_widefield/*.png`, 14
figuras. Objetivo: reconstruir el mismo tipo de figura con nuestra propia grilla LANL + receta de
ruido (`kilonova.photometry.roman_noise`, `kilonova.simulation.extinction`) y compararla
visualmente/cuantitativamente contra el original. Este documento es el plan — no hay código
todavía. Antes de implementar hay decisiones metodológicas abiertas (sección "Supuestos a
confirmar") que cambian el resultado y deben acordarse primero.

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
  `mass_wind`, `velocity_wind`, `angle_index`, `time_days`, espectro rest-frame
  (`lanl_cache.py`).
- Redshift + dimming + magnitudes AB por banda: `spectra.py` /
  `extinction.py::generate_observed_kilonova_spectrum`,
  `compute_roman_ab_magnitudes` — **usar la versión ya corregida** (factor `1/(1+z)`, flujo cero
  no enmascarado; ver `docs/plan_kn_r062_sanity.md`, fix aplicado en `5e11827`/`fa06e59`).
- Redshift grid builder: `extinction.py::build_redshift_grid`.
- Bandas Roman disponibles: `ALL_BANDS_BY_WAVELENGTH` en `roman_noise.py` — R062/Z087/Y106/J129/
  H158/F184/K213 (todas menos K213 aparecen en las figuras de referencia).
- Ruido + 5σ: `roman_noise.py` (zeropoint, NEA, fondo por banda, jitter de zeropoint 0.15 mag).
- STOP rules / definición de "detectado" ya usada en el pipeline KN:
  `extinction.py::_lc_stop_idx_perband` (S/N ≥ 5 con paciencia) y, en el pipeline final,
  `early_windows.py::build_window_from_model` (S/N ≥ 5 en la ventana de cadencia real).

## Supuestos a confirmar antes de implementar

Estas decisiones no están en las imágenes y determinan la fracción detectable — cambiarlas cambia
la figura. Marcar la elegida y por qué.

1. **Definición de "detectable" en un punto (t, z)**: ¿un solo epoch con S/N≥5 en esa banda en
   ese instante de tiempo observador? (más simple, es lo que el eje x/y sugiere: t es una
   coordenada continua, no una ventana de cadencia). O ¿al menos una detección dentro de la
   ventana real de cadencia Roman (wide/deep) que contiene ese t? La primera es coherente con
   cómo se lee el eje (tiempo observador continuo, sin discretizar en épocas de survey) y es la
   opción recomendada para esta reproducción — más simple y no depende de la cadencia HLTDS.
2. **Profundidad/exposición asumida**: la receta de ruido necesita un tiempo de exposición por
   banda. Usar el de `build_tier_constants("wide")` (Roman wide-field HLTDS, coherente con el
   nombre `kn_widefield` de la carpeta de imágenes) — a confirmar que el paper usa la misma
   estrategia de exposición wide y no una genérica de "survey depth".
3. **Marginalización de la grilla LANL**: para cada (t_obs, z) fijo, la "fracción detectable" es
   sobre qué variable — plausible: uniforme sobre `angle_index` (ángulo de vista) × todas las
   combinaciones de `mass_dynamical`/`velocity_dynamical`/`mass_wind`/`velocity_wind` presentes en
   la grilla, salvo en los paneles de apéndice donde una o más masas se fijan. Confirmar si es
   marginalización uniforme por punto de grilla (lo más simple, recomendado) o ponderada por
   algún prior físico.
4. **Extinción**: ¿incluir dust (host + MW, `extinction.py`) o magnitudes limpias? Las figuras no
   lo aclaran. Recomendado: sin extinción en la primera pasada (isolar la física del ruido/grilla
   LANL primero), y una segunda pasada con extinción si la primera no calza.
5. **Curva AT2017gfo**: requiere la traza observada real de AT2017gfo (magnitud vs tiempo por
   banda) redshifteada sintéticamente a cada z, no un punto de la grilla LANL — o el punto de la
   grilla más cercano en parámetros al fit de AT2017gfo (dyn+wind masses/velocities/ángulo
   publicados para ese evento). Confirmar cuál fuente de AT2017gfo usar; si no la tenemos
   disponible, omitir el overlay en la primera pasada y dejarlo como paso 2.
6. **Percentiles de contorno**: 0.05/0.5/0.95 sobre la misma distribución marginalizada del punto
   3 — directo una vez resuelto el punto 3.

## Pipeline propuesto (una vez resueltos los supuestos)

1. Notebook/script en `notebooks/validation/` (o `simulation/`), reusando funciones existentes,
   sin duplicar lógica de `extinction.py`/`roman_noise.py`.
2. Grilla de tiempo observador: log-spaced, mismo rango que las figuras (~0.25–40 d).
3. Grilla de redshift: `build_redshift_grid`, rango 0–1 (0–0.5 para el recorte "texto
   principal").
4. Para cada banda × t_obs × z: tomar todas las filas de grilla LANL válidas (según el corte del
   panel: completo, o fijando `mass_wind`/`mass_dynamical`), interpolar a `time_days` rest-frame
   = t_obs/(1+z), generar magnitud AB observada, aplicar ruido Roman wide, marcar detectado
   (S/N≥5), promediar sobre la marginalización → fracción detectable.
5. Plot: `imshow`/`pcolormesh` t_obs (log) × z, colormap `magma`/`inferno` (igual paleta que el
   original), contornos 0.05/0.5/0.95, eje derecho de distancia de luminosidad (`astropy.cosmology`
   con la misma cosmología que usa el resto del repo — verificar cuál).
6. Repetir por banda y por corte de grilla (9 paneles H-band + 5 paneles multi-banda = 14 figuras,
   igual a la carpeta de referencia).

## Comparación / criterio de éxito

- **Cualitativo (primer filtro)**: forma general del contorno 0.5 (pico en t~1-3 d, caída hacia z
  alto y hacia t grande) y ordenamiento entre bandas (H alcanza z más alto que R, consistente con
  K-correction/NIR) deben coincidir a ojo.
- **Cuantitativo**: comparar el z del contorno 0.5 en el pico (t~1-2 d) banda por banda contra el
  leído de la imagen de referencia (tabla manual, ±0.05 en z como tolerancia inicial).
- **Paneles de apéndice (mw/md fijos)**: verificar el ordenamiento monotónico correcto — más
  masa eyecta ⇒ contorno 0.5 alcanza z más alto (comparar 0.1 vs 0.01 vs 0.001 M☉, y total 0.2 vs
  0.002 M☉) — esto es más robusto para detectar un bug de escala que el valor absoluto.
- Si el z_max del contorno 0.05 keeps siendo sistemáticamente distinto (mismo signo en todas las
  bandas): sospechar la profundidad de exposición asumida (supuesto 2) antes que un bug de física.

## Próximo paso

Resolver los supuestos 1–5 (idealmente citando la sección del paper que los fija, si el PDF está
disponible) y recién ahí escribir el notebook. No implementar con supuestos sin confirmar — esta
figura es puramente de validación, un mismatch metodológico produciría una comparación inútil.
