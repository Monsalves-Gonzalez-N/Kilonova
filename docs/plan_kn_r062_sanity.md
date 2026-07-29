# Sanity check KN / banda R062 — diagnóstico y plan de ejecución

Diagnóstico verificado empíricamente el 2026-07-29 (sesión Fable). Este documento es el plan de
trabajo para la sesión de ejecución (Opus): los fixes van indicados en dirección, no en código;
los tests están diseñados para escribirse ANTES de los fixes y quedar como guardas permanentes.

## Diagnóstico

### Bug A — falta el factor 1/(1+z) en el flujo observado (afecta TODAS las bandas)

`spectra.py::redshift_and_dim_spectrum` y su copia `extinction.py` (~líneas 197–210) redshiftean
con specutils `shift_spectrum_to`, que **solo corre el eje espectral y no toca el flujo**, y luego
aplican únicamente el dimming geométrico `(10pc/d_L)²`. La física correcta para densidad de flujo
en longitud de onda es:

```
f_λ,obs(λ_obs) = f_λ,10pc(λ_obs/(1+z)) · (10pc/d_L)² / (1+z)
```

**Evidencia** (reproducida): integral bolométrica de un espectro plano a z=0.3 →
`E_obs/esperado = 1.3000` (exactamente 1+z); el cociente puntual de flujo confirma que specutils
no reescala.

**Impacto**: todas las KN sintéticas están (1+z) demasiado brillantes — 0.08 mag a z=0.08,
0.28 mag a z=0.3. Infla las detecciones (hoy deep 269 492 / wide 165 066) y el z_max
(0.2938/0.1689). Los contaminantes OU **no** se ven afectados (sus mags vienen de snana). La
copia en `extinction.py` contamina también toda la fotometría de extinción y lo derivado de ella.
Los notebooks `kilonova_dataloader.ipynb` y `Extinction.ipynb` usan la misma receta ("validated
in kilonova_dataloader.ipynb" en el docstring quedó invalidado: la validación no capturó este
factor).

### Bug B — flujo cero mal manejado (la fuente del "inf en R062")

Los espectros LANL son Monte Carlo y tienen bins con flujo 0 (medido: 12.9% de los espectros de
fase temprana tienen algún bin ≤0 dentro de la ventana rest de R062; cuando los hay, mediana 11%
de los bins). `magnitudes_for_bands` enmascara `flux > 0` ANTES de integrar, lo que produce tres
síntomas distintos según el caso:

1. **inf literal** (lo que reportó Opus): por el camino del dataloader
   (`spectrum_to_roman_magnitudes` directo, sin la máscara), galsim integra flujo 0 sobre la
   banda → `mag = +inf`. Reproducido: SED con ceros en el azul da `R062: inf`.
2. **NaN → banda descartada → artefacto de la banda ancla**: con la máscara, si los ceros son
   contiguos al borde azul, el chequeo de cobertura (min/max del array ENMASCARADO) falla → NaN →
   `kn_model_from_spectra` descarta la época → `build_window_from_model` marca `observed=False`.
   Es el artefacto conocido (0.037% filas deep, 1.51% wide, 0 en OU): una firma exclusiva de KN
   que el transformer puede explotar como atajo espurio.
3. **Flujo inventado por interpolación**: si queda un bin MC positivo espurio en el azul, la
   cobertura pasa y la interpolación lineal PUENTEA el gap de ceros. Reproducido: un solo bin de
   1e-12 en 1360 Å convierte `R062: NaN` en `R062: 35.5`. Explica la cola `mag_true` hasta 44.6
   en los parquets actuales.

**Física correcta**: flujo cero es un valor legítimo (cortina de lantánidos, vista edge-on). El
survey observa la banda igual: `flux_true = 0` → realización de solo ruido → no-detección (token
`u` con `mag_limit_5sigma`). **Nunca** `observed=False`.

### Menor C — `mag_err = 1.0857/snr` sin guarda

Con el fix B aparecerán `flux_true = 0` exactos → `snr = 0` → `mag_err = inf` en el parquet.
Los tokens no lo consumen (los `u` usan `mag_limit_5sigma` y los `d` exigen snr≥5), pero el
parquet debe quedar finito o NaN. Hoy no se dispara (snr mínimo ~1e-7), pero hay `mag_err` de
hasta 9.5e6 — sin sentido físico, mismo origen.

## Dirección de los fixes

- **A**: `distance_dimming = (10/d_L)² / (1+z)` en `spectra.py` Y en `extinction.py` (misma
  corrección en los dos notebooks). El camino z=0 (10 pc) queda sin factor.
- **B**: eliminar la máscara `flux > 0`; clip de negativos a 0; integrar con los ceros incluidos.
  Con el grid completo (1002–127 695 Å rest) la cobertura de banda nunca falla → el chequeo de
  cobertura deja de generar NaN. `kn_model_from_spectra` conserva épocas con mag no finita (o
  trabaja en flujo); en `build_window_from_model`, `observed` deja de exigir
  `isfinite(mag_true)`; con flux_true=0: `detected=False`, `mag_err=NaN`, `mag_observed` sale de
  la realización de ruido (el fallback `flux_observed <= 0 → mag_limit` ya existe).
- **C**: guarda `snr == 0 → mag_err = NaN`.
- **Decisión de representación** (tomar en la sesión de ejecución): cómo guardar `mag_true`
  cuando el flujo es 0 — (a) `+inf` permitido y documentado (recomendado: parquet lo soporta y
  los tokens nunca leen `mag_true`), o (b) NaN con `observed=True`. Los tests T5/T6 se escriben
  según la opción elegida.

## Tests a escribir (pytest, SEDs sintéticas, sin datos grandes)

- **T1 — Conservación bolométrica** (captura Bug A; hoy falla por un factor exactamente 1+z):
  espectro plano en f_λ, z ∈ {0.1, 0.3, 1.0}:
  `∫f_obs dλ_obs · d_L² == ∫f_rest dλ_rest · (10pc)²` a <0.1%. Mismo test para la copia de
  `extinction.py`.
- **T2 — Magnitud AB analítica end-to-end**: SED plana en f_ν (f_λ = A/λ², normalizada a M_AB en
  10 pc). Para f_ν plana el resultado es exacto y analítico en TODA banda y todo z:
  `m(z) = M + 5·log10(d_L/10pc) − 2.5·log10(1+z)`.
  El código actual da 2.5·log10(1+z) de más brillo → el test lo captura y además fija la
  convención para siempre (independiente de galsim).
- **T3 — Flujo cero en la banda**: SED con f=0 exacto para λ_rest < 9000 Å y positivo arriba, a
  z=0.05: R062 debe dar flux_true=0 (según la representación elegida) — no NaN, no 35.5. La
  ventana generada desde ese modelo debe tener R062 `observed=True`, `detected=False` (token `u`)
  y todas las columnas de ruido finitas o NaN.
- **T4 — El bin espurio no puentea**: la SED de T3 + un bin de 1e-12 en 1360 Å. Comparar EN
  FLUJO (no en magnitud): la diferencia de flujo integrado en R062 entre T4 y T3 debe ser ≤ la
  contribución analítica del bin (~nada), y ambas realizaciones deben salir `detected=False`.
  Hoy: NaN vs mag 35.5.
- **T5 — Invariante de cadencia/ancla** (mata el artefacto): para cualquier modelo KN con ≥1
  detección, toda (visita, banda) que `cadence_schedule` marca observada dentro de la ventana
  sale `observed=True`. Modelo sintético mínimo + assert.
- **T6 — Regresión smoke**: correr `kn-kilonova-windows --limit-kn 500` a un dir temporal y
  asertar sobre el parquet: sin ±inf en ninguna columna (salvo `mag_true` si se eligió la opción
  (a)), `mag_err` finito o NaN, `snr ≥ 0` finito, y cero filas de banda ancla no-observada dentro
  de la ventana.
- **T7 — Paridad OU/KN** (script de chequeo, no pytest): en los datasets regenerados, la fracción
  de épocas de banda ancla con `observed=False` dentro de la ventana debe ser 0 tanto en OU como
  en KN.

## Orden de ejecución sugerido

1. Commitear primero el diff pendiente (paralelización `--workers`, ya usado para generar los
   datasets del 2026-07-29) para que los fixes vayan sobre base limpia.
2. Escribir T1–T5 → confirmar que fallan como se predice (T1: factor 1.3; T2: −2.5log10(1+z);
   T3: NaN; T4: 35.5).
3. Fix A (spectra.py + extinction.py) → T1/T2 verdes.
4. Fix B + C → T3/T4/T5 verdes.
5. `pytest` + `ruff check` completos; T6 smoke.
6. Regenerar KN (`kn-kilonova-windows --workers 30`, ~73 min local). **Esperado**: detecciones y
   z_max BAJAN (hoy 26.9%/16.5%, z_max 0.294/0.169) — el resultado nulo a z>0.3 se refuerza.
   El artefacto de la banda ancla debe quedar en 0 filas (T7).
7. Actualizar números en memoria/docs y el docstring "validated in kilonova_dataloader.ipynb";
   revisar los dos notebooks.
8. Nota: el fix A invalida también la fotometría de `kn-extinguish` — lo generado con
   `extinction.py` necesitará regeneración cuando se retome esa línea (ver memoria
   dust_generation).
