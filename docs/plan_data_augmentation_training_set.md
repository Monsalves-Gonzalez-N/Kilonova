# Data augmentation desde el training set — plan

Escrito el 2026-09-18. Rama: `izc-low-redshift-contaminants`. Entorno: `kn_class`
(`~/.local/share/mamba/envs/kn_class` — mamba, no aparece en `conda env list`).

---

## Qué es el método, en una frase

Se toma un evento **ya generado** del training set, se usa **su** plantilla y **su** redshift, se
calcula la fase, y de ahí sale el zero point fotométrico de la SED — que es lo que permite moverla
a otro redshift con magnitudes consistentes.

La fase **no se busca, se calcula**:

```
phase0     = (mjd_epoca0 − peak_mjd)/(1+z) + C[plantilla]
zero_point = mediana(mag_true − modelo(banda, phase))
```

`C` son 44 números, uno por plantilla: la fase, en unidades de la plantilla, donde cae el
`peak_mjd` de OpenUniverse. **Están congelados en `data/openuniverse/template_phase_anchor.csv`** y
los lee `kilonova.simulation.window_anchor`; no se recalculan por objeto. Casi todos salen del
borde del modelo de OU, más un desplazamiento medido:

```
C = minphase(plantilla) − (mjd_inicio_modelo − peak_mjd)/(1+z) + delta
```

El `delta` es **+0.20 d** en casi todas las plantillas, y se midió contra el brillo que el izc lee
del pico de la curva completa del padre. **No es medio paso de grilla**, que es lo que decía este
plan antes de la medición del 2026-09-21; ver la sección de esa fecha, más abajo.

**Dos plantillas no siguen la regla** y por eso la tabla manda sobre la fórmula: `SN2011bm`
(−1.70 d) y `SN1987A` (+45.15 d), cuyo modelo OpenUniverse trunca — su plantilla empieza en
−78.95 d y la ventana del survey la corta antes, así que el borde del modelo no es la plantilla.

## Por qué así y no ajustando colores

| | ajuste por colores | fase analítica |
|---|---|---|
| objetos que el ajuste resolvía (n=500) | 0.0217 mag (p90 0.063) | **0.0161** (p90 0.048) |
| objetos que el ajuste **rielaba** (n=117) | 0.0417 (p90 0.099) | **0.0311** (p90 0.073) |
| coste | grilla de 0.25 d + minimización | 12 interpolaciones |

Medido fuera de muestra: `C` calibrado en una mitad, evaluado en la otra. Y sin `C`, o sea
`(mjd0 − peak_mjd)/(1+z)` a secas, el spread es **0.72 mag** — `C` es el 97% del método.

El ajuste por colores tenía un modo de falla **silencioso**: la minimización se pegaba al borde de
la grilla renderizada y devolvía una fase plausible con un zero point malo (28% de los objetos a
z>0.5). La fórmula no puede rielar, y cuando la fase cae fuera de la plantilla se sabe *antes* de
renderizar comparando dos números.

## Presupuesto de error, sobre los buenos

| | mediana | p90 |
|---|---|---|
| spread total del zero point | 0.0204 | 0.0566 |
| entre bandas, misma época → fase fija = **integración** | 0.0070 | 0.0177 |
| entre épocas, misma banda → **fase** | 0.0101 | 0.0385 |

Contra la verdad de OU, el error de fase es **0.02–0.25 d** por plantilla (mediana ~0.10 d), y es
plano en redshift. La atribución no es retórica: el residuo de fase correlaciona 0.58 con el
término entre épocas y 0.21 con el de bandas.

Ninguno de los dos términos domina. El de integración es `galsim.roman` contra el kcor de SNANA.

## Qué ya está hecho

1. **Columna `mjd`** en `early_windows.py:196`. No se deriva de `days_since_detection`: la cadencia
   salta visitas y el cero se corre con la primera detección. Test:
   `test_window_carries_the_absolute_mjd`.
2. **`cc_templates.npz` regenerado** con el ancla de fase en el borde real del archivo
   (`regular_phase_grid`, antes `np.ceil`). Recuperados **0.605 d de mediana en 41 de 44
   plantillas**. Test: `test_the_phase_grid_is_anchored_at_the_file_edge`.
3. **La curva de luz corre sobre toda la plantilla** (`template_phase_grid`), no sobre
   `peak_phase + (−20,+70)`. Esa ventana sobrevive sólo en `rendered_band_curves`, que se aparea
   con `openuniverse_parents.PEAK_PHASE_LIMITS` y no puede cambiar sola. Test:
   `test_the_light_curve_spans_the_whole_template`.
4. **Cuarentena** en `data/openuniverse/_old_izc/` con su propio README.
5. **Header de `download_openuniverse_snana.sh`** corregido y `TARGET_DIR` apuntando a Dropbox.
6. **`kn-run-openuniverse` paralelizado** (`--workers N`) y con **shard por campo** en
   `.early_windows_shards/`. El shard es el checkpoint: un campo ya escrito se salta, asi que
   relanzar reanuda. El tier se cose shard a shard con `pq.ParquetWriter`, lo que quita el pico de
   13 GB de tener el tier entero en RAM. Verificado bit-identico contra la version en serie y contra
   la implementacion anterior, en los dos tiers; el ruido se siembra con `int(object_id)` por objeto,
   no de un stream compartido, que es lo que lo hace seguro. Tests:
   `tests/test_run_all_openuniverse.py`. Smoke test 22 s -> 3 s.

Estado: **98 passed, 1 skipped**, ruff limpio.

## Lo primero al abrir la sesión

Comprobar que la regeneración de los parquets terminó:

```
tail -5 data/openuniverse/early_windows_regen_2026-09-18.log
ls -la data/openuniverse/early_windows_{deep,wide}.parquet
```

Relanzada en paralelo a las 13:26 del 2026-09-18 con 16 workers (~13 GB de pico, 16 cores de 32).
Si murió, **no hay que empezar de cero**: los shards de `.early_windows_shards/` son el checkpoint y
el mismo comando reanuda donde quedó.

```
nohup ~/.local/share/mamba/envs/kn_class/bin/kn-run-openuniverse \
  --source-dir ~/Dropbox/Kilonova/openuniverse2025 --output-dir data/openuniverse --workers 16 \
  > data/openuniverse/early_windows_regen_2026-09-18.log 2>&1 &
```

Los parquets anteriores, sin la columna `mjd`, están en `_old_izc/early_windows_*.parquet.no-mjd`.

## Hecho el 2026-09-21: la medicion limpia, la tabla y el port

### 1. La medicion, rehecha con los artefactos nuevos

`scripts/probes_data_augmentation/anchor_measurement.py`. **La verdad ya no es un ajuste por
colores**: es el brillo que `measure_brightness_offset` lee del PICO de la curva completa del padre
en el hdf5, una cantidad que no depende de la fase. Por objeto se barre un `delta` alrededor de C y
se miran dos criterios -- el que anula el sesgo contra ese brillo y el que minimiza el spread del
zero point, que no necesita nada externo. **Eligen el mismo `delta` en el 93% de las plantillas**,
que es lo que permite congelar C sin arrastrar el hdf5.

Sobre 1320 objetos, 44 plantillas, z 0.3-2.0:

| | delta = 0 | delta ajustado |
|---|---|---|
| sesgo del zero point | -0.0106 mag | **+0.0009** |
| \|residuo\| p50 / p90 | 0.0138 / 0.0743 | **0.0023** / 0.0832 |
| spread p50 | 0.0592 | **0.0184** |

**El `delta` no es medio paso de grilla**: es +0.20 d en casi todas las plantillas. Lo que decia
este plan (+0.35 d de sesgo, corregir restando medio paso) se midio con los artefactos viejos y no
sobrevivio.

**La pregunta del 25%, respondida.** El `ceil` movia el borde en 41 de 44 plantillas, mediana
+0.605 d. De los 1260 objetos con C fiable, el **15.4%** tiene su primera epoca dentro de esa
rebanada, y rinden **igual** que el resto: \|residuo\| p50 0.0020 y p90 0.0071 en los dos grupos.
Se recupera. Solo el **2.46%** cae por debajo del borde nuevo, y de esos se sabe antes de
renderizar. La calidad es plana en redshift (p50 0.0018-0.0030 de z=0.3 a z=2).

Dos plantillas se delataron solas por su residuo y necesitaron barrido propio:

* `SN2011bm`: el barrido de +-1.5 d rielaba; con ventana ancha, **delta = -1.70 d**, residuo 0.003.
* `SN1987A`, la excepcion que este plan anticipaba: **delta = +45.15 d**, residuo 0.001.
  OpenUniverse trunca su modelo, asi que el borde del hdf5 no es el de la plantilla -- pero el
  corte es constante, no objeto a objeto, y una vez medido esa plantilla no tiene nada de especial.

Una trampa del barrido, por si alguien lo repite: un `delta` que empuja la fase fuera de la
plantilla deja de medir en casi todos los objetos, y con tres que queden el criterio del sesgo
encuentra un minimo espurio. Hay que exigir que siga midiendo la mitad de la muestra.

### 2. La tabla, congelada

`data/openuniverse/template_phase_anchor.csv`, 44 plantillas, versionada en git (12 KB, no DVC), la
escribe `scripts/probes_data_augmentation/freeze_C_table.py`. Lleva `C`, `C_edge`, `delta`, los
bordes de fase, cuantos objetos la midieron y con que residuo. Residuo contra el brillo del izc:
**p50 0.0020, p90 0.0046, maximo 0.0117** (`SN1999em`).

### 3. El anclaje, en el repo

`src/kilonova/simulation/window_anchor.py`, con `tests/test_window_anchor.py` (8 tests).

* `phase_anchors()` -- la tabla congelada, cacheada.
* `window_phases(mjd, peak_mjd, z, anchor)` -- la formula, tres restas y una division.
* `uncovered_phases(phases, anchor)` -- **el chequeo de cobertura, POR DELANTE**: dice que fases
  sobran antes de renderizar nada. Era el defecto del ajuste por colores, que devolvia un zero
  point malo en silencio.
* `brightness_from_window(...)` -- el zero point, misma convencion que `measure_brightness_offset`.
* `anchor_window(window, parent, z, ...)` -- todo junto: devuelve la realizacion lista para
  `build_izc_windows`, o **None** si la plantilla no cubre las fases o quedan menos de 6 medidas.

De paso, `intermediate_z_contaminants.band_magnitudes_at_phases` se separo de
`rendered_band_curves`: el anclaje lee el modelo en cuatro fases, y renderizar las 91 de
`REST_FRAME_PHASES` para usar 4 era 20 veces las integraciones. `source_of` se separo de
`build_model` por lo mismo: el chequeo de cobertura quiere los bordes de la plantilla, no el modelo.

Verificado contra la sonda sobre 8 objetos reales de `early_windows_deep.parquet`, z 1.0-1.6: los
mismos numeros, y el zero point coincide con el del hdf5 en 0.0061 mag el peor. Estado: **108
passed, 1 skipped** (los 8 del anclaje incluidos), ruff limpio.

## Pendiente

### 4. Regenerar tokens

`openuniverse_tokens.npz` y `openuniverse_tokens_test.npz` se construyen sobre los
`early_windows_*`, asi que quedan desactualizados. `training/openuniverse_data.py` los reconstruye
solo si el fichero no existe, y `normalization.json` se reajusta en el mismo paso.

### 5. Usar el anclaje para generar la muestra

`window_anchor` existe y esta medido, pero todavia no lo llama nadie del pipeline. Falta decidir
cuantas re-renderizaciones por objeto y a que redshifts, y si el izc pasa a leer el brillo de aqui
en vez del hdf5 -- que es lo que quitaria la dependencia de los 16 GB.

## Decisiones ya tomadas, no reabrir

- **Sin corte en redshift.** Medido: cortar en z>1 no mejora la calidad (spread 0.0204 → 0.0209),
  sólo el rendimiento, y a z>1.5 se pierden plantillas enteras (43 → 39). El óptimo por objeto está
  en z 1.0–1.25, donde se cruzan el término de integración (crece con z) y el de fase (baja).
- **El brillo sale del padre, no de parámetros.** `SNANA_INPUT_CONFIGS.tar` no está en el record de
  Zenodo, así que WGT/MAGOFF/MAGSMEAR y el `GENMAG_SMEAR_MODEL` de las Ia no existen de este lado.
- **Las core-collapse van sin extinción de host.** Es un bug de OU que el pipeline reproduce a
  propósito.

## Abierto, opcional

¿El término de integración (0.007 mag) es artefacto de `galsim.roman` contra el kcor de SNANA?
Se decide comparando contra `data/kcor/kcor_ROMAN.fits`. Si lo es, se puede borrar
`BRIGHTNESS_REST_WAVELENGTH_LIMITS`, que hoy es un guard empírico.
