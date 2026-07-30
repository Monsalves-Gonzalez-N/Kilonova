# `data/openuniverse/` — qué archivo es cuál

Estado a **2026-07-30**. Este directorio acumula varias generaciones de los mismos datos, así que
la regla es: **el nombre limpio es el bueno; cualquier sufijo después de la extensión es una
generación superada que se conserva solo para comparar.** Nada con sufijo debe entrar a un
entrenamiento.

## Vigente

| archivo | qué es | generado por |
|---|---|---|
| `early_windows_{deep,wide}.parquet` | contaminantes OpenUniverse (SN, TDE, SLSN, PISN) | `kn-run-openuniverse`, 2026-07-28 |
| `kn_windows_{deep,wide}.parquet` | kilonovas LANL inyectadas | `kn-kilonova-windows`, 2026-07-30 |

Las magnitudes de los contaminantes vienen del snana de OpenUniverse, que ya trae la
K-corrección y todo lo demás incorporado; este pipeline solo les aplica la receta de ruido y la
ventana. Por eso **ningún fix de la fotometría sintética los toca** — esa fotometría existe
únicamente en el camino de las KN, que parten de espectros LANL en reposo.

## Superado — **borrado el 2026-07-30** (30 GB)

La regla del sufijo sigue en pie para lo que venga, pero de estas generaciones ya no queda copia
local: sus resultados están en las tablas de más abajo, que es lo único que hacía falta conservar.

| archivo borrado | por qué quedó obsoleto |
|---|---|
| `early_windows_{deep,wide}.parquet.stale-cadence` | cadencia incorrecta; corregido en la corrida del 2026-07-28 |
| `kn_windows_{deep,wide}.parquet.buggy-photometry` | los tres bugs de fotometría de abajo (2026-07-29) |
| `kn_windows_{deep,wide}.parquet.missing-angular-factor` | esos tres corregidos, pero sin el factor angular 54 (2026-07-30) |
| `openuniverse_tokens{,_test}.npz.stale-*` | caché de tokens anterior a las regeneraciones |
| `normalization.json.stale-2026-07-24` | normalización de magnitud ajustada sobre los tokens viejos |
| `kilonova_windows_{deep,wide}.hdf5` | clase KN de junio, con los tres bugs y sin factor angular |

Los `.npz` eran **caché derivada**, no fuente: `training/openuniverse_data.py` los reconstruye
cuando el fichero no existe, y `normalization.json` se reajusta en el mismo paso. Los `.hdf5` sí
eran fuente para el training y estaban en DVC: sus `.dvc` **siguen commiteados**, así que un
`dvc pull` los recupera desde el remote del Elements (Mac) — desde esta máquina el remote no es
alcanzable, así que si esa copia no existe, no existen.

Fuera de este directorio se borraron en la misma pasada
`data/dust_generation/lanl_extinguished_photometry.parquet` (17 GB) y
`lanl_extinguished_spectra_test.parquet`, obsoletos por el bug del `1/(1+z)` **y** por el factor
angular (hay que rehacerlos con `kn-extinguish`), y `lanl_spectra.parquet.per-angular-bin` (11 GB),
la versión previa del caché: reproducible con `kn-cache-lanl` sobre `kn_sim_cube_v1` desde un
checkout anterior a `c696ed0`.

También se borró la copia local de la **grilla cruda** `kn_sim_cube_v1`: el tarball (12 GB) y 2698
de los 2700 `.dat` (46 GB). Se conservaron a propósito los dos ficheros del modelo
`md0.1_vd0.05_mw0.1_vw0.05` (`_spec_` + `_mags_`, 54 MB) porque son los que lee el test que fija el
**valor** del factor 54; sin ellos ese test haría skip y nada en esta máquina cazaría una
normalización equivocada. La grilla completa sigue en el Elements
(`configs/paths.yaml: lanl_grid_dir`), que es lo que hace recuperable todo esto.

## El bug grande: el factor angular 54 (4.331 mag)

Detectado el 2026-07-30 comparando contra **Chase et al. 2021 (arXiv:2105.12268)**, que usa esta
misma grilla (900 simulaciones × 54 ángulos) y alcanza z~1 con Roman donde nosotros nos
quedábamos en z=0.24.

Los ficheros `_spec_` de LANL guardan el flujo **por bin angular**, no el equivalente isotrópico.
Un observador situado en el bin *k* ve una fuente cuya luminosidad aparente es la del bin
repartida por toda la esfera: el factor es `4π/ΔΩ_bin`, y con bins uniformes en cos θ eso es
`n_angles` = **54 = 4.331 mag**. Sin él, toda kilonova sintética salía 4.33 mag demasiado débil.

Cómo se verificó (la grilla cruda trae `_lums_` y `_mags_`, magnitudes publicadas por LANL):

1. El espectro integrado bolométricamente da `L_bol/(4π(10pc)²)` del `_lums_` **exactamente**
   (razón 1.0000 en todas las fases) → los espectros están a 10 pc y los leíamos bien.
2. Los bloques de banda de `_lums_` son `L_ν` y encajan con `_mags_` vía `f_ν = L_ν/(4π(10pc)²)`.
3. Pero `νL_ν` de una sola banda supera el bolométrico ~37×, imposible → los dos bloques están en
   normalizaciones distintas.
4. Nuestra magnitud vs la publicada, misma fase y mismo ángulo: desfase **4.354 ± 0.017 mag
   constante en los 54 ángulos** (correlación con el índice angular −0.02). Multiplicando por 54
   el residuo se centra en cero (mediana +0.007, rms 0.16) y lo que queda es la diferencia de
   filtro LSST/2MASS vs Roman.

El factor se aplica ahora en `lanl_cache.isotropic_equivalent_flux`, así que
`lanl_spectra.parquet` **ya guarda flujo observable** (metadata `flux_convention`) y no hay que
aplicarlo otra vez aguas abajo. Fijado por `tests/test_lanl_cache.py`.

### Efecto medido: sin factor → con factor (2026-07-30)

Corrida de `kn-kilonova-windows` de 10:30 a 11:55, grilla de `params.yaml` (100 nodos log en
z ∈ [0.02, 1.0] × 10 000 realizaciones = 1e6), 30 workers.

| | deep | wide |
|---|---|---|
| KN con ≥1 detección / inyectadas | 155 268 → **721 776** / 1e6 | 95 650 → **590 917** / 1e6 |
| z máximo con detección | 0.2411 → **1.000** (borde de la grilla) | 0.1561 → **1.000** |
| z50% (mitad de la grilla detectada) | — → **0.387** | — → **0.241** |
| z5% | — → **>1.0** | — → **0.729** |
| filas | — → 14 435 510 | — → 11 818 340 |

La grilla ahora se satura por arriba: queda 7.0% (deep) y 0.9% (wide) de detección en el último
nodo, z=1. Si se quiere el z5% de deep hay que extender `redshift_max`.

Chequeos de sanidad sobre los ficheros nuevos: **0** filas de banda ancla R062 con
`observed=False` en ambos tiers, **0** `NaN` en `mag_true`, y los `+inf` presentes (42 deep,
3 246 wide) son no-detecciones reales según la convención de abajo.

Contra Chase et al. (z50%=0.29, z5%=0.96, métrica más laxa): **wide** queda por debajo en ambos
percentiles, como se esperaba. **deep** los supera —su z5% pasa de 1.0— coherente con que el
`m_lim` único de Chase se parece más al tier wide, pero hay que decirlo al citar la comparación.

**Al comparar con Chase et al., cuidado con la métrica**: ellos definen detectable como *supera
`m_lim` en el mejor epoch de la curva*, sin cadencia ni ruido, y su Tabla 1 para Roman/R da
z50%=0.29, z95%=0.10 y **z5%=0.96** — el "z~1" del abstract es el z5%, el 5% más brillante de los
modelos, no un horizonte típico. Nuestro pipeline exige SNR≥5 en una visita real con la cadencia
del survey, así que debe quedar por debajo de su z5%.

## Los tres bugs de fotometría corregidos (commits `fa06e59`, `5e11827`)

Diagnóstico completo en `docs/plan_kn_r062_sanity.md`. En corto:

1. **Faltaba el `1/(1+z)`** en `redshift_and_dim_spectrum`: specutils solo estira el eje espectral,
   así que `f_lambda` tiene que cargar el factor. Todas las KN salían `2.5·log10(1+z)` demasiado
   brillantes.
2. **Máscara `flux > 0` antes de integrar**: dejaba huecos que la interpolación lineal cruzaba con
   una recta, inventando flujo en bandas donde la KN está oscura; y cuando no sobrevivía ningún
   bin, la banda fallaba el chequeo de cobertura y la época quedaba `observed=False`. Esto último
   era **exclusivo de las KN** (48 648 filas de banda ancla en wide, 0 en OpenUniverse): un atajo
   correlacionado con la etiqueta metido en los datos de entrenamiento.
3. **Flujo de banda negativo** (~1e-35, ruido de cuadratura sobre una integral nula) → `log10` de
   un negativo → NaN indistinguible de "banda no cubierta".

Convención que quedó fijada y hay que respetar: **`NaN` = banda fuera de la cobertura espectral;
`+inf` = banda cubierta sin flujo**, que es una medición real (no-detección observada, token `u`),
nunca un hueco.

### Efecto medido (`.buggy-photometry` → `.missing-angular-factor`)

Esta tabla compara las dos generaciones **anteriores** entre sí: aísla el efecto de los tres bugs
con el factor angular todavía ausente en ambas. Los números vigentes son los de la sección del
factor 54.

| | deep | wide |
|---|---|---|
| objetos con ≥1 detección | 269 492 → **261 569** | 165 066 → **160 405** |
| z máximo con detección | 0.2938 → **0.2411** | 0.1689 → **0.1561** |
| filas de banda ancla `observed=False` | 1 983 → **0** | 48 648 → **0** |

Verificado cruzando viejo↔nuevo por `(object_id, epoch, band)`: el cambio de `mag_true` es
exactamente `2.5·log10(1+z)` en el 98.98% de las filas de deep y el 91.05% de wide (residuo
mediano ~1e-16, exacto en punto flotante). El resto es el bug 2, concentrado en R062 —el 77% de
las filas desviadas en wide— y en el 90% de los casos la magnitud nueva es **más débil**, que es
la dirección correcta al quitar el flujo inventado.

## Pendiente conocido

`object_id` no es estrictamente único en `kn_windows_*.parquet`: la clave se construye como
`sim_angle_?_z` con los valores redondeados a 4 decimales, y en la corrida del 2026-07-30 hay 3
objetos de deep con 40 filas en vez de 20 (dos realizaciones distintas colisionando en la misma
cadena). Son 3 de 721 776, pero cualquier `groupby('object_id')` aguas abajo fusiona dos curvas de
luz en una. (Aparte hay objetos con menos de 20 filas: ventana más corta, eso es legítimo.)

**`training/openuniverse_data.py` se queda sin clase KN**: consume los
`kilonova_windows_{deep,wide}.hdf5` que se borraron el 2026-07-30, y **no** lee
`kn_windows_*.parquet`. Hasta que exista un conversor parquet→hdf5 o el dataloader aprenda a leer
el parquet, el training no arranca en esta máquina — que es preferible a que arranque en silencio
sobre las KN con los tres bugs, como venía pasando.

`dvc add` del nuevo `lanl_spectra.parquet`: el `.dvc` commiteado sigue apuntando al hash de la
versión por bin angular. Pendiente de hacer en el Mac, que es donde el remote (`/Volumes/Elements`)
es alcanzable.

`src/kilonova/simulation/extinction.py` también llevaba el bug 1 y ya está corregido, pero nada de
lo derivado de `kn-extinguish` se ha regenerado y sus dos salidas están borradas. La etapa
`extinguish` de `dvc.yaml` lo detecta sola porque `lanl_spectra.parquet` es una de sus `deps`.
