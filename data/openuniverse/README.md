# `data/openuniverse/` — qué archivo es cuál

Estado a **2026-07-30**. Este directorio acumula varias generaciones de los mismos datos, así que
la regla es: **el nombre limpio es el bueno; cualquier sufijo después de la extensión es una
generación superada que se conserva solo para comparar.** Nada con sufijo debe entrar a un
entrenamiento.

## Vigente

| archivo | qué es | generado por |
|---|---|---|
| `early_windows_{deep,wide}.parquet` | contaminantes OpenUniverse (SN, TDE, SLSN, PISN) | `kn-run-openuniverse`, 2026-07-28 |
| `kn_windows_{deep,wide}.parquet` | kilonovas LANL inyectadas | `kn-kilonova-windows`, 2026-07-30 (2ª corrida del día: paridad de cadencia) |

Las magnitudes de los contaminantes vienen del snana de OpenUniverse, que ya trae la
K-corrección y todo lo demás incorporado; este pipeline solo les aplica la receta de ruido y la
ventana. Por eso **ningún fix de la fotometría sintética los toca** — esa fotometría existe
únicamente en el camino de las KN, que parten de espectros LANL en reposo.

## Superado — conservado en disco

| archivo | por qué quedó obsoleto |
|---|---|
| `kn_windows_{deep,wide}.parquet.cadence-parity-leak` | la fuga de paridad de la cadencia de más abajo (1ª corrida del 2026-07-30, la del factor angular) |
| `openuniverse_tokens.npz.stale-2026-07-30` | caché de tokens construida sobre esos parquets |

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
cuando el fichero no existe, y `normalization.json` se reajusta en el mismo paso.

Los `kilonova_windows_*.hdf5` sí eran fuente para el training y estaban en DVC. El **2026-07-31 se
borraron también sus `.dvc`**, junto con el remote `elements` que era lo único que los alojaba: la
migración a Dropbox no los llevó, y mantener un remote entero por una generación con tres bugs de
fotometría y sin factor angular solo confundía. Ya no hay forma de recuperarlos con `dvc pull`; la
generación viva es `kn_windows_{deep,wide}.parquet`. Los bytes siguen físicamente en el disco
Elements si alguna vez hiciera falta una arqueología, pero nada del repo apunta ahí.

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
| KN con ≥1 detección / inyectadas | 155 268 → **721 773** / 1e6 | 95 650 → **590 914** / 1e6 |
| z máximo con detección | 0.2411 → **1.000** (borde de la grilla) | 0.1561 → **1.000** |
| z50% (mitad de la grilla detectada) | — → **0.387** | — → **0.241** |
| z5% | — → **>1.0** | — → **0.729** |
| filas | — → 14 435 450 | — → 11 818 280 |

La grilla ahora se satura por arriba: queda 7.0% (deep) y 0.9% (wide) de detección en el último
nodo, z=1. Si se quiere el z5% de deep hay que extender `redshift_max`.

Chequeos de sanidad sobre los ficheros nuevos: **0** filas de banda ancla con `observed=False`
(ojo, la ancla es **Z087 en deep y R062 en wide**, `roman_noise.py: TIER_ANCHOR_BAND` — comprobar
deep contra R062 da 0 pero es vacío, esa banda no existe en ese tier), **0** `NaN` en `mag_true`, y
los `+inf` presentes (42 deep, 3 246 wide) son no-detecciones reales según la convención de abajo.
Las bandas no-ancla salen con `observed=False` en el 50.0% exacto de sus filas, que es la cadencia
por diseño: la ancla en todas las visitas, el resto en visitas alternas (`cadence_schedule`).

### La generación vigente: paridad de la cadencia (2ª corrida del 2026-07-30)

Misma grilla, mismos tres bugs corregidos y mismo factor angular; lo único que cambia es que la
paridad de la cadencia ya se sortea (sección de más abajo). Corrida de 15:22 a 17:56, 30 workers,
108 KN/s.

| | deep | wide |
|---|---|---|
| KN con ≥1 detección / inyectadas | 721 773 → **752 474** / 1e6 | 590 914 → **649 211** / 1e6 |
| filas | 14 435 450 → **15 049 470** | 11 818 280 → **12 984 220** |
| z máximo con detección | **1.000** (borde de la grilla) | **1.000** |

Suben las detecciones (+4.3% deep, +9.9% wide) porque la mitad de las KN caen ahora en la paridad
que observa temprano el otro par de bandas, y algunas que antes no llegaban a SNR≥5 ahora sí. Ojo:
el set de realizaciones **no** es el mismo que el de la 1ª corrida aunque la semilla no cambie —
sortear la paridad extrae un número más por realización y desplaza el stream del RNG.

Chequeos sobre las cuatro fuentes vigentes: `object_id` únicos, **0** `NaN` en `mag_true`, **0**
filas de banda ancla sin observar, **0** filas observadas sin magnitud, **0** `snr==0` con `mag_err`
finito. Los `+inf` (120 deep, 4 791 wide, ninguno en OU) son no-detecciones reales.

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

## El otro bug grande: la paridad de la cadencia (fuga de etiqueta)

Detectado el **2026-07-30**, verificando que los tres bugs de fotometría estuvieran cerrados sobre
los ficheros del factor angular. No es fotometría: es **qué bandas están disponibles**, la misma
familia que el bug 2, y bastante más grande.

El patrón de bandas (`roman_noise.bands_observed_at_visit`) tiene periodo **2 visitas**: pares →
ancla + las dos no-ancla más azules (ZYJ deep / RZY wide), impares → ancla + las dos rojas (ZHF /
RJH). Los contaminantes se apoyan en la secuencia de visitas **del survey**, pero la grilla KN se
reconstruye por objeto desde el merger: `base_epochs = explosion_offset + 5*arange(8)`, con
`EXPLOSION_OFFSET_MAX_DAYS = 5.0` = **un** periodo de cadencia. Su índice de visita 0 es siempre la
primera visita post-merger, o sea siempre paridad par; y como la KN es rápida y casi siempre se
detecta ahí, la fase de la cadencia quedaba impresa en la etiqueta.

La fase del merger dentro del ciclo de 10 d tiene **dos** grados de libertad: el retardo hasta la
primera visita y la **paridad** de esa visita. El offset U[0,5) sí daba bien el retardo; la paridad
no se sorteaba. Ahora `sample_kn_realizations_on_grid` extrae `cadence_parity` aparte y
`cadence_schedule` la aplica vía `visit_index_offset`.

Fase par en la época de primera detección, y lo que se puede sacar solo de la máscara:

| | KN antes | KN después | contaminantes |
|---|---|---|---|
| deep | 95.28% | **46.14%** | 40.28% |
| wide | 88.15% | **40.37%** | 14.26% |
| combinado | 92.07% | **43.46%** | 28.59% |

Lo que se puede sacar de una regla que **solo mira la máscara** ("fase par ⇒ KN"), sin fotometría ni
tiempos:

| | antes | después |
|---|---|---|
| accuracy | **80.73%** | **58.35%** |
| baseline por clase mayoritaria | 54.90% | 53.27% |
| razón de verosimilitud P(par\|KN)/P(par\|OU) | 3.22 | **1.52** |

El `val_acc_noz` del modelo de julio era ~0.91, así que una parte de eso era artefacto. El
transformer lo veía entero: las bandas no observadas se emiten como tokens `n`.

**El sesgo del lado OU no es un bug**: su grilla es la del survey real, y las bandas rojas —las que
detectan primero a los contaminantes de z~1.3— caen en visitas impares. Que las KN, azules y
rápidas, prefieran las pares es física legítima; lo que no lo era es que la prefirieran al 92% por
construcción. Que tras el fix no quede exactamente en 50% también es física: cuando la detección no
cae en la visita 0 cae en la 1, de paridad opuesta, y qué banda dispara primero depende del color.
De ahí que quede un residuo de 5 puntos sobre el baseline: eso es información de color y escala
temporal, aprendible de forma legítima, no un atajo.

Fijado por `test_cadence_parity_selects_the_band_pair_of_the_first_epoch` (las dos paridades dan
pares de banda complementarios en la primera época) y
`test_sample_kn_realizations_on_grid_draws_both_cadence_parities` (50/50 e independiente del
offset).

## Resuelto: `kn_object_id` ya es único

Era `{sim}_{angle}_{offset:.4f}_{z:.4f}` y dejaba fuera el `noise_id`, el único campo único de la
realización, así que dos sorteos colisionaban si coincidían sim, ángulo y nodo de z y sus offsets
redondeaban igual a 4 decimales: **~2 esperadas por millón**, 3 en la corrida del 2026-07-30 (las
mismas en los dos tiers, que comparten el set de realizaciones). No era un fallo de lógica sino el
cumpleaños esperado de esa clave, pero los dos caminos no se comportaban igual: el paralelo
(`_kn_simulation_task`) escribía las dos ventanas —un objeto de 40 filas— y el secuencial
(`build_kn_models`) acumulaba en un dict indexado por el id y **se comía una en silencio**.

Desde el 2026-07-30 el id es `{sim}_{angle}_{offset:.4f}_{z:.4f}_{parity}_{noise_id}`: el
`noise_id` al final lo hace único por construcción, no solo improbable (verificado sobre el sorteo
completo de 1e6: 0 colisiones, frente a 1 con la clave vieja sobre ese mismo sorteo). El
`simulation_id` sigue siendo el **primer** campo porque `training/openuniverse_data.py` lo lee de
ahí para el split anti-fuga por modelo de eyecta. `scripts/dedupe_kn_windows.py`, que hacía la
edición quirúrgica sobre los parquets viejos, queda obsoleto.

## Pendiente conocido

(Hay unos pocos objetos de deep con 15 filas en vez de 20: `N_KN_VISITS = 8` y la ventana es primera
detección + 4 épocas, así que si la detección cae en la visita 6 no quedan 4 épocas detrás. Eso es
legítimo, no un duplicado.)

`dvc add` del nuevo `lanl_spectra.parquet`: el `.dvc` commiteado sigue apuntando al hash de la
versión por bin angular. Pendiente de hacer en el Mac, que es donde el remote (`/Volumes/Elements`)
es alcanzable.

`src/kilonova/simulation/extinction.py` también llevaba el bug 1 y ya está corregido, pero nada de
lo derivado de `kn-extinguish` se ha regenerado y sus dos salidas están borradas. La etapa
`extinguish` de `dvc.yaml` lo detecta sola porque `lanl_spectra.parquet` es una de sus `deps`.
