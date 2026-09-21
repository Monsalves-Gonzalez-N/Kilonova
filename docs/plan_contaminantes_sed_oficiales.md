# Contaminantes desde los SED oficiales, sin padres — plan

> **SUPERADO el 2026-09-18, no implementado.** Este plan proponia asignar la luminosidad sin usar
> los padres. Se decidio lo contrario: el brillo se toma del padre, porque `SNANA_INPUT_CONFIGS.tar`
> (WGT/MAGOFF/MAGSMEAR, `GENMAG_SMEAR_MODEL`) no esta publicado en Zenodo y la via parametrica no
> existe de este lado. El plan vigente es `docs/plan_data_augmentation_training_set.md`.

Decidido el 2026-09-17. Rama: `izc-low-redshift-contaminants`.
Sucede a `docs/plan_templates_oficiales_ou.md`, que migró los templates core-collapse al release
oficial. Esto usa ese mismo release para el resto, y cambia cómo se asigna la luminosidad.

---

## La idea

Hoy cada contaminante se genera re-renderizando un **objeto padre** de OpenUniverse: se le mide el
brillo a la curva de luz del padre en los HDF5 de 16 GB, y se re-renderiza a otro redshift. Eso
existía porque la luminosidad que SNANA le dio a cada objeto es un sorteo que no está en el
catálogo.

Ya no hace falta para 6 de las 7 clases. El pipeline pasa a ser el de las KN: **SED → luminosidad →
redshift → fotometría.** Sólo SN Ia sigue necesitando padre.

## El reparto: 1/7 por etiqueta

Uniforme sobre las 7 etiquetas que lleva el parquet. Desaparece `UNIFORM_CLASS_SHARE = 1/4` y con
él el reparto 17/24–7/24 de SN II, que venía del conteo de templates de OU.

| etiqueta | frac | SED | luminosidad | ¿padre? |
|---|---|---|---|---|
| SN Ia | 1/7 | `SALT3.NIR_WAVEEXT` | `salt2_mB` del catálogo, objeto a objeto | **sí** |
| SN IIP | 1/7 | 17 oficiales | propia + MAGOFF + N(0, MAGSMEAR) | no |
| SN IIL | 1/7 | 7 oficiales | ídem | no |
| SN Ib | 1/7 | 13 oficiales | ídem | no |
| SN Ic | 1/7 | 7 oficiales | ídem | no |
| SN Iax | 1/7 | 919 oficiales | **la propia del SED, a 10 pc** | no |
| TDE | 1/7 | 1 oficial | **la propia del SED, a 10 pc** | no |

SED uniforme dentro de cada clase. Para core-collapse eso **es** lo que hizo OU: su `WGT` es
idéntico entre templates de una misma clase (17 IIP todos 0.033563, 13 Ib todos 0.008316); `WGT`
sólo codifica la tasa *entre* clases, que es justo lo que este reparto sobreescribe a propósito.

**Costo conocido y aceptado:** TDE tiene **un solo SED**, así que sus ~86 000 objetos son ese
template variando sólo en luminosidad, redshift y fase de cadencia. SLSN-I y PISN quedan fuera: 0.1 %
de la población de OU y un template cada uno.

## Lo que está medido, y que es lo que permite esto

Todo del 2026-09-17, contra el release y contra los propios datos de OU.

**1. La receta de SNANA para core-collapse está publicada y reproduce lo que el pipeline mide.**
`SIMGEN_INCLUDE_NON1A.INPUT` trae `WGT`, `MAGOFF` y `MAGSMEAR` por template. Contra la luminosidad
que el pipeline mide hoy de 450 000 padres, sin ajustar nada:

| subtipo | medido | receta | KS |
|---|---|---|---|
| SN IIP | −17.07, σ 1.08 | −17.08, σ 1.10 | 0.014 |
| SN IIL | −18.20, σ 0.91 | −18.21, σ 0.92 | 0.010 |
| SN Ib | −17.53, σ 1.11 | −17.57, σ 1.12 | 0.015 |
| SN Ic | −17.74, σ 1.18 | −17.80, σ 1.18 | 0.029 |

Coinciden percentil por percentil de p5 a p95, a 0.02–0.06 mag.

**2. Para SN Iax y TDE, la calibración propia del SED ES la de OU.** Medido sobre padres reales,
mapeando cada uno a su `template_index`:

| clase | M_B propia del SED | M_B que OU le dio | diferencia |
|---|---|---|---|
| SN Iax | −16.37 | −16.35 | −0.05 |
| TDE | −18.12 | −18.22 | −0.10 |

Sus archivos lo declaran en el header: `flux: erg/s/cm^2/A scaled to 10 pc`.

**3. Los `.SED` de V19 NO declaran calibración y no son de fiar.** No tienen header. Y hay prueba
interna: los MAGOFF de los 7 SN Ic son −1.46, −1.09, −0.61, **−4.56**, 0.00, −0.61, −1.00. Ese
−4.56 es SN2011bm, cuyo archivo leído a 10 pc da M_B = −13.22 contra −16.5 a −17.8 de sus seis
hermanos. Ese archivo está mal normalizado por ~4 mag y su MAGOFF es la reparación. **Por eso
core-collapse usa MAGOFF y las otras clases no**: no es dogma de reproducir a OU, es que ahí la
normalización del archivo está rota.

**4. `SALT3.NIR_WAVEEXT` es idéntico al `salt3-nir` de sncosmo** — escala 1.0000, residual 0.0000
sobre el rango común — y llega a 25 000 Å en vez de 20 000. Adoptarlo elimina el parche
`IA_PAD_WAVELENGTH`, que hoy extiende el modelo plano sobre los últimos 1 000 Å. SN Ia es la peor
clase del residual (0.037 contra 0.006 de Ib/Ic) precisamente por ser la única que no lee el
template de OU.

**5. Todas las clases mapean a un SED por `template_index`.** Iax lleva índices 1–919 discretos
sobre los 999 del release: OU **no** interpoló la grilla SIMSED, sorteó un índice, igual que
core-collapse. (Lo cual ya estaba documentado en el módulo: "ONLY THE FIRST 919 ROWS SHIPPED".)

## Por qué el sesgo brillante de los templates no se resuelve solo

Los 44 templates V19 vienen de supernovas cercanas bien observadas, o sea seleccionadas por
brillantes. Para lo que esta muestra hace, lo que hace falta **no** es el extremo brillante sino una
**cola débil** que solape con las KN: si a bajo z los contaminantes salen sólo brillantes, el atajo
no desaparece, se invierte a "débil a bajo z ⟹ KN". MAGSMEAR es justo lo que ensancha esa cola, de
44 valores discretos a una distribución continua que llega más brillante *y* más débil que cualquier
template individual. La corrección de OU sirve aquí por una razón distinta de por la que OU la hizo.

## Pasos

1. **SN Iax y TDE.** Leer sus `.SED` del release a un archivo con el mismo patrón que
   `cc_templates.npz`. Reemplazan la reconstrucción del notebook de Rutgers (0.05 mag de error) y
   el banco MOSFiT (0.19 mag de sustitución), que es por lo que `669b8a2` las había sacado. Borrar
   `_iax_bank`, `_iax_base_sed`, `tde_source`, `scripts/build_tde_templates.py` y `data/iax/`.
2. **SN Ia.** Registrar `SALT3.NIR_WAVEEXT` como fuente y borrar `_padded_ia_source` /
   `IA_PAD_WAVELENGTH`.
3. **Luminosidad paramétrica para core-collapse.** Guardar MAGOFF/MAGSMEAR y la M_B propia de cada
   template en el archivo, y asignar `peak_absolute_magnitude` sin tocar al padre.
4. **`CLASS_FRACTION` a 1/7 por etiqueta**, y sortear el SED uniformemente dentro de la clase.
5. **Regenerar y validar.** El padre pasa a hacer falta sólo para SN Ia: 1/7 de la muestra en vez de
   toda. El test de no-regresión ya no puede ser el residual banda-a-banda (que compara contra el
   padre); pasa a ser la distribución de magnitud absoluta por clase contra la tabla del punto 1.

## Auditoría de los templates (2026-09-17)

Todo SED que la muestra sortearía, revisado por magnitud absoluta fuera de rango, flujo negativo y
ceros en el óptico.

**Core-collapse, 50 archivos (44 sorteados por OU):**
- `SN2011bm` M_B −13.22, **+4.5 sigma robustas**. Roto, y V19 lo sabía: su MAGOFF es −4.56 mientras
  los otros seis SN Ic van de −1.46 a 0.00. **El MAGOFF es el que delata cuál está roto.**
- `SN1987A` M_B −14.45, +3.1 sigma — pero SN1987A ERA subluminosa, y su MAGOFF es el −0.40 por
  defecto de la clase, no una reparación. Es real, no rota.
- **10 de los 50 llevan celdas de flujo negativo**, hasta 1.1 % (`SN2009bb`, que OU no sortea).
  `roman_light_curve` ya hace `np.clip(..., 0.0, None)`, así que no llega a la fotometría, pero
  `peakmag` y `set_source_peakabsmag` no clipean.

**SN Iax, 1001 archivos: limpios.** Ningún outlier sobre 3 sigma, cero flujo negativo, cero huecos
en el óptico. M_B de −12.36 a −19.72, mediana −16.35.

**TDE y SLSN-I:** un archivo cada uno, sin patologías.

## Cabos sueltos

- ~~gentype 99~~ **RESUELTO**: son fuentes de magnitud FIJA. Variación temporal 0.000 en las seis
  bandas y la misma magnitud en todas ellas (R=Z=Y=J=H=F), o sea espectro plano en AB. Objetos de
  calibración, no transientes; el `z_CMB` que llevan no significa nada. Quedan fuera.
- ~~SN Iax, la relación objeto a objeto~~ **RESUELTO, y era un off-by-one del release.** Ver abajo.
- **`SIGMA_INT: 0.106`** en `SALT3.INFO`, "used in simulation", contra los 0.21 mag de scatter que
  el módulo midió en la fotometría de las Ia. No son el mismo número; entender por qué.


---

## El `template_index` de SN Iax está corrido en uno (2026-09-17)

**`NON1A.LIST` de `SIMSED.SNIax` dice que `template_index` 1 es `SED-Iax-0001.dat`. OpenUniverse
usó `SED-Iax-0000.dat`.** Quien lea la lista al pie de la letra renderiza el template equivocado.

Medido sobre 4 357 padres reales en 40 templates, comparando la magnitud absoluta que OU le dio a
cada objeto contra la que su `.SED` carga por sí solo:

| corrimiento | pendiente | r | sd(OU − propia) |
|---|---|---|---|
| −2 | 0.103 | 0.135 | 1.82 |
| **−1** | **0.902** | **0.998** | **0.15 mag** |
| 0, lo que dice `NON1A.LIST` | 0.037 | 0.055 | 2.04 |
| +1 | −0.008 | −0.013 | 2.20 |

El banco Iax es un sorteo aleatorio fila a fila, así que filas vecinas no se parecen en nada: un
corrimiento de uno decorrelaciona por completo y es **indistinguible de "OU sobreescribió la
calibración"** si no se prueba el corrimiento explícitamente. Con el mapeo correcto la correlación
es 0.998 y **la calibración propia del SED es la de OU**, que es lo que el plan asume.

Encaja con lo que el módulo ya documentaba: el notebook de Rutgers sortea `ssize = 1001` filas
(0 a 1000) y el catálogo de OU lleva `template_index` 1 a 919 — fila *k* ↔ índice *k+1*.

**Hay 1001 archivos (`SED-Iax-0000` a `SED-Iax-1000`) y la lista sólo indexa 0001 a 0999.** Ese
descuadre es la pista de que la lista se generó desfasada.

### Lo que queda por separar

La dispersión **dentro** de un template es 0.71 mag, que no es cero. No puede ser MAGSMEAR sin más,
porque las medianas por template siguen la calibración propia a 0.15 mag. La sospecha concreta:
**esta medición no aplicó la pantalla de polvo de host**, y SN Iax es la única clase que lleva una
explícita (AV mediana 0.440, p84 1.027, RV 3.1). Un AV de 0.44 son ~0.58 mag en B. Repetir la
medición aplicando el `AV`/`RV` que el catálogo da por objeto debería colapsar esos 0.71.

---

## Implementado el 2026-09-17

**SN Ia — hecho.** `SALT3.NIR_WAVEEXT` vendorizado en `data/openuniverse/salt3_nir_waveext/`
(17 MB; sncosmo exige el directorio descomprimido y los ocho archivos, las superficies de varianza
incluidas). `_padded_ia_source` y `IA_PAD_WAVELENGTH` borrados.

Vale menos de lo que yo había supuesto, y una hipótesis se cayó: contra el parche que reemplaza,
cambia **sólo F184 y sólo bajo z = 0.05** — −0.022 mag a z=0.01, −0.019 a z=0.02, −0.011 a z=0.03,
−0.002 a z=0.05, 0.000 arriba; todas las demás bandas 0.0000 a todo redshift. Y **no** explica que
SN Ia sea la peor clase del residual: medido sobre 400 padres, el residual es idéntico a cuatro
decimales con cualquiera de las dos fuentes, porque esa medición renderiza al redshift del *padre*,
que es alto, donde F184 en reposo cae muy dentro del techo de sncosmo. **Qué hace de SN Ia la peor
clase sigue sin explicación.**

De paso corrige un número que el módulo afirmaba: decía que el parche plano difería del modelo
continuado en 0.00001 mag. Contra el modelo real son 0.019 mag a z=0.02.

**TDE — hecho, y es la ganancia grande.** `2019qiz.sed` en `data/openuniverse/tde_template.npz`
(0.7 MB). Fuera el banco MOSFiT, `scripts/build_tde_templates.py`, `data/tde/` y el sorteo de
template por padre — OU tiene **un** SED y sus 3 769 TDE llevan todos `template_index` 1.

| | MOSFiT (antes) | SED oficial (ahora) |
|---|---|---|
| residual banda-a-banda | 0.19 mag | **0.0168 mag** |
| tendencia de color R062→F184 | monótona | −0.002 a −0.004, plana |

Medido sobre 448 padres. Factor 11, y la tendencia de color desaparece.

**SN Iax — archivo construido, falta cablear.** `data/openuniverse/iax_templates.npz`, 356 MB,
919 templates en su grilla nativa (94 fases × 1200 Å, todas idénticas) **sin remuestrear nada**,
escritos en orden de `template_index` de OU con el off-by-one ya aplicado.

Y hay motivo de sobra para cablearlo: medida hoy, la reconstrucción del notebook de Rutgers que
sigue en uso da **0.1913 mag de residual con una tendencia de color monótona de 0.28 mag**
(R062 −0.196 → F184 +0.083) sobre 440 padres. El módulo documenta esa reconstrucción como "0.05 mag
off"; ese número no es el que sale de esta medición y hay que entender la diferencia, pero en
cualquier caso Iax es hoy **la peor clase de la muestra por un factor 10**.

`PARENT_GENTYPES` pasa a `(10, 12, 21, 26, 32, 42)` y `LABEL_BY_GENTYPE` gana 12 y 42.
`CLASS_FRACTION` sigue sin ellos: se cambia a 1/7 cuando Iax esté cableado.

---

## SLSN-I entra, PISN no (2026-09-17)

El argumento que había para dejarlas fuera — "0.1 % de la población de OU" — **es inválido bajo
reparto uniforme por clase**: la fracción de OU sólo importa si muestreas proporcional a OU, que es
justo lo que este diseño no hace. Retirado. Lo que sigue se midió en su lugar.

**Primero, un error de medición que casi las condena a las dos.** Un primer pase dio residuales
banda-a-banda de 0.196 (SLSN), 0.223 (PISN-He) y 0.398 (PISN-Hy) — el orden de las sustituciones
recién retiradas. La causa era que el script renderizaba el SED desnudo, y estas clases llevan
pantalla de polvo de host en el **100 %** de sus objetos:

| clase | fracción con `AV != -9` |
|---|---|
| CC, SN Ia, TDE | 0.000 |
| SN Iax, SLSN-I, PISN-He, PISN-Hy | **1.000** |

Con la pantalla aplicada:

| clase | sin polvo | con polvo | OU − calibración propia | mapeo |
|---|---|---|---|---|
| SLSN-I | 0.196 | **0.0153** | **+0.00** | 1 template |
| PISN-HeCore | 0.223 | 0.0984 | +0.16 | r = 0.972, sin corrimiento |
| PISN-Hydrogenic | 0.398 | 0.1553 | −0.06 | r = 0.990, sin corrimiento |

**SLSN-I entra.** Residual 0.0153, el mismo orden que TDE y mejor que SN Ia, y su calibración
propia a 10 pc es exactamente la de OU: −21.75 contra −21.75. Un solo template (`2016apd.sed`),
1 122 padres. El mapeo no tiene off-by-one (hay un solo índice).

**PISN no, y ahora por una razón dura.** Sus SED llegan a **20 000 Å** en reposo, y el borde rojo de
F184 está en 21 000: no cubre F184 sin extrapolar por debajo de **z = 0.050**. La grilla del izc
empieza en z = 0.010. O sea PISN no se puede renderizar justamente en los bines que esta muestra
existe para llenar, y el módulo prohíbe extrapolar por diseño (ver `SOURCES_BY_LABEL`). Su residual
de 0.10–0.16 es además 6–10× el de las demás.

**El reparto pasa a 1/8**: {Ia, IIP, IIL, Ib, Ic, Iax, TDE, SLSN-I}.

### Un hallazgo lateral que refuerza el punto A

Con el borde rojo de F184 en 21 000 Å, cada modelo tiene un redshift mínimo por cobertura:

| modelo | cubre hasta | F184 sin extrapolar desde |
|---|---|---|
| SALT3.NIR | 25 000 Å | z = 0 |
| SN Iax | 24 990 Å | z = 0 |
| SLSN-I, TDE | 24 000 Å | z = 0 |
| **core-collapse, con NUESTRO recorte** | **20 600 Å** | **z = 0.019** |
| PISN | 20 000 Å | z = 0.050 |

Los archivos oficiales de core-collapse llegan a 25 000 Å; los 20 600 son el recorte que heredamos
del archivo recuperado. **Hoy se están descartando objetos core-collapse por falta de cobertura en
los bines más bajos de la grilla**, y ampliar la grilla los recupera. Es un argumento concreto para
el punto A que antes no existía.

---

## SN Iax, SLSN-I y el reparto 1/8 — implementado

**SN Iax.** `data/openuniverse/iax_templates.npz`, 356 MB, los 919 templates en su grilla nativa
(94 fases × 1200 Å, idéntica en los 919, verificado) **sin remuestrear nada**, escritos en orden de
`template_index` de OU con el off-by-one absorbido. Fuera las ~100 líneas de constantes del replay
del notebook, `_iax_bank`, `_iax_base_sed`, `iax_template`, el warp, y `data/iax/`.
`iax_source(template_index)` lee del archivo bajo `@cache`.

**SLSN-I.** `data/openuniverse/slsn_template.npz`, 0.8 MB, `2016apd.sed`.

**`CLASS_FRACTION` a 1/8 por etiqueta**, y se va `SN_II_SUBTYPE_FRACTION` (el 17/24–7/24 venía de
los conteos de templates de OU: lo correcto cuando el objetivo es la mezcla de OU, irrelevante
cuando no lo es). `PARENT_GENTYPES` es `(10, 12, 21, 26, 32, 40, 42)`.

**Smoke run de 4 000 objetos, 8 workers:**

| | |
|---|---|
| composición | 459–527 por clase sobre 8 clases — uniforme |
| gentypes izc | 210, 212, 221, 226, 232, 240, 242 |
| sin brillo medible | 0 |
| descartados por cobertura | 6 de 4 000 |
| padres distintos | 3 862 para 4 000 objetos |

97 tests pasan, `ruff check` y `ruff format --check` limpios.

### Dónde queda cada clase, medido contra la fotometría propia de OU

| clase | residual banda-a-banda | antes |
|---|---|---|
| SN Ib / Ic / IIL | 0.006 | — |
| SLSN-I | **0.015** | no existía |
| TDE | **0.017** | 0.19 (MOSFiT) |
| SN IIP | 0.016 | — |
| SN Ia | 0.039 | 0.039 (sin explicación) |
| SN Iax | **pendiente de medir** | 0.19 (replay del notebook) |

Falta re-medir SN Iax con su SED oficial: la predicción es que caiga como cayó TDE, de 0.19 a
~0.02, pero eso **no está medido todavía**.

### SN Iax medido con su SED oficial

0.1913 → **0.0121 mag**, sobre los mismos 440 padres y con el mismo método. Factor 16. Y la
tendencia de color, que era la firma de la sustitución, se derrumba:

| | R062 | Z087 | Y106 | J129 | H158 | F184 |
|---|---|---|---|---|---|---|
| replay del notebook | −0.196 | −0.135 | −0.091 | −0.030 | +0.035 | +0.083 |
| **SED oficial** | **+0.052** | +0.021 | +0.004 | +0.000 | −0.002 | +0.000 |

De 0.28 mag de barrido a 0.052, y todo lo que queda está en R062.

### El cuadro completo, todo medido contra la fotometría propia de OU

| clase | residual banda-a-banda | antes |
|---|---|---|
| SN Ib / Ic / IIL | 0.006 | — |
| **SLSN-I** | **0.0088** | no existía |
| **SN Iax** | **0.0121** | 0.1913 |
| SN IIP | 0.016 | — |
| **TDE** | **0.0168** | 0.19 |
| **SN Ia** | **0.039** | 0.039 |

Siete de las ocho clases están entre 0.006 y 0.017. **SN Ia es la única fuera, en 0.039, y sigue
sin explicación** — no es el parche de longitud de onda, que se midió y no cambia nada. Es el único
cabo suelto de fidelidad que queda.

---

## La grilla de core-collapse, completa (2026-09-17)

`cc_templates.npz`: 57 MB → **123.6 MB**, 44 templates.

| | antes | ahora |
|---|---|---|
| longitud de onda | 3000–20600 Å, remuestreada a 5 Å | **1605–25000 Å, la del archivo, sin tocar** |
| fase | (−12, +35) d fija para todos | **199 d de mediana, por template** |
| z mínimo por cobertura de F184 | 0.019 | **0.000** |

**Las longitudes de onda no se tocan.** Los 44 archivos comparten la grilla exactamente —
1605–25000 Å en pasos de 5 Å, 4680 puntos, verificado en los 44— así que no hay nada que
remuestrear. El recorte a 20 600 Å era el borde rojo de F184 a z = 0.02, de modo que cubría F184
sólo por encima de z = 0.019 mientras la grilla del izc empieza en 0.010: se descartaban objetos
por falta de una cobertura que el modelo sí tiene.

**La fase sí se remuestrea, a 1 d, y no es preferencia.** Las fases de los archivos son las épocas
que tuvo la espectroscopía original, o sea salvajemente irregulares — nodos consecutivos a 0.04 d
junto a huecos de 2 d — y el spline 2D de sncosmo se desestabiliza ahí: medido, `SN2016X` sale
**negativo en todo 9000–21000 Å** a 8 d antes de su propio máximo B, donde el archivo no tiene ni
una celda no positiva en ese rango. En grilla regular de 1 d el desborde desaparece.

**Y las fases se recortan, por una razón del modelo y no de la interpolación.** Algunos templates
tienen corridas de flujo cero en el infrarrojo en sus fases extremas — la extensión V19 no tiene
qué extender donde acaba la espectroscopía original — y la peor mide 4750 Å de ancho. Una fase con
el óptico sano y el infrarrojo a cero llega a la ventana como "muy rojo, sin infrarrojo", que es un
artefacto correlacionado con clase. Cada template conserva la corrida contigua más larga de fases,
**que contenga su máximo B**, con flujo positivo en todo 3000–21000 Å. **36 de los 44 no pierden
ninguna fase.**

La corrida tiene que contener el máximo y no ser simplemente la más larga: `SN1994I` tiene un hueco
en el ultravioleta entre +32 y +96 d, y su corrida limpia más larga es todo lo posterior, que
descartaría el pico.

**Lo que el recorte (−12, +35) costaba, además de los 47 d:** decidía dónde `peak_phase` encuentra
el máximo B, y para las SN II lentas lo metía dentro del corte. SN1987A y SN2008bj tienen su
máximo después de +35 d, así que su "máximo" era un artefacto de la ventana.

**Y una corrección que no buscaba:** `parent_peak_magnitudes` toma el pico del padre sobre
(−20, +70) d y `rendered_peak_magnitudes` sobre `REST_FRAME_PHASES`, que es el mismo rango — **pero
el archivo recortado truncaba el render en +35**. Los dos lados de la medición usaban ventanas
distintas. Ahora coinciden. Sobre 132 padres fijos el brillo inferido y las magnitudes de pico no
se mueven en la mediana (0.0000) ni `peak_phase` en ninguno; cambia una minoría, las SN II lentas,
con cola de 0.29 mag en p90.
