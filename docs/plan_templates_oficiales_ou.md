# Migrar a los templates oficiales de OpenUniverse2024 — plan

> **COMPLETADO el 2026-09-17.** La migracion esta hecha y verificada en disco; este documento
> se conserva como registro de por que se hizo, no como trabajo pendiente. El plan vigente es
> `docs/plan_data_augmentation_training_set.md`.

Planificado el 2026-09-17, revisado el mismo día contra el código (y contra una crítica de codex,
que corrigió tres premisas falsas). Rama: `izc-low-redshift-contaminants`.

---

## Qué cambia, en una frase

`scripts/build_openuniverse_cc_templates.py` deja de **recuperar** los templates core-collapse
leyendo `flambda` de los HDF5 de 16 GB de OpenUniverse, y pasa a **leerlos** de los archivos
`.SED` del release oficial. Mismo `.npz` de salida, misma grilla, mismo todo aguas abajo.

Es una simplificación del productor, no un arreglo: el pipeline actual funciona y está validado
(0.008 mag de residual banda-a-banda, plano en redshift). Ver el apéndice para por qué la premisa
del script — *"la extensión al infrarrojo no es pública"* — es falsa desde 2025-01-27.

**Criterio de éxito: reproducir OpenUniverse.** Si OU tiene un color sesgado o se le olvidó el
polvo de host, la muestra izc debe tener exactamente lo mismo. Una desviación de OU es una feature
correlacionada con clase que el clasificador aprende y que el cielo no tiene.

---

## La decisión de diseño que hace esto corto

**El `.npz` se queda.** La tentación es que `_openuniverse_templates()` lea el directorio oficial
directamente y el `.npz` desaparezca. No: el tar son **3.77 GB**, y eso lo convertiría en
dependencia dura del stage `izc_windows` y de los tests. El `.npz` son 24 MB versionados en DVC.

Consecuencia — **nada de esto se toca**:

- `_openuniverse_templates()`, `register_sources()`, `core_collapse_source_by_template_index()`.
- `dvc.yaml` (la dep `data/openuniverse/cc_templates.npz` sigue siendo la misma, línea 81).
- `tests/test_intermediate_z_contaminants.py`, incluidos `_core_collapse_archive_order` y
  `defective_near_infrared_extension`.
- Los HDF5: siguen haciendo falta para `measure_brightness_offset`. El brillo se sigue midiendo
  de la curva de luz del padre (ver "Lo que no se toca").

Y la prueba de no-regresión se abarata: **es un diff `.npz` contra `.npz`**, no una regeneración
de datasets.

---

## Paso 0 — Conseguir la librería

- Record: https://zenodo.org/records/14749318 · `MODELS-1_TRANSIENT_SED.tar` (3.77 GB).
- Extraer **sólo** `NON1ASED.V19_CC+HostXT_WAVEEXT`. Dejarlo fuera del repo
  (`~/Dropbox/Kilonova/openuniverse2025/` o donde caiga); no entra a git ni a DVC.
- Verificar en el directorio: que existan `NON1A.LIST` y `SIMGEN_INCLUDE_NON1A.INPUT` (los dos
  archivos que `read_template_types()` ya lee hoy), y que los `.SED` lleguen a 25 000 Å.
- **Si sólo hay `SED.INFO`** y no esos dos: adaptar `read_template_types()` a ese formato. Es el
  único punto del plan donde el formato real puede obligar a cambiar algo; se resuelve mirando,
  no planificando.

## Paso 1 — Reescribir el script

El diff es más pequeño de lo que parece porque **el mapeo `template_index` → archivo → SNTYPE ya
es explícito y no cambia**: sale de `NON1A.LIST` + `SIMGEN_INCLUDE_NON1A.INPUT` vía
`read_template_types()`, y el cruce contra los gentypes del catálogo ya lo hace
`core_collapse_source_by_template_index()`.

- **Añadir** `read_sed_file()`: un `.SED` NON1ASED es `phase wavelength flux` por línea →
  remuestrear a la misma grilla (`PHASE_LIMITS`, `WAVELENGTH_LIMITS`, `NORMALISATION_*`).
- **Borrar** `open_healpix()`, `object_rest_frame_cube()`, `--local-directory`,
  `--objects-per-template`, `BASE_URL`, `OBSERVER_FRAME_RED_EDGE`, `MAXIMUM_REDSHIFT` y el corte
  `z < 0.187`. Con un SED por template desaparecen también la mediana sobre objetos,
  `object_spread` y `coverage_*`.
- **Conservar** `read_catalogs()`: ya no para elegir objetos sino sólo para saber **qué 44 índices
  sorteó OU**. La librería completa tiene más templates que los que OU usó; hay que seleccionar
  por índice, nunca leer el directorio entero.
- **Conservar** el recorte de fase por template: hoy sale del soporte de los objetos, ahora saldrá
  del soporte del `.SED`. Para el Paso 2 hay que **recortar a la grilla vieja** (fases (−12, 35),
  3000–20600 Å); ampliarla es el punto A de "Después", deliberado y aparte.
- Reescribir el docstring del módulo: la premisa *"el `_WAVEEXT` no es público"* es falsa y todo
  el argumento de la recuperación desde HDF5 se va con ella.

## Paso 2 — Diff `.npz` contra `.npz`

Escribir a `cc_templates_official.npz` y comparar contra el vigente, template a template, en la
grilla vieja. Script desechable en el scratchpad, no en `scripts/`.

- Ambos normalizados igual (8000 Å / fase 0) y en la misma grilla, así que la comparación es
  directa. Máscara: celdas donde el viejo tiene flujo finito y > 0.
- Estadístico: mediana y p99 de `|nuevo/viejo − 1|` por template. **Criterio: mediana < 1 %**, del
  mismo orden que el 0.5 % entre objetos que el archivo recuperado ya pasó.
- Cruzar `template_names` y `labels`: deben salir idénticos entrada por entrada. Si el orden
  cambia, `_core_collapse_archive_order()` rompe silenciosamente y se re-renderiza la clase
  equivocada. **Este check es obligatorio.**
- **Salida: una tabla de 44 filas con la discrepancia, y un veredicto.** Si falla, parar y
  entender antes de seguir; el resto del plan cambia de significado.

## Paso 3 — Gate barato antes de regenerar nada

No regenerar los datasets para descubrir un problema. Con los dos `.npz` en mano:

- Fijar un puñado de padres que cubran los 44 templates y varios redshifts.
- Comparar `rendered_band_curves()` y `measure_brightness_offset()` viejo contra nuevo **sobre los
  mismos padres**. Mirar el brillo inferido, no sólo el `brightness_residual`: ese es dispersión
  entre bandas y un desplazamiento gris común lo dejaría intacto.
- Verificar la convención de fase explícitamente. `peak_phase()` y `REST_FRAME_PHASES` asumen que
  el eje del template y `(mjd − peak_mjd)/(1+z)` del padre son el mismo eje. El archivo recuperado
  alineó su fase al `peak_mjd` de OU; el `.SED` oficial trae su grilla nativa. Un desplazamiento
  aquí mueve todas las curvas.

## Paso 4 — Swap, tests, regeneración única

1. `cc_templates_official.npz` → `cc_templates.npz`, `dvc add`, `dvc push`.
2. `pytest tests/test_intermediate_z_contaminants.py` con `kn_class`. Debe pasar **sin tocar
   ningún test**; si un test hay que tocarlo, entender por qué antes de tocarlo.
3. Regenerar `izc_windows_{deep,wide}.parquet` **una vez** y re-medir la tabla de residuales del
   apéndice. Iguales o mejores. Comparar también objetos aceptados vs rechazados: si la tasa de
   rechazo se mueve, la cobertura del template cambió algo.
4. Recién ahí borrar `scripts/build_v19_extended_templates.py` (el intento con `snsedextend`, ya superado).

---

## Después, y por separado

Ninguna de estas va en el mismo commit que los Pasos 0–4.

**A. Ampliar la grilla.** El oficial llega a 25 000 Å y trae la fase completa; hoy el `.npz` corta
en 20 600 Å y +35 d, y varios templates empiezan en −4 d cuando `REST_FRAME_PHASES` pide desde
−20. Es ganancia neta y es la razón real para migrar, pero **cambia la muestra generada**: máximos,
ventanas aceptadas y cobertura de banda. Por eso va después de demostrar la equivalencia, no
mezclada con ella.

**B. Re-correr la auditoría del infrarrojo.** `defective_near_infrared_extension()` podó
`SOURCES_BY_LABEL` por huecos de flujo cero que venían de *nuestra* extensión con `snsedextend`.
Con SEDs oficiales lo más probable es que ya no haya nada que podar — pero eso se **mide**, no se
asume, y sólo entonces se borran la función y sus tests.

**C. Las clases descartadas.** `MODELS-1` publica TDE, SN Iax, SLSN-I y PISN como NON1ASED oficial,
y las cuatro razones por las que se excluyeron caen. Para clasificar KN esta es probablemente la
ganancia más grande del hallazgo: más diversidad de contaminantes a bajo z. **No es de dos líneas**:
`realization_from_parent()` y `build_model()` siguen usando la reconstrucción Iax y el banco MOSFiT
para TDE, y SLSN/PISN caen en el `ValueError` de clase no soportada. Hay que integrar fuente,
índices, etiquetas, polvo y tests. Ojo con la extinción de host por clase, que hay que seguir
reproduciendo tal cual: SN Iax (12) es la única con pantalla explícita (AV mediana 0.440, p84 1.027,
RV 3.1); TDE (42) y core-collapse ninguna; SN Ia (10) dentro del `c` de SALT.

**D. Medir la función de luminosidad de OU.** Hoy el brillo de cada padre se mide, se usa y se
descarta; agregando los 1.32 M se obtiene empíricamente lo que `SNANA_INPUT_CONFIGS.tar` no
publicó. Viable porque `rendered_peak_magnitudes` depende sólo de (template, redshift): 44 × ~100
nodos en z = 4 400 renders más interpolación, en vez de ~700 k. Antes de creerle: `run_izc_healpix`
excluye los objetos con `brightness_bands == 0`, así que **si las ausencias correlacionan con
brillo, ya sesgan la muestra actual**, no sólo la LF. Auditar padres únicos y separar "HDF5
ausente" de "sin bandas comunes". Y una distribución empírica no identifica `MAGOFF`/`MAGSMEAR` sin
ponderar por selección.

**Lo que se elimina del plan anterior: la fase del "ancla en banda en reposo fija".** Partía de una
premisa falsa. `REFERENCE_MAGNITUDE_BAND` ya es `("bessellb", "ab")` y `build_model()` normaliza
ahí vía `set_source_peakabsmag` — el ancla ya es fija y en reposo. Lo que se arrastra con z no es
el ancla sino **qué bandas observadas entran en la medición**, que es lo que filtra
`BRIGHTNESS_REST_WAVELENGTH_LIMITS`. Si eso deja un error dependiente de z es una pregunta abierta,
pero se responde midiendo el offset contra z (diagnóstico de una tarde, parte de D), no
refactorizando el ancla.

---

## Lo que no se toca

**La asignación de brillo sigue midiéndose de la curva de luz del padre.** No migrar a
"luminosidad del catálogo", por más tentador que se vea. Las Ia de OU cumplen
`salt2_mB + 0.15·x1 − 3.1·c − μ(z) = −19.363447 ± 0.000103` sobre las 224 118: ese ±0.0001 es una
identidad, **el `mB` reportado no lleva dispersión intrínseca alguna**. Pero la fotometría sí:
contra `salt2_mB − μ`, sobre 60 Ia, la mediana da +0.19 mag (offset de sistema, esperado) y el rms
**0.21 mag** objeto a objeto. Es acromático (dentro de un objeto las bandas concuerdan a 0.039 mag)
y no correlaciona con x1, c ni z: es el `GENMAG_SMEAR_MODEL` de SNANA aplicado al flujo y no
devuelto al `mB`. **Un generador que tomara el catálogo al pie de la letra produciría SNe Ia sin
scatter alguno.**

Los padres se siguen sorteando del catálogo **generado** (1.32 M), no del detectado (880 k).

`CLASS_FRACTION` sigue siendo `UNIFORM_CLASS_SHARE` a propósito: lo que se preserva es la
distribución *dentro* de cada clase, no la mezcla entre clases.

Fuera de alcance, explícitamente: validar contra funciones de luminosidad publicadas
(Richardson+2014, Li+2011), y mejorar el realismo físico de los templates.

---

## Apéndice — antecedentes

**El hallazgo.** `build_openuniverse_cc_templates.py:5` afirma que *"la mitad base de ese nombre es
pública y la mitad `_WAVEEXT` no lo es"*. Falso desde el 2025-01-27:
[zenodo.org/records/14749318](https://zenodo.org/records/14749318) publica
`MODELS-1_TRANSIENT_SED.tar` (3.77 GB), `MODELS-2_HOST_CORRELATION.tar` (3.30 GB) y la nota de
release. De la nota: *"sus rangos de longitud de onda han sido extendidos a 25 000 Å para cubrir la
banda K de Roman en el marco en reposo... Convertimos cada modelo SIMSED en un modelo NONIASED"*.
Modelos: SNIa-SALT2+NIR, SNIax, SNCC(II/Ib/Ic), KILONOVA (Kasen+2017), PAIR_INSTABILITY,
SUPERLUMINOUS_SN-I, TIDAL_DISRUPTION_EVENT.

**SIMSED contra NON1ASED, que no es lo que el plan decía en su primera versión.** La nota dice
*"only the NON1ASED model class can output the true SEDs. We therefore converted each SIMSED model
into a NON1ASED model, which is nothing more than how the SED files are listed"*, y su nota al pie
explica por qué: *"the SIMSED class in SNANA pre-computes broadband flux integrals on a grid of
{redshift, phase, passband}, and does not hold the SEDs in memory during the simulation"*.

Es una afirmación sobre **cómo corre SNANA**, no sobre el formato de los archivos. La conversión no
transforma dato alguno: `NON1ASED.SNIax` y `NON1ASED.KN-K17` son **symlinks** a `SIMSED.SNIax` y
`SIMSED.KN-K17`, con un `NON1A.LIST` que lista los mismos `*_SED.dat.gz`.

Dos consecuencias:

- **Core-collapse no está afectado.** `NON1ASED.V19_CC+HostXT_WAVEEXT` es un directorio real con
  `.SED.gz` planos, NON1ASED nativo, nunca fue SIMSED. Y como OU corrió esa clase con NON1ASED, su
  fotometría sale de integrar *estas* SEDs — que es lo que hace comparable el residual.
- **El punto C se encarece.** OU sorteó SN Iax, KN y PISN **con SIMSED**, interpolando sobre la
  grilla de parámetros; el listado NON1ASED es un reempaquetado posterior ("has no impact on the
  OpenUniverse2024 sims"). Reproducir a OU en esas clases no es elegir un template de la lista sino
  reproducir la interpolación SIMSED. TDE y SLSN-I sí son NON1ASED nativos (`-BBFIT`).

La nota confirma en su sección "Mistakes" el bug que el código ya había deducido: *"para los
modelos SNCC usamos por error las SED des-enrojecidas y por lo tanto no modelamos extinción de
host"*. **Reproducir ese bug es obligatorio**: cero extinción para core-collapse, igual que hoy.

**Lo que NO está publicado.** La nota describe un `SNANA_INPUT_CONFIGS.tar` que **no está en el
record** (verificado vía la API de Zenodo). Ahí vivirían `WGT`, `MAGOFF`, `MAGSMEAR` por template y
el `GENMAG_SMEAR_MODEL` de las Ia. Sin eso la vía paramétrica no existe — de ahí el punto D.

**La normalización no importa.** El recuperado está normalizado a 8000 Å/fase 0, el oficial lleva
la nativa de Vincenzi+2019. `build_model()` llama `set_source_peakabsmag`, que la sobreescribe.

**Estado actual.** El script recupera 44 templates leyendo `flambda` de los HDF5 y dividiendo la
grilla observador por (1+z). Sólo usa objetos con z < 0.187, porque la grilla observador acaba en
24 450 Å y sólo abajo de ese z un objeto alcanza 20 600 Å en reposo. Promedia 8 objetos por
template. Validado a 0.5 % entre objetos de un mismo template a z = 0.081, 0.991 y 1.501, y a 2.3 %
contra la base pycoco pública. Residual banda-a-banda contra la fotometría de OU, sobre
`izc_windows_deep` (599 757 objetos):

| z del padre | n | mediana | p90 |
|---|---|---|---|
| 0.0–0.3 | 4 517 | 0.008 | 0.123 |
| 0.3–0.6 | 33 440 | 0.009 | 0.134 |
| 0.6–1.0 | 101 359 | 0.013 | 0.166 |
| 1.0–1.5 | 152 797 | 0.013 | 0.114 |
| 1.5–2.0 | 131 275 | 0.010 | 0.056 |
| 2.0–3.0 | 176 369 | 0.008 | 0.039 |

Por subtipo: Ib/Ic/IIL 0.006, IIP 0.016, Ia 0.037 (la Ia es la única clase donde no leemos el
template de OU sino que usamos SALT). **Esta es la tabla del Paso 4.3.**

---

## Ejecutado el 2026-09-17 — resultados

**Paso 0.** `MODELS-1_TRANSIENT_SED.tar` bajado a `/home/nicolas/nico/openuniverse_models/`, fuera
del repo. `NON1ASED.V19_CC+HostXT_WAVEEXT` trae los dos archivos índice que el script ya leía
(`NON1A.LIST`, `SIMGEN_INCLUDE_NON1A.INPUT`), con SNTYPE 20/22/32/33 — los mismos que
`LABEL_BY_SNTYPE`. 50 `.SED.gz` en el directorio, **44 indexados**: 17 IIP, 7 IIL, 13 Ib, 7 Ic,
idéntico al `.npz` vigente. Los archivos llegan a 25 000 Å y de −18 d a +120 y más.

**Hallazgo lateral:** `SIMGEN_INCLUDE_NON1A.INPUT` **trae `WGT`, `MAGOFF` y `MAGSMEAR` por
template**. El plan los daba por no publicados (viven en el `SNANA_INPUT_CONFIGS.tar` que no está
en el record). Para core-collapse ya no hacen falta: MAGOFF −0.40 / MAGSMEAR 0.40 para las IIP,
−0.35 / 0.65 para las IIL, etc. Eso cambia el punto D de "Después" — la vía paramétrica existe para
core-collapse, y la LF medida pasa de ser la única fuente a ser un contraste contra un número
publicado.

**Paso 1.** Script reescrito: se fue toda la maquinaria HDF5/fsspec, el corte `z < 0.187`, la
mediana sobre 8 objetos y `--local-directory`. Entró `read_sed_file`. El `.npz` conserva el
esquema; se añadió `template_indices`, que antes no se guardaba y era lo que obligaba a
`_core_collapse_archive_order()` a recuperar el mapeo por orden.

**Paso 2 — el criterio del plan estaba mal planteado.** El diff crudo falla (mediana 13.5 %), pero
la culpa es del eje. `peak_phase()` es `Source.peakphase("bessellb")` y `build_model` llama
`set_source_peakabsmag`: **el pipeline referencia todo al máximo B del propio template y
renormaliza el flujo**, así que la fase cero del archivo y su escala global son dos cosas que no
llegan a existir aguas abajo. Comparados con esos dos grados de libertad libres, los 44 templates
coinciden a **mediana 0.3 %, peor 2.3 %**. Son los mismos SEDs.

El desplazamiento de fase ajustado es 0 d para la mayoría de las Ib/Ic y +2 a +13 d para las SN II:
el eje del archivo recuperado era el `peak_mjd` de OU y el del `.SED` es el nativo de V19, y para
una SN II el `peak_mjd` de OU cae al principio de la meseta.

Alineando por `peakphase` en vez de por el ajuste, la mediana queda en 1.4 % con dos outliers,
SN1987A (43 %) y SN2008bj (26 %). Los dos son SN II de subida lenta cuyo máximo B real cae **fuera**
del recorte a +35 d del archivo viejo, así que su `peakphase` era un artefacto del trim. Eso
entrelaza el recorte con el punto A: el trim define `peakphase` y `peakphase` define todo.

**Paso 3 — el gate, sobre 132 padres fijos del healpix 10050, 44 templates, z 0.15–0.30:**

| | recortado (−12, +35) | fase completa (−40, +150) |
|---|---|---|
| brillo inferido, nuevo − viejo (mediana) | −0.0027 mag | −0.0027 mag |
| ídem, máximo \|·\| | 0.060 | 0.225 |
| magnitudes de pico renderizadas (mediana) | −0.0004 mag | −0.0005 mag |
| ídem, p90 \|·\| | 0.091 | 0.391 |
| `peak_phase`, nuevo − viejo | −2.2 d (máx 12.2) | −2.2 d (máx 12.2) |
| **residual banda-a-banda contra OU** | **0.0084 → 0.0079** | 0.0084 → 0.0077 |

El residual contra la fotometría propia de OU **mejora**, que es el número que dice si el pipeline
reproduce el release. Se adopta el recortado: mejora igual y su cola es cuatro veces menor. La
fase completa queda para el punto A, con la evidencia de que también mejora la mediana.

**Paso 4.** Swap hecho, `dvc add` hecho. **51 tests de izc y la suite completa (102) pasan sin
tocar un solo test.** La auditoría del infrarrojo da 0 huecos de flujo cero en los dos archivos y
sólo dispara su segundo test heurístico (35 templates en el recuperado, 37 en este), que es el que
su propio docstring llama poco sólido — el punto B se cierra: no hay nada que podar y
`defective_near_infrared_extension` se queda como está.

`scripts/build_v19_extended_templates.py` borrado. `sncosmo` añadido a `pyproject.toml`: el módulo
lo importa desde siempre y **no estaba declarado**, así que el stage `izc_windows` no corría en un
env limpio (no estaba instalado en ninguno de los 28 envs de esta máquina).

Falta: la regeneración de `izc_windows_{deep,wide}.parquet` (603 869 objetos, corriendo, log en
`izc_regen_2026-09-17.log`) y la tabla de residuales comparada contra la del apéndice, que se
reproduce exactamente desde `izc_windows_deep.parquet.grid-z-2026-09-16`.

### Validación de los SEDs (2026-09-17, tras cancelar la regeneración)

La regeneración se canceló porque nada había validado los SEDs **como SEDs**: todo lo medido hasta
ahí comparaba magnitudes de pico, y un template desplazado en fase da el mismo pico y una curva de
luz equivocada.

La nota de release vuelve esa validación exacta para core-collapse. OU corrió esa clase con
NON1ASED, la clase de SNANA que *sí* guarda las SED en memoria, así que el `flambda (n_mjd x 227)`
de cada objeto en los HDF5 completos **es la SED del modelo**, no un resumen. De-redshifteada tiene
que SER el template. (Para las clases SIMSED no valdría: esas precomputan integrales de banda ancha
sobre una grilla {z, fase, banda} y no guardan la SED.)

Un objeto por template, healpix 10050, z = 0.046–0.309, comparando 4000–9000 Å con una escala
acromática libre y barriendo el desfase de fase:

| | |
|---|---|
| residual con desfase 0 | mediana 4.4 %, p90 10.6 % |
| **residual en el mejor desfase** | **mediana 1.0 %, p90 2.2 %** |
| mejor desfase | 10/44 exactamente 0, mediana −2 d, rango −20 a +16 |

(43 de 44: el objeto de menor z de SN2016X tiene **una sola época** con flujo, así que no hay nada
que comparar. Es un objeto degenerado, no un template malo.)

**Conclusión, en dos partes.**

1. **Los SEDs son los correctos.** 1.0 % contra la SED que OU guardó de sus propios objetos es tan
   bueno como permite interpolar su grilla observador de 227 puntos.
2. **La fase cero del `.SED` NO es el `peak_mjd` de OU**, y el desfase depende del template. Eso
   era falso antes: el archivo recuperado se construía de-redshifteando objetos en torno a
   `peak_mjd`, así que su fase cero *era* la de OU. Un `.SED` trae el eje nativo de V19.

**Por qué el punto 2 no rompe nada, verificado y no argumentado:** `rendered_band_curves` mide las
fases desde `peak_phase`, el máximo B de la propia fuente, y su **único** llamador —
`rendered_peak_magnitudes` — descarta el eje de fase y se queda con las magnitudes
(`for band, (_, magnitudes)`). `measure_brightness_offset` compara máximos sobre ventanas a ambos
lados, y un máximo no depende de dónde esté el cero. Por eso el residual contra OU mejoró en vez de
empeorar. `roman_light_curve` también mide desde el máximo B.

Lo que sí había que arreglar: el docstring de `rendered_band_curves` **afirmaba** que su eje y
`(mjd − peak_mjd)/(1+z)` del padre eran el mismo eje, "which is what makes an overlay of the two
meaningful". Eso pasó a ser falso con la migración y está corregido. No hay ningún overlay de esos
en el código; si alguien hace uno, ahora el docstring le avisa.
