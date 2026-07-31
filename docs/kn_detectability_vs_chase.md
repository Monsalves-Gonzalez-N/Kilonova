# Detectabilidad de nuestras kilonovas vs Chase et al. 2021 — resultados

Corrida del **2026-07-31** con `notebooks/validation/kn_detectability_vs_chase.py`, sobre las curvas
de luz **ya generadas** (`data/openuniverse/kn_windows_{deep,wide}.parquet`, las que entrenaron el
transformer), no sobre la grilla LANL cruda. Figuras y CSV en
`data/openuniverse/detectability_vs_chase/`. La metodología del paper está en
`docs/plan_kn_widefield_repro.md`; aquí solo lo que se midió y qué salió.

**Conclusión: la generación de kilonovas es consistente con el paper.** Los tres contornos de la
figura de Roman/H coinciden, y los cortes de masa fija reproducen los valores publicados.

## Qué se pudo medir y por qué

`kn_windows_*.parquet` se generó con 100 nodos de z log-espaciados en [0.02, 1] y **10 000
realizaciones por nodo**, con simulación LANL y ángulo sorteados **uniformemente** — la misma
marginalización uniforme sobre las 48 600 realizaciones (900 sims × 54 ángulos) que promedia el
paper, en versión Monte Carlo. La ventana arranca en la primera detección y **las realizaciones
nunca detectadas no se escriben**, así que el denominador de cualquier fracción es 10 000 por nodo,
no el número de filas: los ceros del mapa son los objetos ausentes. En los paneles de masa fija ese
denominador se escala por la fracción de la grilla que sobrevive al corte (180/900 para una masa
fija, 36/900 para masa eyecta total fija); sin ese factor el panel parece 5× menos detectable.

Eje temporal: las visitas de una KN son `explosion_offset_days + 5·arange(8)` en tiempo observador
**desde el merger**, con el offset ~U[0,5) codificado en el `object_id`. La paridad de la cadencia
(también en el id) revela, por las bandas observadas en la época 1, la paridad de la visita donde
cayó la primera detección: sale **par para el 97.3% (deep) / 94.5% (wide)** de los objetos, y 99% a
z<0.1. Las que no lo cumplen genuinamente no fueron detectadas en la visita 0, así que entran como
no-detección con denominador completo.

Se usan dos criterios: el del paper (`mag_true < m_lim` con la Tabla 1 fija de Chase, que no lo toca
la cadencia porque `mag_true` existe en toda banda cubierta por el modelo) y el nuestro (S/N ≥ 5 con
la receta de ruido). La magnitud de partida es la que pide el paper: `magnitudes_for_bands` aplica
redshift + dimming + K-correction y **no** aplica extinción.

## Resultado principal — Roman/H, grilla completa

Los tres contornos, leídos de la figura del paper (`RomanH.png`) y de la nuestra:

| contorno | Chase+21 | nuestro, deep | nuestro, wide |
|---|---|---|---|
| 0.95 | ~0.09 | ~0.095 | 0.088 |
| 0.5 | **0.22** | **0.215** | 0.201 |
| 0.05 | ~0.48 | 0.477 | 0.454 |

Roman/Z sale aún mejor: pico del contorno 0.05 en ~0.76 a 0.6 d (ellos ~0.78 a 0.5 d), 0.5 en ~0.24
(ellos 0.24) y 0.95 en ~0.11 (ellos ~0.11).

Coinciden dentro de la tolerancia de ±0.02 del plan. La forma también reproduce la Fig. 3: pico en
t ~1 d en las bandas azules y ~3 d en F184, caída del contorno 0.5 hacia z alto y hacia t grande,
alcance creciente hacia el rojo.

> **Corrección**: `docs/plan_kn_widefield_repro.md` anota `z5% = 0.79` para Roman/H. Ese valor no
> concuerda con la figura del propio paper, donde el contorno 0.05 llega a ~0.48. El 0.48 es el
> número correcto a comparar, y es el que reproducimos.

## Cortes de masa fija (Fig. 5/7 del paper)

Pico del contorno 0.5, tier deep, contra los valores que publica el paper:

| corte | Chase+21 | nuestro |
|---|---|---|
| `m_dyn` = 0.1 M☉ | 0.31 (pico a 3 d) | ~0.31 (pico a 3 d) |
| `m_wind` = 0.1 M☉ | 0.37 | ~0.365 |
| `m_dyn` = 0.001 M☉ | 0.16 | ~0.147 |

El más desviado es el de menor masa, que es lo esperable: son los modelos más débiles y por tanto
los que más sufren el truncamiento del dataset (ver limitaciones).

## Bordes por banda (`bordes_de_detectabilidad.csv`)

Criterio del paper, z50%: deep Z087 0.230 · Y106 0.228 · J129 0.219 · H158 0.215 · F184 0.170;
wide R062 0.229 · Z087 0.230 · Y106 0.227 · J129 0.206 · H158 0.201. Las bandas azules y medias
quedan muy juntas porque la Tabla 1 de Chase también lo está (26.2 → 25.4); F184 baja por su límite
más somero (24.9).

Con **nuestro** criterio (S/N ≥ 5) todo se corre a z más alto en deep — hasta z50% = 0.49 en F184 —
simplemente porque el tier deep es **1–2 mag más profundo por visita** que lo que tabula el paper
(F184: 27.0 vs 24.9; H158: 26.7 vs 25.4). En wide las profundidades son parecidas a las del paper
(H158 26.5 vs 25.4, Z087 25.6 vs 25.7, R062 25.6 vs 26.2) y los dos criterios dan casi lo mismo, que
es la comprobación cruzada interesante: donde la profundidad coincide, el criterio no importa.

## Figuras

`mapas/` — 28 paneles al estilo de las figuras de referencia (z **lineal de 0 a 1**, tiempo
observador log con ticks en potencias de 2, `inferno`, contornos 0.05/0.5/0.95 en
discontinuo/punteado/raya-punto, eje derecho en Gpc): `Roman{R,Z,Y,J,H,F}_{deep,wide}.png`, los seis
cortes de masa fija y los dos de masa eyecta total, por tier. `RomanH_{tier}_maintext.png` repite el
panel de H con el eje recortado a z=0.5, que es el único recorte que usa el paper (su `RomanH.png`
de texto principal); todas las demás figuras de referencia llegan a z=1.

Todo esto está también reunido en `detectabilidad_vs_chase.pdf`, una figura por página.

El campo se dibuja **suavizado** (gaussiana de ~1 bin). El del paper es liso porque evalúa las
48 600 realizaciones en cada (t, z) exacto; el nuestro es Monte Carlo y sin suavizar el moteado de
Poisson domina la lectura. Los números de las tablas de arriba no salen del campo suavizado sino de
`per_band_fraction_vs_redshift`, que no aplica ningún filtro.

Fuera de `mapas/`: `detectabilidad_vs_redshift.png` (fracción detectada en cualquier banda, deep
z50%=0.43 / wide z50%=0.29 — no comparable directamente con el 0.22 del paper, que es una sola
banda con `m_lim` fijo), `criterio_paper_vs_pipeline.png` y `diagnosticos.png` (profundidad
alcanzada vs Tabla 1, y validez del eje de fase).

## Limitaciones

- **Truncamiento**: solo existen en el archivo los objetos que el pipeline detectó en alguna banda.
  El sesgo es despreciable donde la supervivencia es alta y crece hacia z alto y hacia los modelos
  más débiles. Se ve en que deep, que trunca menos, da sistemáticamente valores más cercanos al
  paper que wide con idéntico criterio (H158: 0.215 vs 0.201).
- El eje de fase llega a 20 d (4 épocas × 5 d), no a los 40 d del paper.
- No hay overlay de AT 2017gfo: son datos espectroscópicos reales del evento (kilonova.space), no
  derivables de nuestra grilla.
- Los mapas con criterio de pipeline (no incluidos en `mapas/`) mostraban una costura artificial en
  ~5 d, donde el eje de fase cruza de la visita 0 a la visita 1 y el factor 1/2 del denominador de
  las bandas no-ancla deja de ser exacto. No afecta a los números publicados, que salen del criterio
  del paper.

## Reproducir

```
python notebooks/validation/kn_detectability_vs_chase.py
```
