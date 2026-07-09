# Definición de un token (OpenUniverse dataloader → `KilonovaTransformer`)

Fuente: `openuniverse_data.py` (alimenta a `model.py` / `train_lightning.py`).

## Qué es un "token"

Cada **medición fotométrica individual** (un objeto, un día de visita, una banda) es un
token. Una curva de luz = una secuencia de tokens (hasta 3 épocas × 6 bandas = máx. 18
tokens para KN; los contaminantes tienen hasta 4 épocas). Los tokens de un objeto se
ordenan por `(día, banda)`.

## Campos por token

| campo                  | qué es                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `delta_time`           | día de la visita menos el día "ancla" de la ventana (float, sin normalizar; se codifica luego con Time2Vec) |
| `band_index`           | índice de banda, 0–5 (ver vocabulario de bandas abajo)                  |
| `token_type_index`     | tipo de observación: `d`=detección (0), `u`=upper limit / no detectado (1), `n`=no observado en esa banda/época (2) |
| `magnitude`             | magnitud **normalizada**: `(mag_observada - MAG_MEAN) / MAG_STD` si es detección; si no hay medición, se pone `0.0` y se marca con `magnitude_mask=0` |
| `sigma_magnitude`       | error de magnitud **normalizado**: `(mag_err - SIGMA_MAG_MEAN) / SIGMA_MAG_STD`; solo las detecciones tienen error real, el resto es `0.0` con `sigma_mask=0` |
| `magnitude_mask`        | 1.0 si `magnitude` es un valor real, 0.0 si es relleno                   |
| `sigma_mask`            | 1.0 si `sigma_magnitude` es un valor real, 0.0 si es relleno             |

`MAG_MEAN`, `MAG_STD`, `SIGMA_MAG_MEAN`, `SIGMA_MAG_STD` son estadísticas globales
**ajustadas solo en el split de train** (para no filtrar información de val/test).

### Cómo se decide `token_type` y `magnitude` cruda (antes de normalizar)

- **Detección** (`detected=True` o, si falta el flag, `snr >= 5`): `token_type='d'`,
  `magnitude = mag_observed`, `sigma = mag_err`.
- **Upper limit** (observado pero no detectado): `token_type='u'`,
  `magnitude = mag_limit_5sigma` (el límite de detección a 5σ), sin `sigma` (NaN → mask 0).
- **No observado** en esa banda/época: `token_type='n'`, `magnitude` y `sigma` son NaN
  (mask 0 en ambos) — así el modelo "sabe" que esa banda no tiene información esa noche,
  en vez de simplemente omitir el token.

## Vocabulario de bandas (`band_index`)

6 bandas Roman, mapeadas a una letra (`R`, `Z`, `Y`, `J`, `H`, `F`):

| índice | letra | banda Roman |
|---|---|---|
| 0 | R | R062 |
| 1 | Z | Z087 |
| 2 | Y | Y106 |
| 3 | J | J129 |
| 4 | H | H158 |
| 5 | F | F184 |

`deep` y `wide` se combinan en un solo modelo sobre este vocabulario de 6 bandas; una
banda que un tier no observa simplemente nunca genera token (igual que una banda
realmente ausente).

## Redshift (features globales, no por token)

- `redshift`: el z verdadero del objeto (o `NaN`/0.0 si se oculta — ver dropout abajo).
- `redshift_error`: siempre `0.0` en este dataset (OpenUniverse no trae error de z).
- `has_redshift`: 1.0 si el modelo recibe z en este ejemplo, 0.0 si no.
- **Redshift dropout** (`REDSHIFT_DROPOUT_PROBABILITY = 0.50` en train): la mitad de las
  veces se oculta el z real al modelo (`has_redshift=0`), para que aprenda a clasificar
  también sin z. En validación se evalúan ambos regímenes por separado
  (`val_acc_z` vs `val_acc_noz`).
- Si no hay z (`has_redshift=False`), el modelo usa un token aprendido `no_redshift_token`
  en vez de proyectar `(redshift, redshift_error)` (ver `GlobalTokens` en `model.py`).

## Augmentation de ventana (solo train)

- `SHIFT_PROBABILITY = 0.20`: con 20% de probabilidad se desliza la ventana una época
  hacia adelante (simula detección tardía), siempre que la segunda visita tenga al menos
  una detección real.

## Etiqueta (nivel objeto, no token)

Tarea **binaria**: `GROUP_ORDER = ['other', 'KN']` → `label 0 = other` (cualquier
contaminante: SN II/Ia/Ib/Ic/Iax, TDE, SLSN-I, PISN), `label 1 = KN`.

## Batch (después de `collate_token_windows`)

Los tokens de cada objeto se apilan y se rellenan (padding) al largo máximo del batch,
generando además una `padding_mask` (True = posición de relleno, se ignora en la
atención). Es la estructura que `model.py::TokenEmbedding` consume directamente.
