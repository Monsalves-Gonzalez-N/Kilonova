# Arquitectura: KilonovaTransformer (`model.py`)

Set-transformer sobre tokens de fotometría (una curva de luz = una secuencia de
observaciones). Usado por `train_local.ipynb` (OpenUniverse, clasificación binaria
`{other, KN}`) vía el wrapper `LitKilonova` en `train_lightning.py`.

## Hiperparámetros usados en el entrenamiento actual (train_local.ipynb)

| parámetro       | valor |
|------------------|------:|
| `d_model`         | 192 |
| `num_heads`       | 6 |
| `num_layers`      | 6 (bloques encoder) |
| `d_feedforward`   | 768 (4× `d_model`, ratio estándar transformer) |
| `dropout`         | 0.1 |
| `num_classes`     | 2 (`other`, `KN`) |

(Los defaults de `LitKilonova`/`model.py` son más chicos —`d_model=128, num_heads=4,
num_layers=4, d_feedforward=512`— pero el notebook los sobreescribe con los valores de
arriba, un modelo más grande, ya que la GPU de 8GB tiene holgura de sobra con secuencias
≤22 tokens.)

## Pipeline de tokens (entrada)

Cada observación fotométrica es un **token**, con features:

- **Banda** (`band_index`): embedding de 16 dim, vocabulario de 6 bandas Roman
  (R062, Z087, Y106, J129, H158, F184).
- **Tipo de token** (`token_type_index`): embedding de 16 dim, 3 tipos
  `{d=detección, u=upper limit/no detectado, n=?}`.
- **Magnitud**: proyección lineal de 4 valores → `d_model`
  (`magnitude`, `sigma_magnitude`, `magnitude_mask`, `sigma_mask`).
- **Tiempo** (`delta_time`): **Time2Vec** — un término lineal (carrier de decline-rate) +
  varias sinusoides de baja frecuencia (6 frecuencias, periodos ~10–60 días sin escalar),
  proyectado a `d_model`.

`content = LayerNorm(Linear(concat(banda, tipo, magnitud)))`, y luego se **suma** la
codificación temporal (`content + time_encoding(delta_time)`).

## Tokens globales

- **`[CLS]`**: token aprendido, se usa su salida final para clasificar.
- **Token de redshift**: si el objeto tiene z conocido, se proyecta `(redshift,
  redshift_error)` a `d_model`; si no, se usa un token aprendido `no_redshift_token`
  (permite entrenar/evaluar en los regímenes "con z" y "sin z" del mismo modelo).

La secuencia de entrada al encoder es: `[CLS, token_z, token_1, token_2, ..., token_N]`
(+ dropout de entrada), con una máscara de padding para ignorar tokens inválidos.

## Encoder

- Pre-norm: `x = x + Dropout(Attention(LayerNorm(x)))`, luego
  `x = x + FFN(LayerNorm(x))` (sin dropout extra en la conexión residual del FFN).
- **Multi-head attention** estándar (scaled dot-product, escala `1/sqrt(head_dim)`),
  implementada a mano (no `nn.MultiheadAttention`).
- **FFN**: `Linear(d_model→d_feedforward) → GELU → Dropout → Linear(d_feedforward→d_model)`.
- `num_layers` bloques idénticos apilados.

## Cabeza de clasificación

`classification_head(LayerNorm(encoded[:, 0]))` — toma solo la salida del token `[CLS]`
y produce los logits de las `num_classes` clases.

## Notas de entrenamiento (train_local.ipynb)

- Optimizador: AdamW, scheduler coseno con warmup (5 épocas), hasta 150 épocas con
  EarlyStopping (patience 25).
- Selección de modelo por `val_acc_noz` (sin usar z), porque KN y contaminantes están casi
  disjuntos en redshift y `val_acc_z` resulta artificialmente perfecto.
- Se guardan los top-5 checkpoints para promediar pesos ("model soup") en vez de depender
  de una sola época.
- Pesos de clase con `mode='sqrt'` para suavizar el desbalance `other` (~1.6M) vs
  `KN` (~324k) — ver `other_class_breakdown.md` y `kilonova_parameters.md`.
