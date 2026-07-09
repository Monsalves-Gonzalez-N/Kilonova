# Kilonova transformer — paquete de entrenamiento (Colab GPU, datos reales)

Entrena `KilonovaTransformer` sobre la **fotometría real de Hourglass**. Listo para correr en
la GPU gratis de Colab — no necesita los espectros LANL de 10 GB.

```
kilonova_colab/
├── train_colab.ipynb          # ← abrir esto en Colab
├── train.py                   # loop de entrenamiento (GPU, class weights, checkpoint)
├── model.py                   # KilonovaTransformer (extraído de transformer_architecture.ipynb)
├── hourglass_data.py          # DataLoaders Hourglass (extraído de hourglass_eda.ipynb)
├── requirements.txt
├── data/dust_generation/
│   ├── hourglass_objects.parquet      # 5.6 MB
│   └── hourglass_photometry.parquet   # 141 MB
└── notebooks/                 # los notebooks originales, solo de referencia
    ├── transformer_architecture.ipynb
    └── hourglass_eda.ipynb
```

`model.py` y `hourglass_data.py` son el **código núcleo extraído verbatim** de los dos
notebooks (sin las celdas de EDA/plots). El batch que produce `hourglass_data.collate_token_windows`
tiene exactamente las llaves que consume `KilonovaTransformer.forward()`.

## Correr en Colab (GPU)

1. Sube `kilonova_colab.zip` a Google Drive.
2. Nuevo notebook Colab → **Runtime → Change runtime type → GPU (T4)**.
3. Abre `train_colab.ipynb` (o pega sus celdas) y ejecútalas. La primera monta Drive,
   descomprime y hace `%cd` a la carpeta; el resto entrena.

Equivalente en una línea desde una celda:
```python
!python train.py --data-dir data/dust_generation --epochs 30 --batch-size 64
```

## Qué entrena

- **Clases** `{Ia, II, other, KN}`. KN (índice 3) queda **vacío** en este set survey-only — el
  modelo aprende a separar los tres contaminantes. La inyección de KN llega después con el
  dataloader LANL (no incluido aquí); la arquitectura ya reserva el slot.
- **Split** estratificado y agrupado por `cid` (70/15/15), sin fuga entre train/val/test.
- **Normalización** de magnitud ajustada solo en train.
- **Class weights** inversos a la frecuencia (Ia/II dominan, `other` es raro).
- Guarda el mejor checkpoint por val-accuracy en `kilonova_transformer.pt`.

Verificado localmente (env `KN_class`): 44 591 objetos de train, ~75 k parámetros,
val-acc ≈ 0.72 tras 1 epoch en CPU. En la GPU de Colab cada epoch es mucho más rápido.

## NO incluido

Los espectros de KN (`lanl_spectra.parquet`, ~10 GB) y el pipeline de inyección
(`kilonova_dataloader.ipynb`, requiere astropy/dustmaps/speclite/pyphot) — demasiado grandes
para Drive/Colab free. Por eso la clase KN está vacía por ahora.
