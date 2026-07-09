# Kilonova transformer — entrenamiento (OpenUniverse, GPU local)

Entrena `KilonovaTransformer` sobre las **ventanas tempranas de OpenUniverse**: contaminantes
Roman (`early_windows_{deep,wide}.parquet`) + kilonovas LANL inyectadas
(`kilonova_windows_{deep,wide}.hdf5`), generados por el pipeline de `src/kilonova/`.
Clasificación binaria `{KN, other}` (~1.9M objetos, ~324k KN).

```
training/
├── train_local.ipynb      # ← notebook de entrenamiento (RTX 4060 local)
├── train_lightning.py     # LightningModule: AdamW, class weights, checkpoints, early stopping
├── model.py               # KilonovaTransformer (encoder-only, ver notebooks/transformer_architecture.ipynb)
├── openuniverse_data.py   # DataLoaders: split 90/5/5, KN por simulation_id, contaminantes por object_id
├── evaluation.ipynb       # métricas, matriz de confusión, contaminantes, model soup
├── docs/                  # tokens, arquitectura, parámetros KN, desglose de la clase "other"
└── requirements.txt
```

## Correr

```bash
python train_lightning.py --data-dir <dir con parquet/hdf5> --epochs 50
```

Los datos viven fuera del repo (ver `configs/paths.yaml` / DVC); los checkpoints y
`lightning_logs/` se quedan locales.

## Qué entrena

- **Clases** `{KN, other}` con pesos inversos a la frecuencia.
- **Split** 90/5/5: KN separadas por `simulation_id`, contaminantes por `object_id`
  estratificado por clase — sin fuga entre train/val/test.
- **Selección de modelo** por `val_acc_noz` (KN y contaminantes son casi disjuntos en
  redshift, así que `val_acc_z` es un artefacto).
- `evaluation.ipynb` incluye el *model soup* de los mejores checkpoints.

## Hourglass

La fotometría de Hourglass ya **no** se usa para generar datos de entrenamiento; solo sirve
para validar la receta de ruido (`notebooks/validation/validate_noise_recipe.ipynb`,
`docs/hourglass_noise_recipe.md`).
