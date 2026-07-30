# Parámetros del grid de kilonovas (LANL)

Fuente: `/home/nicolas/nico/git/Kilonova/data/dust_generation/lanl_catalog.parquet`
(900 simulaciones únicas por `simulation_id`)

## Parámetros físicos del grid (por simulación)

| parámetro            | descripción                                    | valores                         |
|-----------------------|------------------------------------------------|----------------------------------|
| `run_type`            | tipo de merger/run                              | `TP`, `TS`                       |
| `wind`                | componente de viento (wind ejecta)              | `wind1`, `wind2`                 |
| `mass_dynamical`      | masa eyectada dinámica (M☉)                     | 0.001, 0.003, 0.01, 0.03, 0.1     |
| `velocity_dynamical`  | velocidad de la eyecta dinámica (c)             | 0.05, 0.15, 0.3                   |
| `mass_wind`           | masa eyectada de viento (M☉)                    | 0.001, 0.003, 0.01, 0.03, 0.1     |
| `velocity_wind`       | velocidad de la eyecta de viento (c)            | 0.05, 0.15, 0.3                   |
| `angle_index`         | índice de ángulo de observación (0–53, 54 bins) | 0–53                              |

Nota: la eyecta se modela con **dos componentes** (dinámica + viento), cada una con su
propia masa y velocidad — no es solo "mass ejected" sino `mass_dynamical` + `mass_wind`
(idem para velocidad). El ángulo de visión (`angle_index`) es el otro parámetro geométrico
por realización.

## Parámetros por ventana/realización (no por simulación física)

Estos vienen de `kn_windows_{deep,wide}.parquet` (uno por objeto/ventana muestreada). En el parquet
no son columnas propias: van codificados en el `object_id`,
`{simulation_id}_{angle_index}_{explosion_offset:.4f}_{z:.4f}_{cadence_parity}_{noise_id}`:

| parámetro                | descripción                                              |
|----------------------------|-----------------------------------------------------------|
| `simulation_id`            | referencia a la simulación física (tabla de arriba); va primero porque el split anti-fuga lo lee de ahí |
| `angle_index`              | ángulo de observación usado en esta realización           |
| `explosion_offset_days`    | offset temporal de la explosión respecto al muestreo       |
| `redshift`                 | corrimiento al rojo cosmológico aplicado                   |
| `cadence_parity`           | paridad de la primera visita: el otro grado de libertad de la fase del merger en el ciclo de 10 d de la cadencia |
| `noise_id`                 | id de la realización; hace el `object_id` único por construcción |
| `n_detected`                | número de épocas detectadas en la ventana                 |
| `gentype`                   | código de tipo generativo (50 = KN)                        |

## Resumen

- 900 simulaciones LANL únicas, parametrizadas por `run_type`, `wind`, masa y velocidad
  dinámica/viento (5 valores de masa × 3 de velocidad × 2 componentes × wind/run_type).
- 54 ángulos de observación (`angle_index`) por simulación.
- Además de wind/masa/ángulo: falta considerar **velocidad** (dinámica y de viento) y
  **run_type** — son los parámetros que probablemente se te escapaban.
