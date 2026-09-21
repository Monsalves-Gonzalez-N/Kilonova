# Sondas del data augmentation — exploratorio

**No son parte del pipeline.** Se corren a mano, escriben CSV al lado, y existen para medir el
método antes de portarlo a `src/`. Plan: `docs/plan_data_augmentation_training_set.md`.

`anchor_measurement.csv` y `template_C_2026-09-21.csv` son de la medición del **2026-09-21**, ya
con los artefactos regenerados: ésos sí son el resultado vigente, y de ellos sale la tabla
congelada `data/openuniverse/template_phase_anchor.csv`.

Los demás `.csv` son los resultados del **2026-09-18 ANTES** de regenerar
`cc_templates.npz` (que tenía el `ceil` en el borde de fase) y antes de ampliar la ventana de
render. Sirven como línea base contra la que comparar; **no como resultado vigente**.

| script | qué mide |
|---|---|
| `batch_color_phase.py` | ajuste de fase por colores sobre una muestra, separado por plantilla |
| `phase_truth.py` | la fase ajustada contra la verdad de OU (`peak_mjd` del catálogo padre) |
| `fast_anchor.py` | la fórmula analítica, calibrando `C` en una mitad y midiendo en la otra |
| `fast_anchor2.py` | variante que pega la fase al borde temprano (probado: **no funciona**, 0.195 mag) |
| `C_from_edge.py` | `C` derivada del borde del modelo de OU, sin muestra de calibración |
| `predict_C.py` | si `C` es un pico de banda de la plantilla (probado: **no lo es**) |
| `minphase_table.py` | cobertura de fase por plantilla: minphase, maxphase, span, `C`, % cubierto |
| `window_recovery.py` | cuántos objetos recupera ampliar la ventana de render |
| `ceil_loss.py` | fase perdida por el `np.ceil`, contra los `.SED` crudos del release |
| `anchor_measurement.py` | LA MEDICION VIGENTE: barre un `delta` alrededor de C y lo fija contra el brillo del izc |
| `freeze_C_table.py` | congela `data/openuniverse/template_phase_anchor.csv` desde lo anterior |
| `regenerate_at_low_z.py` | el metodo completo sobre un objeto: fase analitica -> zero point -> re-render a z bajo |
| `test_color_phase.py` | un solo objeto, para inspeccionar a mano |

Los CSV **por objeto** de los barridos no estan en git (ver `.gitignore`): son megabytes y se
regeneran corriendo la sonda. Lo que se versiona es el resumen por plantilla y, sobre todo, la
tabla congelada, que vive en `data/openuniverse/template_phase_anchor.csv`.

Rutas que asumen: `~/Dropbox/Kilonova/openuniverse2025` (catálogos + hdf5),
`/home/nicolas/nico/openuniverse_models/` (el tar oficial desempacado), y el env `kn_class`.
La constante `SCRATCH` de cada script apunta al scratchpad de la sesión que los escribió: hay que
cambiarla a este directorio.
