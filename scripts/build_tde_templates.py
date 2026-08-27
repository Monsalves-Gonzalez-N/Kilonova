"""Build the izc TDE template bank from MOSFiT's tde model.

Runs in the isolated `mosfit_gen` environment; MOSFiT pins numpy <= 1.26.4 and cannot be installed
alongside the pipeline. Output is the photosphere history only -- phase, temperature, radius -- and
the Planck spectrum is built from it at load time, which keeps the file at kilobytes instead of the
hundreds of megabytes a sampled SED cube would take.
"""

import json
import os
import warnings

import h5py
import numpy as np
from mosfit.model import Model
from mosfit.printer import Printer

warnings.filterwarnings("ignore")

PHASE = np.arange(-25.0, 121.0, 1.0)
INTEGRATION_TIMES = np.arange(0.0, 2500.0, 1.0)
PARAMETERS = ["bhmass", "starmass", "b", "efficiency", "Rph0", "lphoto", "Tviscous"]

realizations = json.loads(bytes(h5py.File("tdegen/products/walkers.h5", "r")["entry_json"][:]).decode())
realizations = list(realizations.values())[0]["models"][0]["realizations"]
print(f"realizaciones sorteadas: {len(realizations)}")

model = Model(model="tde", printer=Printer(quiet=True), output_path=os.getcwd())
model.load_data(
    {},
    event_name="synthetic",
    time_list=[0.0, 400.0],
    band_list=["V"],
    user_fixed_parameters=["covariance", "redshift", 0.02, "lumdist", 87.0],
)

# MOSFiT's priors are fitting priors, deliberately uninformative, not a population model: drawn
# blind they put the peak photosphere temperature anywhere from 5e2 to 1e6 K, and only a third of
# the draws land where TDEs are actually observed. The physics (the shape of the decline, the near
# constancy of T, the R(t) history) comes from MOSFiT; the population is anchored to the observed
# temperature range of optically selected TDEs, van Velzen et al. (2021).
OBSERVED_PEAK_TEMPERATURE = (1.5e4, 5.0e4)
# Same anchor on the other axis. MOSFiT's efficiency, black hole mass and stellar mass priors put
# the peak bolometric luminosity anywhere from 1e38 to 1e45 erg/s, and half the draws are orders of
# magnitude off any TDE ever observed.
OBSERVED_PEAK_LUMINOSITY = (1.0e43, 1.0e45)

kept, dropped = (
    [],
    {"nopeak": 0, "nonfinite": 0, "tooshort": 0, "failed": 0, "temperature": 0, "luminosity": 0},
)
for realization in realizations:
    drawn = {key: float(realization["parameters"][key]["value"]) for key in PARAMETERS}
    inputs = dict(
        dense_times=INTEGRATION_TIMES,
        rest_times=INTEGRATION_TIMES,
        resttexplosion=0.0,
        Leddlim=1.0,
        **drawn,
    )
    try:
        for name in ["fallback", "viscous", "tde_photosphere"]:
            output = model._modules[name].process(**inputs)
            inputs.update(output)
            if "dense_luminosities" in output:
                inputs["luminosities"] = np.asarray(output["dense_luminosities"])
    except Exception:
        dropped["failed"] += 1
        continue

    luminosity = np.asarray(inputs["luminosities"], dtype=float)
    temperature = np.asarray(inputs["temperaturephot"], dtype=float)
    radius = np.asarray(inputs["radiusphot"], dtype=float)
    if not (np.all(np.isfinite(temperature)) and np.all(np.isfinite(radius))):
        finite = np.isfinite(temperature) & np.isfinite(radius) & np.isfinite(luminosity)
        if finite.sum() < 200:
            dropped["nonfinite"] += 1
            continue
        luminosity, temperature, radius = luminosity[finite], temperature[finite], radius[finite]
        times = INTEGRATION_TIMES[finite]
    else:
        times = INTEGRATION_TIMES
    if luminosity.max() <= 0.0:
        dropped["nopeak"] += 1
        continue

    peak_time = times[np.argmax(luminosity)]
    shifted = times - peak_time
    if shifted.min() > PHASE.min() or shifted.max() < PHASE.max():
        dropped["tooshort"] += 1
        continue
    peak_temperature = np.interp(0.0, shifted, temperature)
    if not OBSERVED_PEAK_TEMPERATURE[0] <= peak_temperature <= OBSERVED_PEAK_TEMPERATURE[1]:
        dropped["temperature"] += 1
        continue
    peak_luminosity = np.interp(0.0, shifted, luminosity)
    if not OBSERVED_PEAK_LUMINOSITY[0] <= peak_luminosity <= OBSERVED_PEAK_LUMINOSITY[1]:
        dropped["luminosity"] += 1
        continue
    kept.append(
        (
            np.interp(PHASE, shifted, luminosity),
            np.interp(PHASE, shifted, temperature),
            np.interp(PHASE, shifted, radius),
            [drawn[key] for key in PARAMETERS],
        )
    )

print(f"conservadas: {len(kept)}   descartadas: {dropped}")
if kept:
    np.savez_compressed(
        "tde_templates.npz",
        phase=PHASE.astype(np.float32),
        luminosity=np.array([k[0] for k in kept], dtype=np.float64),
        temperature=np.array([k[1] for k in kept], dtype=np.float32),
        radius=np.array([k[2] for k in kept], dtype=np.float32),
        parameters=np.array([k[3] for k in kept], dtype=np.float64),
        parameter_names=np.array(PARAMETERS),
    )
    peak_temperature = np.array([k[1] for k in kept])[:, 25]
    print(
        f"T en el pico: min={peak_temperature.min():.0f} "
        f"mediana={np.median(peak_temperature):.0f} max={peak_temperature.max():.0f} K"
    )
