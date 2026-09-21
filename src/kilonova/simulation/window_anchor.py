"""Anclar una ventana ya generada a su plantilla, y leerle el brillo.

QUE PROBLEMA RESUELVE. El training set tiene 717 864 contaminantes de OpenUniverse, casi todos a
redshift alto, y las clases se solapan poco justo donde el clasificador las necesita: abajo. El izc
rellena esos bins re-renderizando padres, y para eso le lee el brillo a la curva de luz COMPLETA
del padre, que vive en 16 GB de hdf5 en un disco externo. Este modulo hace lo mismo a partir de la
VENTANA -- las cuatro epocas que ya estan en el parquet -- y sin abrir el hdf5:

    phase0     = (mjd - peak_mjd)/(1 + z) + C[plantilla]
    zero_point = mediana(mag_true - modelo(banda, phase0))

La fase no se busca, se calcula. `C` es el numero, uno por plantilla, que lleva el `peak_mjd` del
catalogo a la fase nativa de la plantilla; esta congelado en `template_phase_anchor.csv` y como se
midio esta en `scripts/probes_data_augmentation/`. Medido sobre 1320 objetos contra el brillo que
`intermediate_z_contaminants.measure_brightness_offset` saca del hdf5 -- una cantidad que no
depende de la fase en absoluto -- el zero point de aqui coincide con el de alla en 0.0023 mag de
mediana, y el residuo es plano entre z = 0.3 y z = 2.

EL CHEQUEO DE COBERTURA VA POR DELANTE, y eso es la mitad del punto. El metodo anterior ajustaba la
fase minimizando colores y tenia un modo de falla silencioso: la minimizacion se pegaba al borde de
la grilla renderizada y devolvia una fase plausible con un zero point malo, en el 28% de los
objetos a z > 0.5. Una formula no puede rielar, y si la fase cae fuera de la plantilla se sabe
comparando dos numeros ANTES de renderizar nada. `anchor_window` devuelve None ahi, y
`uncovered_phases` dice cuales sobran."""

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from kilonova.simulation.intermediate_z_contaminants import (
    GENTYPE_BY_LABEL,
    REFERENCE_ABSOLUTE_MAGNITUDE,
    apply_brightness_offset,
    band_magnitudes_at_phases,
    realization_from_parent,
    source_of,
)
from kilonova.simulation.openuniverse_parents import CORE_COLLAPSE_GENTYPES

PHASE_ANCHOR_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "openuniverse" / "template_phase_anchor.csv"
)

# Medidas por debajo de las cuales el zero point es una anecdota y no una mediana. Una epoca
# observa 3 bandas, asi que esto son dos epocas: con una sola, el error de fase entra entero en el
# brillo y no hay forma de verlo en el spread.
MINIMUM_MEASUREMENTS = 6


@cache
def phase_anchors(path=PHASE_ANCHOR_PATH):
    """{familia: fila de `template_phase_anchor.csv`} -- C y los bordes de cada modelo.

    Es un dato versionado y no un calculo: derivar `C` pide el hdf5 de OpenUniverse, y hacerlo por
    objeto seria recalcular 48 numeros un millon de veces."""
    table = pd.read_csv(path)
    return {_as_family(row.family): row for row in table.itertuples()}


def _as_family(value):
    """Las familias core-collapse son enteros; las otras cuatro, el nombre de la clase."""
    text = str(value)
    return int(text) if text.lstrip("-").isdigit() else text


# Las clases cuya familia es la plantilla y no la clase, derivadas y no escritas a mano: son las
# que OpenUniverse agrupa bajo un gentype core-collapse.
CORE_COLLAPSE_LABELS = frozenset(
    label for label, gentype in GENTYPE_BY_LABEL.items() if gentype in CORE_COLLAPSE_GENTYPES
)


def family_of_realization(realization):
    """`family_of` para una realizacion ya sorteada, que no lleva el gentype sino su etiqueta."""
    if realization["label"] in CORE_COLLAPSE_LABELS:
        return int(realization["parent_template_index"])
    return str(realization["label"])


def family_of(parent):
    """La clave con la que un padre busca su C.

    UNA POR PLANTILLA EN LAS CORE-COLLAPSE, UNA POR CLASE EN EL RESTO. Las 44 core-collapse son 44
    SED distintas, cada una con su fase cero; las otras cuatro clases tienen un modelo solo -- o,
    en SN Iax, un banco de 919 que comparte el cero de fase del SED base -- y una sola C las cubre.
    Medido: el residuo por clase queda en 0.003-0.013 mag, del mismo orden que el de una plantilla
    core-collapse."""
    if int(parent["gentype"]) in CORE_COLLAPSE_GENTYPES:
        return int(parent["template_index"])
    return str(parent["label"])


# Lo unico que el anclaje lee de una ventana. `observed` decide que filas cuentan y `snana_id`
# es el campo, que es como se saca un healpix sin leer el parquet entero.
WINDOW_COLUMNS = ("object_id", "snana_id", "mjd", "band", "mag_true", "observed")


def windows_by_parent(windows_directory, healpix, tiers=("deep", "wide")):
    """{parent_key: sus filas observadas} de los objetos de un healpix, de los early windows.

    Los tiers se leen en orden y el PRIMERO que trae a un objeto se queda con el: deep antes que
    wide porque observa una banda mas, y wide detras para no perder al objeto que solo esta ahi.
    El brillo es del objeto y no del tier, asi que cual de los dos lo mide no cambia la respuesta
    -- lo que cambia es cuantas medidas entran en la mediana."""
    import pyarrow.parquet as pq

    windows = {}
    for tier in tiers:
        path = Path(windows_directory) / f"early_windows_{tier}.parquet"
        if not path.exists():
            continue
        table = pq.read_table(
            path, columns=list(WINDOW_COLUMNS), filters=[("snana_id", "==", f"snana_{healpix}")]
        ).to_pandas()
        table = table[table["observed"] & np.isfinite(table["mag_true"])]
        for parent_key, block in table.groupby("object_id", sort=False):
            windows.setdefault(str(parent_key), block)
    return windows


def parents_with_windows(windows_directory, tiers=("deep", "wide")):
    """Los `parent_key` que tienen ventana, o sea los padres que este camino puede anclar.

    QUE ESTO NO SEA TODO EL CATALOGO ES LA LIMITACION DEL METODO: solo entran los objetos que el
    survey detecto a SU redshift, y a cada z los que faltan son los mas debiles. Se asume: son 48
    modelos repartidos en 1.3 millones de objetos, asi que la parte debil de cada plantilla llega
    igual por los objetos que si se detectaron mas abajo."""
    import pyarrow.parquet as pq

    keys = set()
    for tier in tiers:
        path = Path(windows_directory) / f"early_windows_{tier}.parquet"
        if path.exists():
            keys.update(pq.read_table(path, columns=["object_id"])["object_id"].to_pylist())
    return keys


def window_phases(mjds, peak_mjd, redshift, anchor):
    """La fase nativa de la plantilla en cada `mjd` observado. Tres restas y una division."""
    return (np.asarray(mjds, dtype=float) - float(peak_mjd)) / (1.0 + float(redshift)) + float(anchor.C)


def uncovered_phases(phases, anchor):
    """Las fases que la plantilla no tiene. Vacio = se puede anclar; esto se mira ANTES de render."""
    phases = np.asarray(phases, dtype=float)
    return phases[(phases < float(anchor.minphase)) | (phases > float(anchor.maxphase))]


def brightness_from_window(realization, window, phases, cosmology=None):
    """(offset, spread, medidas) del brillo del objeto, leido de SU ventana.

    Misma convencion que `measure_brightness_offset`, con la que se puede comparar objeto a objeto:
    el offset es magnitudes respecto de `REFERENCE_ABSOLUTE_MAGNITUDE`, positivo = mas debil. El
    modelo se renderiza al redshift al que la ventana FUE OBSERVADA, no a aquel al que se la va a
    mover: lo que se esta midiendo es el brillo intrinseco del padre.

    El spread no corrige nada, igual que alla. Es el desacuerdo entre epocas y bandas de un numero
    que deberia ser uno solo, o sea lo que queda de error de fase, y sirve de bandera por objeto:
    las plantillas cuyo `C` estaba mal se delataron por ahi antes de que nadie mirase el hdf5."""
    redshift = float(realization["parent_redshift"])
    at_reference = dict(realization, peak_absolute_magnitude=REFERENCE_ABSOLUTE_MAGNITUDE)
    rendered = band_magnitudes_at_phases(at_reference, redshift, phases, cosmology)
    phase_index = {float(phase): index for index, phase in enumerate(phases)}
    differences = []
    for row in window.itertuples():
        magnitudes = rendered.get(row.band)
        index = phase_index.get(float(row.phase))
        if magnitudes is None or index is None or not np.isfinite(row.mag_true):
            continue
        difference = float(row.mag_true) - float(magnitudes[index])
        if np.isfinite(difference):
            differences.append(difference)
    if len(differences) < MINIMUM_MEASUREMENTS:
        return float("nan"), float("nan"), len(differences)
    return (float(np.median(differences)),
            float(max(differences) - min(differences)),
            len(differences))


def brightness_for_realization(realization, window, anchor, cosmology=None):
    """(offset, spread, medidas) de una realizacion ya sorteada, con la cobertura POR DELANTE.

    Devuelve (nan, nan, 0) cuando la plantilla no cubre alguna fase observada o cuando quedan menos
    de `MINIMUM_MEASUREMENTS` magnitudes utiles -- las dos razones para descartar el objeto. Es el
    nucleo que comparten `anchor_window`, que parte de una fila del catalogo, y
    `measure_population_brightness_from_windows`, que parte de una poblacion ya sorteada."""
    nothing = (float("nan"), float("nan"), 0)
    if window is None or not len(window):
        return nothing
    phases = window_phases(window["mjd"], realization["parent_peak_mjd"],
                           realization["parent_redshift"], anchor)
    unique_phases = np.unique(phases)
    if len(uncovered_phases(unique_phases, anchor)):
        return nothing
    # Una segunda red, por si la tabla y el archivo de plantillas se desincronizan: los bordes que
    # decidieron la cobertura son los del CSV, y quien renderiza es la fuente.
    source = source_of(realization)
    if unique_phases.min() < source.minphase() or unique_phases.max() > source.maxphase():
        return nothing
    return brightness_from_window(realization, window.assign(phase=phases), unique_phases, cosmology)


def measure_population_brightness_from_windows(population, windows, anchors=None, cosmology=None):
    """Escribe el brillo de cada realizacion cuyo padre tiene ventana, in place.

    El espejo de `intermediate_z_contaminants.measure_population_brightness`, que hace lo mismo
    abriendo los 16 GB de hdf5. Devuelve cuantas se quedaron sin brillo: un padre sin ventana, o
    uno cuyas fases se salen de su plantilla, no se puede re-renderizar y el llamador lo descarta.

    La medida se cachea POR PADRE. Un padre se re-renderiza muchas veces -- las clases raras,
    decenas -- y todas sus copias tienen el mismo modelo al mismo redshift de origen, asi que el
    render que cuesta la medida se hace una vez."""
    anchors = phase_anchors() if anchors is None else anchors
    unmeasured = 0
    measured_by_parent = {}
    for realization in population:
        parent_key = realization["parent_key"]
        if parent_key not in measured_by_parent:
            anchor = anchors.get(family_of_realization(realization))
            measured_by_parent[parent_key] = (
                (float("nan"), float("nan"), 0) if anchor is None
                else brightness_for_realization(realization, windows.get(parent_key), anchor, cosmology)
            )
        offset, spread, measurements = measured_by_parent[parent_key]
        if measurements == 0:
            unmeasured += 1
            continue
        apply_brightness_offset(realization, offset, spread, measurements)
    return unmeasured


def anchor_window(window, parent, redshift, source_by_template_index, random_generator,
                  index=0, anchors=None, cosmology=None):
    """El padre de `window`, listo para renderizar a `redshift`, con el brillo de su propia ventana.

    `index` es el del izc: numera las re-renderizaciones de un padre y siembra su ruido.

    `window` son las filas OBSERVADAS de un objeto en `early_windows_{tier}.parquet` -- necesita
    `mjd`, `band` y `mag_true`, y nada mas -- y `parent` su fila de `read_parent_catalog`. El
    resultado es lo mismo que devuelve `realization_from_parent` seguido de
    `apply_brightness_offset`, o sea lo que `build_izc_windows` sabe renderizar.

    Devuelve None cuando la plantilla no cubre alguna de las fases observadas, o cuando quedan
    menos de `MINIMUM_MEASUREMENTS` magnitudes utiles. Las dos son razones para descartar el
    objeto, no para devolver un numero peor."""
    anchors = phase_anchors() if anchors is None else anchors
    anchor = anchors.get(family_of(parent))
    if anchor is None:
        return None
    realization = realization_from_parent(
        parent, int(index), float(redshift), source_by_template_index, random_generator
    )
    offset, spread, measurements = brightness_for_realization(
        realization, window, anchor, cosmology
    )
    if measurements == 0:
        return None
    return apply_brightness_offset(realization, offset, spread, measurements)
