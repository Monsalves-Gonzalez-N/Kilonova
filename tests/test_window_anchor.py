"""El anclaje de una ventana a su plantilla: kilonova.simulation.window_anchor.

El test que importa es el round trip: se genera una ventana con un brillo conocido, se la vuelve a
leer por donde la leeria el data augmentation -- `mjd`, `band`, `mag_true` y nada mas -- y tiene que
salir el mismo brillo. Lo demas son las dos redes del modulo: que la tabla congelada siga siendo la
del archivo de plantillas, y que una fase fuera de la plantilla se descarte en vez de anclarse mal.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from kilonova.simulation.intermediate_z_contaminants import (
    OPENUNIVERSE_ARCHIVE_PATH,
    _core_collapse_archive_order,
    apply_brightness_offset,
    build_izc_windows,
    core_collapse_source_by_template_index,
    measure_brightness_offset,
    peak_phase,
    realization_from_parent,
    rendered_peak_magnitudes,
    source_of,
)
from kilonova.simulation.window_anchor import (
    MINIMUM_MEASUREMENTS,
    anchor_window,
    brightness_from_window,
    family_of,
    family_of_realization,
    measure_population_brightness_from_windows,
    phase_anchors,
    uncovered_phases,
    window_phases,
)
from test_intermediate_z_contaminants import TEST_BRIGHTNESS_OFFSET, synthetic_parent_catalog

galsim = pytest.importorskip("galsim")

TEST_REDSHIFT = 0.4


def one_core_collapse_parent(template_index=None):
    """Un padre core-collapse con `peak_mjd` en cero, que es donde el eje del izc pone el maximo."""
    catalog = synthetic_parent_catalog()
    core_collapse = catalog[catalog.gentype.isin((21, 26, 32))]
    row = (core_collapse.iloc[0] if template_index is None
           else core_collapse[core_collapse.template_index == template_index].iloc[0]).copy()
    row["peak_mjd"] = 0.0
    row["redshift"] = TEST_REDSHIFT
    return catalog, row


def anchor_of(realization):
    """La fila que `phase_anchors` devolveria si el eje de tiempo fuese el que genera el izc.

    `build_izc_windows` escribe en `mjd` los dias OBSERVADOS desde el maximo en B, asi que sobre esa
    ventana el papel de C lo hace `peak_phase`: la fase de la plantilla donde cae ese maximo. La
    aritmetica que se prueba es la misma que en el parquet de OpenUniverse, donde el cero es el
    `peak_mjd` del catalogo."""
    source = source_of(realization)
    return SimpleNamespace(C=peak_phase(realization),
                           minphase=float(source.minphase()),
                           maxphase=float(source.maxphase()))


def one_window(random_generator=None):
    catalog, parent = one_core_collapse_parent()
    by_index = core_collapse_source_by_template_index(catalog)
    random_generator = random_generator or np.random.default_rng(3)
    realization = realization_from_parent(parent, 0, TEST_REDSHIFT, by_index, random_generator)
    apply_brightness_offset(realization, TEST_BRIGHTNESS_OFFSET, 0.0, 5)
    window, _ = build_izc_windows([realization], "deep")
    return catalog, parent, by_index, realization, window[window.observed]


def test_the_frozen_table_is_the_template_archive_itself():
    """C es un dato versionado, y sus bordes tienen que ser los del archivo que se renderiza."""
    anchors = phase_anchors()
    names, _ = _core_collapse_archive_order()
    by_template = sorted((key, anchor) for key, anchor in anchors.items() if isinstance(key, int))
    assert len(by_template) == len(names)
    with np.load(OPENUNIVERSE_ARCHIVE_PATH) as archive:
        for position, (_, anchor) in enumerate(by_template):
            phase = archive[f"phase_{position}"]
            assert anchor.source == names[position]
            assert anchor.minphase == pytest.approx(float(phase.min()), abs=1e-2)
            assert anchor.maxphase == pytest.approx(float(phase.max()), abs=1e-2)
            assert np.isfinite(anchor.C)


def test_every_class_the_sample_renders_has_an_anchor():
    """Las cuatro clases que no son core-collapse van por clase: un modelo, una C."""
    anchors = phase_anchors()
    for label in ("SN Ia", "SN Iax", "SLSN-I", "TDE"):
        assert label in anchors, f"{label} no tiene C y el izc lo genera"
        assert np.isfinite(anchors[label].C)
        assert anchors[label].minphase < anchors[label].maxphase


def test_the_family_is_the_template_for_core_collapse_and_the_class_for_the_rest():
    assert family_of({"gentype": 32, "template_index": 731, "label": None}) == 731
    assert family_of({"gentype": 21, "template_index": 703, "label": None}) == 703
    # Un SN Iax lleva `template_index` 1..919 y NO es una familia: su banco comparte el cero.
    assert family_of({"gentype": 12, "template_index": 42, "label": "SN Iax"}) == "SN Iax"
    assert family_of({"gentype": 10, "template_index": 0, "label": "SN Ia"}) == "SN Ia"


def test_the_phase_is_an_arithmetic_and_not_a_search():
    anchor = SimpleNamespace(C=-5.0, minphase=-10.0, maxphase=100.0)
    # Dos epocas separadas por 10 dias observados son 5 dias de plantilla a z = 1.
    phases = window_phases([62000.0, 62010.0], peak_mjd=62000.0, redshift=1.0, anchor=anchor)
    assert phases == pytest.approx([-5.0, 0.0])


def test_a_phase_outside_the_template_is_named_and_not_anchored():
    """El modo de falla que el ajuste por colores tenia en silencio: aqui se sabe antes de render."""
    anchor = SimpleNamespace(C=0.0, minphase=-10.0, maxphase=50.0)
    outside = uncovered_phases([-12.0, -9.0, 0.0, 60.0], anchor)
    assert sorted(outside) == [-12.0, 60.0]
    assert len(uncovered_phases([-9.0, 0.0], anchor)) == 0


def test_the_brightness_round_trips_through_the_window():
    """Lo unico que entra es (mjd, band, mag_true); lo que sale es el brillo con que se genero."""
    catalog, parent, by_index, realization, window = one_window()
    anchors = {int(parent.template_index): anchor_of(realization)}
    anchored = anchor_window(window, parent, TEST_REDSHIFT, by_index,
                             np.random.default_rng(3), anchors=anchors)
    assert anchored is not None
    assert anchored["brightness_offset"] == pytest.approx(TEST_BRIGHTNESS_OFFSET, abs=0.05)
    assert anchored["brightness_bands"] >= MINIMUM_MEASUREMENTS
    # El spread es el error de fase que queda, y es lo que delata a un `C` malo.
    assert anchored["brightness_residual"] < 0.15


def test_the_window_brightness_is_the_brightness_the_hdf5_path_measures():
    """El mismo numero por los dos caminos: el pico de la curva completa y las cuatro epocas."""
    catalog, parent, by_index, realization, window = one_window()
    anchor = anchor_of(realization)
    phases = np.unique(window_phases(window["mjd"], parent.peak_mjd, parent.redshift, anchor))
    offset, spread, measurements = brightness_from_window(
        realization, window.assign(phase=window_phases(
            window["mjd"], parent.peak_mjd, parent.redshift, anchor)), phases)
    peaks = rendered_peak_magnitudes(realization, TEST_REDSHIFT)
    by_peak, _, _ = measure_brightness_offset(realization, peaks)
    assert offset == pytest.approx(by_peak, abs=0.05)
    assert measurements >= MINIMUM_MEASUREMENTS
    assert spread < 0.15


def test_an_object_whose_phases_fall_before_the_template_is_dropped():
    catalog, parent, by_index, realization, window = one_window()
    anchor = anchor_of(realization)
    # El padre explota mas tarde de lo que dice el catalogo: las fases se van por delante del borde.
    moved = parent.copy()
    moved["peak_mjd"] = parent.peak_mjd + 500.0
    anchors = {int(parent.template_index): anchor}
    assert anchor_window(window, moved, TEST_REDSHIFT, by_index,
                         np.random.default_rng(3), anchors=anchors) is None


def test_a_template_with_no_anchor_is_dropped():
    catalog, parent, by_index, realization, window = one_window()
    assert anchor_window(window, parent, TEST_REDSHIFT, by_index,
                         np.random.default_rng(3), anchors={}) is None


def test_the_anchor_does_not_need_the_hdf5():
    """Lo que el modulo lee de la ventana, y nada mas: si sobra una columna, que falle aqui."""
    catalog, parent, by_index, realization, window = one_window()
    anchors = {int(parent.template_index): anchor_of(realization)}
    minimal = window[["mjd", "band", "mag_true"]]
    anchored = anchor_window(minimal, parent, 0.1, by_index, np.random.default_rng(3),
                             anchors=anchors)
    assert anchored is not None
    assert anchored["redshift"] == pytest.approx(0.1)
    assert anchored["parent_redshift"] == pytest.approx(TEST_REDSHIFT)


def test_a_population_takes_its_brightness_from_its_parents_window():
    """El camino que usa la stage: una poblacion sorteada, y el brillo de la ventana del padre.

    El espejo de `measure_population_brightness`, que abre los 16 GB de hdf5 para lo mismo."""
    catalog, parent, by_index, realization, window = one_window()
    anchors = {family_of_realization(realization): anchor_of(realization)}
    # Tres copias del MISMO padre a tres redshifts, que es como el izc lo sortea.
    population = [
        realization_from_parent(parent, index, redshift, by_index, np.random.default_rng(index))
        for index, redshift in enumerate((0.05, 0.1, 0.2))
    ]
    unmeasured = measure_population_brightness_from_windows(
        population, {parent.parent_key: window}, anchors=anchors
    )
    assert unmeasured == 0
    for one in population:
        assert one["brightness_offset"] == pytest.approx(TEST_BRIGHTNESS_OFFSET, abs=0.05)
        assert one["brightness_bands"] >= MINIMUM_MEASUREMENTS


def test_a_parent_without_a_window_is_counted_and_left_alone():
    """Un padre que el survey no detecto no tiene ventana, y este camino no puede inventarselo."""
    catalog, parent, by_index, realization, window = one_window()
    anchors = {family_of_realization(realization): anchor_of(realization)}
    population = [realization_from_parent(parent, 0, 0.1, by_index, np.random.default_rng(0))]
    assert measure_population_brightness_from_windows(population, {}, anchors=anchors) == 1
    assert not np.isfinite(population[0]["brightness_offset"])
