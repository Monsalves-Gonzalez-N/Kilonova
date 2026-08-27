"""GIF de curvas de luz para la presentacion, en dos layouts:

- "event" (default): UN evento por frame, sus dos tiers lado a lado (deep | wide), rotando entre
  clases (KN, SN II, SN Ia, SN Ic, SN Ib, KN, ...).
- "grid": la figura estatica completa (5 clases x 2 tiers) con otros objetos en cada frame.

Reusa las funciones del notebook openuniverse_hdf5_lightcurve_error.ipynb ejecutando sus celdas,
para no duplicar la receta de ruido ni el dibujo de los paneles. Los ejes se autoescalan en cada
frame igual que en la figura estatica: a 4 s por frame el ojo lee cada frame como una figura
aparte, no como una animacion.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator
from PIL import Image

# Ticks por eje en el layout "event". Fijarlos evita que la densidad de ticks cambie de un frame al
# siguiente (300 dias contra 15).

NOTEBOOK_PATH = Path(__file__).with_name("openuniverse_hdf5_lightcurve_error.ipynb")
NOTEBOOK_CELLS = [1, 3, 11, 12]
# La celda 12 termina construyendo y dibujando la figura del paper (minutos de seleccion); aca solo
# se necesitan sus definiciones.
CELL_12_STOP_MARKER = "class_lightcurve_rows, paper_figure_objects, class_census"

# El GIF NO sigue a PANEL_CLASSES del notebook: la figura del paper crecio a las 10 clases del
# catalogo y este grid es de 5x2 (una clase por fila, 5 filas). Estas son sus 5 clases y su
# figsize cuadrado, independientes de lo que haga la figura estatica.
GIF_PANEL_CLASSES = ["KN", "SN II", "SN Ia", "SN Ic", "SN Ib"]
GRID_FIGSIZE = (13, 13)


def load_notebook_namespace(hdf5_path, catalog_path):
    """Ejecuta las celdas del notebook que definen la receta y el dibujo, y devuelve su namespace."""
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    namespace = {"__name__": "__notebook__", "display": lambda *args, **kwargs: None}
    for cell_index in NOTEBOOK_CELLS:
        source = "".join(notebook["cells"][cell_index]["source"])
        if CELL_12_STOP_MARKER in source:
            source = source[: source.index(CELL_12_STOP_MARKER)]
        exec(compile(source, f"<cell {cell_index}>", "exec"), namespace)
        if cell_index == 1:
            namespace["HDF5_PATH"] = str(hdf5_path)
            namespace["CATALOG_PATH"] = str(catalog_path)
    plt.close("all")
    return namespace


def collect_openuniverse_pool(namespace, class_catalog, hdf5, target_z, min_detections, scan_limit, wanted):
    """Hasta `wanted` objetos de la clase cerca de target_z, bien detectados en LOS DOS tiers.

    Mismo criterio que select_openuniverse_example del notebook, pero acumulando candidatos en vez
    de quedarse con uno: la seleccion es la parte cara, y un solo barrido alcanza para todo el GIF.
    """
    ordered = class_catalog.copy()
    ordered["z_distance"] = (ordered["z_CMB"] - target_z).abs()
    ordered = ordered.sort_values("z_distance").head(scan_limit)
    pool = []
    for _, row in ordered.iterrows():
        model_6band = namespace["model_from_hdf5_group"](
            hdf5[str(int(row["id"]))], {"bands": namespace["ALL_ROMAN_BANDS"]}
        )
        by_tier = namespace["photometry_in_both_tiers"](model_6band, None, int(row["id"]))
        if by_tier is None:
            continue
        detections_by_tier = {
            tier: int(by_tier[tier]["detected"].sum()) for tier in namespace["TIER_COLUMNS"]
        }
        if min(detections_by_tier.values()) < min_detections:
            continue
        pool.append({"object_id": int(row["id"]), "z_CMB": float(row["z_CMB"]), "by_tier": by_tier})
        if len(pool) >= wanted:
            break
    return pool


def collect_kilonova_pool(namespace, redshift, wanted, rng, attempts, minimum_ejecta_mass, min_detections):
    """`wanted` KN de la grilla LANL por el mismo camino que el dataset, ordenadas por deteccion.

    Se sortean `attempts` realizaciones de una vez (build_kn_models es lo caro) y se conservan las
    mejores, en vez de llamar a build_kilonova_example una vez por frame.
    """
    paths = namespace["load_paths"]()
    lanl_spectra_path = str(paths.lanl_spectra)
    lanl_catalog = namespace["load_lanl_catalog_metadata"](lanl_spectra_path)
    wavelength_rest_aa = namespace["load_lanl_wavelength_grid"](lanl_spectra_path)
    simulation_time_grids = namespace["build_simulation_time_grids"](lanl_catalog)
    simulation_pool = sorted(simulation_time_grids)
    if minimum_ejecta_mass is not None:
        massive = namespace["massive_simulation_pool"](minimum_ejecta_mass)
        simulation_pool = [simulation_id for simulation_id in simulation_pool if simulation_id in massive]

    realizations = namespace["sample_kn_realizations_on_grid"](
        [redshift], realizations_per_redshift=attempts, simulation_pool=simulation_pool, rng=rng
    )
    kn_models = namespace["build_kn_models"](
        realizations,
        simulation_time_grids,
        wavelength_rest_aa,
        lanl_spectra_path,
        bands=namespace["ALL_ROMAN_BANDS"],
    )
    scored = []
    for _kn_object_id, (model_6band, base_epochs, realization) in kn_models.items():
        by_tier = namespace["photometry_in_both_tiers"](
            model_6band, base_epochs, namespace["KN_SEED_OFFSET"] + realization["noise_id"]
        )
        if by_tier is None:
            continue
        detections = min(int(by_tier[tier]["detected"].sum()) for tier in namespace["TIER_COLUMNS"])
        if detections < min_detections:
            continue
        scored.append((detections, realization, by_tier))
    scored.sort(key=lambda entry: entry[0], reverse=True)
    return [
        {
            "object_id": f"sim{realization['simulation_id']}_angle{realization['angle_index']}",
            "z_CMB": redshift,
            "by_tier": by_tier,
        }
        for _detections, realization, by_tier in scored[:wanted]
    ]


def build_pools(
    namespace,
    wanted_per_class,
    target_z,
    kilonova_redshift,
    min_detections,
    kilonova_min_detections,
    scan_limit,
    kilonova_attempts,
    kilonova_minimum_ejecta_mass,
    seed,
):
    """`wanted_per_class` objetos por clase, cada uno con su fotometria en los dos tiers."""
    catalog = pd.read_parquet(namespace["CATALOG_PATH"])
    catalog["label"] = catalog["gentype"].map(namespace["GENTYPE_LABEL"])
    rng = np.random.default_rng(seed)

    pools = {
        "KN": collect_kilonova_pool(
            namespace,
            kilonova_redshift,
            wanted_per_class,
            rng,
            kilonova_attempts,
            kilonova_minimum_ejecta_mass,
            kilonova_min_detections,
        )
    }
    print(f"KN: {len(pools['KN'])} realizaciones", flush=True)
    with h5py.File(namespace["HDF5_PATH"], "r") as hdf5:
        present = set(hdf5.keys())
        catalog = catalog[catalog["id"].astype(str).isin(present)]
        for class_label in GIF_PANEL_CLASSES[1:]:
            pools[class_label] = collect_openuniverse_pool(
                namespace,
                catalog[catalog["label"] == class_label],
                hdf5,
                target_z,
                min_detections,
                scan_limit,
                wanted_per_class,
            )
            print(f"{class_label}: {len(pools[class_label])} objetos", flush=True)
    for class_label, pool in pools.items():
        for entry in pool:
            entry["label"] = class_label
    return pools


def compose_event_frames(namespace, pools, number_of_frames):
    """Un evento por frame, rotando entre clases: KN, SN II, SN Ia, SN Ic, SN Ib, KN, ...

    Rotar en vez de agrupar por clase mantiene la KN visible cada 5 frames (20 s) y hace que dos
    frames consecutivos nunca sean la misma clase.
    """
    frames = []
    position = 0
    while len(frames) < number_of_frames:
        added = False
        for class_label in GIF_PANEL_CLASSES:
            if position < len(pools[class_label]) and len(frames) < number_of_frames:
                frames.append(pools[class_label][position])
                added = True
        if not added:
            break
        position += 1
    return frames


def compose_grid_frames(namespace, pools, number_of_frames):
    """Un frame = las 5 clases a la vez, como la figura estatica."""
    available = min(len(pool) for pool in pools.values())
    return [
        [pools[class_label][frame_index] for class_label in GIF_PANEL_CLASSES]
        for frame_index in range(min(number_of_frames, available))
    ]


def draw_event_frame(namespace, entry, style, dpi, figsize):
    """Un solo evento por frame: deep | wide del MISMO objeto, con la clase y el z en el titulo.

    Se conservan las dos columnas porque son el mismo evento: quitar wide dejaria sin mostrar que la
    misma curva, un tier mas arriba, se queda en limites. Sin color por tier (style["tier_color"]
    en False): las dos columnas estan siempre en el mismo orden, el color no agregaba informacion y
    competia con el de las bandas.
    """
    figure, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi, sharex=True, sharey=True)
    extents = []
    for column_position, tier in enumerate(namespace["TIER_COLUMNS"]):
        extents.append(
            namespace["plot_class_panel"](entry["by_tier"][tier], axes[column_position], tier, "", style)
        )
    magnitude_low = min(extent[0] for extent in extents)
    magnitude_high = max(extent[1] for extent in extents)
    day_low = min(extent[2] for extent in extents)
    day_high = max(extent[3] for extent in extents)
    margin = max(0.2, 0.05 * (magnitude_high - magnitude_low))
    span = day_high - day_low
    for ax, tier in zip(axes, namespace["TIER_COLUMNS"], strict=True):
        ax.set_ylim(magnitude_high + margin, magnitude_low - margin)
        ax.set_xlim(day_low - 0.05 * span, day_high + 0.05 * span)
        ax.set_xlabel("days since first detection (deep)", fontsize=style["axis_labelsize"])
        ax.set_title(f"{tier.capitalize()} tier", fontsize=style["title_size"], pad=10, fontweight="bold")
        day_ticks = namespace["axis_ticks"](*ax.get_xlim())
        magnitude_ticks = namespace["axis_ticks"](*ax.get_ylim())
        ax.xaxis.set_major_locator(FixedLocator(day_ticks))
        ax.yaxis.set_major_locator(FixedLocator(magnitude_ticks))
        ax.set_xlim(day_ticks[0], day_ticks[-1])
        ax.set_ylim(magnitude_ticks[-1], magnitude_ticks[0])
    axes[0].set_ylabel("AB magnitude", fontsize=style["axis_labelsize"])

    band_color = namespace["BAND_COLOR"]
    band_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=band_color[band],
            mec="k",
            mew=0.3,
            ms=style["marker_size"] + 1.5,
            label=band,
        )
        for band in namespace["ROMAN_BANDS_BY_WAVELENGTH"]
    ]
    symbol_handles = [
        plt.Line2D(
            [], [], marker="o", ls="", color="0.4", ms=style["marker_size"] + 1.5, label="detection (S/N ≥ 5)"
        ),
        plt.Line2D(
            [],
            [],
            marker="v",
            ls="",
            mfc="none",
            color="0.4",
            ms=style["marker_size"] + 1.5,
            label="5σ upper limit",
        ),
    ]
    figure.legend(
        handles=band_handles + symbol_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=len(band_handles) + len(symbol_handles),
        frameon=False,
        fontsize=style["legend_size"],
    )
    figure.suptitle(
        f"{entry['label']}   z = {entry['z_CMB']:.2f}",
        fontsize=style["title_size"] + 8,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return figure_to_image(figure)


def draw_frame(namespace, frame, style, dpi):
    """Un frame del GIF con el mismo layout que draw_class_lightcurves(style='presentation').

    Los ejes se autoescalan por fila y por frame, igual que en la figura estatica: fijarlos a la
    union de los 30 frames hacia que las curvas mas cortas (SN Ib de 80 d contra SN II de 300 d)
    quedaran apretadas en una esquina.
    """
    figure, axes = plt.subplots(5, 2, figsize=GRID_FIGSIZE, dpi=dpi, sharex="row", sharey="row")
    for row_position, (letter, entry) in enumerate(zip("abcde", frame, strict=True)):
        extents = []
        for column_position, tier in enumerate(namespace["TIER_COLUMNS"]):
            ax = axes[row_position, column_position]
            extents.append(
                namespace["plot_class_panel"](
                    entry["by_tier"][tier],
                    ax,
                    tier,
                    f"({letter}{column_position + 1}) {entry['label']}, z={entry['z_CMB']:.2f}",
                    style,
                )
            )
        magnitude_low = min(extent[0] for extent in extents)
        magnitude_high = max(extent[1] for extent in extents)
        day_low = min(extent[2] for extent in extents)
        day_high = max(extent[3] for extent in extents)
        margin = max(0.2, 0.05 * (magnitude_high - magnitude_low))
        span = day_high - day_low
        for ax in axes[row_position]:
            ax.set_ylim(magnitude_high + margin, magnitude_low - margin)
            ax.set_xlim(day_low - 0.05 * span, day_high + 0.05 * span)
        axes[row_position, 0].set_ylabel("AB magnitude", fontsize=style["axis_labelsize"])

    tier_accent, tier_tint = namespace["TIER_ACCENT"], namespace["TIER_TINT"]
    for column_position, tier in enumerate(namespace["TIER_COLUMNS"]):
        axes[0, column_position].set_title(
            f"{tier.capitalize()} tier",
            fontsize=style["title_size"],
            pad=10,
            color=tier_accent[tier],
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=tier_tint[tier],
                edgecolor=tier_accent[tier],
                linewidth=2.0,
            ),
        )
        axes[-1, column_position].set_xlabel(
            "days since first detection (deep)", fontsize=style["axis_labelsize"]
        )

    band_color = namespace["BAND_COLOR"]
    band_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=band_color[band],
            mec="k",
            mew=0.3,
            ms=style["marker_size"] + 1.5,
            label=band,
        )
        for band in namespace["ROMAN_BANDS_BY_WAVELENGTH"]
    ]
    symbol_handles = [
        plt.Line2D(
            [], [], marker="o", ls="", color="0.4", ms=style["marker_size"] + 1.5, label="detection (S/N ≥ 5)"
        ),
        plt.Line2D(
            [],
            [],
            marker="v",
            ls="",
            mfc="none",
            color="0.4",
            ms=style["marker_size"] + 1.5,
            label="5σ upper limit",
        ),
    ]
    figure.legend(
        handles=band_handles + symbol_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=style["legend_columns"],
        frameon=False,
        fontsize=style["legend_size"],
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure_to_image(figure)


def figure_to_image(figure):
    figure.canvas.draw()
    image = Image.frombuffer(
        "RGBA", figure.canvas.get_width_height(), figure.canvas.buffer_rgba(), "raw", "RGBA", 0, 1
    ).convert("RGB")
    plt.close(figure)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--seconds-per-frame", type=float, default=4.0)
    parser.add_argument(
        "--layout",
        choices=["event", "grid"],
        default="event",
        help="event: un objeto por frame (deep|wide). grid: las 5 clases a la vez.",
    )
    parser.add_argument("--pixels", type=int, default=1400, help="ancho del GIF en pixeles")
    parser.add_argument("--aspect", type=float, default=1.9, help="ancho/alto en layout event")
    parser.add_argument("--target-z", type=float, default=0.5)
    parser.add_argument("--kilonova-redshift", type=float, default=0.5)
    parser.add_argument("--min-detections", type=int, default=10)
    parser.add_argument("--kilonova-min-detections", type=int, default=4)
    parser.add_argument("--scan-limit", type=int, default=600)
    parser.add_argument("--kilonova-attempts", type=int, default=200)
    parser.add_argument("--kilonova-minimum-ejecta-mass", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--hdf5", default=None)
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()

    from kilonova.config import load_paths

    paths = load_paths()
    hdf5_path = arguments.hdf5 or paths.openuniverse_hdf5
    catalog_path = arguments.catalog or paths.openuniverse_catalog
    output_path = Path(
        arguments.output or paths.output_dir / f"presentation_class_lightcurves_{arguments.layout}.gif"
    )

    namespace = load_notebook_namespace(hdf5_path, catalog_path)
    style = dict(namespace["FIGURE_STYLES"]["presentation"])

    number_of_classes = len(GIF_PANEL_CLASSES)
    # En layout "event" cada frame gasta UN objeto de UNA clase, asi que el pool por clase es 5 veces
    # mas chico que el numero de frames.
    wanted_per_class = (
        -(-arguments.frames // number_of_classes) if arguments.layout == "event" else arguments.frames
    )
    pools = build_pools(
        namespace,
        wanted_per_class,
        arguments.target_z,
        arguments.kilonova_redshift,
        arguments.min_detections,
        arguments.kilonova_min_detections,
        arguments.scan_limit,
        arguments.kilonova_attempts,
        arguments.kilonova_minimum_ejecta_mass,
        arguments.seed,
    )

    if arguments.layout == "event":
        frames = compose_event_frames(namespace, pools, arguments.frames)
        style.update(
            {
                "marker_size": 8.0,
                "limit_size": 110,
                "limit_linewidth": 1.8,
                "tick_labelsize": 14,
                "axis_labelsize": 16,
                "title_size": 20,
                "legend_size": 14,
                "tier_color": False,
            }
        )
        figsize = (arguments.pixels / 100.0, arguments.pixels / 100.0 / arguments.aspect)
        dpi = 100.0
        images = [draw_event_frame(namespace, frame, style, dpi, figsize) for frame in frames]
        summary = pd.DataFrame(
            [
                {
                    "frame": frame_index,
                    "label": frame["label"],
                    "object_id": frame["object_id"],
                    "z_CMB": frame["z_CMB"],
                    **{
                        f"detections_{tier}": int(frame["by_tier"][tier]["detected"].sum())
                        for tier in namespace["TIER_COLUMNS"]
                    },
                }
                for frame_index, frame in enumerate(frames)
            ]
        )
    else:
        frames = compose_grid_frames(namespace, pools, arguments.frames)
        dpi = arguments.pixels / GRID_FIGSIZE[0]
        images = [draw_frame(namespace, frame, style, dpi) for frame in frames]
        summary = pd.DataFrame(
            [
                {"frame": frame_index, **{entry["label"]: entry["object_id"] for entry in frame}}
                for frame_index, frame in enumerate(frames)
            ]
        )
    print(f"{len(frames)} frames", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(arguments.seconds_per_frame * 1000),
        loop=0,
        optimize=True,
    )

    # "_gif_objects.md" y no "_objects.md": ese nombre ya lo usa la tabla de la figura estatica.
    summary_path = output_path.with_name(output_path.stem + "_gif_objects.md")
    summary_path.write_text(summary.to_markdown(index=False) + "\n")
    print(f"wrote {output_path}\nwrote {summary_path}")


if __name__ == "__main__":
    main()
