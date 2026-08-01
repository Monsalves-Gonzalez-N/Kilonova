"""Detectabilidad de nuestras kilonovas frente a Chase et al. 2021 (arXiv:2105.12268).

Reproduce el tipo de figura del paper -- fraccion detectable en el plano (tiempo observador desde
el merger) x (redshift) -- pero **sobre las curvas de luz ya generadas**, `kn_windows_{deep,wide}
.parquet`, las mismas que entrenaron el transformer. Ver `docs/plan_kn_widefield_repro.md` para la
metodologia del paper; aqui se documenta solo lo que cambia al medirla sobre este dataset.

Por que el dataset admite esta medida
-------------------------------------
`kn_windows_*.parquet` se genero con 100 nodos de z log-espaciados en [0.02, 1] y
**10 000 realizaciones por nodo** (`params.yaml: kilonova_windows`), cada una con simulacion LANL y
angulo sorteados **uniformemente** -- que es exactamente la marginalizacion uniforme sobre las
48 600 realizaciones (900 sims x 54 angulos) que promedia el paper, en version Monte Carlo.

La ventana empieza en la primera deteccion y las realizaciones **nunca detectadas no se escriben**,
asi que el denominador de toda fraccion es el numero de realizaciones sorteadas (10 000), no el
numero de filas del archivo. Esa es la clave: los ceros del mapa son los objetos ausentes.

Eje temporal
------------
Las visitas sinteticas de una KN son `explosion_offset_days + 5*arange(8)` en tiempo observador
**desde el merger**, con el offset ~ U[0, 5) codificado en el `object_id`. Si la primera deteccion
cae en la visita 0, la fase de la epoca k es `offset + days_since_detection`. La paridad de la
cadencia (tambien en el `object_id`) fija que par de bandas se observa en cada visita, asi que las
bandas observadas en la epoca 1 revelan la **paridad** de la visita donde se detecto: sale par para
el 94.5% de los objetos (99% a z<0.1), y las realizaciones que no lo cumplen genuinamente NO fueron
detectadas en la visita 0. El mapa se construye solo con las de paridad par y denominador completo,
de modo que las demas cuentan como no-deteccion, que es lo correcto para el primer bloque de fase.

Como cada realizacion tiene exactamente una visita en cada bloque de 5 dias y el offset es uniforme,
el denominador de un bin de fase de ancho `w` es `10000 * w / 5` por nodo de z (por 1/2 mas para
las bandas no-ancla, que solo se observan en visitas alternas).

Dos criterios de deteccion
--------------------------
- **Paper**: `mag_true < m_lim` con la Tabla 1 fija de Chase. Comparable 1:1 con sus figuras, y no
  lo toca la cadencia porque `mag_true` existe en toda banda y epoca cubierta por el modelo.
- **Nuestro**: `detected`, o sea S/N >= 5 con la receta de ruido y el limite 5sigma por visita.

Uso: python notebooks/validation/kn_detectability_vs_chase.py [--output-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from kilonova.config import load_paths

# Tabla 1 de Chase et al. 2021: magnitud limite 5sigma por banda Roman (Hounsell+18, Scolnic+18).
CHASE_LIMITING_MAGNITUDE = {
    "R062": 26.2, "Z087": 25.7, "Y106": 25.6, "J129": 25.5, "H158": 25.4, "F184": 24.9,
}
# Valores publicados a igualar (Roman/H, grilla completa). z05 corregido a 0.48: el 0.79 de
# `docs/plan_kn_widefield_repro.md` no concuerda con la figura del propio paper, donde el contorno
# 0.05 llega a ~0.48 (ver `docs/kn_detectability_vs_chase.md`).
CHASE_ROMAN_H = {"z95": 0.10, "z50": 0.22, "z05": 0.48}
# Planck 2020 tal como lo usa el paper -- no los defaults de Planck18 de astropy.
CHASE_COSMOLOGY = FlatLambdaCDM(H0=67.4, Om0=0.315)

REALIZATIONS_PER_REDSHIFT = 10_000
BASE_CADENCE_DAYS = 5.0
BANDS_BY_WAVELENGTH = ["R062", "Z087", "Y106", "J129", "H158", "F184"]
ANCHOR_BAND = {"wide": "R062", "deep": "Z087"}
# Par de bandas de la visita par de cada tier (la impar es el otro par); fija la paridad observada.
EVEN_VISIT_BANDS = {"wide": {"Z087", "Y106"}, "deep": {"Y106", "J129"}}

# Bins log-espaciados como el eje x del paper. El denominador de un bin es proporcional a su ancho:
# con offset ~ U[0,5) y una visita por bloque de 5 dias, la densidad de visitas es uniforme en fase.
PHASE_MIN_DAYS, PHASE_MAX_DAYS = 0.2, 20.0
MAP_PHASE_EDGES = np.geomspace(PHASE_MIN_DAYS, PHASE_MAX_DAYS, 41)
# Los bordes z50%/z05%/z95% se miden sobre bins mucho mas anchos (factor 2). El paper evalua las
# 48 600 realizaciones en cada t exacto y no tiene ruido; nuestra version es Monte Carlo, y un maximo
# sobre 40 bins finos -- el mas estrecho recibe solo ~50 realizaciones por nodo de z -- se sesga al
# alza por fluctuaciones. Con bins de factor 2 el menos poblado recibe ~1000 y el sesgo desaparece.
STATISTIC_PHASE_EDGES = np.geomspace(0.5, 16.0, 6)
CONTOUR_LEVELS = (0.05, 0.5, 0.95)
# Trazo de cada nivel. Se dibujan en blanco con un reborde oscuro: el gris claro de antes se perdia
# sobre el amarillo saturado justo donde vive el contorno de 0.95.
CONTOUR_DASHES = {0.05: (0, (6, 3)), 0.5: (0, (1, 2.2)), 0.95: (0, (7, 2.5, 1.5, 2.5))}
CONTOUR_OUTLINE = [path_effects.withStroke(linewidth=4.4, foreground="0.12", alpha=0.85)]

# Tipografia serif como las figuras de referencia.
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})


def band_colors(bands):
    """Orden fijo por longitud de onda sobre una rampa perceptualmente uniforme: el color codifica
    la banda y ademas su orden espectral, y sobrevive a la vision de color anomala."""
    ramp = plt.get_cmap("viridis")
    return {band: ramp(0.08 + 0.84 * index / max(len(bands) - 1, 1)) for index, band in enumerate(bands)}


def load_tier(tier, output_dir):
    """Curvas de un tier + las columnas que el object_id codifica (offset, paridad de la cadencia).

    object_id = simulation_id_angle_index_offset_redshift_parity_noiseid
    (`early_windows.kn_object_id`)."""
    frame = pd.read_parquet(
        output_dir / f"kn_windows_{tier}.parquet",
        columns=[
            "object_id",
            "z_CMB",
            "epoch",
            "days_since_detection",
            "band",
            "observed",
            "mag_true",
            "detected",
            "mag_limit_5sigma",
        ],
    )
    fields = frame.object_id.str.split("_")
    frame["simulation_id"] = fields.str[0].astype(int)
    frame["explosion_offset_days"] = fields.str[2].astype(float)
    frame["cadence_parity"] = fields.str[4].astype(int)
    return frame


def attach_ejecta_masses(frame, lanl_catalog_path):
    """Anade mass_dynamical / mass_wind por simulation_id, para los cortes de masa fija (Fig. 5/7).

    El object_id solo guarda el simulation_id; las masas viven en el catalogo de la grilla LANL."""
    catalog = pd.read_parquet(lanl_catalog_path, columns=["simulation_id", "mass_dynamical", "mass_wind"])
    masses = catalog.drop_duplicates("simulation_id").set_index("simulation_id")
    return frame.join(masses, on="simulation_id")


def first_detection_visit_is_even(frame, tier):
    """Serie object_id -> True si la epoca 1 cayo en una visita de paridad par.

    Las bandas no-ancla observadas en la epoca 1 dan la paridad de esa visita; comparada con la
    `cadence_parity` sorteada dice si la primera deteccion ocurrio en una visita par (compatible con
    la visita 0) o impar."""
    first_epoch = frame[(frame.epoch == 1) & frame.observed & (frame.band != ANCHOR_BAND[tier])]
    visit_is_even = first_epoch.band.isin(EVEN_VISIT_BANDS[tier])
    by_object = (
        pd.DataFrame(
            {"object_id": first_epoch.object_id, "even": visit_is_even, "parity": first_epoch.cadence_parity}
        )
        .groupby("object_id")
        .first()
    )
    return by_object.even.eq(by_object.parity == 0)


def survival_fraction(frame):
    """Fraccion de realizaciones detectada en alguna banda, por nodo de z: los objetos que existen
    en el archivo sobre las 10 000 sorteadas."""
    objects = frame.drop_duplicates("object_id")
    counts = objects.groupby("z_CMB").size()
    return counts / REALIZATIONS_PER_REDSHIFT


def redshift_at_fraction(redshifts, fractions, level):
    """z donde la fraccion cruza `level` por ultima vez (borde de detectabilidad)."""
    above = np.where(fractions >= level)[0]
    if len(above) == 0:
        return np.nan
    last = above[-1]
    if last + 1 >= len(fractions):
        return np.nan  # la grilla se satura: el cruce esta fuera de z_max
    return float(
        np.interp(level, [fractions[last + 1], fractions[last]], [redshifts[last + 1], redshifts[last]])
    )


def detectability_edge(redshifts, fractions, level):
    """z del borde de detectabilidad en cada columna de fase: el ultimo cruce del nivel a lo largo
    de z (`fractions` es n_z x n_fase).

    Sustituye a `contour` sobre el campo 2D. La fraccion decrece con z, asi que el borde ES una
    curva z(t) de un solo valor; `contour` no lo sabe y, alrededor de 0.95, donde el ruido Monte
    Carlo de un campo casi saturado cruza el nivel muchas veces, devolvia una maraña de islas en vez
    de una linea. Aqui cada columna aporta un punto y el resultado es una linea limpia; las columnas
    que no alcanzan el nivel quedan en NaN y cortan el trazo, que es lo correcto."""
    return np.array(
        [redshift_at_fraction(redshifts, fractions[:, column], level) for column in range(fractions.shape[1])]
    )


def detectability_map(
    frame, band, criterion, tier, phase_edges=MAP_PHASE_EDGES, grid_fraction=1.0, redshift_nodes=None
):
    """Fraccion detectable en (fase desde el merger) x (z) para una banda.

    `criterion` es 'paper' (mag_true < Tabla 1 de Chase) o 'pipeline' (S/N >= 5). Devuelve
    (redshifts, centros de fase, matriz n_z x n_fase).

    `grid_fraction` es la fraccion de las 900 simulaciones LANL que sobrevive al corte del panel
    (1 para la grilla completa, 180/900 para una masa fija, 36/900 para masa eyecta total fija). Las
    realizaciones se sortearon uniformemente sobre la grilla, asi que solo esa fraccion de las 10 000
    por nodo de z pertenece al corte, y el denominador debe escalarse: sin esto un panel de masa fija
    parece cinco veces menos detectable de lo que es.

    `redshift_nodes` fija la grilla de z. Sin el, un corte de masa la deduce de sus propias filas y
    se queda en el ultimo nodo con alguna deteccion, dejando el panel cortado en blanco justo donde
    la fraccion ya es cero -- que es informacion, no ausencia de ella."""
    band_rows = frame[frame.band == band].copy()
    if criterion == "paper":
        hit = band_rows.mag_true < CHASE_LIMITING_MAGNITUDE[band]
        visits_per_realization = 1.0
    else:
        hit = band_rows.detected.to_numpy()
        # Las bandas no-ancla solo se observan en visitas alternas: la mitad de las realizaciones
        # aporta una visita a cada bin de fase.
        visits_per_realization = 1.0 if band == ANCHOR_BAND[tier] else 0.5
    band_rows = band_rows[np.asarray(hit) & np.isfinite(band_rows.mag_true)]

    redshifts = np.sort(frame.z_CMB.unique()) if redshift_nodes is None else np.asarray(redshift_nodes)
    phase = band_rows.explosion_offset_days + band_rows.days_since_detection
    counts, _, _ = np.histogram2d(
        band_rows.z_CMB, phase, bins=[np.append(redshifts, redshifts[-1] * 1.001), phase_edges]
    )
    widths = np.diff(phase_edges)
    denominator = (
        REALIZATIONS_PER_REDSHIFT * grid_fraction * (widths / BASE_CADENCE_DAYS) * visits_per_realization
    )
    return redshifts, np.sqrt(phase_edges[:-1] * phase_edges[1:]), counts / denominator


def per_band_fraction_vs_redshift(frame, band, criterion, tier):
    """Fraccion detectable por nodo de z, maximizada sobre la fase: el borde de detectabilidad que
    el paper reporta como z50%/z05%/z95% es el del mejor momento de la curva."""
    redshifts, _, fractions = detectability_map(frame, band, criterion, tier, STATISTIC_PHASE_EDGES)
    return redshifts, fractions.max(axis=1)


def style_axes(axes):
    axes.tick_params(labelsize=9)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    axes.grid(alpha=0.18, linewidth=0.6)
    axes.set_axisbelow(True)


def figure_survival(tiers, output_path):
    """Fraccion detectable (cualquier banda) vs z, por tier, contra los valores publicados."""
    figure, axes = plt.subplots(figsize=(7.2, 4.6))
    colors = {"deep": "#1f4e79", "wide": "#c8102e"}
    label_offsets = {"deep": (10, 12), "wide": (-72, -20)}
    for tier, frame in tiers.items():
        fraction = survival_fraction(frame)
        redshifts, values = fraction.index.to_numpy(), fraction.to_numpy()
        axes.plot(redshifts, values, lw=2, color=colors[tier], label=f"{tier} (S/N$\\geq$5, cualquier banda)")
        z50 = redshift_at_fraction(redshifts, values, 0.5)
        if np.isfinite(z50):
            axes.plot([z50], [0.5], "o", ms=7, color=colors[tier], mec="white", mew=1.2, zorder=5)
            axes.annotate(
                f"$z_{{50\\%}}$ = {z50:.2f}",
                (z50, 0.5),
                textcoords="offset points",
                xytext=label_offsets[tier],
                fontsize=9,
                color=colors[tier],
            )
    for level in (0.95, 0.5, 0.05):
        axes.axhline(level, color="0.55", lw=0.8, ls=":", zorder=0)
    for key, level in (("z95", 0.95), ("z50", 0.5), ("z05", 0.05)):
        axes.plot([CHASE_ROMAN_H[key]], [level], "*", ms=13, color="0.25", zorder=6)
    axes.set_xscale("log")
    axes.set_xlabel("Redshift")
    axes.set_ylabel("Fraccion detectable")
    axes.set_ylim(0, 1.03)
    axes.set_title(
        "Fraccion de kilonovas detectada vs redshift\n(curvas ya generadas, 10 000 realizaciones por nodo)",
        fontsize=11,
    )
    handles, labels = axes.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="*", ls="", ms=11, color="0.25"))
    labels.append("Chase+21, Roman/H (grilla completa)")
    axes.legend(handles, labels, fontsize=8.5, frameon=False, loc="lower left")
    style_axes(axes)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def luminosity_distance_ticks(redshift_limits, cosmology=CHASE_COSMOLOGY):
    """(posiciones en z, etiquetas en Gpc) para el eje derecho: valores redondos de d_L dentro del
    rango de z del panel, colocados en la posicion z que les corresponde."""
    low, high = redshift_limits
    reference = np.geomspace(max(low, 1e-3), high, 512)
    distances = cosmology.luminosity_distance(reference).to("Gpc").value
    candidates = np.array([0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0])
    inside = candidates[(candidates >= distances.min()) & (candidates <= distances.max())]
    return np.interp(inside, distances, reference), [f"{value:g}" for value in inside]


def figure_paper_style(
    frame,
    tier,
    band,
    criterion,
    output_path,
    redshift_max=1.0,
    subtitle=None,
    grid_fraction=1.0,
    redshift_nodes=None,
):
    """Un panel al estilo de las figuras de referencia (Chase+21 Fig. 3/5/7).

    Reproduce su composicion: redshift **lineal** en el eje y, tiempo observador log con ticks en
    potencias de 2, colormap `inferno`, contornos gruesos 0.05/0.5/0.95 en gris con trazo
    discontinuo/punteado/raya-punto, eje derecho de distancia de luminosidad en Gpc y barra de color
    "Fraccion Detectable".

    El campo se dibuja suavizado (gaussiana de ~1 bin). El del paper es liso porque evalua las
    48 600 realizaciones en cada (t, z) exacto; el nuestro es Monte Carlo y sin suavizar el moteado
    de Poisson domina la lectura. Los numeros publicados (z50%/z95%) NO salen de este campo
    suavizado sino de `per_band_fraction_vs_redshift`, que no aplica ningun filtro."""
    redshifts, phases, fractions = detectability_map(
        frame, band, criterion, tier, grid_fraction=grid_fraction, redshift_nodes=redshift_nodes
    )
    smoothed = gaussian_filter(fractions, sigma=1.2, mode="nearest")

    # La grilla de z arranca en 0.02 y el eje del paper en 0: sin esto el panel parece cortado por
    # abajo, justo donde la fraccion satura. Se extiende replicando el nodo mas bajo hasta z=0, que
    # es exacto salvo una franja de 0.02 de ancho en la que la fraccion solo puede ser mayor (la
    # detectabilidad decrece con z), asi que el sesgo es a la baja y despreciable.
    redshifts = np.concatenate([[0.0], redshifts])
    smoothed = np.vstack([smoothed[0], smoothed])

    figure, axes = plt.subplots(figsize=(6.4, 4.4))
    mesh = axes.pcolormesh(phases, redshifts, smoothed, cmap="inferno", vmin=0, vmax=1, shading="gouraud")
    for level in CONTOUR_LEVELS:
        if smoothed.max() <= level:
            continue
        axes.plot(
            phases,
            detectability_edge(redshifts, smoothed, level),
            color="white",
            lw=2.2,
            ls=CONTOUR_DASHES[level],
            path_effects=CONTOUR_OUTLINE,
        )
    axes.set_xscale("log")
    axes.set_xlim(0.25, PHASE_MAX_DAYS)
    axes.set_xticks([0.25, 0.5, 1, 2, 4, 8, 16])
    axes.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8", "16"])
    axes.minorticks_off()
    axes.set_ylim(0.0, redshift_max)
    axes.set_xlabel("Observer-Frame Time (d)", fontsize=13)
    axes.set_ylabel("Redshift", fontsize=13)
    title = f"Roman/${band[0]}$-band ({tier})" + (f": {subtitle}" if subtitle else "")
    axes.set_title(title, fontsize=12)
    axes.tick_params(labelsize=11)

    distance_axes = axes.twinx()
    distance_axes.set_ylim(axes.get_ylim())
    positions, labels = luminosity_distance_ticks(axes.get_ylim())
    distance_axes.set_yticks(positions)
    distance_axes.set_yticklabels(labels)
    distance_axes.set_ylabel("Luminosity Distance [Gpc]", fontsize=12)
    distance_axes.tick_params(labelsize=11)

    # Handles largos y finos: con el trazo grueso y el handle corto de antes los tres patrones de
    # guiones se veian como la misma barra gris y la leyenda no distinguia un nivel de otro.
    handles = [
        Line2D([], [], color="0.15", lw=1.8, ls=CONTOUR_DASHES[level]) for level in CONTOUR_LEVELS
    ]
    axes.legend(
        handles,
        [f"{level:g}" for level in CONTOUR_LEVELS],
        fontsize=10,
        loc="upper right",
        framealpha=0.85,
        handlelength=4.0,
    )

    colorbar = figure.colorbar(mesh, ax=[axes, distance_axes], fraction=0.045, pad=0.14)
    colorbar.set_label("Fraction Detectable", fontsize=12)
    colorbar.ax.tick_params(labelsize=10)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def figure_criteria(tiers, output_path):
    """Fraccion detectable vs z por banda y por criterio, con el z50% de cada una."""
    figure, axes_grid = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    summary = []
    for axes, (tier, frame) in zip(axes_grid, tiers.items(), strict=True):
        bands = [band for band in BANDS_BY_WAVELENGTH if band in set(frame.band)]
        colors = band_colors(bands)
        for band in bands:
            for criterion, style in (("paper", "-"), ("pipeline", "--")):
                redshifts, fractions = per_band_fraction_vs_redshift(frame, band, criterion, tier)
                axes.plot(redshifts, fractions, style, lw=1.8, color=colors[band], alpha=0.95)
                summary.append(
                    {
                        "tier": tier,
                        "band": band,
                        "criterio": criterion,
                        "z95": redshift_at_fraction(redshifts, fractions, 0.95),
                        "z50": redshift_at_fraction(redshifts, fractions, 0.5),
                        "z05": redshift_at_fraction(redshifts, fractions, 0.05),
                    }
                )
        axes.axvline(CHASE_ROMAN_H["z50"], color="0.35", lw=1.0, ls=":")
        axes.annotate(
            "Chase+21 Roman/H\n$z_{50\\%}$ = 0.22",
            (CHASE_ROMAN_H["z50"], 0.30),
            textcoords="offset points",
            xytext=(8, 0),
            fontsize=8,
            color="0.35",
        )
        axes.axhline(0.5, color="0.55", lw=0.8, ls=":")
        axes.set_xscale("log")
        axes.set_xlabel("Redshift")
        axes.set_title(f"tier {tier}", fontsize=10)
        style_axes(axes)
        handles = [Line2D([], [], color=colors[band], lw=2) for band in bands]
        handles += [
            Line2D([], [], color="0.3", lw=1.8, ls="-"),
            Line2D([], [], color="0.3", lw=1.8, ls="--"),
        ]
        axes.legend(
            handles,
            bands + ["$m<m_{lim}$ (paper)", "S/N$\\geq$5 (pipeline)"],
            fontsize=8,
            frameon=False,
            ncol=2,
        )
    axes_grid[0].set_ylabel("Fraccion detectable (max sobre la fase)")
    axes_grid[0].set_ylim(0, 1.03)
    figure.suptitle("Detectabilidad por banda: criterio del paper vs criterio del pipeline", fontsize=11)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return pd.DataFrame(summary)


def figure_diagnostics(tiers, parity_by_tier, output_path):
    """Los dos supuestos que sostienen la comparacion: profundidad alcanzada y eje de fase."""
    figure, (depth_axes, parity_axes) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    offsets = {"deep": -0.18, "wide": 0.18}
    colors = {"deep": "#1f4e79", "wide": "#c8102e"}
    positions = {band: index for index, band in enumerate(BANDS_BY_WAVELENGTH)}
    for tier, frame in tiers.items():
        observed = frame[frame.observed & np.isfinite(frame.mag_true)]
        medians = observed.groupby("band").mag_limit_5sigma.median()
        x = [positions[band] + offsets[tier] for band in medians.index]
        depth_axes.bar(x, medians.to_numpy(), width=0.34, color=colors[tier], label=tier)
    depth_axes.plot(
        [positions[band] for band in CHASE_LIMITING_MAGNITUDE],
        list(CHASE_LIMITING_MAGNITUDE.values()),
        "*",
        ms=13,
        color="0.2",
        ls="",
        label="Chase+21, Tabla 1",
    )
    depth_axes.set_xticks(list(positions.values()))
    depth_axes.set_xticklabels(list(positions), fontsize=9)
    depth_axes.set_ylim(24, 27.6)
    depth_axes.set_ylabel("Magnitud limite 5$\\sigma$ (AB)")
    depth_axes.set_title("Profundidad por visita alcanzada vs la tabulada por el paper", fontsize=10)
    depth_axes.legend(fontsize=8.5, frameon=False)
    style_axes(depth_axes)

    for tier, parity in parity_by_tier.items():
        frame = tiers[tier]
        redshift_by_object = frame.drop_duplicates("object_id").set_index("object_id").z_CMB
        aligned = pd.DataFrame({"even": parity, "z": redshift_by_object.reindex(parity.index)})
        by_redshift = aligned.groupby("z").even.mean()
        parity_axes.plot(by_redshift.index, by_redshift.to_numpy(), lw=1.8, color=colors[tier], label=tier)
    parity_axes.set_xscale("log")
    parity_axes.set_ylim(0.5, 1.02)
    parity_axes.set_xlabel("Redshift")
    parity_axes.set_ylabel("Fraccion con 1a deteccion en visita par")
    parity_axes.set_title(
        "Validez del eje de fase\n(par $\\Rightarrow$ compatible con la visita 0)", fontsize=10
    )
    parity_axes.legend(fontsize=8.5, frameon=False)
    style_axes(parity_axes)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="donde escribir las figuras")
    arguments = parser.parse_args()

    paths = load_paths()
    figures_dir = arguments.output_dir or (paths.output_dir / "detectability_vs_chase")
    figures_dir.mkdir(parents=True, exist_ok=True)

    tiers, parity_by_tier, filtered = {}, {}, {}
    for tier in ("deep", "wide"):
        frame = load_tier(tier, paths.output_dir)
        parity = first_detection_visit_is_even(frame, tier)
        parity_by_tier[tier] = parity
        tiers[tier] = frame
        # El mapa de fase solo usa los objetos cuya 1a deteccion es compatible con la visita 0; los
        # demas quedan como no-deteccion, que es lo que fueron en el primer bloque de fase.
        filtered[tier] = attach_ejecta_masses(
            frame[frame.object_id.isin(parity[parity].index)], paths.lanl_catalog
        )
        print(
            f"{tier}: {frame.object_id.nunique()} objetos, "
            f"{parity.mean():.3f} con 1a deteccion en visita par"
        )

    # Fraccion de la grilla LANL (900 simulaciones) que sobrevive a cada corte de masa.
    grid_masses = pd.read_parquet(
        paths.lanl_catalog, columns=["simulation_id", "mass_dynamical", "mass_wind"]
    ).drop_duplicates("simulation_id")

    maps_dir = figures_dir / "mapas"
    maps_dir.mkdir(exist_ok=True)
    for tier, frame in filtered.items():
        redshift_nodes = np.sort(frame.z_CMB.unique())
        bands = [band for band in BANDS_BY_WAVELENGTH if band in set(frame.band)]
        for band in bands:
            figure_paper_style(frame, tier, band, "paper", maps_dir / f"Roman{band[0]}_{tier}.png")
        # Version de texto principal de H158: el mismo panel con el eje de redshift recortado a 0.5
        # (las figuras de referencia usan 1.0 en todo salvo esa).
        figure_paper_style(
            frame, tier, "H158", "paper", maps_dir / f"RomanH_{tier}_maintext.png", redshift_max=0.5
        )
        # Cortes de masa fija (Fig. 5 del paper): una sola masa fijada, el resto de la grilla libre.
        for column, prefix in (("mass_dynamical", "md"), ("mass_wind", "mw")):
            for mass in (0.1, 0.01, 0.001):
                selected = np.isclose(grid_masses[column], mass)
                figure_paper_style(
                    frame[np.isclose(frame[column], mass)],
                    tier,
                    "H158",
                    "paper",
                    maps_dir / f"RomanH_{prefix}_{mass:g}_{tier}.png",
                    subtitle=f"$m_{{{'dyn' if prefix == 'md' else 'wind'}}}$ = {mass:g} $M_\\odot$",
                    grid_fraction=selected.mean(),
                    redshift_nodes=redshift_nodes,
                )
        # Masa eyecta total fija (Fig. 7): el unico par (dyn, wind) de la grilla que suma el total.
        for mass in (0.1, 0.001):
            selected = np.isclose(grid_masses.mass_dynamical, mass) & np.isclose(grid_masses.mass_wind, mass)
            figure_paper_style(
                frame[np.isclose(frame.mass_dynamical, mass) & np.isclose(frame.mass_wind, mass)],
                tier,
                "H158",
                "paper",
                maps_dir / f"RomanH_mw_{mass:g}_md_{mass:g}_{tier}.png",
                subtitle=f"$m_{{ej}}$ = {2 * mass:g} $M_\\odot$ (dyn = wind = {mass:g})",
                grid_fraction=selected.mean(),
                redshift_nodes=redshift_nodes,
            )

    summary = figure_criteria(filtered, figures_dir / "criterio_paper_vs_pipeline.png")
    figure_diagnostics(tiers, parity_by_tier, figures_dir / "diagnosticos.png")

    summary.to_csv(figures_dir / "bordes_de_detectabilidad.csv", index=False)
    print("\nBordes de detectabilidad (z al que la fraccion cruza cada nivel):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nChase+21 Roman/H (grilla completa): z95%={CHASE_ROMAN_H['z95']}, "
          f"z50%={CHASE_ROMAN_H['z50']}, z05%={CHASE_ROMAN_H['z05']}")
    print(f"\nFiguras en {figures_dir}")


if __name__ == "__main__":
    main()
