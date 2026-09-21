"""Congela la tabla de C del punto 2 del plan, desde las mediciones de `anchor_measurement.py`.

Entrada: el CSV por objeto de la corrida principal (barrido de delta +-1.5 d) y los CSV de las
plantillas que necesitaron un barrido propio. Salida: un CSV versionado en el repo, con el C de
cada plantilla y el residuo con que se midio, para que nadie tenga que recalcularlo por objeto.
"""

import argparse

import numpy as np
import pandas as pd

OUTPUT = "data/openuniverse/template_phase_anchor.csv"
# Minimo de objetos que tienen que seguir midiendo en un `delta` para creerle: empujar la fase
# fuera de la plantilla deja tres objetos y un minimo espurio (le paso a SN2011bm a -18.5 d).
MINIMUM_FRACTION = 0.5


def delta_by_template(measured):
    """{familia: (delta, |residuo| p50, spread p50, n)} por el criterio del sesgo contra izc.

    La familia es el `template_index` en las core-collapse, donde cada plantilla es una SED
    distinta, y la clase en las otras cuatro, donde el modelo es uno solo -- o, en SN Iax, un banco
    de 919 que comparte el cero de fase. Medido: una sola C cubre cada una de esas clases."""
    columns = {float(c[3:]): c for c in measured.columns if c.startswith("zp_")}
    spreads = {float(c[3:]): c for c in measured.columns if c.startswith("sp_")}
    chosen = {}
    for family, block in measured.groupby("family"):
        floor = max(4, int(MINIMUM_FRACTION * len(block)))
        scores = {}
        for delta, column in columns.items():
            residual = block[column] - block.offset_izc
            if residual.notna().sum() >= floor:
                scores[delta] = abs(np.nanmedian(residual))
        if not scores:
            continue
        delta = min(scores, key=scores.get)
        residual = (block[columns[delta]] - block.offset_izc).abs()
        chosen[family] = (delta, float(np.nanmedian(residual)),
                          float(np.nanmedian(block[spreads[delta]])),
                          int(residual.notna().sum()))
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", nargs="*",
                        default=["scripts/probes_data_augmentation/anchor_measurement.csv"],
                        help="CSV por objeto de las corridas de barrido +-1.5 d")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="CSV de plantillas con barrido propio; pisan a los de la corrida principal")
    parser.add_argument("--table", nargs="*",
                        default=["scripts/probes_data_augmentation/template_C_2026-09-21.csv"],
                        help="de donde salen source/label/minphase/maxphase")
    parser.add_argument("--out", default=OUTPUT)
    args = parser.parse_args()

    reference = pd.concat([pd.read_csv(one) for one in args.table]).set_index("family")
    chosen, origin = {}, {}
    for path in args.main:
        for family, values in delta_by_template(pd.read_csv(path)).items():
            chosen[family] = values
            origin[family] = "barrido +-1.5 d"
    for path in args.extra:
        for template_index, values in delta_by_template(pd.read_csv(path)).items():
            chosen[template_index] = values
            origin[template_index] = "barrido propio"

    rows = []
    for family, (delta, residual, spread, n) in sorted(chosen.items(), key=lambda one: str(one[0])):
        row = reference.loc[family]
        rows.append(dict(family=family, source=row.source, label=row.label,
                         C=row.C_edge + delta, C_edge=row.C_edge, delta=delta,
                         minphase=row.minphase, maxphase=row.maxphase,
                         objects=n, residual_mag=residual, spread_mag=spread,
                         origin=origin[family]))
    table = pd.DataFrame(rows)
    table.round(4).to_csv(args.out, index=False)
    print(table.round(3).to_string(index=False))
    print(f"\n{len(table)} plantillas -> {args.out}")
    print(f"residuo contra el brillo del izc: p50 {table.residual_mag.median():.4f}  "
          f"p90 {table.residual_mag.quantile(0.9):.4f}  max {table.residual_mag.max():.4f} "
          f"({table.loc[table.residual_mag.idxmax(), 'source']})")


if __name__ == "__main__":
    main()
