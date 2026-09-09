"""Measure the brightness offset of each izc class against the OpenUniverse objects it stands in for.

The izc luminosity functions are handles on a model, not measured magnitudes, and normalising a
template in rest-frame B fixes its brightness in B while leaving its brightness in Y106 to the
template's own B - Y colour -- which is exactly what these libraries disagree on. So the medians of
`PEAK_ABSOLUTE_MAGNITUDE` carry a calibration offset, and this script is where that offset comes
from. It used to be an ad-hoc measurement recorded only in the module's comments.

The statistic is the same on both sides and assumes nothing: generate izc objects at redshifts
resampled from OpenUniverse's own low-redshift objects of that class, push them through the same
window builder and the same detection cut, and take the brightest Y106 `mag_true` of the window
minus the distance modulus. No K-correction is assumed anywhere -- each template supplies its own
colours -- and the two populations are compared after the same selection.

Two classes are deliberately NOT calibrated and are reported only: SN Iax, whose brightness comes
from the template bank OpenUniverse itself drew from, and TDE, whose brightness comes from MOSFiT's
physics. Moving either would make the class agree by construction instead of by model.

WHY ONE BAND SETS THE OFFSET AND THE OTHER FOUR ARE ONLY REPORTED. The offset is one scalar per
class, so measuring it in five bands would measure the same number five times rather than fit five
numbers; what a scalar cannot move is the COLOUR, which the templates set. Y106 anchors because it
is mid-range, in both tiers, and at these redshifts still samples the rest-frame red optical, where
these templates are observed rather than extrapolated. So the brightness block below is unchanged
and its numbers stay reproducible, and the colour block is a diagnostic that adjusts nothing.

The colour statistic is deliberately NOT the raw per-band median. Two things corrupt that, and the
same pair already produced a false result in the CSP comparison: per-band medians are taken over
different subsets of objects, because a band a faint object was not detected in simply drops out,
and they carry the distance modulus and the brightness offset inside them. So the colour is formed
PER OBJECT, M(band) - M(Y106) on objects detected in both, and only then compared between
populations. That differences away the distance and the brightness and leaves the shape of the SED,
which is the thing the two template libraries actually disagree on.

It is a peak-to-peak colour: each band contributes its own brightest point, which need not be the
same epoch. That is the same convention the brightness statistic already uses, and it is the one
available without assuming a time origin the two populations share.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kilonova.photometry.roman_noise import ALL_BANDS_BY_WAVELENGTH
from kilonova.simulation import intermediate_z_contaminants as izc

# The izc labels that stand in for one OpenUniverse gentype, and the gentype they stand in for.
# The SN II subtypes are pooled the way OpenUniverse pools them, because the offset is measured
# against a population OpenUniverse never separated.
CALIBRATED = {
    "SN Ib": 21,
    "SN Ic": 26,
    "SN II": 32,
    "SN Ia": 10,
    "SN Iax": 12,
    "TDE": 42,
}
ADJUSTABLE = ("SN Ib", "SN Ic", "SN II")
SUBTYPES_OF = {"SN II": ("SN IIP", "SN IIL", "SN IIn")}
MAXIMUM_REDSHIFT = 0.45
# The band the offset is measured in, and the band every reported colour is formed against. See the
# module docstring for why it is this one and why there is only one.
ANCHOR_BAND = "Y106"
# Below this many objects a median is not reported at all rather than reported with a large error:
# the classes where a band drops out are exactly the ones where the survivors are a bright,
# selected subset, and a median over a handful of them is not a population statistic.
MINIMUM_COLOUR_OBJECTS = 25


def peak_absolute_magnitudes(windows, cosmology):
    """Per object and band, the brightest `mag_true` of the window minus the distance modulus.

    Indexed by (object_id, z_CMB), one column per band. A band the object was never detected in is
    NaN rather than absent, so a colour can be formed only where both of its bands exist and the
    two populations are never compared over different subsets of objects.
    """
    rows = windows[np.isfinite(windows["mag_true"])]
    peak = rows.groupby(["object_id", "z_CMB", "band"])["mag_true"].min().reset_index()
    peak["absolute"] = peak["mag_true"] - cosmology.distmod(peak["z_CMB"].to_numpy()).value
    return peak.pivot_table(index=["object_id", "z_CMB"], columns="band", values="absolute")


def openuniverse_peak_absolute_magnitudes(windows, gentype, cosmology):
    """The table above for the OpenUniverse objects of one gentype below MAXIMUM_REDSHIFT."""
    rows = windows[(windows["gentype"] == gentype) & (windows["z_CMB"] < MAXIMUM_REDSHIFT)]
    return peak_absolute_magnitudes(rows, cosmology)


def izc_peak_absolute_magnitudes(label, redshifts, random_generator, cosmology):
    """The same table for izc objects of one class, generated at the given redshifts."""
    wanted = set(SUBTYPES_OF.get(label, (label,)))
    population = []
    while len(population) < len(redshifts):
        batch = izc.draw_population(4 * len(redshifts), np.repeat(redshifts, 4), random_generator)
        population.extend(one for one in batch if one["label"] in wanted)
    population = population[: len(redshifts)]
    for one, redshift in zip(population, redshifts, strict=True):
        one["redshift"] = float(redshift)
    windows, _ = izc.build_izc_windows(population, "deep", cosmology=cosmology)
    return peak_absolute_magnitudes(windows, cosmology)


def median_and_error(values):
    """Median and its standard error, the 1.253 sigma/sqrt(n) of the asymptotic normal case."""
    return float(np.median(values)), float(1.253 * np.std(values) / np.sqrt(len(values)))


def anchor_column(table):
    """M(ANCHOR_BAND) of the objects detected in it, in the table's own order."""
    if ANCHOR_BAND not in table.columns:
        return table.iloc[:0].index, np.array([])
    detected = table[ANCHOR_BAND].dropna()
    return detected.index, detected.to_numpy()


def colour_records(label, openuniverse_table, izc_table):
    """Per band, the median M(band) - M(ANCHOR_BAND) of each population and the difference.

    The colour is formed per object before any median is taken, which differences away both the
    distance modulus and the class brightness offset. What is left is the shape of the SED, and a
    residual here is the two template libraries disagreeing about colour -- something the scalar
    offset of the brightness block cannot move and does not try to.
    """
    bands = [
        band
        for band in ALL_BANDS_BY_WAVELENGTH
        if band != ANCHOR_BAND and band in openuniverse_table.columns and band in izc_table.columns
    ]
    records = []
    for band in bands:
        openuniverse_colour = (openuniverse_table[band] - openuniverse_table[ANCHOR_BAND]).dropna()
        izc_colour = (izc_table[band] - izc_table[ANCHOR_BAND]).dropna()
        if len(openuniverse_colour) < MINIMUM_COLOUR_OBJECTS or len(izc_colour) < MINIMUM_COLOUR_OBJECTS:
            records.append(
                {
                    "label": label,
                    "band": band,
                    "colour": f"{band} - {ANCHOR_BAND}",
                    "openuniverse_median": np.nan,
                    "openuniverse_sigma": np.nan,
                    "openuniverse_n": len(openuniverse_colour),
                    "izc_median": np.nan,
                    "izc_sigma": np.nan,
                    "izc_n": len(izc_colour),
                    "residual": np.nan,
                    "residual_error": np.nan,
                }
            )
            continue
        openuniverse_median, openuniverse_error = median_and_error(openuniverse_colour.to_numpy())
        izc_median, izc_error = median_and_error(izc_colour.to_numpy())
        records.append(
            {
                "label": label,
                "band": band,
                "colour": f"{band} - {ANCHOR_BAND}",
                "openuniverse_median": openuniverse_median,
                # The width matters as much as the median: a colour the two populations agree on but
                # that one of them draws far more tightly than the other is still separable, and a
                # class whose colour has almost no scatter is a handle the classifier can take hold of.
                "openuniverse_sigma": float(np.std(openuniverse_colour.to_numpy())),
                "openuniverse_n": len(openuniverse_colour),
                "izc_median": izc_median,
                "izc_sigma": float(np.std(izc_colour.to_numpy())),
                "izc_n": len(izc_colour),
                "residual": openuniverse_median - izc_median,
                # Both sides are medians of finite samples here, unlike the brightness offset where the
                # OpenUniverse side is large enough to ignore, so the error carries both.
                "residual_error": float(np.hypot(openuniverse_error, izc_error)),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=Path("data/openuniverse/early_windows_deep.parquet"))
    parser.add_argument("--objects", type=int, default=600, help="izc objects generated per class")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classes", nargs="*", default=list(CALIBRATED), choices=list(CALIBRATED))
    parser.add_argument("--output", type=Path, default=Path("data/csp/izc_brightness_calibration.csv"))
    parser.add_argument(
        "--colour-output",
        type=Path,
        default=Path("data/csp/izc_colour_residuals.csv"),
        help="where the per-band colour diagnostic goes; it adjusts nothing and is reported only",
    )
    parser.add_argument(
        "--apply-above-sigma",
        type=float,
        default=2.0,
        help="report which offsets are worth applying: an offset smaller than this many standard "
        "errors of the median is noise, and applying it would tune the class to one draw",
    )
    arguments = parser.parse_args()

    cosmology = izc.openuniverse_cosmology()
    random_generator = np.random.default_rng(arguments.seed)
    # Read through pyarrow and dictionary-encode the two string columns before handing them to
    # pandas. The file is 17.5 million rows and `object_id` and `band` are `large_string`, so the
    # default path builds 35 million Python str objects and the process is killed for memory on a
    # 16 GB machine. `object_id` has one value per object and `band` has seven, so as categoricals
    # they cost a fraction of that and group and compare exactly the same.
    import pyarrow.parquet as pq

    table = pq.read_table(arguments.windows, columns=["object_id", "gentype", "z_CMB", "band", "mag_true"])
    for name in ("object_id", "band"):
        index = table.schema.get_field_index(name)
        table = table.set_column(index, name, table.column(name).dictionary_encode())
    windows = table.to_pandas()
    del table

    records = []
    colours = []
    for label in arguments.classes:
        gentype = CALIBRATED[label]
        openuniverse_table = openuniverse_peak_absolute_magnitudes(windows, gentype, cosmology)
        # Redshifts are resampled from the objects detected in the ANCHOR_BAND, not from every
        # object detected in any band: the izc side is compared against them in that band.
        anchor_index, openuniverse = anchor_column(openuniverse_table)
        redshift = anchor_index.get_level_values("z_CMB").to_numpy()
        drawn = random_generator.choice(redshift, size=arguments.objects, replace=True)
        izc_table = izc_peak_absolute_magnitudes(label, drawn, random_generator, cosmology)
        _, generated = anchor_column(izc_table)
        colours.extend(colour_records(label, openuniverse_table, izc_table))
        # Standard error of the median, which is what says whether an offset is worth applying.
        error = 1.253 * np.std(generated) / np.sqrt(len(generated))
        records.append(
            {
                "label": label,
                "openuniverse_median": np.median(openuniverse),
                "openuniverse_sigma": np.std(openuniverse),
                "openuniverse_n": len(openuniverse),
                "izc_median": np.median(generated),
                "izc_sigma": np.std(generated),
                "izc_n": len(generated),
                "offset": np.median(openuniverse) - np.median(generated),
                "offset_error": error,
                "adjustable": label in ADJUSTABLE,
            }
        )
        latest = records[-1]
        note = "" if label in ADJUSTABLE else "   (reported, not applied)"
        print(
            f"{label:<7s} OU {latest['openuniverse_median']:8.3f} "
            f"(sigma {latest['openuniverse_sigma']:.2f}, n {latest['openuniverse_n']:5d})   "
            f"izc {latest['izc_median']:8.3f} (sigma {latest['izc_sigma']:.2f}, "
            f"n {latest['izc_n']:4d})   offset {latest['offset']:+.3f} +- {error:.3f}{note}"
        )

    print()
    for record in records:
        significant = abs(record["offset"]) > arguments.apply_above_sigma * record["offset_error"]
        if record["adjustable"] and significant:
            key = record["label"] if record["label"] in izc.PEAK_ABSOLUTE_MAGNITUDE else "SN IIP"
            moved = record["offset"] + izc.PEAK_ABSOLUTE_MAGNITUDE[key][0]
            print(
                f"{record['label']:<7s} apply {record['offset']:+.3f}: "
                f"PEAK_ABSOLUTE_MAGNITUDE median moves to {moved:.3f}"
            )
        elif record["adjustable"]:
            errors = abs(record["offset"]) / record["offset_error"]
            print(
                f"{record['label']:<7s} converged: {record['offset']:+.3f} is {errors:.1f} "
                f"standard errors, below the {arguments.apply_above_sigma:.1f} threshold"
            )

    colour_table = pd.DataFrame(colours)
    print()
    print(f"Colour residuals against {ANCHOR_BAND}, per object before any median. Diagnostic only: nothing")
    print("below is adjusted, because a scalar luminosity offset cannot move a colour.")
    print()
    for label in arguments.classes:
        rows = colour_table[colour_table["label"] == label]
        for _, row in rows.iterrows():
            if not np.isfinite(row["residual"]):
                print(
                    f"{label:<7s} {row['colour']:<13s} too few objects "
                    f"(OU n {row['openuniverse_n']:.0f}, izc n {row['izc_n']:.0f})"
                )
                continue
            print(
                f"{label:<7s} {row['colour']:<13s} "
                f"OU {row['openuniverse_median']:+7.3f} (sigma {row['openuniverse_sigma']:.3f}, "
                f"n {row['openuniverse_n']:5.0f})   "
                f"izc {row['izc_median']:+7.3f} (sigma {row['izc_sigma']:.3f}, "
                f"n {row['izc_n']:4.0f})   "
                f"residual {row['residual']:+.3f} +- {row['residual_error']:.3f}"
            )

    table = pd.DataFrame(records)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(arguments.output, index=False)
    arguments.colour_output.parent.mkdir(parents=True, exist_ok=True)
    colour_table.to_csv(arguments.colour_output, index=False)
    print("\nwrote", arguments.output)
    print("wrote", arguments.colour_output)


if __name__ == "__main__":
    main()
