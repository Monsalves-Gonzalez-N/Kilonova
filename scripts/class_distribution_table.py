"""Class distribution of the OpenUniverse contaminant windows, as the LaTeX table of the paper.

Counts unique objects per class and tier in early_windows_{deep,wide}.parquet, plus the
re-rendered low-redshift contaminants of izc_windows_{deep,wide}.parquet with --with-izc, and
writes a complete table environment to output_dir, so the paper includes it with \\input and
nobody copies numbers by hand. Run it against the SAME parquets the token cache was built from:
the izc parquets are only the training set if dvc.lock points at them. Every object in those
parquets has at least one (visit, band) with S/N >= 5 by construction of early_windows.py; the
--minimum-detections flag exists only to report how the census would shrink under a stricter cut.
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from kilonova.config import load_paths, require

TIERS = ["deep", "wide"]
# Order of the rows: decreasing abundance, the same as the paper table.
CLASS_ORDER = ["SN II", "SN Ia", "SN Ic", "SN Ib", "SN Iax", "TDE", "SLSN-I", "PISN-He", "PISN-H"]
CLASS_LATEX = {
    "SN II": "SN~II",
    "SN Ia": "SN~Ia",
    "SN Ic": "SN~Ic",
    "SN Ib": "SN~Ib",
    "SN Iax": "SN~Iax",
    "TDE": "TDE",
    "SLSN-I": "SLSN-I",
    "PISN-He": "PISN-He",
    "PISN-H": "PISN-H",
    "KN": "KN",
}
CAPTION = (
    r"Simulated sample per HLTDS tier: OpenUniverse contaminants and LANL kilonovae, after the "
    r"Roman noise model of Eq.~\ref{eq:noise}. An object is kept when at least one epoch and "
    r"band reaches $\mathrm{S/N} \geq 5$."
)


def count_objects_per_class(parquet_path, minimum_detections):
    table = pq.read_table(parquet_path, columns=["object_id", "label", "detected"]).to_pandas()
    detections_per_object = table.groupby("object_id")["detected"].sum()
    kept_objects = detections_per_object.index[detections_per_object >= minimum_detections]
    labels = table.drop_duplicates("object_id").set_index("object_id")["label"]
    return labels.loc[kept_objects].value_counts()


def build_class_distribution(windows_dir, minimum_detections=1, with_izc=False, with_kn=False):
    sources = ["early_windows", "izc_windows"] if with_izc else ["early_windows"]
    if with_kn:
        sources.append("kn_windows")
    counts = {}
    for tier in TIERS:
        counts[tier] = pd.Series(dtype=int)
        for source in sources:
            parquet_path = require(Path(windows_dir) / f"{source}_{tier}.parquet", f"{source}_{tier}")
            class_counts = count_objects_per_class(parquet_path, minimum_detections)
            counts[tier] = counts[tier].add(class_counts, fill_value=0)
    distribution = pd.DataFrame(counts).fillna(0).astype(int)
    row_order = CLASS_ORDER + ["KN"] if with_kn else CLASS_ORDER
    distribution = distribution.reindex(row_order).fillna(0).astype(int)
    distribution["total"] = distribution[TIERS].sum(axis=1)
    distribution["percent"] = 100.0 * distribution["total"] / distribution["total"].sum()
    return distribution


def format_percent(percent):
    if percent < 0.05:
        return r"$\sim$0.0\%"
    return f"{percent:.1f}\\%"


def class_distribution_to_latex(distribution, caption=CAPTION, label="tab:class_distribution"):
    lines = [
        r"\begin{table}[!htb]",
        r"    \centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \begin{tabular}{lrrrr}",
        r"        \hline",
        r"        Class & Deep & Wide & Total & \% \\",
        r"        \hline",
    ]
    for class_name, row in distribution.iterrows():
        if class_name == "KN":
            lines.append(r"        \hline")
        lines.append(
            f"        {CLASS_LATEX[class_name]:<8} & {int(row['deep']):>6d} & {int(row['wide']):>6d} & "
            f"{int(row['total']):>7d} & {format_percent(row['percent'])} \\\\"
        )
    totals = distribution[TIERS + ["total"]].sum()
    lines += [
        r"        \hline",
        f"        Total    & {int(totals['deep']):>6d} & {int(totals['wide']):>6d} "
        f"& {int(totals['total']):>7d} & 100\\% \\\\",
        r"        \hline",
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--windows-dir", type=Path, help="directory with early_windows_{deep,wide}.parquet")
    parser.add_argument(
        "--output",
        type=Path,
        help="destination .tex (default: output_dir/class_distribution_table.tex)",
    )
    parser.add_argument("--minimum-detections", type=int, default=1)
    parser.add_argument(
        "--with-izc", action="store_true", help="add the izc_windows_{tier}.parquet contaminants"
    )
    parser.add_argument("--with-kn", action="store_true", help="add a KN row from kn_windows_{tier}.parquet")
    arguments = parser.parse_args()

    paths = load_paths()
    windows_dir = arguments.windows_dir or require(paths.output_dir, "output_dir")
    default_output = Path(require(paths.output_dir, "output_dir")) / "class_distribution_table.tex"
    output_path = arguments.output or default_output

    distribution = build_class_distribution(
        windows_dir, arguments.minimum_detections, arguments.with_izc, arguments.with_kn
    )
    print(distribution.to_string())
    print(f"total: {distribution['total'].sum()}")
    output_path.write_text(class_distribution_to_latex(distribution))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
