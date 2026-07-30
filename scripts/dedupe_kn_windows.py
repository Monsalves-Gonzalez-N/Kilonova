"""Drop the duplicate realization of every colliding kn_object_id, in place.

Three object_ids per tier name two different realizations (same LANL model, angle, redshift node
and explosion offset to 4 decimals, different noise draw) because kn_object_id leaves out the
noise_id. Keep the first block, drop the second: `duplicated(keep='first')` over
(object_id, epoch, band) marks exactly the second block's rows.

Writes a temp file next to the target, verifies it, then replaces atomically.
"""

import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

KEY = ["object_id", "epoch", "band"]


def dedupe(path):
    keys = pq.read_table(path, columns=KEY).to_pandas()
    drop_mask = keys.duplicated(subset=KEY, keep="first").to_numpy()
    n_drop = int(drop_mask.sum())
    ids = sorted(keys.loc[drop_mask, "object_id"].unique())
    print(f"{os.path.basename(path)}: {n_drop} filas a borrar, {len(ids)} object_id: {ids}")
    if n_drop == 0:
        return

    table = pq.read_table(path)
    kept = table.filter(pa.array(~drop_mask))
    assert kept.schema.equals(table.schema), "el schema cambio al filtrar"

    temp_path = f"{path}.dedupe-tmp"
    pq.write_table(kept, temp_path, compression="snappy")

    check = pq.read_table(temp_path, columns=KEY).to_pandas()
    assert len(check) == table.num_rows - n_drop, "conteo de filas inesperado"
    assert not check.duplicated(subset=KEY).any(), "siguen quedando duplicados"
    assert check.object_id.nunique() == keys.object_id.nunique(), "se perdio algun object_id"
    written = pq.ParquetFile(temp_path)
    assert written.schema_arrow.metadata == pq.ParquetFile(path).schema_arrow.metadata, "metadata perdida"

    os.replace(temp_path, path)
    print(f"  -> {len(check):,} filas, {check.object_id.nunique():,} objetos, {len(ids)} duplicadas fuera")


if __name__ == "__main__":
    for tier_path in sys.argv[1:]:
        dedupe(tier_path)
