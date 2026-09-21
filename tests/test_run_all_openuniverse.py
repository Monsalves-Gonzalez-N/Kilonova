"""kn-run-openuniverse: los shards por campo, el reanudado y que paralelizar no cambie el resultado.

Paralelizar es seguro porque el ruido no sale de un stream compartido: `build_early_window` lo
siembra con `int(object_id)`, asi que un campo da las mismas filas lo renderice el proceso que lo
renderice. Este test es lo que sostiene esa afirmacion.
"""

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("galsim")

from kilonova.cli import run_all_openuniverse as runner  # noqa: E402


def write_field(directory, snana_id, n_objects=6):
    """Un par snana_<id>.{hdf5,parquet} sintetico, con la forma que el runner espera."""
    mjd = np.arange(0.0, 60.0, 1.0)
    ids = [1000 + one for one in range(n_objects)]
    with h5py.File(directory / f"{snana_id}.hdf5", "w") as hdf5:
        for index, object_id in enumerate(ids):
            group = hdf5.create_group(str(object_id))
            group.create_dataset("mjd", data=mjd)
            for band in "RZYJHFK":
                group.create_dataset("mag_" + band, data=np.full(len(mjd), 21.0 + 0.1 * index))
    pd.DataFrame({"id": ids, "z_CMB": np.full(n_objects, 0.2), "gentype": np.full(n_objects, 32)}).to_parquet(
        directory / f"{snana_id}.parquet", index=False
    )


def build(tmp_path, workers):
    source = tmp_path / f"source_{workers}"
    source.mkdir()
    for snana_id in ("snana_1", "snana_2", "snana_3"):
        write_field(source, snana_id)
    output = tmp_path / f"out_{workers}" / "early_windows_deep.parquet"
    output.parent.mkdir(parents=True)
    written = runner.process_tier(
        "deep", sorted(str(one) for one in source.glob("*.hdf5")),
        output, tmp_path / f"shards_{workers}", workers=workers,
    )
    assert written
    return pd.read_parquet(output).sort_values(["object_id", "epoch", "band"]).reset_index(drop=True)


def test_parallel_gives_exactly_what_serial_gives(tmp_path):
    assert build(tmp_path, 1).equals(build(tmp_path, 3))


def test_a_field_already_sharded_is_not_recomputed(tmp_path):
    """El shard es el checkpoint: relanzar reanuda en vez de rehacer el campo entero."""
    source = tmp_path / "source"
    source.mkdir()
    write_field(source, "snana_1")
    shards = tmp_path / "shards"
    payload = ("deep", str(source / "snana_1.hdf5"), None, str(shards))

    shards.mkdir()
    snana_id, n_detected, _, _, note = runner.process_field(payload)
    assert note is None and n_detected > 0
    assert runner.shard_path(shards, "deep", "snana_1").exists()

    again = runner.process_field(payload)
    assert again[4] == "ya estaba"
    assert again[1] is None, "no vuelve a contar detecciones porque no vuelve a abrir el hdf5"
