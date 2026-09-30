"""Reconstruir la cache de tokens, y recortar de ella la del test.

POR QUE HACE FALTA UN SCRIPT. `openuniverse_data._load_or_build` cachea por NOMBRE de fichero, no
por contenido: si los parquets se regeneran con el mismo nombre, el entrenamiento carga la cache
vieja y no avisa. Regenerar los datos obliga a borrar la cache a mano, y esto lo hace explicito.

`openuniverse_tokens_test.npz` es el recorte solo-test de la cache, y hasta ahora se cortaba a
mano. Se corta del MISMO split que el entrenamiento usa -- `_leakage_aware_split` con su semilla --
asi que sale de la cache que el entrenamiento carga, que desde 2026-09 es la del izc.

    python training/build_token_cache.py --izc --cut-test
"""

import argparse
import inspect
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openuniverse_data import (  # noqa: E402
    GROUP_KEY_VERSION,
    _leakage_aware_split,
    _load_or_build,
    build_dataloaders,
)

# El split del recorte tiene que ser EL MISMO que el del entrenamiento, asi que sus parametros se
# leen de la firma de `build_dataloaders` en vez de copiarse aqui: si alla cambia la semilla, esto
# la sigue en vez de partir el test por otro lado y no decirlo.
_defaults = inspect.signature(build_dataloaders).parameters
SPLIT_FRACTIONS = _defaults["fractions"].default
SPLIT_SEED = _defaults["split_seed"].default

DATA_DIR = "data/openuniverse"
QUARANTINE = os.path.join(DATA_DIR, "_old_izc")
TOKEN_KEYS = ("day", "band_index", "token_type_index", "mag", "sigma_mag")
META_KEYS = ("offsets", "orig_label", "redshift", "group_key", "is_kn", "is_izc")


def quarantine(path, suffix):
    """Mover una cache superada a la cuarentena en vez de borrarla. Ver su README."""
    if not os.path.exists(path):
        return None
    os.makedirs(QUARANTINE, exist_ok=True)
    target = os.path.join(QUARANTINE, f"{os.path.basename(path)}.{suffix}")
    shutil.move(path, target)
    return target


def cut_test(cache_path, output_path):
    """El recorte solo-test, con el split que el entrenamiento hace: grupos enteros, misma semilla."""
    cached = np.load(cache_path, allow_pickle=False)
    meta = {key: cached[key] for key in META_KEYS}
    _, _, test = _leakage_aware_split(meta, SPLIT_FRACTIONS, SPLIT_SEED)
    test = np.sort(np.asarray(test))
    # El izc es augmentation solo de train: el recorte de test tiene que salir puro OpenUniverse.
    assert not meta["is_izc"][test].any(), "izc objects in the test cut"
    offsets = cached["offsets"]
    # `offsets` marca donde empieza y termina cada objeto en el array plano de tokens, asi que el
    # recorte no es un slice: hay que copiar los tokens de cada objeto y rehacer los offsets.
    # `cached[key]` DESCOMPRIME EL ARRAY ENTERO en cada acceso: dentro del bucle eso era releer
    # 0.33 GB por objeto, 211 690 veces, y el proceso moria por memoria antes de escribir nada.
    # Las posiciones se calculan primero y cada array se toca UNA vez.
    lengths = offsets[test + 1] - offsets[test]
    positions = np.concatenate([np.arange(offsets[index], offsets[index + 1]) for index in test])
    new_offsets = np.concatenate([[0], np.cumsum(lengths)])
    big = {key: cached[key][positions] for key in TOKEN_KEYS}
    small = {key: meta[key][test] for key in META_KEYS if key != "offsets"}
    np.savez(
        output_path,
        **big,
        offsets=np.asarray(new_offsets, dtype=offsets.dtype),
        **small,
        group_key_version=GROUP_KEY_VERSION,
        source_names=cached["source_names"] if "source_names" in cached else np.array([]),
    )
    return len(test), len(big["day"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--izc", dest="izc", action="store_true", default=True)
    parser.add_argument("--no-izc", dest="izc", action="store_false")
    parser.add_argument("--cut-test", action="store_true", help="rehacer openuniverse_tokens_test.npz")
    parser.add_argument("--suffix", default=time.strftime("stale-%Y-%m-%d"))
    arguments = parser.parse_args()

    data_dir = arguments.data_dir
    cache_name = "openuniverse_tokens_izc.npz" if arguments.izc else "openuniverse_tokens.npz"
    cache_path = os.path.join(data_dir, cache_name)
    moved = quarantine(cache_path, arguments.suffix)
    if moved:
        print(f"cache anterior -> {moved}")

    start = time.time()
    _load_or_build(
        cache_path,
        kn_deep=os.path.join(data_dir, "kn_windows_deep.parquet"),
        kn_wide=os.path.join(data_dir, "kn_windows_wide.parquet"),
        contaminant_deep=os.path.join(data_dir, "early_windows_deep.parquet"),
        contaminant_wide=os.path.join(data_dir, "early_windows_wide.parquet"),
        izc_deep=os.path.join(data_dir, "izc_windows_deep.parquet") if arguments.izc else None,
        izc_wide=os.path.join(data_dir, "izc_windows_wide.parquet") if arguments.izc else None,
    )
    print(f"{cache_path}: {os.path.getsize(cache_path)/1e9:.2f} GB en {time.time()-start:.0f} s")

    if arguments.cut_test:
        test_path = os.path.join(data_dir, "openuniverse_tokens_test.npz")
        moved = quarantine(test_path, arguments.suffix)
        if moved:
            print(f"recorte anterior -> {moved}")
        objects, tokens = cut_test(cache_path, test_path)
        print(
            f"{test_path}: {objects} objetos, {tokens} tokens, "
            f"{os.path.getsize(test_path)/1e6:.0f} MB (recortado de {cache_name})"
        )


if __name__ == "__main__":
    main()
