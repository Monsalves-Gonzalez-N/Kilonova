"""Central path configuration for the pipelines.

Every data location lives in ``configs/paths.yaml`` (or a file pointed to by the
``KN_PATHS_FILE`` environment variable) instead of being hardcoded in the code.
Precedence, highest first:

    CLI flag  >  environment variable KN_<FIELD>  >  YAML file

Relative paths in the YAML are resolved against the repository root, so the
committed defaults work on any checkout; external-volume paths stay absolute.
"""

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS_FILE = REPOSITORY_ROOT / "configs" / "paths.yaml"


@dataclass
class PathsConfig:
    openuniverse_source: Path | None = None  # directory with snana_*.hdf5 + snana_*.parquet pairs
    openuniverse_hdf5: Path | None = None  # single-field SED hdf5 (kn-early-windows)
    openuniverse_catalog: Path | None = None  # single-field object catalog parquet
    lanl_spectra: Path | None = None  # cached LANL rest-frame spectra parquet (kn-cache-lanl output)
    lanl_grid_dir: Path | None = None  # raw LANL kn_sim_cube_v1 grid of .dat spectra
    lanl_catalog: Path | None = None  # LANL grid index parquet (built if missing)
    kcor: Path | None = None  # SNANA kcor FITS with the Roman FilterTrans extension
    hourglass_objects: Path | None = None  # Hourglass positions parquet, used to sample MW E(B-V)
    output_dir: Path | None = None  # where pipeline products are written


def _resolve(raw_value):
    path = Path(str(raw_value)).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path


def load_paths(paths_file=None):
    if paths_file is None:
        paths_file = Path(os.environ.get("KN_PATHS_FILE", DEFAULT_PATHS_FILE))
    yaml_values = {}
    if Path(paths_file).exists():
        with open(paths_file) as file_handle:
            yaml_values = yaml.safe_load(file_handle) or {}

    values = {}
    for field in fields(PathsConfig):
        environment_value = os.environ.get(f"KN_{field.name.upper()}")
        raw_value = environment_value if environment_value is not None else yaml_values.get(field.name)
        values[field.name] = _resolve(raw_value) if raw_value is not None else None
    return PathsConfig(**values)


def require(path, field_name):
    """Resolve a configured path or exit with a friendly, actionable message."""
    if path is None:
        raise SystemExit(
            f"path '{field_name}' is not configured: set it in configs/paths.yaml, "
            f"via KN_{field_name.upper()}, or with the corresponding CLI flag"
        )
    path = Path(path)
    if not path.exists():
        hint = " (is the external volume mounted?)" if str(path).startswith("/Volumes") else ""
        raise SystemExit(f"{field_name}: {path} does not exist{hint}")
    return path
