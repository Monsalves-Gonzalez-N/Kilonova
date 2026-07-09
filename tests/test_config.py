from kilonova.config import REPOSITORY_ROOT, PathsConfig, load_paths, require


def test_yaml_paths_load_and_relative_resolution(tmp_path):
    paths_file = tmp_path / "paths.yaml"
    paths_file.write_text("lanl_spectra: data/lanl_spectra.parquet\nkcor: /tmp/kcor.fits\n")
    paths = load_paths(paths_file)
    assert paths.lanl_spectra == REPOSITORY_ROOT / "data" / "lanl_spectra.parquet"
    assert str(paths.kcor) == "/tmp/kcor.fits"
    assert paths.output_dir is None


def test_environment_overrides_yaml(tmp_path, monkeypatch):
    paths_file = tmp_path / "paths.yaml"
    paths_file.write_text("kcor: /from/yaml.fits\n")
    monkeypatch.setenv("KN_KCOR", "/from/env.fits")
    paths = load_paths(paths_file)
    assert str(paths.kcor) == "/from/env.fits"


def test_kn_paths_file_environment_variable(tmp_path, monkeypatch):
    paths_file = tmp_path / "other.yaml"
    paths_file.write_text("output_dir: /somewhere/out\n")
    monkeypatch.setenv("KN_PATHS_FILE", str(paths_file))
    assert str(load_paths().output_dir) == "/somewhere/out"


def test_require_missing_and_unconfigured(tmp_path):
    existing = tmp_path / "present.txt"
    existing.write_text("x")
    assert require(existing, "kcor") == existing

    for bad_value, expected_fragment in [(None, "not configured"), (tmp_path / "absent", "does not exist")]:
        try:
            require(bad_value, "kcor")
        except SystemExit as error:
            assert expected_fragment in str(error)
        else:
            raise AssertionError("require should exit")


def test_all_fields_none_by_default():
    assert all(value is None for value in vars(PathsConfig()).values())
