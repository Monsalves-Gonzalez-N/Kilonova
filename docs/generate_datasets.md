# Dataset generation runbook (training laptop)

Regenerate the FOUR classification datasets after two changes committed on 2026-07-28:

1. **Cadence fix** (`c8a3956`): `bands_observed_at_visit` now alternates the published HLTDS
   sequences — wide `RZY`/`RJH`, deep `ZYJ`/`ZHF`. The previous parquets were generated with the
   wrong filter pair per visit (`RZJ`/`RYH`, `ZYH`/`ZJF`) and must ALL be regenerated.
2. **KN injection replaced** (`15d0b8a`): the per-transient same-z injection is gone. Kilonovae
   are now an independent population over a log redshift grid (`kn-kilonova-windows`), written to
   their own `kn_windows_{tier}.parquet` — the OpenUniverse parquets contain contaminants only.

| Dataset | Command | Output |
|---|---|---|
| Contaminants deep + wide | `kn-run-openuniverse` | `data/openuniverse/early_windows_{deep,wide}.parquet` |
| Kilonovae deep + wide | `kn-kilonova-windows` | `data/openuniverse/kn_windows_{deep,wide}.parquet` |

## Prerequisites

```bash
git pull
conda activate kilonova
pip install -e .            # registers the new kn-kilonova-windows entry point
dvc pull data/dust_generation/lanl_spectra.parquet.dvc   # 10G, needed by kn-kilonova-windows
```

Mounted volumes:

- `/Volumes/T7/openuniverse2025` — OpenUniverse snana hdf5 + parquet pairs (33 fields), read by
  `kn-run-openuniverse`.
- `/Volumes/Elements` — DVC remote (`/Volumes/Elements/dvc-kilonova`), needed only for
  `dvc pull` / `dvc push`.

All paths come from `configs/paths.yaml` (override with `KN_<FIELD>` env vars or CLI flags).

## 1. Smoke tests (minutes)

```bash
kn-early-windows --limit-ou 5 --output-dir /tmp/smoke
kn-kilonova-windows --limit-kn 20 --output-dir /tmp/smoke
pytest && ruff check src tests
```

Sanity check on the smoke output before the full runs — the cadence per visit must be exactly:

- deep: epoch 1 observes Z087+Y106+J129, epoch 2 observes Z087+H158+F184, alternating;
- wide: epoch 1 observes R062+Z087+Y106, epoch 2 observes R062+J129+H158, alternating.

(The `observed` column of the parquet encodes this; the anchor band Z087/R062 is observed at
every epoch.)

## 2. Contaminants (expensive — the reason this runs here)

```bash
kn-run-openuniverse
```

Writes `early_windows_{deep,wide}.parquet` to `output_dir` (default `data/openuniverse`),
overwriting the stale wrong-cadence files. Same runtime as the previous full run.

## 3. Kilonovae (much cheaper, reads only the LANL parquet)

```bash
kn-kilonova-windows
```

Defaults: log grid z = 0.01 → 1.0, 50 nodes, 200 realizations per node (10k KNe; most skip —
LANL KNe only detect at z ≲ 0.1, by design). All knobs are flags: `--redshift-min/max`,
`--n-redshift`, `--redshift-spacing {log,linear}`, `--realizations-per-redshift`, `--seed`.
Realizations are shared across tiers (the same KN observed in deep and wide); the KN `object_id`
string is `{simulation_id}_{angle_index}_{offset:.4f}_{z:.4f}` and rows carry
`gentype=50` / `label="KN"`. Schema: same 14 columns as the contaminant parquets.

## 4. Publish (never git — DVC)

```bash
dvc add data/openuniverse/early_windows_deep.parquet data/openuniverse/early_windows_wide.parquet
dvc add data/openuniverse/kn_windows_deep.parquet data/openuniverse/kn_windows_wide.parquet
dvc push
git add data/openuniverse/*.dvc data/.gitignore
git commit -m "Regenerate datasets: fixed cadence + KN redshift grid"
git push
```

The parquets themselves must never be committed to git; only the `.dvc` pointers.
