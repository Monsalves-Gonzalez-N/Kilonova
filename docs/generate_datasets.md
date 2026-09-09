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
| Low-redshift contaminants (izc) | `kn-izc-windows` | `data/openuniverse/izc_windows_{deep,wide}.parquet` |

## Prerequisites

```bash
git pull
conda activate kilonova
pip install -e .            # registers the new kn-kilonova-windows entry point
dvc pull data/dust_generation/lanl_spectra.parquet.dvc   # 10G, needed by kn-kilonova-windows
```

Mounted volumes:

- `/Volumes/T7/openuniverse2025` — OpenUniverse snana hdf5 + parquet pairs (33 fields), read by
  `kn-run-openuniverse` and `kn-izc-windows`. `configs/paths.yaml` points `openuniverse_source` at
  the Dropbox copy, whose files are **placeholders that read as empty rather than as an error**
  unless they are marked "available offline" in Finder, so the izc run refuses a zero-byte file
  instead of generating a sample with no brightness in it. The simplest thing is to pass
  `--source /Volumes/T7/openuniverse2025`.
- `/Volumes/Elements` — DVC remote (`/Volumes/Elements/dvc-kilonova`), needed only for
  `dvc pull` / `dvc push`.

All paths come from `configs/paths.yaml` (override with `KN_<FIELD>` env vars or CLI flags).

## 1. Smoke tests (minutes)

```bash
kn-early-windows --limit-ou 5 --output-dir /tmp/smoke
kn-kilonova-windows --limit-kn 20 --output-dir /tmp/smoke
kn-izc-windows --source /Volumes/T7/openuniverse2025 --limit-objects 400 --workers 4 --output-dir /tmp/smoke
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

## 3b. Low-redshift contaminants (needs the hdf5 AND the two window sets above)

```bash
kn-izc-windows --source /Volumes/T7/openuniverse2025 --deficit-scale 0.9 --workers 6
```

Re-renders OpenUniverse objects at the redshifts the survey has none at. FOUR classes — SN Ia,
SN Ib, SN Ic and SN II (IIP + IIL) — and only those: every class it generates is one whose
spectral model is OpenUniverse's own, so the sample redistributes OpenUniverse's population in
redshift rather than adding to it. SN Iax and TDE were dropped for that reason; the models are
still in `intermediate_z_contaminants`, unused. It reads, in order:

- `data/openuniverse/snana_catalogs/*.parquet`, the 33 catalogues (135 MB, gitignored input) —
  which object to re-render, with its template, its SALT parameters and its host screen;
- `kn_windows_{tier}.parquet` and `early_windows_{tier}.parquet` — the redshift DEFICIT, bin by
  bin, which is how many objects to generate and at what redshift. Both files must be the ones the
  training set will use, or the sample fills a hole that is not there;
- `snana_{healpix}.hdf5` — the parent's full light curve, which is where its brightness comes from.

The deficit is 670 595 objects; `--deficit-scale 0.9` fills 90 % of every bin, which is 603 537
objects and about 1.5 h on 6 workers. The 0.9 is deliberate: at 1.0 the SN Ib and SN Ic shares ask
for 13 % more objects than OpenUniverse has parents of those classes, so they get drawn with
replacement; at 0.9 that falls to 2 %. The unfilled 10 % sits in the bins above z = 0.45, where
OpenUniverse already has contaminants of its own. Smoke test with `--limit-objects 400`, which
keeps the deficit's shape and only changes the count.

## 3c. Train

```bash
python training/train_lightning.py --data-dir data/openuniverse
```

The izc sample is IN by default — it exists to remove the redshift shortcut, so training without
it is the ablation, and that is what `--no-izc` selects. The two mixes use different token caches
(`openuniverse_tokens_izc.npz` and `openuniverse_tokens.npz`), so switching does not silently reuse
the wrong one.

## 4. Publish (never git — DVC)

```bash
dvc add data/openuniverse/early_windows_deep.parquet data/openuniverse/early_windows_wide.parquet
dvc add data/openuniverse/kn_windows_deep.parquet data/openuniverse/kn_windows_wide.parquet
dvc add data/openuniverse/izc_windows_deep.parquet data/openuniverse/izc_windows_wide.parquet
dvc push
git add data/openuniverse/*.dvc data/.gitignore
git commit -m "Regenerate datasets: fixed cadence + KN redshift grid"
git push
```

The parquets themselves must never be committed to git; only the `.dvc` pointers.
