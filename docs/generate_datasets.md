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

The deficit is 670 595 objects; `--deficit-scale 0.9` fills 90 % of **every** bin — the shortfall
is spread uniformly over the whole range, it does not concentrate at high z — which is 603 537
objects and 40 min on 6 workers. That leaves (OpenUniverse + izc) / KN at 0.80–1.10 per bin in
deep, and it is enough: the shortcut this sample exists to remove was 29 contaminants against
239 369 kilonovae in the first bin, and a residual 12 % imbalance is not exploitable the way a
factor of 8000 was. Smoke test with `--limit-objects 400`, which keeps the deficit's shape and
only changes the count.

Parent reuse is NOT what that knob controls. Parents are drawn **with replacement**, so collisions
set the reuse rather than the size of the pool: SN Ib and SN Ic come out at 1.59 copies per parent
at 0.9 and would be 1.67 at 1.0, even though they ask for about as many objects as OpenUniverse has
parents of those classes. What contains it is the leakage-aware split on `parent_key`, which keeps
a parent and all of its copies on one side.

## 3c. Train

```bash
python training/train_lightning.py --data-dir data/openuniverse
```

The izc sample is IN by default — it exists to remove the redshift shortcut, so training without
it is the ablation, and that is what `--no-izc` selects. The two mixes use different token caches
(`openuniverse_tokens_izc.npz` and `openuniverse_tokens.npz`), so switching does not silently reuse
the wrong one.

## 4. Publish (never git — DVC)

All six parquets are outputs of `dvc.yaml` STAGES, not free-standing `dvc add` files, so `dvc add`
refuses them ("overlaps with an output of stage"). Record them with `dvc commit`, which writes the
hashes of what is on disk into `dvc.lock` without re-running anything:

```bash
dvc commit -f early_windows kilonova_windows izc_windows
dvc push
git add dvc.lock dvc.yaml params.yaml
git commit -m "Regenerar los datasets"
git push
```

The parquets themselves must never be committed to git; only `dvc.lock`.

## 5. On the other machine

```bash
git clone https://github.com/Monsalves-Gonzalez-N/Kilonova.git && cd Kilonova
pip install -e .
dvc pull                     # the six parquets, ~2.6 GB
python training/train_lightning.py --data-dir data/openuniverse
```

`dvc pull` reads the Dropbox remote (`~/Dropbox/Kilonova/dvc-kilonova`), so Dropbox has to have
finished syncing there AND the files have to be real rather than placeholders — mark the folder
"available offline" in Finder first. A placeholder reads as an EMPTY FILE with no error, which is
the failure mode this repository has hit before.
