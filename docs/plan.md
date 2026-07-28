# Kilonova classification in Roman — plan

> **Status note (2026-07-08).** The project now trains **only on OpenUniverse**; Hourglass is no
> longer a training stage — it survives solely as the source of the photometric noise recipe
> (embedded in `kilonova.photometry.roman_noise`, provenance in `docs/hourglass_noise_recipe.md`
> and `notebooks/validation/validate_noise_recipe.ipynb`). The "two-stage Hourglass →
> OpenUniverse" sections below are therefore superseded and kept for the record; the token
> specification, model spec and evaluation sections remain current.
> Pending review by the author.

> **Status note (2026-07-28).** The **matched-twin construction is superseded**: LANL kilonovae
> are too faint in Roman to have analogues among the OpenUniverse contaminants, so injecting a KN
> twin per low-z contaminant at that contaminant's z was discarded. KNe are now simulated as a
> **LANL grid over a grid of redshifts**, independent of the contaminant population, with the same
> noise recipe and window logic as the contaminants. The z-leakage risk that motivated the twin
> (batch prior p(KN | z)) still applies and its diagnostics remain current; the mitigation is now
> the choice of the KN redshift grid, not per-object z-matching. Sections mentioning "twin" /
> "matched-twin" are kept for the record.

## Goal

Classify kilonovae (KNe) among transients observed by the Roman Space Telescope (launch ~Sept 2026; NIR, deep + wide modes). Isolate candidates for follow-up: **recall over precision**, thresholded against a follow-up budget.

**Paper scope**: the supervised transformer + early-classification degradation matrix is the paper — the first transient-classification transformer for Roman. The model is **developed and tested on Hourglass** (stage 1), then **fine-tuned on OpenUniverse** (stage 2, same KN-injection pipeline) as a simulation-to-simulation domain-transfer result: OpenUniverse delivers the full 5-band Roman filter set with **no wide/deep tier split**, i.e. the more realistic and complete band coverage Hourglass lacks (see Two-stage training). Contrastive pretraining and **real-data** (post-launch Roman) fine-tuning remain future work (second paper).

## Data

- **Contaminants — stage 1 (development)**: Hourglass simulated catalog (~10k events), dominated by SN Ia and SN II, redshift up to ~3. Contains **zero KNe**. Band×tier coverage is incomplete (wide `RZYJ`, deep `YJHF` — see coverage caveat); this is the set the architecture is built and the degradation matrix is first measured on.
- **Contaminants — stage 2 (fine-tuning)**: OpenUniverse Roman TDS transient light curves. Full **5-band** Roman filter set with **no wide/deep tier split**, so no per-tier missing-filter pattern — the more realistic and complete coverage. The fine-tuning target (see Two-stage training).
- **Kilonovae**: injected from the LANL model grid (`lanl_spectra.parquet`), simulated over a **grid of redshifts** with the same noise recipe, cadence and window logic as the contaminants (superseding the matched-twin construction — see the 2026-07-28 status note). Grid combinations reach ~3e8 light curves, many degenerate (different masses / wind / angle, identical light curve).
- KNe likely only classifiable out to z ≈ 0.5, but injections span the full detectable z range — the 5σ detectability cut does the physical filtering, no hard z cut.
- Observing structure: max 3 epochs spaced 5 days, up to 4 bands per epoch → **max 12 observations per event** the model sees at once. A **4th detection epoch is kept per object** as a buffer for the shift augmentation (see Early classification).
- **Simulation coverage caveat**: Hourglass simulated only 4 of the 5 ROTAC tier bands (wide `RZYJ`, missing H; deep `YJHF`, missing Z), whereas OpenUniverse carries the full filter set and was not split into wide/deep tiers at all. The two simulations therefore have **different, complementary band×tier coverage** — which is exactly what the two-stage training exploits: build on Hourglass's tiered-but-incomplete grid, then fine-tune onto OpenUniverse's complete-but-untiered one (see Two-stage training). Both are accepted as-is for the methodology paper (same spirit as the Hourglass scope decision) and recalibrated once ROTAC/Hourglass 2.0 and the real Roman cadence exist. This coverage mismatch is the reason the token spec drops the explicit mode embedding (see Token specification): an encoder that fingerprinted Hourglass's wide/deep grid would not transfer to OpenUniverse.

## Token specification

One token per visit (band × time):

| Field | Encoding |
|---|---|
| Δt since first detection | continuous time2vec — **linear term carries the decline rate**; few low-freq periodic terms only (no Fourier modulation: 3 epochs over ~15 d have no resolvable/physical periodicity) |
| Band | categorical embedding |
| Magnitude + σ_mag | continuous, **globally normalized** (never per-object — apparent brightness is signal) |
| Type ∈ {detection, 5σ upper limit, instrumental gap} | categorical embedding |

- Work in **magnitudes** with 5σ detection / non-detection convention (what Roman will deliver). Upper limit (`u`) token carries the **per-visit 5σ limiting magnitude** computed from that visit's `fluxcal_err` (`mag_5σ = ZP − 2.5·log10(5·fluxcal_err)`) + flag — never a fixed survey depth, since the depth actually reached is the information content of a non-detection. The same `5σ-from-fluxcal_err` recipe is applied to injected KNe so the limit cannot become a pipeline tell.
- Three magnitude-information levels, distinguished by the type embedding: `d` detection (real mag + σ_mag), `u` upper limit (5σ limit, source is *fainter* than this, σ masked), `n` instrumental gap (magnitude channel **masked** — no information, distinct from an upper limit). The `n` token still carries (Δt, band).
- **No explicit deep/wide mode token.** What a mode embedding would carry — the depth actually reached — is already in the tokens continuously: `σ_mag` for detections and the per-visit 5σ limiting magnitude for non-detections. In Hourglass the band set is moreover almost a deterministic tag of the tier (wide simulated as `RZYJ`, deep as `YJHF`; only Y/J overlap, and there depth separates them), so `band` already implies the mode and an explicit embedding is redundant. Worse, it would bake the **incomplete** Hourglass band×tier grid into the encoder as a fingerprint, widening the domain gap to OpenUniverse (no wide/deep split) and to real Roman (full 5-band tiers `RZYJH` / `ZYJHF`). We drop it and keep the encoder dependent only on portable per-token physics (Δt, band, mag, σ_mag, type) + the global `[Z]`. The residual tier systematics not captured by σ_mag are an accepted, smaller bias than the OOD fingerprint.
- Include the gap token: Roman cadence is deterministic, "not looked at" vs "looked and nothing" is information.
- **No pre-detection upper-limit token.** First detection is the time anchor (Δt = 0); we do not carry prior non-detections — the real Roman pipeline won't deliver them cleanly and keeping them risks a pipeline tell. The late-onset regime (a transient first caught when already ~5 days old) is covered instead by the **shift augmentation** (see Early classification).
- **Redshift as a global token** `[Z]` embedding (z, σ_z), or a learned `[NO_Z]` token. Three regimes: spec-z, photo-z with error, no z.

## Redshift error model

Hourglass provides z (likely from the transient, not the host). Convolve with σ(mag) photo-z error distributions taken from external catalogs; realistic regime proportions as augmentation. Use the **same host-error model for injected KNe and native contaminants** so z quality cannot become a spurious injected-vs-native feature. Recalibrate via fine-tuning once real Roman photo-z distributions exist.

## Redshift leakage and the matched-twin construction

> **Superseded (2026-07-28).** The twin construction below was discarded — LANL KNe are too faint
> to have OU analogues at a shared z. The leakage analysis and the diagnostics remain current; the
> mitigation now lives in the design of the KN redshift grid.

z plays two distinct roles in the classifier:

- **z × photometry interactions** (legitimate, wanted): z + apparent magnitude → luminosity; z × timescale → time dilation. A mag-24 transient at z = 1 is too luminous to be a KN.
- **Marginal class prior p(KN | z)** (artifact, must be controlled): class-balanced batches concentrate all KNe at z < 0.5 (detectability) while contaminants spread to z ~ 3. Locally at low z the batch then says "~80% of events are KN", and the lazy shortcut "low z ⇒ KN" achieves good loss **without looking at the light curve**. In reality even at z < 0.5 contaminants outnumber KNe by ~100:1.

The two directions are independent: "z > 0.5 ⇒ not KN" is physics and must be learned; "z < 0.5 ⇒ probably KN" exists only in the balanced batch and must not be.

**Construction**: for each low-z contaminant in the batch, generate its **KN twin under identical conditions** — same z, same σ_z (or same `[NO_Z]`), same cadence and depths (the per-visit 5σ limits, which now also carry the tier information the dropped mode token would have). High-z contaminants enter the batch without a twin. Within each pair z is identical and cannot separate the classes; only the photometry differs, so the z shortcut yields 50% error and the gradient is forced onto light-curve features. The z > 0.5 boundary is still learned from the untwinned high-z contaminants.

- The **z-regime {spec-z, photo-z, no-z} must be inherited from the twin contaminant** — symmetric z-dropout across classes, otherwise `[NO_Z]` becomes a class feature (same symmetry rule as σ_z and extinction).
- Real prevalence (~1% KN even at low z) enters only at threshold calibration against the follow-up budget — never in batch densities.

**Diagnostics** (paper figures / sanity checks):

- False-positive rate of contaminants **binned in z**: flat FPR within the detectable range = model uses the light curve; FPR concentrated at low z = the shortcut leaked in.
- **z-ablation at inference**: mask the z token on the test set and measure the recall drop. Moderate drop = z used as interaction (fine); collapse = over-reliance.

## Model

Small transformer encoder (2–4 layers, d_model 64–128) + `[CLS]` + softmax head. The transformer is justified by missing bands, three token types and the z token — not sequence length.

**Architecture spec**:

| Component | Choice |
|---|---|
| Type | Encoder-only, pre-LayerNorm, bidirectional attention |
| Layers / d_model / heads | 2–4 / 64–128 / 4 |
| Time | time2vec on Δt, **linear term emphasized** (decline-rate carrier); minimal low-freq sinusoids — **no order-based positional encoding, no high-freq Fourier modulation** |
| Aggregation | learnable `[CLS]` (+ global `[Z]` token) |
| Head | small MLP → softmax over 4 classes |
| Regularization | dropout 0.1–0.3, weight decay — model must stay small (~10k real contaminants) |

- Tokens are a **set indexed by (time, band)**, not an ordered sequence — time enters as a continuous feature inside the token, never as positional encoding. This is the single most important architectural choice: attention can only compute decline rates (Δmag/Δt — the discriminative physics, KN days vs SN weeks) if each token carries a good continuous representation of its time. That decline rate is co-equal with the **detection / upper-limit / gap pattern** (a source that fades below 5σ between epochs is KN-like) as the discriminative signal; periodicity is not a signal here — 3 epochs over ~15 days cannot resolve a frequency and transients are not pulsators. So time2vec's **linear term** is the workhorse (it is what lets attention difference token times into a decline rate); we keep only a few low-frequency sinusoids and add **no high-frequency Fourier modulation**.
- Token fusion: time2vec(Δt) + band embedding + type embedding + (mag, σ_mag) projection → d_model. The **type embedding {detection, upper limit, gap}** is the channel carrying the detection/non-detection pattern — co-equal with the decline rate as the KN discriminator.
- Efficient-attention variants (Informer etc.), decoders and encoder-decoders are irrelevant at ≤12 tokens; causal masking for early classification is rejected in favor of random prefix truncation (simpler, 3 forwards cost nothing).
- Precedent: this is essentially a small ATAT with upper-limit/gap tokens and a global z token. References: **ATAT** (Cabrera-Vives et al. 2024, A&A, arXiv:2405.03078), **Astromer** (Donoso-Oliva et al. 2023, A&A 670, A54, arXiv:2205.01677), **t2** (Allam & McEwen, arXiv:2105.06178).
- **Baseline for the paper**: LightGBM/MLP on summary features (per-band decline rate, colors, peak mag, z) so the transformer's degradation matrix is measured against something.

**Multiclass**: `{Ia, II, other (lumped), KN}`. Class-balanced batches: KNe come from the redshift-grid injection (oversampled to balance), minority contaminant classes oversampled. Real prevalence (~100 KNe vs 10k contaminants over the cycle) enters only at threshold calibration, never in training.

## On-the-fly KN generation (no materialized 300M curves)

Factorization:

1. **Precompute once**: rest-frame grid photometry on `(model, phase, band, z)` — interpolation over the existing Ray-parallel spectrum cache + Roman AB photometry.
2. **Per batch**: sample (model, extinction, z from the KN redshift grid) + apply the tier's own observing conditions (cadence, per-visit depths, z-regime — same recipe as the contaminants) + interpolate grid + noise draw. Pure indexing + Gaussian noise in the dataloader, CPU-only. Extinction follows the **same treatment applied to contaminants**.

Generator policies:

- Sample explosion phase relative to the cadence so "first detection" emerges naturally — never anchor explosion to a visit.
- Injections with zero detections are discarded (they would never enter a transient pipeline).
- Sample roughly flat in **observable space** (peak brightness, color, decline rate), not in physical parameter space, to avoid over-representing degenerate grid corners.

## Early classification (3 epochs)

One model, random prefix truncation during training: {epoch 1 (≤4 tokens), epochs 1–2 (≤8), full (≤12)}. Epochs measured from first detection. Evaluation = inference on prefixes; no separate per-epoch models.

**Shift augmentation (late onset).** Per object, with independent probability **0.20 each time it is drawn into a batch**, slide the observation window forward by one epoch: drop epoch 1, re-anchor Δt at epoch 2, feed epochs {2, 3, 4}. Applied only to **shift-eligible** objects — a detection in epoch 1 *and* a detection in epoch 2 (≥2 detection visits); this is why the 4th detection epoch is kept. The shift uses only real detections (no synthetic pre-detection limit) and simulates the ~20% of events the uniform [0, 5)-day explosion phase catches ≥4 days old. The draw is **stochastic in the dataloader (fresh each epoch), not a fixed split**; because it is conditional on eligibility, the shifted fraction of a batch is `0.20 × (fraction eligible)`, not 0.20 of all objects. Injected KNe **get the same shift policy** so the regime stays symmetric across classes.

**Single-epoch fast faders are hard negatives, not discards.** Objects detected in only one visit (`n_detection_epochs == 1`) decay too fast for a second epoch — morphologically the closest contaminants to a KN. They stay in the training set as hard negatives, are never shifted, and get their own **class × z census** (parallel to `tab:class_redshift`, restricted to single-epoch detections) to expose which classes the recall-first model will most confuse with KNe.

## Injection bias risk

All training KNe come from our injection pipeline; all contaminants from the Hourglass pipeline. If the noise/photometry recipes differ, the classifier learns "my pipeline vs theirs" instead of "KN vs not".

- **Cheap test (do now)**: apply our noise recipe to Hourglass's true magnitudes and compare σ_mag and 5σ-limit distributions against what Hourglass reports, per band and mode. One histogram-comparison cell; builds on the existing SN noise recipe validation notebook.
- **Full test (deferred)**: inject SNe Ia with our pipeline under matched conditions and train an injected-vs-native discriminator; ~50% accuracy = pipelines indistinguishable. Post-baseline, if time allows.

## Two-stage training: Hourglass → OpenUniverse

The architecture is **developed and validated on Hourglass** (stage 1), then **fine-tuned on OpenUniverse** (stage 2). Both stages use the same supervised objective and the same on-the-fly KN-injection / matched-twin construction; only the contaminant simulation and its observing grid change.

- **Why two stages.** Hourglass is tiered (wide/deep) but band-incomplete; OpenUniverse is band-complete (full 5-band Roman set) but untiered. Stage 1 establishes the architecture and the early-classification degradation matrix on the catalog we know best; stage 2 adapts the encoder to the more realistic, complete-coverage simulation — a closer proxy to real Roman — and **measures how the model transfers across simulations**. The sim→sim generalization is itself a paper result, not just a checkpoint.
- **What makes the transfer possible.** The encoder depends only on **portable per-token physics** (Δt, band, mag, σ_mag, type) + the global `[Z]`, with **no mode embedding** (see Token specification). Nothing in the input layer is specific to Hourglass's wide/deep grid, so the same weights accept OpenUniverse's untiered, full-band tokens directly. The depth information a mode token would have carried still rides in continuously via σ_mag and the per-visit 5σ limit — both well-defined in OpenUniverse too.
- **KNe in stage 2.** Same injection pipeline: KN twins are generated for OpenUniverse's low-z contaminants under **OpenUniverse's own observing conditions** (full-band cadence, per-visit 5σ depths, z, z-regime). No wide/deep split to inherit; the twin simply spans whatever bands the contaminant was observed in. The injection-bias controls (matched noise recipe, symmetric z/σ_z/extinction treatment) apply unchanged.
- **Fine-tuning recipe.** Initialize from the stage-1 checkpoint; continue supervised training on OpenUniverse batches at a reduced learning rate. Optionally **replay a fraction of Hourglass batches** to limit forgetting of the tiered regime. **Re-fit the global magnitude normalization is *not* allowed** to use OpenUniverse test data — fit on the OpenUniverse train split (or reuse stage-1 stats if distributions match, decided from a histogram check). Report the degradation matrix for **both** stages so the transfer cost is explicit.

## Evaluation

Central result: **recall at fixed candidate budget** × {epoch 1, 2, 3} × {spec-z, photo-z, no z} — the 3×3 degradation matrix and the paper's key figure. Calibrate scores (temperature scaling); choose threshold on the PR curve from an operational budget ("N candidates per week we can vet"). Calibration and PR curves are computed on a validation set **re-weighted to realistic prevalence**, not on balanced batches. The **no-z column must include contaminants over the full z range** — without z, a high-z SN is only separable by shape (weeks-long dilated rise vs days-long KN evolution) and colors, which is the real use case for events without a counterpart. Include the z-leakage diagnostics (FPR binned in z, z-ablation).

## Execution order (paper)

1. **Token spec + dataset module**: preprocessing emits a clean per-visit table (`cid, mjd, band, fluxcal, fluxcal_err, token_type, mode, n_detection_epochs`, kept rows `first_detection ≤ mjd ≤ 4th_detection_epoch`). `mode` is **metadata only** — kept for the per-band-and-mode noise validation and for matching the twin's depths, never tokenized (see Token specification). The dataloader computes Δt from first detection, converts to (mag, σ_mag), applies **global** normalization (stats fit once on the train set, reused for injected KNe), draws the shift augmentation, and assembles token tensors + mask + the global `[Z]` token. The on-the-fly KN generator paired to contaminant conditions plugs in at this stage.
2. **Supervised transformer (stage 1, Hourglass)** with truncation + z-regime augmentation.
3. **Cheap noise-recipe validation** (in parallel with 2).
4. **Degradation matrix evaluation (stage 1).**
5. **Fine-tune on OpenUniverse (stage 2)**: port the dataset module to the OpenUniverse TDS light curves (full 5-band, untiered — the per-visit table and on-the-fly KN injection are reused unchanged since they carry no mode token), fit/validate the magnitude normalization on the OpenUniverse train split, fine-tune from the stage-1 checkpoint (reduced lr, optional Hourglass replay), and **re-run the degradation matrix (stage 2)** to report the cross-simulation transfer.

## Future work (second paper / post-launch)

- **SupCon contrastive pretraining**, positive hierarchy: same curve with jitter-within-σ / token dropout (~0.1/token) / z-resample-within-error / truncation > same KN under different observing conditions > same class. Never make views invariant to z across arbitrary values — only within σ_z. Linear probe vs the supervised baseline.
- **Fine-tuning on real Roman data** (contaminants only, no KNe): self-supervised domain adaptation of the encoder with replay of simulated batches; optional kNN-to-simulated-KNe in embedding space as a second, recall-boosting detector.

## Open items

- Source file/format of the final Hourglass table (the one explored in `hourglass_eda.ipynb`).
- OpenUniverse TDS port (stage 2): map the OpenUniverse light curves into the same per-visit token table as Hourglass (`cid, mjd, band, fluxcal, fluxcal_err, token_type, n_detection_epochs`), confirm the 5-band / untiered handling needs no schema change, and decide whether to refit or reuse the stage-1 magnitude normalization.
- Photo-z error catalog to adopt for σ(mag).
- Definition of the follow-up candidate budget N.
