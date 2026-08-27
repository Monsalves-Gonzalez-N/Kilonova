"""What the classifier does when the anchor band is missing from the first visit.

The HLTDS cadence observes the anchor band (the bluest of the tier) at every visit, so in the whole
training set the first epoch has exactly three observed bands and the anchor is always one of them.
A window whose anchor slot holds a `not observed` token is therefore a configuration the model has
never been shown, and the answer it gives there is not defined by anything it learned.

The `n` token is not encoded as an upper limit -- it carries its own type embedding, a zero
magnitude and a zero magnitude mask, where `u` carries the 5 sigma limit with the mask set -- but
this measures whether the model nonetheless arrives at the same answer.

Run from the repository root:  python training/diagnostics/anchor_band_ablation.py
"""

import numpy as np
from split_inference import (
    NOT_OBSERVED_TOKEN,
    first_epoch_positions,
    load_model,
    load_test_split,
    predict,
)

SAMPLE_PER_CLASS = 20000
DECISION_THRESHOLD = 0.5
RANDOM_SEED = 0


def remove_anchor_band(tokens, meta, index):
    """Turn the bluest observed band of each first epoch into a `not observed` token.

    Returns the modified copy and the subset of `index` it could be applied to. The band ordering
    is by wavelength, so the bluest observed band IS the tier anchor: R062 for wide, Z087 for deep.
    Which tier an object belongs to never has to be looked up."""
    ablated = {field: values.copy() for field, values in tokens.items()}
    applied = []
    for object_index in index:
        positions = first_epoch_positions(tokens, meta, object_index)
        observed = positions[tokens["token_type_index"][positions] != NOT_OBSERVED_TOKEN]
        if len(observed) == 0:
            continue
        anchor_position = observed[np.argmin(tokens["band_index"][observed])]
        ablated["token_type_index"][anchor_position] = NOT_OBSERVED_TOKEN
        ablated["mag"][anchor_position] = np.nan
        ablated["sigma_mag"][anchor_position] = np.nan
        applied.append(object_index)
    return ablated, np.array(applied)


def main():
    tokens, meta = load_test_split()
    model, device = load_model()
    is_kn = meta["is_kn"]

    random_generator = np.random.default_rng(RANDOM_SEED)
    index = np.arange(len(is_kn))
    sample = np.concatenate(
        [
            random_generator.choice(index[is_kn], SAMPLE_PER_CLASS, replace=False),
            random_generator.choice(index[~is_kn], SAMPLE_PER_CLASS, replace=False),
        ]
    )

    print("how often is the anchor band already missing in the test split?")
    observed_per_window = []
    for object_index in sample:
        positions = first_epoch_positions(tokens, meta, object_index)
        token_types = tokens["token_type_index"][positions]
        observed_per_window.append(int((token_types != NOT_OBSERVED_TOKEN).sum()))
    counts = np.bincount(observed_per_window)
    for observed_bands, count in enumerate(counts):
        if count == 0:
            continue
        share = count / len(sample)
        print(f"  {observed_bands} observed bands in the first epoch: {count} ({share:.2%})")

    ablated, sample = remove_anchor_band(tokens, meta, sample)
    baseline = predict(model, tokens, meta, sample, device)
    without_anchor = predict(model, ablated, meta, sample, device)
    sample_is_kn = is_kn[sample]

    print("\nremoving the anchor band from the first visit")
    print(f"{'':16s} {'n':>7} {'median P(KN)':>26} {'fraction above threshold':>26}")
    for name, mask in [("KN", sample_is_kn), ("contaminants", ~sample_is_kn)]:
        before, after = baseline[mask], without_anchor[mask]
        print(
            f"{name:16s} {int(mask.sum()):7d} "
            f"{np.median(before):11.4f} -> {np.median(after):11.4f} "
            f"{(before >= DECISION_THRESHOLD).mean():11.4f} -> {(after >= DECISION_THRESHOLD).mean():11.4f}"
        )
    contaminants = ~sample_is_kn
    flipped = contaminants & (baseline < DECISION_THRESHOLD) & (without_anchor >= DECISION_THRESHOLD)
    print(
        f"\ncontaminants that become KN once the anchor is gone: {int(flipped.sum())} "
        f"({flipped.sum() / contaminants.sum():.2%} of contaminants)"
    )


if __name__ == "__main__":
    main()
