"""How much of the classifier's decision is the single number "how bright is it".

The kilonova and contaminant populations were generated over nearly disjoint redshift ranges --
median z 0.08 against 1.26 -- which makes them nearly disjoint in apparent magnitude too. "Bright
implies kilonova" is therefore an almost perfect rule inside the dataset and a false one in the sky,
and a model is free to learn it instead of colour. This script measures whether it did.

Three measurements, in order of how hard they are to argue with:

  * false positive rate against redshift. Descriptive: it shows the effect but cannot separate
    redshift from everything correlated with it.
  * the trivial rule. Compares the transformer against ranking objects by their brightest first-epoch
    detection -- one number, no model.
  * the intervention. Shifts every magnitude token of a window, detections and 5 sigma limits alike,
    by the same amount. Colours, signal-to-noise, sigma_mag and the source-to-limit relation are all
    preserved exactly; the only thing that moves is where the window sits on the magnitude axis. A
    model reading physics does not care. This one is causal rather than correlational.

The faint end is not the problem and the intervention is not symmetric on purpose: "too faint to be
a kilonova" is a real limit, since Roman will not find one at z = 3. It is the bright end that is
unearned.

Run from the repository root:  python training/diagnostics/brightness_shortcut.py
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from split_inference import (
    DETECTION_TOKEN,
    NOT_OBSERVED_TOKEN,
    first_epoch_positions,
    load_model,
    load_test_split,
    predict,
)

DECISION_THRESHOLD = 0.5
REDSHIFT_EDGES = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 3.01]
MAGNITUDE_SHIFTS = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
SAMPLE_SIZE = 15000
RANDOM_SEED = 0


def brightest_detection(tokens, meta, index):
    """The brightest first-epoch detection of each object, NaN when it has none."""
    brightest = np.full(len(index), np.nan)
    for position, object_index in enumerate(index):
        positions = first_epoch_positions(tokens, meta, object_index)
        detections = positions[tokens["token_type_index"][positions] == DETECTION_TOKEN]
        if len(detections):
            brightest[position] = tokens["mag"][detections].min()
    return brightest


def magnitude_token_positions(tokens, meta, index):
    """Every first-epoch token that carries a magnitude, detections and upper limits alike."""
    positions = []
    for object_index in index:
        first_epoch = first_epoch_positions(tokens, meta, object_index)
        token_types = tokens["token_type_index"][first_epoch]
        positions.append(first_epoch[token_types != NOT_OBSERVED_TOKEN])
    return np.concatenate(positions)


def false_positive_rate_by_redshift(model, tokens, meta, device):
    index = np.arange(len(meta["is_kn"]))
    probability = predict(model, tokens, meta, index, device)
    is_kn, redshift = meta["is_kn"], meta["redshift"]
    print("false positive rate against redshift (one epoch, no redshift given)")
    print(f"{'redshift':>14} {'KN':>8} {'recall':>8} {'other':>8} {'FPR':>8}")
    for low, high in zip(REDSHIFT_EDGES[:-1], REDSHIFT_EDGES[1:], strict=True):
        in_bin = (redshift >= low) & (redshift < high)
        kilonovae = in_bin & is_kn
        contaminants = in_bin & ~is_kn
        recall = (probability[kilonovae] >= DECISION_THRESHOLD).mean() if kilonovae.sum() else np.nan
        false_positive = (
            (probability[contaminants] >= DECISION_THRESHOLD).mean() if contaminants.sum() else np.nan
        )
        print(
            f"{low:6.2f}-{high:6.2f} {int(kilonovae.sum()):8d} {recall:8.4f} "
            f"{int(contaminants.sum()):8d} {false_positive:8.4f}"
        )
    return probability


def compare_with_trivial_rule(probability, tokens, meta):
    index = np.arange(len(meta["is_kn"]))
    brightest = brightest_detection(tokens, meta, index)
    usable = np.isfinite(brightest)
    label = meta["is_kn"][usable].astype(int)
    score, magnitude = probability[usable], brightest[usable]

    print(f"\nthe transformer against ranking by brightness alone ({int(usable.sum())} objects)")
    print(f"{'':34s} {'ROC AUC':>9} {'PR AUC':>9}")
    print(
        f"{'transformer (one epoch, no z)':34s} "
        f"{roc_auc_score(label, score):9.4f} {average_precision_score(label, score):9.4f}"
    )
    print(
        f"{'trivial rule: brighter is KN':34s} "
        f"{roc_auc_score(label, -magnitude):9.4f} {average_precision_score(label, -magnitude):9.4f}"
    )
    print("\nSpearman correlation between P(KN) and the brightest detection")
    for name, mask in [
        ("all", np.ones(len(label), dtype=bool)),
        ("contaminants only", label == 0),
        ("kilonovae only", label == 1),
    ]:
        print(f"  {name:20s} rho = {spearmanr(score[mask], magnitude[mask]).statistic:+.4f}")


def shift_magnitude_scale(model, tokens, meta, device):
    """Move a window along the magnitude axis and watch the decision follow."""
    is_kn, redshift = meta["is_kn"], meta["redshift"]
    random_generator = np.random.default_rng(RANDOM_SEED)
    groups = {
        "contaminants z > 0.8": np.flatnonzero((~is_kn) & (redshift > 0.8)),
        "contaminants 0.4 < z < 0.8": np.flatnonzero((~is_kn) & (redshift > 0.4) & (redshift < 0.8)),
        "kilonovae z < 0.05": np.flatnonzero(is_kn & (redshift < 0.05)),
    }
    for name, population in groups.items():
        index = random_generator.choice(population, min(SAMPLE_SIZE, len(population)), replace=False)
        positions = magnitude_token_positions(tokens, meta, index)
        print(f"\n{name}  (n={len(index)})")
        print(f"{'shift [mag]':>12} {'median P(KN)':>14} {'fraction above threshold':>26}")
        for shift in MAGNITUDE_SHIFTS:
            shifted = {field: values.copy() for field, values in tokens.items()}
            shifted["mag"][positions] += shift
            probability = predict(model, shifted, meta, index, device)
            marker = "  <- untouched" if shift == 0.0 else ""
            print(
                f"{shift:+12.1f} {np.median(probability):14.4f} "
                f"{(probability >= DECISION_THRESHOLD).mean():26.4f}{marker}"
            )


def main():
    tokens, meta = load_test_split()
    model, device = load_model()
    probability = false_positive_rate_by_redshift(model, tokens, meta, device)
    compare_with_trivial_rule(probability, tokens, meta)
    shift_magnitude_scale(model, tokens, meta, device)


if __name__ == "__main__":
    main()
