"""Prior correction and calibration (trd.md §10.3).

Arms are trained on a downsampled negative distribution, so `Arm.predict` returns
probabilities on the *sampled* distribution, not the true one. The correction back to the
population prior is applied here, by the evaluator, exactly once — never inside an arm.
Applying it in two places, or in one arm but not another, would make the probability
metrics incomparable across the very arms the project exists to compare.

The correction is strictly monotone, so AUC and every ranking metric are unaffected and
are computed on uncorrected scores. Log-loss and Brier are reported twice — sampled and
corrected, each explicitly labelled.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def prior_correct(p_sampled: np.ndarray, *, w: float) -> np.ndarray:
    """Map sampled-distribution probabilities back to the population prior.

    `w` is the negative downsampling rate: the fraction of true negatives that survived
    into the training set (prd.md §5). With one positive per `ratio` negatives drawn from
    a much larger candidate space, `w` is small and the correction is large.

    p_true = p / (p + (1 - p) / w)
    """
    if not 0.0 < w <= 1.0:
        raise ValueError(f"downsampling rate w must be in (0, 1], got {w}")
    p = np.asarray(p_sampled, dtype=np.float64)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities outside [0, 1]")
    return p / (p + (1.0 - p) / w)


def downsampling_rate(n_negatives_kept: int, n_negatives_total: int) -> float:
    """`w` from counts, so the rate is measured rather than assumed."""
    if n_negatives_total <= 0:
        raise ValueError("n_negatives_total must be positive")
    if not 0 < n_negatives_kept <= n_negatives_total:
        raise ValueError(f"kept {n_negatives_kept} of {n_negatives_total} negatives")
    return n_negatives_kept / n_negatives_total


def fit_isotonic(p_val: np.ndarray, y_val: np.ndarray) -> IsotonicRegression:
    """Isotonic recalibration, fitted on validation only.

    Used for arms 9/10, whose in-batch sampled softmax gives them a different negative
    distribution by design, so their probabilities are not on the same scale as arms 1-8
    (trd.md §9b.4). Isotonic is monotone, so it too leaves ranking metrics untouched.
    """
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y_val)


def reliability_curve(
    y: np.ndarray, p: np.ndarray, *, bins: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Mean predicted probability and observed frequency per equal-width bin.

    Empty bins are dropped rather than reported as zero, which would draw a calibration
    curve that dives to the origin for reasons that have nothing to do with the model.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError(f"y {y.shape} and p {p.shape} differ in shape")

    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)

    predicted: list[float] = []
    observed: list[float] = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        predicted.append(float(p[mask].mean()))
        observed.append(float(y[mask].mean()))
    return np.asarray(predicted), np.asarray(observed)
