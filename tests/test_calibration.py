"""Prior correction (trd.md §15, §10.3).

Two properties, and the second is the one that keeps the metric table coherent: the
correction is strictly monotone, so AUC and every ranking metric are identical before and
after. That is why ranking metrics are computed on uncorrected scores and only the
probability metrics are reported twice.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from contentsignal.eval.calibration import (
    downsampling_rate,
    fit_isotonic,
    prior_correct,
    reliability_curve,
)


def test_round_trip_recovers_the_true_base_rate() -> None:
    """Given a known base rate and downsampling rate, correction recovers the prior.

    Construction: a population with base rate 0.002, negatives downsampled at w. A model
    that is perfectly calibrated on the *sampled* distribution predicts the sampled base
    rate; correcting it must land back on the population base rate.
    """
    rng = np.random.default_rng(0)
    true_rate = 0.002
    w = 0.01  # 1% of negatives survive sampling

    sampled_rate = true_rate / (true_rate + (1.0 - true_rate) * w)
    p_sampled = np.full(200_000, sampled_rate)
    corrected = prior_correct(p_sampled, w=w)

    assert np.allclose(corrected, true_rate, rtol=1e-9)

    # And empirically: draw labels at the corrected rate and confirm the mean matches
    # within Monte-Carlo error.
    drawn = rng.random(corrected.size) < corrected
    assert abs(drawn.mean() - true_rate) < 4.0 * np.sqrt(true_rate / corrected.size)


def test_correction_is_strictly_monotone_so_auc_is_unchanged() -> None:
    rng = np.random.default_rng(1)
    n = 50_000
    y = (rng.random(n) < 0.3).astype(int)
    p = np.clip(rng.beta(2, 5, size=n) + 0.3 * y, 1e-6, 1 - 1e-6)

    corrected = prior_correct(p, w=0.05)

    order_before = np.argsort(p, kind="stable")
    order_after = np.argsort(corrected, kind="stable")
    assert np.array_equal(order_before, order_after)
    assert roc_auc_score(y, p) == pytest.approx(roc_auc_score(y, corrected))


def test_correction_raises_the_probabilities() -> None:
    """Downsampling negatives inflates the sampled rate, so correction moves down."""
    p = np.array([0.1, 0.5, 0.9])
    assert np.all(prior_correct(p, w=0.1) < p)


def test_w_of_one_is_the_identity() -> None:
    """No downsampling means no correction."""
    p = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(prior_correct(p, w=1.0), p)


def test_endpoints_are_fixed() -> None:
    out = prior_correct(np.array([0.0, 1.0]), w=0.01)
    assert out[0] == 0.0
    assert out[1] == 1.0


@pytest.mark.parametrize("w", [0.0, -0.1, 1.5])
def test_invalid_downsampling_rate_rejected(w: float) -> None:
    with pytest.raises(ValueError, match="downsampling rate"):
        prior_correct(np.array([0.5]), w=w)


def test_probabilities_outside_unit_interval_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        prior_correct(np.array([0.5, 1.2]), w=0.5)


def test_downsampling_rate_from_counts() -> None:
    assert downsampling_rate(1_000, 100_000) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        downsampling_rate(200, 100)


def test_reliability_curve_drops_empty_bins() -> None:
    """Empty bins are omitted, not reported as zero.

    A zero-filled empty bin draws a calibration curve that dives to the origin for
    reasons that have nothing to do with the model.
    """
    p = np.concatenate([np.full(100, 0.05), np.full(100, 0.95)])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    predicted, observed = reliability_curve(y, p, bins=10)

    assert len(predicted) == 2
    assert np.allclose(predicted, [0.05, 0.95])
    assert np.allclose(observed, [0.0, 1.0])


def test_isotonic_is_monotone() -> None:
    rng = np.random.default_rng(2)
    p = rng.random(2_000)
    y = (rng.random(2_000) < p).astype(int)
    iso = fit_isotonic(p, y)

    grid = np.linspace(0.0, 1.0, 100)
    fitted = iso.predict(grid)
    assert np.all(np.diff(fitted) >= -1e-12)
