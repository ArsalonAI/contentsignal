"""Negative sampling (trd.md §15, §7).

The reproducibility tests carry more weight than they look like they do. Arms 1-8 assert
a shared row-set digest before training, and that assertion is the only thing standing
between "text adds 0.005 AUC" and "we drew different negatives this time".
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter

import numpy as np
import polars as pl
import pytest

from contentsignal.config import SamplerConfig
from contentsignal.sampling.negatives import (
    AliasTable,
    candidate_pool,
    sample_negatives,
    window_seed,
)
from tests.conftest import AS_OF

CFG = SamplerConfig(ratio=10, pop_exponent=0.75, pop_lookback_weeks=12, seed=17)
WINDOW = "rank_w4"


@pytest.fixture
def pool(synthetic_txns: pl.LazyFrame) -> pl.DataFrame:
    return candidate_pool(synthetic_txns, as_of=AS_OF, cfg=CFG)


@pytest.fixture
def negatives(synthetic_positives: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    return sample_negatives(synthetic_positives, pool=pool, cfg=CFG, window_name=WINDOW)


def test_ratio_exact(synthetic_positives: pl.DataFrame, negatives: pl.DataFrame) -> None:
    """Each customer gets exactly ratio x their own positive count."""
    pos = synthetic_positives.group_by("customer_idx").len().sort("customer_idx")
    neg = negatives.group_by("customer_idx").len().sort("customer_idx")
    joined = pos.join(neg, on="customer_idx", suffix="_neg")

    assert joined.height == pos.height, "a customer with positives received no negatives"
    assert (joined["len_neg"] == joined["len"] * CFG.ratio).all()


def test_no_positive_sampled_as_negative(
    synthetic_positives: pl.DataFrame, negatives: pl.DataFrame
) -> None:
    """The two sets are disjoint per (customer, article)."""
    overlap = negatives.join(synthetic_positives, on=["customer_idx", "article_id"], how="inner")
    assert overlap.height == 0, f"{overlap.height} positives were also sampled as negatives"


def test_no_duplicate_negatives(negatives: pl.DataFrame) -> None:
    """Per customer, a negative article appears once.

    A duplicate would silently upweight that customer's loss contribution.
    """
    assert negatives.height == negatives.unique(["customer_idx", "article_id"]).height


def test_labels_are_zero(negatives: pl.DataFrame) -> None:
    assert (negatives["y"] == 0).all()
    assert negatives["y"].dtype == pl.Int8


def test_determinism(synthetic_positives: pl.DataFrame, pool: pl.DataFrame) -> None:
    """Two runs at the same seed produce byte-identical Parquet."""
    first = sample_negatives(synthetic_positives, pool=pool, cfg=CFG, window_name=WINDOW)
    second = sample_negatives(synthetic_positives, pool=pool, cfg=CFG, window_name=WINDOW)

    import io

    a, b = io.BytesIO(), io.BytesIO()
    first.write_parquet(a, compression="zstd")
    second.write_parquet(b, compression="zstd")
    assert a.getvalue() == b.getvalue()


def test_different_windows_draw_differently(
    synthetic_positives: pl.DataFrame, pool: pl.DataFrame
) -> None:
    """Windows are independent draws, but both derive from the one root seed."""
    a = sample_negatives(synthetic_positives, pool=pool, cfg=CFG, window_name="ret_w1")
    b = sample_negatives(synthetic_positives, pool=pool, cfg=CFG, window_name="ret_w2")
    assert not a.equals(b)


def test_window_seed_is_stable_across_processes() -> None:
    """The per-window seed must not depend on PYTHONHASHSEED.

    trd.md §7 originally specified `cfg.seed ^ hash(W.name)`. Python salts `str` hashing
    per process, so that would give a different seed on every run and quietly break the
    byte-identical row-set invariant. This test is why the implementation uses crc32.
    """
    prog = (
        "from contentsignal.sampling.negatives import window_seed;print(window_seed(17, 'rank_w4'))"
    )
    seeds = {
        subprocess.run(
            [sys.executable, "-c", prog],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": str(salt), "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for salt in (0, 1, 12345)
    }
    assert len(seeds) == 1, f"window_seed varies with PYTHONHASHSEED: {seeds}"
    assert seeds.pop() == str(window_seed(17, "rank_w4"))


def test_popularity_weighting() -> None:
    """Over many draws, empirical frequency is proportional to weight."""
    values = np.arange(50, dtype=np.int64)
    weights = (1.0 + np.arange(50, dtype=np.float64)) ** 0.75
    table = AliasTable(values, weights, seed=7)

    draws = table.draw(400_000)
    counts = Counter(draws.tolist())
    empirical = np.array([counts[v] for v in values], dtype=np.float64) / draws.size
    expected = weights / weights.sum()

    # Relative error, so the low-weight tail is held to the same standard as the head.
    assert np.max(np.abs(empirical - expected) / expected) < 0.05


def test_uniform_sensitivity_is_actually_uniform() -> None:
    """pop_exponent = 0.0 gives every eligible article the same weight (prd.md §5)."""
    values = np.arange(20, dtype=np.int64)
    weights = np.ones(20, dtype=np.float64)
    draws = AliasTable(values, weights, seed=3).draw(200_000)
    counts = np.array([np.sum(draws == v) for v in values], dtype=np.float64) / draws.size
    assert np.max(np.abs(counts - 0.05)) < 0.005


def test_candidate_pool_excludes_articles_with_no_prior_history(
    synthetic_txns: pl.LazyFrame, pool: pl.DataFrame
) -> None:
    """A negative must be an article that already existed before the window opened.

    Otherwise the model learns to reject items that had not been launched yet, and the
    cold-start slice stops meaning anything.
    """
    prior_articles = set(
        synthetic_txns.filter(pl.col("t_dat") < AS_OF)
        .select("article_id")
        .unique()
        .collect()
        .get_column("article_id")
        .to_list()
    )
    assert set(pool.get_column("article_id").to_list()) == prior_articles


def test_candidate_pool_is_invariant_to_future_deletion(synthetic_txns: pl.LazyFrame) -> None:
    """Deletion invariance (trd.md §15) applied to the sampler's own pool."""
    full = candidate_pool(synthetic_txns, as_of=AS_OF, cfg=CFG)
    truncated = candidate_pool(synthetic_txns.filter(pl.col("t_dat") < AS_OF), as_of=AS_OF, cfg=CFG)
    assert full.equals(truncated)


def test_negatives_come_from_the_pool(negatives: pl.DataFrame, pool: pl.DataFrame) -> None:
    assert set(negatives.get_column("article_id").to_list()) <= set(
        pool.get_column("article_id").to_list()
    )


def test_pool_too_small_fails_loudly(synthetic_positives: pl.DataFrame) -> None:
    """A pool that cannot supply `ratio` distinct negatives is an error, not a short draw.

    Silently returning fewer negatives would change the effective sampling rate, and with
    it the prior correction, without anything in the logs saying so.
    """
    tiny = pl.DataFrame(
        {"article_id": np.arange(5, dtype=np.int32), "weight": np.ones(5, dtype=np.float64)}
    )
    with pytest.raises(ValueError, match="distinct negatives"):
        sample_negatives(synthetic_positives, pool=tiny, cfg=CFG, window_name=WINDOW)
