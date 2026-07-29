"""Popularity-weighted negative sampling (trd.md §5.3, §7).

Ten negatives per positive, drawn from articles that already existed before the window
opened, weighted by `popularity ** 0.75`. Three properties matter and are tested:

* **Ratio-exact per customer** — a customer with 3 positives gets exactly 30 negatives.
* **Disjoint from positives** — a bought article is never also that customer's negative.
* **Byte-reproducible** — two runs at the same seed produce identical output, which is
  what lets arms 1-8 assert a shared row-set digest and conclude that a small ΔAUC is a
  real difference rather than a different random draw (trd.md §4.5).
"""

from __future__ import annotations

import zlib
from datetime import date, timedelta

import numpy as np
import polars as pl

from contentsignal.config import SamplerConfig
from contentsignal.features.base import history

__all__ = [
    "AliasTable",
    "SamplerConfig",
    "candidate_pool",
    "sample_negatives",
    "window_seed",
]

# Rejection sampling guard. |seen| is typically 2-5 against a pool of tens of thousands,
# so the expected rejection rate is well under 1%; this converts a pathological case into
# a loud failure rather than a hang (trd.md §7).
_MAX_REJECTION_ROUNDS = 64


def window_seed(root_seed: int, window_name: str) -> int:
    """Derive a per-window seed that is stable across processes.

    trd.md §7 wrote this as `cfg.seed ^ hash(W.name)`, but Python salts `hash()` on
    `str` per process (PYTHONHASHSEED), so that expression yields a different seed on
    every run. That would break `test_determinism` and, worse, silently break the
    byte-identical row-set invariant the whole arm comparison rests on. `zlib.crc32` is
    stable across processes, platforms, and Python versions.
    """
    return (root_seed ^ zlib.crc32(window_name.encode("utf-8"))) & 0xFFFFFFFF


class AliasTable:
    """Walker's alias method: O(1) draws from a fixed discrete distribution.

    The pool is ~40-60k articles and is redrawn from millions of times per window, so
    per-draw cost dominates table construction by orders of magnitude.
    """

    def __init__(self, values: np.ndarray, weights: np.ndarray, *, seed: int) -> None:
        if values.shape != weights.shape:
            raise ValueError(f"values {values.shape} and weights {weights.shape} differ in shape")
        if values.size == 0:
            raise ValueError("cannot build an alias table over an empty pool")
        if np.any(weights < 0):
            raise ValueError("negative sampling weights")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("sampling weights sum to zero")

        self.values = np.asarray(values)
        self._rng = np.random.default_rng(seed)
        n = self.values.size

        scaled = np.asarray(weights, dtype=np.float64) * (n / total)
        self.prob = np.zeros(n, dtype=np.float64)
        self.alias = np.zeros(n, dtype=np.int64)

        small = [i for i in range(n) if scaled[i] < 1.0]
        large = [i for i in range(n) if scaled[i] >= 1.0]
        while small and large:
            s = small.pop()
            g = large.pop()
            self.prob[s] = scaled[s]
            self.alias[s] = g
            scaled[g] = scaled[g] - (1.0 - scaled[s])
            (small if scaled[g] < 1.0 else large).append(g)
        for i in large + small:
            self.prob[i] = 1.0
            self.alias[i] = i

    def draw(self, n: int) -> np.ndarray:
        """`n` values sampled with replacement, proportional to weight."""
        idx = self._rng.integers(0, self.values.size, size=n)
        flip = self._rng.random(n) >= self.prob[idx]
        chosen = np.where(flip, self.alias[idx], idx)
        return self.values[chosen]


def candidate_pool(txns: pl.LazyFrame, *, as_of: date, cfg: SamplerConfig) -> pl.DataFrame:
    """Articles eligible as negatives, with sampling weight.

    Only articles with at least one transaction before the cutoff. An article with no
    prior history could not have been shown, so proposing it as a negative would teach
    the model to reject items that simply did not exist yet — and it would make the
    cold-start slice incoherent (trd.md §7).
    """
    lookback_start = as_of - timedelta(weeks=cfg.pop_lookback_weeks)
    prior = history(txns, as_of=as_of)

    popularity = (
        prior.filter(pl.col("t_dat") >= lookback_start).group_by("article_id").agg(pop=pl.len())
    )
    # Existence is judged on all prior history; popularity only on the lookback window.
    eligible = prior.select(pl.col("article_id").unique())

    # Deviation from trd.md §7, deliberate: the spec sets weight = pop_12w ** 0.75 while
    # defining the pool as everything with any prior history. Read literally, an article
    # that sold before the lookback but not inside it gets weight 0 — in the pool yet
    # unsamplable, which is the same as being absent but harder to notice. The +1
    # (Laplace) floor keeps "in the pool" and "can be drawn" the same statement. It also
    # preserves the uniform sensitivity check: at pop_exponent = 0.0 every weight is 1.
    return (
        eligible.join(popularity, on="article_id", how="left")
        .with_columns(pop=pl.col("pop").fill_null(0).cast(pl.Float64))
        .with_columns(weight=(pl.col("pop") + 1.0) ** cfg.pop_exponent)
        .select("article_id", "weight")
        .sort("article_id")
        .collect()
    )


def sample_negatives(
    positives: pl.DataFrame,  # customer_idx, article_id
    *,
    pool: pl.DataFrame,  # article_id, weight
    cfg: SamplerConfig,
    window_name: str,
) -> pl.DataFrame:
    """`cfg.ratio` distinct negatives per positive, per customer.

    `drawn` is a set: a customer never receives the same negative twice, because
    duplicate negatives would silently reweight the loss for that customer.
    """
    seed = window_seed(cfg.seed, window_name)
    alias = AliasTable(
        pool.get_column("article_id").to_numpy(),
        pool.get_column("weight").to_numpy(),
        seed=seed,
    )
    pool_size = pool.height

    grouped = (
        positives.select("customer_idx", "article_id")
        .group_by("customer_idx")
        .agg(pl.col("article_id").alias("seen"))
        .sort("customer_idx")  # deterministic iteration order, independent of hash order
    )

    out_customers: list[np.ndarray] = []
    out_articles: list[np.ndarray] = []

    for customer, seen_list in zip(
        grouped.get_column("customer_idx").to_list(),
        grouped.get_column("seen").to_list(),
        strict=True,
    ):
        seen = set(seen_list)
        need = cfg.ratio * len(seen_list)
        if need > pool_size - len(seen):
            raise ValueError(
                f"customer {customer}: need {need} distinct negatives but the pool holds "
                f"only {pool_size - len(seen)} unseen articles"
            )

        drawn: set[int] = set()
        rounds = 0
        while len(drawn) < need:
            if rounds >= _MAX_REJECTION_ROUNDS:
                raise RuntimeError(
                    f"pathological rejection rate for customer {customer}: "
                    f"{len(drawn)}/{need} after {rounds} rounds over a pool of {pool_size}"
                )
            batch = alias.draw(need - len(drawn) + 8)  # slack for rejections
            for a in batch.tolist():
                if a not in seen and a not in drawn:
                    drawn.add(a)
                    if len(drawn) == need:
                        break
            rounds += 1

        articles = np.fromiter(sorted(drawn), dtype=np.int64, count=need)
        out_articles.append(articles)
        out_customers.append(np.full(need, customer, dtype=np.int64))

    if not out_customers:
        return pl.DataFrame(schema={"customer_idx": pl.Int32, "article_id": pl.Int32, "y": pl.Int8})

    return pl.DataFrame(
        {
            "customer_idx": np.concatenate(out_customers),
            "article_id": np.concatenate(out_articles),
        }
    ).with_columns(
        pl.col("customer_idx").cast(pl.Int32),
        pl.col("article_id").cast(pl.Int32),
        y=pl.lit(0, dtype=pl.Int8),
    )
