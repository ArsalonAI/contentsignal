"""Shared fixtures.

The synthetic transaction frame is deliberately small and fully deterministic. It exists
so the invariants — cutoff strictness, ratio exactness, reproducibility — can be proven
before the real 31.8M-row dataset is available, and so they stay fast to re-prove
afterwards. It is not a statistical stand-in for H&M data and is not used for any
modelling claim.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

# The cutoff every leakage test pivots on. Chosen so the synthetic frame has substantial
# history on both sides of it, including transactions landing exactly on the boundary.
AS_OF = date(2020, 6, 1)

N_CUSTOMERS = 40
N_ARTICLES = 120
N_ROWS = 900
FIRST_DAY = date(2020, 1, 6)
LAST_DAY = date(2020, 6, 30)


@pytest.fixture(scope="session")
def synthetic_txns_df() -> pl.DataFrame:
    """Transactions matching the artifacts/parquet/transactions.parquet schema (trd.md §4.1).

    Article popularity is skewed by construction (a Zipf-like draw), because a uniform
    catalogue would let popularity-weighted sampling pass its test vacuously.
    """
    rng = np.random.default_rng(20200601)
    span = (LAST_DAY - FIRST_DAY).days

    # Zipf-ish popularity: article 0 is the bestseller, article 119 barely sells.
    weights = 1.0 / (1.0 + np.arange(N_ARTICLES))
    weights /= weights.sum()

    day_offsets = rng.integers(0, span + 1, size=N_ROWS)
    frame = pl.DataFrame(
        {
            "t_dat": [FIRST_DAY + timedelta(days=int(d)) for d in day_offsets],
            "customer_idx": rng.integers(0, N_CUSTOMERS, size=N_ROWS).astype(np.int32),
            "article_id": rng.choice(N_ARTICLES, size=N_ROWS, p=weights).astype(np.int32),
            "price": rng.uniform(0.01, 0.06, size=N_ROWS).astype(np.float32),
            "sales_channel_id": rng.integers(1, 3, size=N_ROWS).astype(np.int8),
        }
    )

    # Guarantee transactions exactly on the cutoff. The strict `<` in history() must
    # exclude these; an off-by-one here leaks the first day of the label window.
    boundary = pl.DataFrame(
        {
            "t_dat": [AS_OF] * 5,
            "customer_idx": np.arange(5, dtype=np.int32),
            "article_id": np.arange(5, dtype=np.int32),
            "price": np.full(5, 0.02, dtype=np.float32),
            "sales_channel_id": np.full(5, 2, dtype=np.int8),
        }
    )

    return pl.concat([frame, boundary]).sort("t_dat", "customer_idx", "article_id")


@pytest.fixture
def synthetic_txns(synthetic_txns_df: pl.DataFrame) -> pl.LazyFrame:
    return synthetic_txns_df.lazy()


@pytest.fixture
def synthetic_positives(synthetic_txns_df: pl.DataFrame) -> pl.DataFrame:
    """Distinct (customer, article) pairs bought on or after the cutoff.

    Mirrors how real positives are built: purchases inside the label window, deduplicated
    so a repeat purchase is one row.
    """
    return (
        synthetic_txns_df.filter(pl.col("t_dat") >= AS_OF)
        .select("customer_idx", "article_id")
        .unique()
        .sort("customer_idx", "article_id")
    )
