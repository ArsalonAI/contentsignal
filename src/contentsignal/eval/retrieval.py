"""Stage-1 metrics — how much of the truth ever reaches the ranker (trd.md §10.2).

`recall@K` is the ceiling on the whole pipeline. At `recall@100 = 0.60`, forty percent of
real purchases are invisible to stage 2 permanently, and no ranker recovers them. Locating
that ceiling is the point of the project, so these are first-class metrics rather than
diagnostics.

Two things this module is careful about:

* **The denominator is every positive in the window**, not every retrieved positive.
  A purchase the retriever missed is exactly what `recall@K` is meant to count against it.
* **`coverage` exists because recall alone can be gamed.** A retriever that returns the same
  500 bestsellers to every customer can post an acceptable `recall@K` and be useless as a
  recommender. Recall does not notice; coverage does.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from contentsignal.eval.aggregate import mean_or_zero

CUSTOMER = "customer_idx"
ARTICLE = "article_id"
RANK = "retrieval_rank"


def _top_k(retrieved: pl.DataFrame, *, k: int, rank_col: str = RANK) -> pl.DataFrame:
    """Truncate to the top `k` per customer.

    Truncation is what makes the whole `k_sweep` cost a single retrieval pass at
    `max(k_sweep)` instead of one pass per depth (trd.md §10.3).
    """
    if rank_col not in retrieved.columns:
        raise KeyError(
            f"{rank_col!r} not in retrieved frame; columns are {sorted(retrieved.columns)}. "
            "recall@K is defined on a ranked list, so the rank must be materialized."
        )
    return retrieved.filter(pl.col(rank_col) <= k)


def recall_at_k_per_customer(
    retrieved: pl.DataFrame,
    all_positives: pl.DataFrame,
    *,
    k: int,
    rank_col: str = RANK,
) -> pl.DataFrame:
    """Per-customer `recall@k`, for customer-level bootstrapping.

    `all_positives` is the window's FULL positive set. A positive absent from `retrieved`
    contributes zero to the numerator and still counts in the denominator — dropping it
    instead would report the retriever's accuracy on the items it already found, which is
    not a measure of retrieval.
    """
    hits = (
        _top_k(retrieved, k=k, rank_col=rank_col)
        .join(all_positives.select(CUSTOMER, ARTICLE), on=[CUSTOMER, ARTICLE], how="semi")
        .group_by(CUSTOMER)
        .agg(_hits=pl.len())
    )
    totals = all_positives.group_by(CUSTOMER).agg(_n_positives=pl.len())
    return (
        totals.join(hits, on=CUSTOMER, how="left")
        .with_columns(_hits=pl.col("_hits").fill_null(0))
        .with_columns(recall=pl.col("_hits") / pl.col("_n_positives"))
        .select(CUSTOMER, "recall", "_hits", "_n_positives")
        .sort(CUSTOMER)
    )


def recall_at_k(
    retrieved: pl.DataFrame,
    all_positives: pl.DataFrame,
    *,
    k: int,
    rank_col: str = RANK,
) -> float:
    """Mean per-customer `recall@k`. Non-decreasing in `k`, by construction."""
    per = recall_at_k_per_customer(retrieved, all_positives, k=k, rank_col=rank_col)
    return mean_or_zero(per["recall"])


def recall_at_k_sweep(
    retrieved: pl.DataFrame,
    all_positives: pl.DataFrame,
    *,
    ks: tuple[int, ...],
    rank_col: str = RANK,
) -> dict[int, float]:
    """`recall@k` for every `k`, from one retrieval pass.

    This is the retrieval axis of H1, and it costs nothing beyond the pass already taken —
    which is precisely the asymmetry H1 is testing, since the ranker axis costs a training
    run per point.
    """
    return {k: recall_at_k(retrieved, all_positives, k=k, rank_col=rank_col) for k in sorted(ks)}


def cold_start_recall_at_k(
    retrieved: pl.DataFrame,
    all_positives: pl.DataFrame,
    articles: pl.DataFrame,
    *,
    k: int,
    threshold: int,
    prior_col: str = "art_prior_purchases",
    rank_col: str = RANK,
) -> float:
    """`recall@k` restricted to articles with fewer than `threshold` prior purchases.

    The slice H2 lives on: popularity features are ~0 here while the product description is
    complete from day one. Only customers with at least one cold-start positive are
    averaged — a customer who bought nothing cold has no defined cold-start recall, and
    scoring them zero would dilute the slice with customers it does not describe.
    """
    cold = articles.filter(pl.col(prior_col) < threshold).select(ARTICLE)
    cold_positives = all_positives.join(cold, on=ARTICLE, how="semi")
    if cold_positives.is_empty():
        raise ValueError(
            f"no positives fall in the cold-start slice ({prior_col} < {threshold}); "
            "the threshold is set once at M1 from the measured distribution (trd.md §10.5)"
        )
    per = recall_at_k_per_customer(retrieved, cold_positives, k=k, rank_col=rank_col)
    return mean_or_zero(per["recall"])


def coverage(retrieved: pl.DataFrame, *, catalog_size: int, k: int, rank_col: str = RANK) -> float:
    """Fraction of the catalog appearing in ANY customer's top `k`.

    Guards the failure recall cannot see: a retriever collapsed onto the head of the
    popularity distribution returns the same bestsellers to everyone, scores acceptably on
    recall, and is useless as a recommender.
    """
    if catalog_size <= 0:
        raise ValueError(f"catalog_size must be positive, got {catalog_size}")
    distinct = _top_k(retrieved, k=k, rank_col=rank_col)[ARTICLE].n_unique()
    return distinct / catalog_size


def popularity_rho(
    scored: pl.DataFrame,
    articles: pl.DataFrame,
    *,
    score_col: str = "retrieval_score",
    pop_col: str = "art_pop_12w",
) -> float:
    """Spearman correlation between retriever score and log article popularity.

    The collapse diagnostic. A high positive value means the retriever learned to re-rank by
    popularity — which the tabular features already carry, measured exactly — so the towers
    added nothing. A high *negative* value means the log-Q correction is missing or broken:
    in-batch negatives arrive proportional to popularity, so an uncorrected model is
    rewarded for demoting popular items until it inverts them (trd.md §9.5).
    """
    joined = scored.join(articles.select(ARTICLE, pop_col), on=ARTICLE, how="inner")
    if joined.height < 2:
        raise ValueError("need at least two scored articles with popularity to correlate")
    scores = _rankdata(joined[score_col].to_numpy())
    pops = _rankdata(np.log1p(joined[pop_col].to_numpy()))

    # Zero variance means every article received the same rank. For scores that is embedding
    # collapse — the loss can look healthy while the retriever ranks nothing — and it is a
    # louder failure than the correlation it makes undefined. Returning NaN here would let
    # it pass through the report as a blank cell.
    for name, ranks in (("scores", scores), ("popularity", pops)):
        if float(ranks.std()) == 0.0:
            raise ValueError(
                f"{name} are constant across {joined.height} articles, so the rank "
                "correlation is undefined. For scores this is embedding collapse; check "
                "tower norms and pairwise cosine spread (trd.md §9.5)"
            )

    # Spearman is Pearson on ranks; computing it here avoids a scipy dependency in the hot
    # evaluation path.
    return float(np.corrcoef(scores, pops)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared. Equivalent to `scipy.stats.rankdata` for this use."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # Share ranks within tied groups so a constant score does not fake a correlation.
    unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(unique) < len(x):
        sums = np.zeros(len(unique), dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks
