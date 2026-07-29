"""Per-customer ranking metrics, and the end-to-end variants that make them honest.

Every metric here is computed per customer and then averaged, because a slate is
per-customer and a row-pooled average would let heavy buyers dominate.

The load-bearing distinction in this module is **what goes in the denominator**:

* `map_at_k` / `ndcg_at_k` count only the positives present in the frame they are given —
  i.e. the ones stage 1 retrieved.
* `map_at_k_e2e` / `ndcg_at_k_e2e` count *every* positive in the window. A positive the
  retriever never surfaced cannot appear in the ranking, so it scores as a miss.

Ranking-only metrics therefore overstate the pipeline by exactly the retriever's miss rate,
and **nothing in the output looks wrong when they do** — which is what makes this the most
likely silent error in a two-stage evaluation (trd.md §10.4). Both are reported, always
labelled, and `tests/test_e2e_metrics.py` asserts the inequality between them.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from contentsignal.eval.aggregate import mean_or_zero

CUSTOMER = "customer_idx"
ARTICLE = "article_id"


def _idcg_table(max_k: int) -> np.ndarray:
    """`table[m]` = DCG of a perfect ranking with `m` relevant items in the top slots.

    Precomputed once rather than per customer: the ideal DCG depends only on how many
    relevant items there are, not on which ones.
    """
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))
    return np.concatenate([[0.0], np.cumsum(discounts)])


def _ranked_top_k(scored: pl.DataFrame, *, score_col: str, k: int) -> pl.DataFrame:
    """Order each customer's candidates by score and keep the top `k`.

    Ties are broken by `article_id` so the ordering is deterministic across runs and
    platforms. Without that, two runs of the same model could report different MAP@12 from
    identical scores.
    """
    return (
        scored.sort([CUSTOMER, score_col, ARTICLE], descending=[False, True, False])
        .with_columns(_rank=pl.int_range(1, pl.len() + 1).over(CUSTOMER))
        .filter(pl.col("_rank") <= k)
    )


def _relevant_counts(scored: pl.DataFrame, all_positives: pl.DataFrame | None) -> pl.DataFrame:
    """Per-customer count of relevant items — the metric denominator.

    `all_positives` present  -> end-to-end: every positive in the window counts, including
                                those the retriever never returned.
    `all_positives` is None  -> ranking-only: just the positives inside `scored`.
    """
    if all_positives is None:
        return scored.group_by(CUSTOMER).agg(_n_relevant=pl.col("y").sum())
    return all_positives.group_by(CUSTOMER).agg(_n_relevant=pl.len())


def _per_customer(
    scored: pl.DataFrame,
    all_positives: pl.DataFrame | None,
    *,
    score_col: str,
    k: int,
) -> pl.DataFrame:
    """Per-customer AP@k, NDCG@k and recall@k in one pass.

    Returned per customer rather than aggregated so `eval/bootstrap.py` can resample over
    customers — the only resampling unit that gives honest intervals here, since rows
    within a customer share history features and basket composition (trd.md §10.6).
    """
    counts = _relevant_counts(scored, all_positives)
    top = _ranked_top_k(scored, score_col=score_col, k=k)

    idcg = _idcg_table(k)
    per = (
        top.with_columns(
            _hits=pl.col("y").cum_sum().over(CUSTOMER),
            _discount=1.0 / (pl.col("_rank") + 1).log(2.0),
        )
        .with_columns(
            _precision=pl.col("_hits") / pl.col("_rank"),
        )
        .group_by(CUSTOMER)
        .agg(
            _ap_num=(pl.col("_precision") * pl.col("y")).sum(),
            _dcg=(pl.col("_discount") * pl.col("y")).sum(),
            _retrieved_hits=pl.col("y").sum(),
        )
    )

    # A customer with no relevant items has no defined AP, NDCG or recall, so they are
    # dropped rather than scored as zero — averaging in a meaningless zero would drag every
    # metric toward the share of customers who happen to have nothing to find.
    return (
        per.join(counts, on=CUSTOMER, how="full", coalesce=True)
        .fill_null(0)
        .filter(pl.col("_n_relevant") > 0)
        .with_columns(_denom=pl.min_horizontal(pl.col("_n_relevant"), pl.lit(k)))
        .with_columns(
            ap=pl.col("_ap_num") / pl.col("_denom"),
            ndcg=pl.col("_dcg")
            / pl.col("_denom")
            .cast(pl.Int64)
            .map_elements(lambda m: float(idcg[m]), return_dtype=pl.Float64),
            recall=pl.col("_retrieved_hits") / pl.col("_n_relevant"),
        )
        .select(CUSTOMER, "ap", "ndcg", "recall", "_n_relevant")
        .sort(CUSTOMER)
    )


def per_customer_metrics(
    scored: pl.DataFrame,
    *,
    all_positives: pl.DataFrame | None = None,
    score_col: str = "score",
    k: int = 12,
) -> pl.DataFrame:
    """Per-customer AP@k / NDCG@k / recall@k. Pass `all_positives` for end-to-end."""
    return _per_customer(scored, all_positives, score_col=score_col, k=k)


def map_at_k(scored: pl.DataFrame, *, score_col: str = "score", k: int = 12) -> float:
    """MAP@k over the candidates in `scored` only. Not the pipeline number."""
    return mean_or_zero(_per_customer(scored, None, score_col=score_col, k=k)["ap"])


def ndcg_at_k(scored: pl.DataFrame, *, score_col: str = "score", k: int = 12) -> float:
    """NDCG@k over the candidates in `scored` only. Not the pipeline number."""
    return mean_or_zero(_per_customer(scored, None, score_col=score_col, k=k)["ndcg"])


def precision_at_k(scored: pl.DataFrame, *, score_col: str = "score", k: int = 12) -> float:
    top = _ranked_top_k(scored, score_col=score_col, k=k)
    per = top.group_by(CUSTOMER).agg(p=pl.col("y").sum() / pl.len())
    return mean_or_zero(per["p"])


def map_at_k_e2e(
    scored: pl.DataFrame,
    all_positives: pl.DataFrame,
    *,
    score_col: str = "score",
    k: int = 12,
) -> float:
    """Pipeline MAP@k: every window positive counts, retrieved or not.

    This is the headline. `map_at_k` is reported beside it so the gap — the retriever's
    miss rate — is visible rather than absorbed.
    """
    return mean_or_zero(_per_customer(scored, all_positives, score_col=score_col, k=k)["ap"])


def ndcg_at_k_e2e(
    scored: pl.DataFrame,
    all_positives: pl.DataFrame,
    *,
    score_col: str = "score",
    k: int = 12,
) -> float:
    """Pipeline NDCG@k: every window positive counts, retrieved or not."""
    return mean_or_zero(_per_customer(scored, all_positives, score_col=score_col, k=k)["ndcg"])


def recall_at_k_e2e(
    scored: pl.DataFrame,
    all_positives: pl.DataFrame,
    *,
    score_col: str = "score",
    k: int = 12,
) -> float:
    """Pipeline recall@k. Bounded above by the retriever's recall@K, by construction."""
    return mean_or_zero(_per_customer(scored, all_positives, score_col=score_col, k=k)["recall"])
