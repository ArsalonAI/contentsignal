"""The end-to-end accounting invariant (trd.md §10.4, §15).

This file guards the most likely silent error in a two-stage evaluation: computing MAP@12
over the candidates stage 1 returned, and reporting it as the pipeline's performance. Every
positive the retriever missed has quietly left the denominator, so the number is inflated by
exactly the retriever's miss rate — **and nothing in the output looks wrong.**

The assertions are all forms of one statement: *a ranker cannot fix what retrieval never
surfaced.*
"""

from __future__ import annotations

import polars as pl
import pytest

from contentsignal.eval.ranking import (
    map_at_k,
    map_at_k_e2e,
    ndcg_at_k,
    ndcg_at_k_e2e,
    per_customer_metrics,
    recall_at_k_e2e,
)

K = 12


def _scored(rows: list[tuple[int, int, float, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "customer_idx": pl.Int32,
            "article_id": pl.Int32,
            "score": pl.Float64,
            "y": pl.Int8,
        },
        orient="row",
    )


def _positives(rows: list[tuple[int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema={"customer_idx": pl.Int32, "article_id": pl.Int32}, orient="row"
    )


# --- the invariant ------------------------------------------------------------------


def test_e2e_metrics_count_unretrieved_positives() -> None:
    """A positive stage 1 missed must drag the pipeline number down.

    Customer 1 bought two articles; the retriever surfaced only one. Ranking-only MAP sees a
    perfect list (one candidate, and it is the answer). End-to-end MAP knows there were two
    real purchases and only one was reachable.
    """
    scored = _scored([(1, 10, 0.9, 1), (1, 11, 0.5, 0)])
    all_positives = _positives([(1, 10), (1, 99)])  # 99 was never retrieved

    ranking_only = map_at_k(scored, k=K)
    end_to_end = map_at_k_e2e(scored, all_positives, k=K)

    assert ranking_only == pytest.approx(1.0)
    assert end_to_end == pytest.approx(0.5)
    assert end_to_end < ranking_only


def test_e2e_is_never_above_ranking_only() -> None:
    """The general form, over a mixed population.

    This is the inequality that must hold for every arm in `reports/results.md`. If they are
    ever equal on real data, the accounting is broken and every headline number is inflated.
    """
    scored = _scored(
        [
            (1, 10, 0.9, 1),
            (1, 11, 0.8, 0),
            (2, 20, 0.7, 1),
            (2, 21, 0.6, 1),
            (3, 30, 0.5, 0),
            (3, 31, 0.4, 1),
        ]
    )
    all_positives = _positives([(1, 10), (1, 98), (2, 20), (2, 21), (3, 31), (3, 97)])

    assert map_at_k_e2e(scored, all_positives, k=K) < map_at_k(scored, k=K)
    assert ndcg_at_k_e2e(scored, all_positives, k=K) < ndcg_at_k(scored, k=K)


def test_perfect_retrieval_makes_the_two_agree() -> None:
    """When nothing is missed the gap closes — so the gap measures misses, not a bug.

    Without this, a metric that always returned something smaller would pass the test above
    for the wrong reason.
    """
    scored = _scored([(1, 10, 0.9, 1), (1, 11, 0.8, 0), (2, 20, 0.7, 1)])
    all_positives = _positives([(1, 10), (2, 20)])

    assert map_at_k_e2e(scored, all_positives, k=K) == pytest.approx(map_at_k(scored, k=K))
    assert ndcg_at_k_e2e(scored, all_positives, k=K) == pytest.approx(ndcg_at_k(scored, k=K))


def test_e2e_recall_bounded_by_retrieval_recall() -> None:
    """Pipeline recall@12 cannot exceed what retrieval handed over.

    The ceiling, asserted rather than assumed. Here retrieval found 2 of customer 1's 4
    purchases, so no ranker can push pipeline recall past 0.5.
    """
    scored = _scored([(1, 10, 0.9, 1), (1, 11, 0.8, 1), (1, 12, 0.7, 0)])
    all_positives = _positives([(1, 10), (1, 11), (1, 96), (1, 97)])

    assert recall_at_k_e2e(scored, all_positives, k=K) == pytest.approx(0.5)


def test_total_retrieval_failure_scores_zero_not_missing() -> None:
    """A customer the retriever failed on entirely must score zero, not disappear.

    Customer 2 bought something and stage 1 returned none of it, so customer 2 has no rows
    in the scored frame at all. Dropping them would let a retriever improve its pipeline
    number by failing harder — every customer it completely misses would leave the average.
    """
    scored = _scored([(1, 10, 0.9, 1)])
    all_positives = _positives([(1, 10), (2, 20)])

    per = per_customer_metrics(scored, all_positives=all_positives, k=K)
    assert per["customer_idx"].to_list() == [1, 2]
    assert per["ap"].to_list() == [1.0, 0.0]
    assert map_at_k_e2e(scored, all_positives, k=K) == pytest.approx(0.5)


def test_a_better_ranker_cannot_beat_the_ceiling() -> None:
    """Reordering the candidate list changes MAP but never pipeline recall.

    The mechanical statement of H1's premise: past the retrieval ceiling, ranker work moves
    ordering metrics and leaves reachability untouched.
    """
    good = _scored([(1, 10, 0.9, 1), (1, 11, 0.1, 0)])
    bad = _scored([(1, 10, 0.1, 1), (1, 11, 0.9, 0)])
    all_positives = _positives([(1, 10), (1, 95)])

    assert map_at_k_e2e(good, all_positives, k=K) > map_at_k_e2e(bad, all_positives, k=K)
    assert recall_at_k_e2e(good, all_positives, k=K) == recall_at_k_e2e(bad, all_positives, k=K)


# --- metric mechanics ----------------------------------------------------------------


def test_map_denominator_is_capped_at_k() -> None:
    """MAP@k divides by min(n_relevant, k) — the H&M competition's definition.

    Without the cap, a customer with 50 purchases could never score above 12/50 on a
    12-slot slate, and the metric would mostly measure basket size.
    """
    scored = _scored([(1, i, 1.0 - i / 100, 1) for i in range(3)])
    many = _positives([(1, i) for i in range(50)])

    assert map_at_k_e2e(scored, many, k=3) == pytest.approx(1.0)


def test_customers_with_no_positives_are_dropped_not_zeroed() -> None:
    """A customer with nothing to find has no defined AP; averaging in a zero would make
    every metric partly a measure of how many such customers exist."""
    scored = _scored([(1, 10, 0.9, 1), (2, 20, 0.9, 0)])
    all_positives = _positives([(1, 10)])

    per = per_customer_metrics(scored, all_positives=all_positives, k=K)
    assert per["customer_idx"].to_list() == [1]
    assert map_at_k_e2e(scored, all_positives, k=K) == pytest.approx(1.0)


def test_ranking_is_deterministic_under_ties() -> None:
    """Equal scores resolve by article_id, so two runs report identical MAP.

    Without a deterministic tiebreak, identical model outputs could produce different
    numbers on different platforms — and the report would not be reproducible.
    """
    tied = _scored([(1, 11, 0.5, 0), (1, 10, 0.5, 1)])
    assert map_at_k(tied, k=K) == pytest.approx(1.0)
