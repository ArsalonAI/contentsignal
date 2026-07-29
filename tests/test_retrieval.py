"""Stage-1 metrics and the tower contract (trd.md §15).

Two halves. The metric half runs now — `recall@K`, coverage and the popularity diagnostic
are pure computation and need neither data nor a trained model, and they encode the pipeline
ceiling that everything downstream is interpreted against.

The tower half is skipped until `models/twotower.py` lands at M3, then becomes live. It is
written now rather than later because the property it guards —
`CustomerTower.forward` taking no item argument — is easy to break by accident while
chasing offline metrics, and the consequence is that retrieval stops being possible at all.
"""

from __future__ import annotations

import importlib.util

import polars as pl
import pytest

from contentsignal.eval.retrieval import (
    cold_start_recall_at_k,
    coverage,
    popularity_rho,
    recall_at_k,
    recall_at_k_per_customer,
    recall_at_k_sweep,
)

# Scoped to the tower test alone, never module-level. A module-level `importorskip` would
# skip the metric tests in this file too, and they run today — a file that goes green while
# covering nothing is worse than no file, which is the same reasoning behind
# `test_every_builder_module_is_imported_by_the_package` in test_leakage.py.
_HAS_TWOTOWER = importlib.util.find_spec("contentsignal.models.twotower") is not None
needs_twotower = pytest.mark.skipif(
    not _HAS_TWOTOWER, reason="models/twotower.py lands at M3 (trd.md §16)"
)


def _retrieved(rows: list[tuple[int, int, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "customer_idx": pl.Int32,
            "article_id": pl.Int32,
            "retrieval_rank": pl.Int16,
            "retrieval_score": pl.Float32,
        },
        orient="row",
    )


def _positives(rows: list[tuple[int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema={"customer_idx": pl.Int32, "article_id": pl.Int32}, orient="row"
    )


def _articles(rows: list[tuple[int, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"article_id": pl.Int32, "art_prior_purchases": pl.Int32, "art_pop_12w": pl.Int32},
        orient="row",
    )


# --- recall@K: the pipeline ceiling ---------------------------------------------------


def test_recall_counts_unretrieved_positives_against_the_retriever() -> None:
    """The denominator is every purchase in the window, not every retrieved purchase.

    Dropping the misses would report accuracy on the items already found, which is not a
    measure of retrieval at all.
    """
    retrieved = _retrieved([(1, 10, 1, 0.9), (1, 11, 2, 0.8)])
    all_positives = _positives([(1, 10), (1, 99)])  # 99 never retrieved

    assert recall_at_k(retrieved, all_positives, k=10) == pytest.approx(0.5)


def test_recall_at_k_is_monotone_in_k() -> None:
    """Non-decreasing in k, always. A violation means the truncation is wrong, which would
    silently corrupt the whole K sweep that H1's retrieval axis is built from."""
    retrieved = _retrieved([(1, 10 + i, i + 1, 1.0 - i / 100) for i in range(20)])
    all_positives = _positives([(1, 10), (1, 15), (1, 25), (1, 999)])

    sweep = recall_at_k_sweep(retrieved, all_positives, ks=(1, 5, 10, 20))
    values = [sweep[k] for k in (1, 5, 10, 20)]
    assert values == sorted(values), values
    assert sweep[20] == pytest.approx(0.75)  # 3 of 4 reachable; 999 never retrieved


def test_recall_sweep_matches_individual_calls() -> None:
    """The sweep is a truncation of one pass, so it must agree with computing each k alone."""
    retrieved = _retrieved([(1, 10 + i, i + 1, 1.0 - i / 100) for i in range(10)])
    all_positives = _positives([(1, 12), (1, 18)])

    sweep = recall_at_k_sweep(retrieved, all_positives, ks=(3, 5, 10))
    for k, value in sweep.items():
        assert value == pytest.approx(recall_at_k(retrieved, all_positives, k=k))


def test_recall_is_per_customer_then_averaged() -> None:
    """Not row-pooled: a heavy buyer must not outvote a light one.

    Customer 1 finds 1 of 1; customer 2 finds 1 of 3. Row-pooling gives 2/4 = 0.5; the
    per-customer mean gives (1.0 + 0.333)/2 = 0.667.
    """
    retrieved = _retrieved([(1, 10, 1, 0.9), (2, 20, 1, 0.9)])
    all_positives = _positives([(1, 10), (2, 20), (2, 21), (2, 22)])

    assert recall_at_k(retrieved, all_positives, k=10) == pytest.approx(2 / 3, abs=1e-6)


def test_customer_with_zero_retrieved_hits_scores_zero_not_dropped() -> None:
    """A customer the retriever failed entirely must count as a zero, not vanish.

    Dropping them would hide total retrieval failures — the opposite of what recall is for.
    """
    retrieved = _retrieved([(1, 10, 1, 0.9), (2, 77, 1, 0.9)])
    all_positives = _positives([(1, 10), (2, 20)])

    per = recall_at_k_per_customer(retrieved, all_positives, k=10)
    assert per["customer_idx"].to_list() == [1, 2]
    assert per["recall"].to_list() == [1.0, 0.0]
    assert recall_at_k(retrieved, all_positives, k=10) == pytest.approx(0.5)


def test_recall_requires_a_materialized_rank() -> None:
    """recall@K is defined on a ranked list; a frame without ranks fails loudly."""
    unranked = pl.DataFrame(
        {"customer_idx": [1], "article_id": [10]},
        schema={"customer_idx": pl.Int32, "article_id": pl.Int32},
    )
    with pytest.raises(KeyError, match="retrieval_rank"):
        recall_at_k(unranked, _positives([(1, 10)]), k=10)


# --- cold start: the slice H2 lives on ------------------------------------------------


def test_cold_start_recall_restricts_to_cold_articles() -> None:
    """Only cold positives count, and only customers who bought one are averaged.

    Article 10 is established (200 prior purchases), 99 is cold (1). The retriever found
    the established one and missed the cold one, so cold-start recall is 0 even though
    overall recall is 0.5 — which is the asymmetry H2 exists to detect.
    """
    retrieved = _retrieved([(1, 10, 1, 0.9), (1, 11, 2, 0.8)])
    all_positives = _positives([(1, 10), (1, 99)])
    articles = _articles([(10, 200, 200), (11, 50, 50), (99, 1, 1)])

    assert recall_at_k(retrieved, all_positives, k=10) == pytest.approx(0.5)
    assert cold_start_recall_at_k(
        retrieved, all_positives, articles, k=10, threshold=10
    ) == pytest.approx(0.0)


def test_cold_start_recall_credits_a_cold_hit() -> None:
    retrieved = _retrieved([(1, 99, 1, 0.9)])
    all_positives = _positives([(1, 99)])
    articles = _articles([(99, 1, 1)])

    assert cold_start_recall_at_k(
        retrieved, all_positives, articles, k=10, threshold=10
    ) == pytest.approx(1.0)


def test_empty_cold_slice_raises_rather_than_returning_zero() -> None:
    """A zero would read as 'the retriever failed on cold items'; the truth is 'there were
    none'. The threshold is registered at M1, so an empty slice means it is misconfigured."""
    retrieved = _retrieved([(1, 10, 1, 0.9)])
    all_positives = _positives([(1, 10)])
    articles = _articles([(10, 500, 500)])

    with pytest.raises(ValueError, match="cold-start slice"):
        cold_start_recall_at_k(retrieved, all_positives, articles, k=10, threshold=10)


# --- coverage: the failure recall cannot see ------------------------------------------


def test_coverage_detects_a_bestseller_collapse() -> None:
    """A retriever returning the same two articles to everyone has high recall potential
    and near-zero coverage. Recall does not notice; this does."""
    collapsed = _retrieved(
        [(c, a, r, 0.9) for c in (1, 2, 3) for r, a in enumerate([10, 11], start=1)]
    )
    diverse = _retrieved([(c, 10 * c + r, r, 0.9) for c in (1, 2, 3) for r in (1, 2)])

    assert coverage(collapsed, catalog_size=100, k=2) == pytest.approx(0.02)
    assert coverage(diverse, catalog_size=100, k=2) == pytest.approx(0.06)


def test_coverage_respects_k() -> None:
    retrieved = _retrieved([(1, 10 + i, i + 1, 0.9) for i in range(10)])
    assert coverage(retrieved, catalog_size=100, k=3) == pytest.approx(0.03)
    assert coverage(retrieved, catalog_size=100, k=10) == pytest.approx(0.10)


def test_coverage_rejects_nonpositive_catalog() -> None:
    with pytest.raises(ValueError, match="catalog_size"):
        coverage(_retrieved([(1, 10, 1, 0.9)]), catalog_size=0, k=1)


# --- the popularity collapse diagnostic ----------------------------------------------


def test_popularity_rho_flags_a_popularity_proxy() -> None:
    """Scores that track popularity exactly mean the towers learned nothing the tabular
    features did not already carry, measured more precisely."""
    scored = _retrieved([(1, i, i + 1, float(10 - i)) for i in range(10)])
    articles = _articles([(i, 0, 10 - i) for i in range(10)])

    assert popularity_rho(scored, articles) == pytest.approx(1.0, abs=1e-6)


def test_popularity_rho_flags_an_inverted_ranker() -> None:
    """Strongly negative rho is the signature of a missing log-Q correction: in-batch
    negatives arrive proportional to popularity, so an uncorrected model is rewarded for
    demoting popular items until it ranks them backwards (trd.md §9.5)."""
    scored = _retrieved([(1, i, i + 1, float(i)) for i in range(10)])
    articles = _articles([(i, 0, 10 - i) for i in range(10)])

    assert popularity_rho(scored, articles) == pytest.approx(-1.0, abs=1e-6)


def test_constant_scores_raise_rather_than_returning_nan() -> None:
    """Identical scores for every article is embedding collapse, not a correlation of zero.

    The rank correlation is genuinely undefined there. Raising names the failure; returning
    NaN would let it through `make report` as a blank cell that reads like a formatting bug.
    """
    scored = _retrieved([(1, i, i + 1, 0.5) for i in range(10)])
    articles = _articles([(i, 0, 10 - i) for i in range(10)])

    with pytest.raises(ValueError, match="collapse"):
        popularity_rho(scored, articles)


def test_popularity_rho_shares_ranks_across_ties() -> None:
    """Partial ties must not fake a perfect correlation.

    Five articles share the top score and five share the bottom. A tie-blind ranking would
    impose an arbitrary order inside each group and report rho ≈ 1.
    """
    scored = _retrieved([(1, i, i + 1, 1.0 if i < 5 else 0.0) for i in range(10)])
    articles = _articles([(i, 0, 10 - i) for i in range(10)])

    assert abs(popularity_rho(scored, articles)) < 1.0


# --- the tower contract (live from M3) -----------------------------------------------


@needs_twotower
def test_customer_tower_is_item_independent() -> None:
    """`CustomerTower.forward` must accept no item argument.

    This is the property that keeps both towers precomputable. Candidate-conditioned
    attention would score better offline, but the customer vector would then depend on the
    item, so all 105k article vectors could no longer be precomputed — retrieval would need
    105k forward passes per request and the cost analysis would describe nothing
    deployable (trd.md §5.3).
    """
    import inspect

    from contentsignal.models import twotower

    params = inspect.signature(twotower.CustomerTower.forward).parameters
    forbidden = {"item", "items", "item_vec", "item_ids", "candidate", "candidates"}
    assert not forbidden & set(params), (
        f"CustomerTower.forward accepts item arguments {sorted(forbidden & set(params))}; "
        "that breaks the two-tower factorization and with it retrieval itself"
    )
