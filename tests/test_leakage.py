"""Leakage guards (trd.md §15, §5.1).

M1 gates everything: no model is trained until these pass. Two kinds of check live here.

* **Deletion invariance** — the general form of every leakage check in this project.
  Recompute a feature on data truncated at the window start; if the value changes, the
  feature was reading the future. Parameterized over `ALL_BUILDERS`, so every builder
  added later is covered without anyone remembering to wire it up.
* **Signature enforcement** — `as_of` is keyword-only and required, which is what makes
  the leak unwritable rather than merely discouraged.

The builder list is empty until M2. The harness is live, not a stub: the moment a builder
registers, it is subject to the property.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

# Importing the package (not just `features.base`) is what populates ALL_BUILDERS:
# registration is a side effect of importing each builder module, and
# `features/__init__.py` is where those imports live. Parametrization below reads the
# list at import time, so this must come first or the harness silently covers nothing.
import contentsignal.features  # noqa: F401
from contentsignal.features.base import (
    ALL_BUILDERS,
    AsOfContractViolation,
    assert_as_of_is_enforced,
    history,
    register_builder,
)
from contentsignal.splits.temporal import candidate_windows, windows_for_role
from tests.conftest import AS_OF, N_CUSTOMERS

_NO_BUILDERS_YET = [
    pytest.param(
        None,
        marks=pytest.mark.skip(reason="no feature builders registered yet — they land at M2"),
    )
]


@pytest.fixture
def entities() -> pl.DataFrame:
    return pl.DataFrame({"customer_idx": range(N_CUSTOMERS)}, schema={"customer_idx": pl.Int32})


# --- the property ------------------------------------------------------------------


@pytest.mark.parametrize("builder", ALL_BUILDERS or _NO_BUILDERS_YET)
def test_features_are_invariant_to_future_deletion(
    builder: object, synthetic_txns: pl.LazyFrame, entities: pl.DataFrame
) -> None:
    """Recomputing without the future must change nothing."""
    full = builder.build(synthetic_txns, as_of=AS_OF, entities=entities)  # type: ignore[attr-defined]
    truncated = builder.build(  # type: ignore[attr-defined]
        synthetic_txns.filter(pl.col("t_dat") < AS_OF), as_of=AS_OF, entities=entities
    )
    assert_frame_equal(full, truncated)


def test_all_builders_declare_their_columns() -> None:
    for builder in ALL_BUILDERS:
        assert builder.name, "builder has no name"
        assert builder.columns, f"{builder.name} declares no output columns"


def test_every_builder_module_is_imported_by_the_package() -> None:
    """A builder in a module nobody imports is a builder nobody leak-tests.

    `ALL_BUILDERS` is populated by import side effect, so an unimported builder module
    leaves the deletion-invariance harness green while covering nothing — a worse
    outcome than having no harness. This checks the package actually pulls in every
    module that sits next to `base.py`.
    """
    package_dir = Path(contentsignal.features.__file__).parent
    modules = {p.stem for p in package_dir.glob("*.py") if p.stem not in {"__init__", "base"}}
    missing = {m for m in modules if f"contentsignal.features.{m}" not in sys.modules}
    assert not missing, (
        f"feature modules not imported by contentsignal/features/__init__.py: "
        f"{sorted(missing)} — their builders would never be leak-tested"
    )


# --- the retriever's leakage boundary ------------------------------------------------


def test_retriever_windows_precede_candidate_windows() -> None:
    """The retriever must never train on a window it later retrieves candidates for.

    This leak has no analogue in a single-stage design, and it is the one most likely to be
    introduced by a reasonable-looking change — training the retriever on all eight train
    windows is the obvious default and gives better retrieval metrics.

    The mechanism: a retriever that saw window W's purchases has memorized them and ranks
    them first. The ranker then trains on candidate lists where the correct answer sits at
    rank 1 for a reason that will never hold at serving time, learns to trust rank 1, and
    collapses in production. Nothing downstream detects it — the offline numbers improve.
    """
    trained_on = windows_for_role("retriever")
    served = candidate_windows()
    assert trained_on, "no retriever-role windows configured"
    assert served, "no candidate windows configured"

    last_train = max(w.end for w in trained_on)
    for w in served:
        assert w.start > last_train, (
            f"candidate window {w.name} starts {w.start}, but the retriever trained through "
            f"{last_train} — it has seen {w.name}'s purchases and will rank them first"
        )


def test_retriever_and_candidate_windows_are_disjoint() -> None:
    """Belt and braces on the same property, stated as a set relation.

    The date comparison above would pass if a window appeared in both lists with the
    ordering still intact; this cannot.
    """
    trained = {w.name for w in windows_for_role("retriever")}
    served = {w.name for w in candidate_windows()}
    assert not trained & served, f"windows used for both training and retrieval: {trained & served}"


# --- history() is the only sanctioned read -----------------------------------------


def test_history_excludes_the_cutoff_date(synthetic_txns: pl.LazyFrame) -> None:
    """The cutoff is strict: `t_dat < as_of`, never `<=`.

    A transaction dated exactly `as_of` falls on the first day of the label window.
    Admitting it leaks a day of the thing being predicted — the cheapest possible way to
    invalidate every number downstream, and invisible in the metrics.
    """
    on_boundary = synthetic_txns.filter(pl.col("t_dat") == AS_OF).collect().height
    assert on_boundary > 0, "fixture must contain transactions exactly on the cutoff"

    seen = history(synthetic_txns, as_of=AS_OF).collect()
    assert seen.filter(pl.col("t_dat") >= AS_OF).height == 0
    assert seen.get_column("t_dat").max() < AS_OF


def test_history_keeps_everything_before_the_cutoff(synthetic_txns: pl.LazyFrame) -> None:
    """Strictness must not become over-truncation: nothing before the cutoff is dropped."""
    expected = synthetic_txns.filter(pl.col("t_dat") < AS_OF).collect()
    assert_frame_equal(history(synthetic_txns, as_of=AS_OF).collect(), expected)


def test_history_at_dataset_start_is_empty(synthetic_txns: pl.LazyFrame) -> None:
    assert history(synthetic_txns, as_of=date(2000, 1, 1)).collect().height == 0


# --- signature enforcement ----------------------------------------------------------


def test_no_builder_signature_lacks_as_of() -> None:
    """Every registered builder takes `as_of` as a required keyword-only argument."""
    for builder in ALL_BUILDERS:
        assert_as_of_is_enforced(builder)


def test_builder_without_as_of_is_rejected() -> None:
    """The guard bites. Without this, the contract is a comment."""

    class NoCutoff:
        name = "no_cutoff"
        columns = ("x",)

        def build(self, txns: pl.LazyFrame, *, entities: pl.DataFrame) -> pl.DataFrame:
            return txns.collect()

    with pytest.raises(AsOfContractViolation, match="no `as_of` parameter"):
        register_builder(NoCutoff())


def test_builder_with_defaulted_as_of_is_rejected() -> None:
    """There is no sensible default cutoff; omitting it must be an error."""

    class DefaultedCutoff:
        name = "defaulted_cutoff"
        columns = ("x",)

        def build(
            self,
            txns: pl.LazyFrame,
            *,
            as_of: date = date(2020, 1, 1),
            entities: pl.DataFrame | None = None,
        ) -> pl.DataFrame:
            return txns.collect()

    with pytest.raises(AsOfContractViolation, match="default"):
        register_builder(DefaultedCutoff())


def test_builder_with_positional_as_of_is_rejected() -> None:
    """Positional cutoffs can be supplied by accident, or shift when a param is added."""

    class PositionalCutoff:
        name = "positional_cutoff"
        columns = ("x",)

        def build(  # type: ignore[misc]
            self, txns: pl.LazyFrame, as_of: date, *, entities: pl.DataFrame | None = None
        ) -> pl.DataFrame:
            return txns.collect()

    with pytest.raises(AsOfContractViolation, match="keyword-only"):
        register_builder(PositionalCutoff())


def test_rejected_builders_are_not_registered() -> None:
    """A failed registration must not leave a half-registered builder behind."""
    names = [getattr(b, "name", None) for b in ALL_BUILDERS]
    assert "no_cutoff" not in names
    assert "defaulted_cutoff" not in names
    assert "positional_cutoff" not in names
