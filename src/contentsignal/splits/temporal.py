"""Temporal windows — the spine of every leakage guarantee in the project.

Ten contiguous, non-overlapping 14-day windows read from `conf/split.yaml` (trd.md §3.1).
The important member is `Window.as_of`: the exclusive cutoff before which all features for
that window must be computed. Feature code never constructs a cutoff itself — it receives
`as_of` from a `Window`.

Windows carry a `role` as well as a `split`, because the retriever is itself a trained model
and therefore needs its own leakage boundary. Four windows train the retriever; the next
four train the rankers on candidates the *frozen* retriever generated. A retriever trained
on a window it later retrieves for has memorized that window's labels and will rank them
first — for a reason that never holds at serving time, when the retriever has never seen
tomorrow. `assert_role_ordering` is what prevents that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import polars as pl

from contentsignal.config import SplitConfig, load_split_config

Split = Literal["train", "val", "test"]
Role = Literal["retriever", "ranker", "val", "test"]

#: Roles whose windows receive candidates from the frozen retriever. Ordered.
CANDIDATE_ROLES: tuple[Role, ...] = ("ranker", "val", "test")


@dataclass(frozen=True)
class Window:
    name: str
    split: Split
    role: Role
    start: date  # inclusive
    end: date  # inclusive

    @property
    def as_of(self) -> date:
        """Exclusive feature cutoff. Features may read `t_dat < as_of`, never `>=`.

        This is `start`, not `start - 1 day`: a transaction landing exactly on the first
        day of the window belongs to the label period, so it must not be visible to a
        feature. The strict `<` in `features.base.history` is what enforces that.
        """
        return self.start

    @property
    def length_days(self) -> int:
        return (self.end - self.start).days + 1


def load_windows(cfg: SplitConfig | None = None) -> list[Window]:
    """Materialize the windows from config, validating their geometry on the way out."""
    cfg = cfg or load_split_config()
    windows = [
        Window(name=w.name, split=w.split, role=w.role, start=w.start, end=w.end)
        for w in cfg.windows
    ]
    assert_contiguous_non_overlapping(windows)
    assert_split_ordering(windows)
    assert_role_ordering(windows)
    return windows


def window_by_name(name: str, cfg: SplitConfig | None = None) -> Window:
    for w in load_windows(cfg):
        if w.name == name:
            return w
    known = ", ".join(w.name for w in load_windows(cfg))
    raise KeyError(f"unknown window {name!r}; known windows: {known}")


def windows_for_split(split: Split, cfg: SplitConfig | None = None) -> list[Window]:
    return [w for w in load_windows(cfg) if w.split == split]


def windows_for_role(role: Role, cfg: SplitConfig | None = None) -> list[Window]:
    return [w for w in load_windows(cfg) if w.role == role]


def candidate_windows(cfg: SplitConfig | None = None) -> list[Window]:
    """Every window the frozen retriever generates candidates for: ranker, val, test.

    The retriever must not have trained on any of these (`assert_role_ordering`).
    """
    return [w for w in load_windows(cfg) if w.role in CANDIDATE_ROLES]


def assert_contiguous_non_overlapping(ws: Sequence[Window]) -> None:
    """Each window starts the day after the previous one ends. No gaps, no overlaps.

    A gap would silently drop transactions from every row set; an overlap would let the
    same purchase be a label in one window and history in another.
    """
    if not ws:
        raise ValueError("no windows configured")
    for prev, cur in zip(ws, ws[1:], strict=False):  # pairwise; the tail has no successor
        if cur.start != prev.end + timedelta(days=1):
            gap = (cur.start - prev.end).days - 1
            kind = "gap" if gap > 0 else "overlap"
            raise ValueError(
                f"{kind} between {prev.name} (ends {prev.end}) and {cur.name} (starts {cur.start})"
            )


def assert_split_ordering(ws: Sequence[Window]) -> None:
    """Train strictly precedes val, which strictly precedes test.

    Chronological, never random: a shuffled split would let a model see the future of
    the very customers it is scored on (prd.md §4).
    """
    order: dict[str, int] = {"train": 0, "val": 1, "test": 2}
    seen = [order[w.split] for w in ws]
    if seen != sorted(seen):
        raise ValueError(f"splits are out of order: {[(w.name, w.split) for w in ws]}")

    train_end = max((w.end for w in ws if w.split == "train"), default=None)
    val = [w for w in ws if w.split == "val"]
    test = [w for w in ws if w.split == "test"]
    if len(val) != 1 or len(test) != 1:
        raise ValueError(
            f"expected exactly one val and one test window, got {len(val)}/{len(test)}"
        )
    if train_end is None or not (train_end < val[0].start < test[0].start):
        raise ValueError("expected train.end < val.start < test.start")


def assert_role_ordering(ws: Sequence[Window]) -> None:
    """Every retriever-training window ends strictly before any window it retrieves for.

    This is the leakage boundary the second stage introduces, and it has no analogue in a
    single-stage design. If the retriever trains on a window it later generates candidates
    for, it has seen those purchases and will rank them first — so the ranker trains on
    candidate lists where the answer sits at rank 1 for a reason that will never hold at
    serving time, learns to trust rank 1, and collapses in production (prd.md §5).

    `tests/test_leakage.py::test_retriever_windows_precede_candidate_windows` asserts the
    same property independently, so a regression here cannot hide behind a passing loader.
    """
    order: dict[Role, int] = {"retriever": 0, "ranker": 1, "val": 2, "test": 3}
    seen = [order[w.role] for w in ws]
    if seen != sorted(seen):
        raise ValueError(f"roles are out of order: {[(w.name, w.role) for w in ws]}")

    retriever = [w for w in ws if w.role == "retriever"]
    candidates = [w for w in ws if w.role in CANDIDATE_ROLES]
    if not retriever:
        raise ValueError("no windows have role 'retriever'; the retriever has nothing to train on")
    if not candidates:
        raise ValueError(f"no windows have a candidate role ({', '.join(CANDIDATE_ROLES)})")

    last_train = max(w.end for w in retriever)
    first_candidate = min(w.start for w in candidates)
    if last_train >= first_candidate:
        overlap = [w.name for w in candidates if w.start <= last_train]
        raise ValueError(
            f"retriever training ends {last_train} but candidate windows start "
            f"{first_candidate}; the retriever would have trained on {overlap}, whose "
            "labels it would then rank first"
        )


def eligible_customers(txns: pl.LazyFrame, w: Window, cohort: pl.Series) -> pl.Series:
    """Cohort customers with at least one purchase inside `w` (trd.md §3.3).

    Rows exist only for customers who transacted during the window, so results are
    conditional on the customer transacting — stated in reports, not buried.
    """
    keys = pl.DataFrame({"customer_idx": cohort}).lazy()
    found = (
        txns.filter(pl.col("t_dat").is_between(w.start, w.end, closed="both"))
        .join(keys, on="customer_idx", how="semi")
        .select(pl.col("customer_idx").unique())
        .sort("customer_idx")
        .collect()
    )
    return found.get_column("customer_idx")


def positives(txns: pl.LazyFrame, w: Window, eligible: pl.Series) -> pl.DataFrame:
    """Distinct (customer_idx, article_id) purchased in `w` by eligible customers.

    Repeat purchases of the same article collapse to one row. Purchase count is
    deliberately not carried as a feature: it is a function of the label window and
    would leak (trd.md §3.3).
    """
    keys = pl.DataFrame({"customer_idx": eligible}).lazy()
    return (
        txns.filter(pl.col("t_dat").is_between(w.start, w.end, closed="both"))
        .join(keys, on="customer_idx", how="semi")
        .select("customer_idx", "article_id")
        .unique()
        .sort("customer_idx", "article_id")
        .with_columns(y=pl.lit(1, dtype=pl.Int8))
        .collect()
    )
