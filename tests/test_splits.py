"""Window geometry (trd.md §15, §3.1).

If the windows are wrong, every downstream number is wrong in a way no later test can
detect — a one-day overlap between train and val is invisible in the metrics and fatal to
the conclusion.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from contentsignal.config import SplitConfig, load_split_config
from contentsignal.splits.temporal import (
    assert_contiguous_non_overlapping,
    assert_role_ordering,
    assert_split_ordering,
    candidate_windows,
    load_windows,
    window_by_name,
    windows_for_role,
    windows_for_split,
)

WINDOW_DAYS = 14


@pytest.fixture(scope="module")
def cfg() -> SplitConfig:
    return load_split_config()


def test_ten_windows_eight_one_one(cfg: SplitConfig) -> None:
    ws = load_windows(cfg)
    assert len(ws) == 10
    assert len(windows_for_split("train", cfg)) == 8
    assert len(windows_for_split("val", cfg)) == 1
    assert len(windows_for_split("test", cfg)) == 1


def test_train_windows_split_four_four_by_role(cfg: SplitConfig) -> None:
    """Four windows train the retriever, four train the rankers.

    The split is what buys the leakage boundary: the retriever never sees a window it
    later generates candidates for.
    """
    assert len(windows_for_role("retriever", cfg)) == 4
    assert len(windows_for_role("ranker", cfg)) == 4
    assert len(windows_for_role("val", cfg)) == 1
    assert len(windows_for_role("test", cfg)) == 1


def test_candidate_windows_are_ranker_val_and_test(cfg: SplitConfig) -> None:
    """Every window the frozen retriever serves — and no retriever-training window."""
    served = candidate_windows(cfg)
    assert [w.name for w in served] == ["rank_w1", "rank_w2", "rank_w3", "rank_w4", "val", "test"]
    assert not any(w.role == "retriever" for w in served)


def test_every_window_is_fourteen_days(cfg: SplitConfig) -> None:
    for w in load_windows(cfg):
        assert w.length_days == WINDOW_DAYS, f"{w.name} is {w.length_days} days"


def test_windows_are_contiguous_and_non_overlapping(cfg: SplitConfig) -> None:
    ws = load_windows(cfg)
    assert_contiguous_non_overlapping(ws)
    for prev, cur in zip(ws, ws[1:], strict=False):
        assert cur.start == prev.end + timedelta(days=1)


def test_train_precedes_val_precedes_test(cfg: SplitConfig) -> None:
    ws = load_windows(cfg)
    assert_split_ordering(ws)
    train_end = max(w.end for w in ws if w.split == "train")
    val = windows_for_split("val", cfg)[0]
    test = windows_for_split("test", cfg)[0]
    assert train_end < val.start < test.start


def test_windows_lie_inside_the_dataset_range(cfg: SplitConfig) -> None:
    for w in load_windows(cfg):
        assert cfg.dataset_start <= w.start
        assert w.end <= cfg.dataset_end


def test_last_window_ends_on_the_last_day(cfg: SplitConfig) -> None:
    """The test window ends on the dataset's final day — no data left unused past it."""
    assert windows_for_split("test", cfg)[0].end == cfg.dataset_end


def test_as_of_is_the_window_start(cfg: SplitConfig) -> None:
    """The cutoff is exclusive and equals `start`: nothing inside the window is visible."""
    for w in load_windows(cfg):
        assert w.as_of == w.start


def test_window_by_name_rejects_unknown(cfg: SplitConfig) -> None:
    assert window_by_name("val", cfg).split == "val"
    with pytest.raises(KeyError, match="unknown window"):
        window_by_name("ret_w99", cfg)


def test_retriever_training_precedes_everything_it_serves(cfg: SplitConfig) -> None:
    """The leakage boundary the second stage introduces.

    A retriever trained on a window it later retrieves for has memorized that window's
    purchases and will rank them first — so the ranker trains on candidate lists where the
    answer sits at rank 1 for a reason that never holds at serving time. Asserted again in
    test_leakage.py, deliberately: this is a property worth two independent checks.
    """
    ws = load_windows(cfg)
    assert_role_ordering(ws)
    last_train = max(w.end for w in windows_for_role("retriever", cfg))
    first_served = min(w.start for w in candidate_windows(cfg))
    assert last_train < first_served


# --- the geometry checks must actually bite ---------------------------------------

_ROLE_FOR_SPLIT: dict[str, str] = {"train": "retriever", "val": "val", "test": "test"}


def _spec(
    name: str,
    split: str,
    start: date,
    days: int = WINDOW_DAYS,
    role: str | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "split": split,
        "role": role or _ROLE_FOR_SPLIT[split],
        "start": start,
        "end": start + timedelta(days=days - 1),
    }


def _config(*windows: dict[str, object]) -> dict[str, object]:
    return {
        "window_length_days": WINDOW_DAYS,
        "dataset_start": date(2018, 9, 20),
        "dataset_end": date(2020, 9, 22),
        "windows": windows,
    }


def test_gap_between_windows_is_rejected() -> None:
    bad = _config(
        _spec("a", "train", date(2020, 5, 6)),
        _spec("b", "val", date(2020, 5, 21)),  # one day late
    )
    with pytest.raises(ValueError, match="gap between"):
        load_windows(SplitConfig.model_validate(bad))


def test_overlapping_windows_are_rejected() -> None:
    bad = _config(
        _spec("a", "train", date(2020, 5, 6)),
        _spec("b", "val", date(2020, 5, 19)),  # starts before a ends
    )
    with pytest.raises(ValueError, match="overlap between"):
        load_windows(SplitConfig.model_validate(bad))


def test_wrong_length_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="is not 14 days"):
        SplitConfig.model_validate(_config(_spec("a", "train", date(2020, 5, 6), days=13)))


def test_window_outside_dataset_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside the dataset range"):
        SplitConfig.model_validate(_config(_spec("a", "train", date(2020, 9, 20))))


def test_val_after_test_is_rejected() -> None:
    bad = _config(
        _spec("a", "train", date(2020, 5, 6)),
        _spec("b", "test", date(2020, 5, 20)),
        _spec("c", "val", date(2020, 6, 3)),
    )
    with pytest.raises(ValueError, match="out of order"):
        load_windows(SplitConfig.model_validate(bad))


# --- the role checks must bite too ------------------------------------------------


def test_ranker_window_before_retriever_window_is_rejected() -> None:
    """Roles out of chronological order means the retriever trained on the future."""
    bad = _config(
        _spec("a", "train", date(2020, 5, 6), role="ranker"),
        _spec("b", "train", date(2020, 5, 20), role="retriever"),
        _spec("c", "val", date(2020, 6, 3)),
        _spec("d", "test", date(2020, 6, 17)),
    )
    with pytest.raises(ValueError, match="out of order"):
        load_windows(SplitConfig.model_validate(bad))


def test_config_with_no_retriever_window_is_rejected() -> None:
    """Without a retriever window there is nothing to train stage 1 on."""
    bad = _config(
        _spec("a", "train", date(2020, 5, 6), role="ranker"),
        _spec("b", "val", date(2020, 5, 20)),
        _spec("c", "test", date(2020, 6, 3)),
    )
    with pytest.raises(ValueError, match="role 'retriever'"):
        load_windows(SplitConfig.model_validate(bad))


def test_role_disagreeing_with_split_is_rejected() -> None:
    """A val window mislabelled `ranker` would quietly enter the ranker's training set —
    the exact leak roles exist to prevent, so it fails at config parse time."""
    bad = _config(
        _spec("a", "train", date(2020, 5, 6), role="retriever"),
        _spec("b", "val", date(2020, 5, 20), role="ranker"),
    )
    with pytest.raises(ValueError, match="requires split"):
        SplitConfig.model_validate(bad)
