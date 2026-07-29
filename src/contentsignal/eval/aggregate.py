"""Aggregation helpers shared by the metric modules.

Exists for one reason: `pl.Series.mean()` is typed as a broad union (it has to be — it is
defined over every dtype), so calling `float()` on it fails strict type checking at every
metric. Wrapping it once documents the empty-input decision in a single place instead of
repeating a `or 0.0` at a dozen call sites where its meaning is invisible.
"""

from __future__ import annotations

import polars as pl


def mean_or_zero(values: pl.Series) -> float:
    """Mean of a numeric series; 0.0 when it is empty.

    Empty means no customer in this slice had anything to find — a cold-start slice with no
    cold purchases, say. Zero is the right answer for a *metric* there (nothing was found
    because nothing was findable), but callers that need to distinguish "scored zero" from
    "nothing to score" must check the height themselves; `eval/retrieval.py` raises on an
    empty cold-start slice for exactly that reason.
    """
    if values.is_empty():
        return 0.0
    result = values.mean()
    if result is None:
        return 0.0
    return float(result)  # type: ignore[arg-type]
