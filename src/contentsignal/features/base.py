"""The `as_of` contract — the interface that makes temporal leakage unwritable.

Every feature builder reads transactions through `history()` and nowhere else. `as_of` is
keyword-only and has no default, so a builder that forgets the cutoff does not compile
into existence — it raises at import time via `ALL_BUILDERS` registration, and
`tests/test_leakage.py` asserts the property directly (trd.md §5.1, §15).

The registry exists so that a builder added later is automatically subjected to the
deletion-invariance property test. Nobody has to remember to wire it up.
"""

from __future__ import annotations

import inspect
from datetime import date
from typing import Protocol, TypeVar, runtime_checkable

import polars as pl


def history(txns: pl.LazyFrame, *, as_of: date) -> pl.LazyFrame:
    """The ONLY sanctioned way to read transactions inside a feature builder.

    Returns strictly-before-cutoff rows. Every builder calls this first; no builder
    touches the raw LazyFrame. Enforced by review and by tests/test_leakage.py.

    The comparison is strict `<`. A transaction dated exactly `as_of` falls on the first
    day of the label window, so admitting it would leak a day of the thing being
    predicted — the cheapest possible way to invalidate every number downstream.
    """
    return txns.filter(pl.col("t_dat") < as_of)


@runtime_checkable
class FeatureBuilder(Protocol):
    """One feature group: built per window, stored independently, joined at train time.

    An arm is defined by which groups it receives, which is what makes the arms
    comparable — they differ only in feature groups, never in data or tuning.
    """

    name: str  # -> artifacts/features/{name}/
    columns: tuple[str, ...]  # declared output columns, asserted on build

    def build(
        self,
        txns: pl.LazyFrame,
        *,
        as_of: date,
        entities: pl.DataFrame,  # the keys to produce rows for
    ) -> pl.DataFrame: ...


ALL_BUILDERS: list[FeatureBuilder] = []

B = TypeVar("B", bound=FeatureBuilder)


class AsOfContractViolation(TypeError):
    """Raised when a builder's `build` signature does not enforce the cutoff."""


def assert_as_of_is_enforced(builder: object) -> None:
    """`build` must take `as_of` as keyword-only and required.

    Checked at registration rather than only in the test suite, so a violation fails at
    import time in the pipeline too — not just under pytest.

    Accepts a class or an instance. On a class, `self` simply stays in the signature,
    which does not affect the `as_of` parameter being checked.
    """
    who = getattr(builder, "name", None) or getattr(builder, "__name__", repr(builder))
    build = getattr(builder, "build", None)
    if build is None:
        raise AsOfContractViolation(f"{who} has no `build` method")
    sig = inspect.signature(build)

    param = sig.parameters.get("as_of")
    if param is None:
        raise AsOfContractViolation(f"{who}.build has no `as_of` parameter: {sig}")
    if param.kind is not inspect.Parameter.KEYWORD_ONLY:
        raise AsOfContractViolation(
            f"{who}.build takes `as_of` as {param.kind.description}, not keyword-only. "
            "A positional cutoff can be supplied by accident or by position shift."
        )
    if param.default is not inspect.Parameter.empty:
        raise AsOfContractViolation(
            f"{who}.build gives `as_of` a default ({param.default!r}). There is no "
            "sensible default cutoff; omitting it must be an error, not a silent read "
            "of all history."
        )


def register_builder(builder: B) -> B:
    """Add a builder to `ALL_BUILDERS` after checking it honours the cutoff contract.

    Use as a decorator on the class, or call with an instance.
    """
    assert_as_of_is_enforced(builder)
    ALL_BUILDERS.append(builder)
    return builder


def assert_declared_columns(builder: FeatureBuilder, out: pl.DataFrame) -> None:
    """A builder's output must match the columns it declares (trd.md §5.1).

    Silent column drift would change an arm's feature set without changing its
    definition, which breaks the only thing that makes arms comparable.
    """
    missing = [c for c in builder.columns if c not in out.columns]
    if missing:
        raise ValueError(f"{builder.name}: declared columns missing from output: {missing}")
