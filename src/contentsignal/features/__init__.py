"""Feature groups — the stage-2 ranker's inputs (trd.md §6).

Five groups, ~47 columns: `customer`, `article`, `categorical`, `cross`, `retrieval`. Each
is built and stored independently per window, then joined at train time. Every ranker arm
receives every group, so the arms differ only in architecture — never in data or tuning.

The eleven columns in `categorical` are non-negotiable for every arm. Almost every text
field in `articles.csv` has a 1:1 categorical twin, so withholding them from a baseline
while feeding the same information to the encoder as text would credit the encoder with
information the baseline never had (prd.md §3).

There is no `text_item` or `text_customer` group. Hand-built similarity features like
`sim_taste_cos` existed because *trees cannot compute a dot product* between a customer
taste vector and an item vector — a two-tower computes exactly that, learned end-to-end, so
stage 1 subsumes them (trd.md §7).

**Every builder module must be imported here.** `ALL_BUILDERS` is populated as a side
effect of import, and `tests/test_leakage.py` applies deletion invariance to whatever is
in that list. A builder in a module nobody imports is a builder nobody leak-tests, which
is worse than having no harness at all — the suite would go green while the property went
unchecked. The imports land alongside the builders at M2.
"""

from __future__ import annotations

from contentsignal.features.base import (
    ALL_BUILDERS,
    AsOfContractViolation,
    FeatureBuilder,
    assert_as_of_is_enforced,
    assert_declared_columns,
    history,
    register_builder,
)

# M2: from contentsignal.features import article, categorical, cross, customer  # noqa: F401
# M5: from contentsignal.features import retrieval  # noqa: F401

__all__ = [
    "ALL_BUILDERS",
    "AsOfContractViolation",
    "FeatureBuilder",
    "assert_as_of_is_enforced",
    "assert_declared_columns",
    "history",
    "register_builder",
]
