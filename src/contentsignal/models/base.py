"""The arm protocol (trd.md §5.5).

A common interface so arms are interchangeable and the training CLI is arm-agnostic. An
arm is defined by *which feature groups it receives* — that is the ablation axis, and it
is why the arms are comparable to each other at all.

`predict` returns probabilities on the sampled distribution. Prior correction lives in
`contentsignal.eval.calibration` and is applied by the evaluator, so it happens exactly
once and identically for every arm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import polars as pl


@runtime_checkable
class Arm(Protocol):
    name: str
    feature_groups: tuple[str, ...]

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        *,
        valid: tuple[pl.DataFrame, np.ndarray] | None,
    ) -> None: ...

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """SAMPLED-distribution probabilities. Never prior-corrected here."""
        ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> Arm: ...
