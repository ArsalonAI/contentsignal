"""Typed configuration and the config digest that makes every stage idempotent.

Two jobs:

1. Parse `conf/*.yaml` into validated models. `conf/split.yaml` is the single source of
   truth for window boundaries (trd.md §3.1) — no module hardcodes a date.
2. Produce `config_sha256`, a stable digest of the config a stage ran under. Every CLI
   command skips work when its outputs already exist with a matching digest (trd.md §13),
   and the digest is logged as an MLflow param (trd.md §11) so a run's numbers can always
   be traced back to the exact configuration that produced them.

The digest must be stable across processes and Python versions. It is computed over
canonical JSON — sorted keys, no whitespace, dates as ISO strings — never over `repr()`
or `hash()`, both of which vary between runs.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date, timedelta
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

Split = Literal["train", "val", "test"]

# What a window is FOR, orthogonal to its ML split. The retriever is itself a trained model,
# so it needs its own leakage boundary — see `WindowSpec.role` and trd.md §3.1.
Role = Literal["retriever", "ranker", "val", "test"]


class Frozen(BaseModel):
    """Base for every config model: immutable, and unknown keys are an error.

    `extra="forbid"` is deliberate. A typo'd key in a YAML file would otherwise be
    silently ignored, and the run would proceed under settings nobody chose.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class WindowSpec(Frozen):
    """One window's dates and roles as written in `conf/split.yaml`. Bounds inclusive."""

    name: str
    split: Split
    role: Role
    start: date
    end: date

    @model_validator(mode="after")
    def _check_role_agrees_with_split(self) -> WindowSpec:
        """`retriever` and `ranker` are train roles; `val`/`test` roles match their split.

        Caught here rather than downstream because a `val` window mislabelled as `ranker`
        would quietly enter the ranker's training set — the exact leak the roles exist to
        prevent.
        """
        expected: dict[Role, Split] = {
            "retriever": "train",
            "ranker": "train",
            "val": "val",
            "test": "test",
        }
        if expected[self.role] != self.split:
            raise ValueError(
                f"window {self.name}: role {self.role!r} requires split "
                f"{expected[self.role]!r}, got {self.split!r}"
            )
        return self


class SplitConfig(Frozen):
    window_length_days: int = 14
    dataset_start: date
    dataset_end: date
    windows: tuple[WindowSpec, ...]

    @model_validator(mode="after")
    def _check_lengths_and_bounds(self) -> SplitConfig:
        span = timedelta(days=self.window_length_days - 1)  # bounds are inclusive
        for w in self.windows:
            if w.end - w.start != span:
                raise ValueError(
                    f"window {w.name}: {w.start}..{w.end} is not "
                    f"{self.window_length_days} days (inclusive)"
                )
            if w.start < self.dataset_start or w.end > self.dataset_end:
                raise ValueError(
                    f"window {w.name}: {w.start}..{w.end} falls outside the dataset "
                    f"range {self.dataset_start}..{self.dataset_end}"
                )
        return self


class PathsConfig(Frozen):
    raw: Path
    artifacts: Path
    parquet: Path
    candidates: Path
    rows_random: Path
    features: Path
    vectors: Path
    mlruns: Path


class KaggleConfig(Frozen):
    competition: str


class CohortConfig(Frozen):
    size: int
    seed: int
    qualify_start: date
    qualify_end: date


class CandidatesConfig(Frozen):
    """Candidate generation and the ranker row budget (trd.md §3.4, §8).

    `k_sweep` is evaluated by truncating a single retrieval pass at `max(k_sweep)`, so the
    whole sweep costs one pass and no retraining.
    """

    k: int = 100
    k_sweep: tuple[int, ...] = (20, 50, 100, 200, 500, 1000)
    target_train_rows: int = 5_000_000
    row_budget_tolerance: float = 0.2
    train_customer_cap: int | None = None

    @model_validator(mode="after")
    def _check_k_within_sweep(self) -> CandidatesConfig:
        """`k` must be reachable by truncating the sweep, and the sweep must be sorted.

        If `k` exceeded `max(k_sweep)`, the main grid's candidate depth could not be
        derived from the sweep's single retrieval pass, and the two would silently
        disagree about what `recall@k` refers to.
        """
        if not self.k_sweep:
            raise ValueError("k_sweep must not be empty")
        if list(self.k_sweep) != sorted(self.k_sweep):
            raise ValueError(f"k_sweep must be ascending, got {self.k_sweep}")
        if self.k > max(self.k_sweep):
            raise ValueError(
                f"candidates.k ({self.k}) exceeds max(k_sweep) ({max(self.k_sweep)}); "
                "the sweep is computed by truncating one retrieval pass, so it must reach k"
            )
        return self

    @property
    def retrieval_depth(self) -> int:
        """The depth actually retrieved: enough to serve both `k` and the whole sweep."""
        return max(self.k, max(self.k_sweep))


class SamplerConfig(Frozen):
    """Random negative sampling — the H3 arm only (trd.md §8.3).

    The main grid's negatives are the retriever's top-K non-purchases. This sampler exists
    so H3 can measure what training on the wrong negative distribution costs.

    `seed` is the root seed. Per-window seeds derive from it deterministically; see
    `contentsignal.sampling.negatives.window_seed`.
    """

    ratio: int = 10
    pop_exponent: float = 0.75
    pop_lookback_weeks: int = 12
    seed: int = 17


class EvalConfig(Frozen):
    """Slice boundaries and bootstrap settings (trd.md §10.5, §10.6).

    Slice thresholds are set once at M1 from measured distributions and registered.
    Retuning one after seeing results is p-hacking with extra steps.
    """

    cold_start_threshold: int = 10
    low_history_threshold: int = 3
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 17
    ndcg_k: int = 12


class DataConfig(Frozen):
    paths: PathsConfig
    kaggle: KaggleConfig
    cohort: CohortConfig
    candidates: CandidatesConfig
    sampler: SamplerConfig
    eval: EvalConfig


def repo_root() -> Path:
    """The directory holding `conf/` and `pyproject.toml`.

    Walks up from this file so commands work from any working directory; the package is
    installed editable, so `__file__` sits inside the repo.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "conf").is_dir():
            return parent
    return Path.cwd()


def conf_dir() -> Path:
    """`$CONTENTSIGNAL_CONF` if set, else `<repo root>/conf`."""
    override = os.environ.get("CONTENTSIGNAL_CONF")
    return Path(override) if override else repo_root() / "conf"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(loaded).__name__}")
    return loaded


@cache
def load_split_config(path: Path | None = None) -> SplitConfig:
    return SplitConfig.model_validate(_read_yaml(path or conf_dir() / "split.yaml"))


@cache
def load_data_config(path: Path | None = None) -> DataConfig:
    return DataConfig.model_validate(_read_yaml(path or conf_dir() / "data.yaml"))


def config_sha256(cfg: BaseModel | dict[str, Any]) -> str:
    """Stable digest of a config, for idempotency checks and run provenance.

    Canonical JSON — sorted keys, tight separators, `mode="json"` so dates and paths
    serialize as strings rather than objects whose repr could drift.
    """
    payload: Any = cfg.model_dump(mode="json") if isinstance(cfg, BaseModel) else cfg
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ConfigHashMismatch(RuntimeError):
    """Raised when an existing artifact was produced under a different config."""


def outputs_are_current(stamp_path: Path, digest: str, *, force: bool = False) -> bool:
    """True when a stage may skip its work.

    Every stage writes `stamp_path` containing the digest of the config it ran under.
    The stage is skippable only if that file exists and matches (trd.md §13).
    """
    if force or not stamp_path.is_file():
        return False
    return stamp_path.read_text(encoding="utf-8").strip() == digest


def write_stamp(stamp_path: Path, digest: str) -> None:
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(digest + "\n", encoding="utf-8")
