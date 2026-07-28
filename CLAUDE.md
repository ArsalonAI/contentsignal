# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

**This repo currently contains no code.** It holds two specification documents:

- `prd.md` — what is being built and why (approved)
- `trd.md` — how: schemas, signatures, algorithms, resource budgets, test assertions

Every command, module, and artifact path referenced below is **specified in `trd.md` but
not yet implemented**. Do not assume anything runs. Implementation follows the M0–M9
milestones in `trd.md` §16.

Before writing code, read `trd.md` §5 (module contracts) and §6 (feature specification).
They are precise enough that implementation is mechanical, and deviating from them silently
breaks the experiment's validity rather than just its style.

## What this project is

A research repo testing one falsifiable hypothesis on the H&M Personalized Fashion
Recommendations dataset: **does fine-tuning a transformer encoder on product text add
measurable predictive lift over tabular features alone for purchase propensity, and what
does that lift cost to serve?**

It is an experiment, not a product. The output is a results table and a written finding.
Correctness of the *comparison* matters more than the performance of any single model.

## Planned commands (`trd.md` §13)

A `typer` CLI installed as `contentsignal`, with a `Makefile` chaining `make m1` … `make m9`
to match milestones. Every command is idempotent and skips work when outputs exist with a
matching config hash, unless `--force`.

```
contentsignal ingest
contentsignal sample
contentsignal build-features --group G --window W
contentsignal embed --variant a|b --source frozen|contrastive
contentsignal finetune --variant a|b --seed N
contentsignal train --arm A --variant V --seed N
contentsignal evaluate --arm A ...
contentsignal bench --config C
contentsignal report
```

Environment: **Python 3.11 via `uv`** — system Python is 3.9.6 and cannot be used.
Tests: `pytest`, with a single test as `pytest tests/test_leakage.py::test_name`.

**M0 is blocked** until the H&M competition rules are accepted on Kaggle and
`~/.kaggle/kaggle.json` exists with mode `600`. Neither is true on this machine.

## Architecture

The pipeline is a linear chain of idempotent stages writing immutable artifacts under
`artifacts/` (gitignored):

```
Kaggle CSVs → parquet → sampled row sets → feature groups ─┐
                                                            ├→ arms → metrics JSON → report
              article text → encoder → embedding cache ────┘
```

Three structural ideas carry the whole design:

**1. Everything is windowed and `as_of`-gated.** Data is cut into ten 14-day windows
(8 train, 1 val, 1 test). Every feature for a row in window *W* is computed strictly from
transactions before `W.start`. This is not a convention — it is enforced by the type
signature (see Invariants).

**2. Feature groups are the ablation axes.** `customer`, `article`, `categorical`, `cross`,
`text_item`, `text_customer`. Each is built and stored independently, then joined at train
time. An arm is defined by which groups it receives. This is why arms are comparable: they
differ only in feature groups, never in data or tuning.

**3. The encoder is an offline feature extractor, not an online model.** It encodes ~105k
unique article texts once into a cache; everything downstream reads from that cache.
Embeddings are stored as `.npy` + a sorted `ids.npy` (not Parquet) so the serving benchmark
can `mmap` them and lookup is `np.searchsorted`.

### How the customer enters the text arm

Non-obvious and load-bearing. The encoder never reads a customer column. Personalization
enters two ways:

- **Taste vector** — the mean embedding of articles a customer bought before `as_of`, and
  similarity features between it and the candidate article (`sim_taste_cos` et al.).
- **Contrastive fine-tuning** — positive pairs are two articles bought by the *same*
  customer. Customer information enters through *which pairs are positives*.

This exists because item-side embedding columns hold the same value for every customer, and
a feature constant across customers cannot reorder items differently per customer. Without
the customer side, per-customer ranking metrics would show near-zero lift for structural
reasons regardless of whether text carries signal. See `prd.md` §6.

## Invariants

These are the things that quietly invalidate results if broken. Most were chosen against a
specific failure mode; the reasoning is in the referenced sections.

**The `as_of` contract.** `FeatureBuilder.build` takes `as_of` as keyword-only and required.
Feature code reads transactions only through `features.base.history(txns, as_of=...)`.
Never add an overload or default that permits reading without a cutoff — the signature is
the enforcement mechanism (`trd.md` §5.1).

**Row sets are byte-identical across arms.** `artifacts/rows/{window}.parquet` is written
once and digested in `rows_manifest.json`; every arm asserts the digest before training.
Never resample per arm — a ΔAUC of 0.005 is otherwise indistinguishable from a different
random draw (`trd.md` §4.5).

**The tabular baseline receives all eleven categorical columns.** Withholding them while
feeding the same information to the encoder as text credits the encoder with information
the baseline was never given. This is the most common way this experiment is run wrong
(`prd.md` §6).

**Hyperparameters are tuned on arm 3 only, then frozen.** Tuning each arm separately
confounds "better features" with "more tuning budget" (`trd.md` §9.4).

**`max_bin=63` is a memory requirement, not a tuning choice.** It takes the 5M × 80 matrix
from 1.6 GB raw to 400 MB binned, which is what makes LightGBM fit in 8 GB (`trd.md` §14).

**The test split is read once, at M9.** M3–M8 report validation numbers. The pre-registered
success criterion (ΔAUC ≥ 0.005 with a customer-level bootstrap CI excluding zero) is
applied to that single evaluation (`prd.md` §1).

**Bootstrap resamples customers, not rows.** Rows within a customer share history features
and basket composition; row-level resampling gives CIs narrow enough for noise to clear the
significance bar (`trd.md` §10.2).

**Prior correction is applied by the evaluator, never inside an arm.** `Arm.predict` returns
sampled-distribution probabilities. Correction happens once, in one place (`trd.md` §10.3).

**No number in `reports/results.md` is hand-typed.** Runs dual-write to MLflow
(`artifacts/mlruns`, local file backend) and to git-committed
`reports/metrics/{run_name}.json`; `make report` regenerates the tables from the JSON
(`trd.md` §11).

## Claims that must not be overstated

The project's credibility rests on these, and they are easy to soften by accident:

- **The label is purchase, not click or add-to-cart.** H&M ships purchase transactions only;
  no impression or funnel events exist in the dataset. Do not describe this as engagement,
  click, or ATC prediction (`prd.md` §3).
- **`price` is scaled and anonymized, not currency.** All revenue and AOV figures are
  relative lift ratios. Never report AOV in dollars (`prd.md` §2).
- **Rows exist only for customers who transacted in the window.** Results are conditional on
  the customer transacting and do not describe the full customer base (`trd.md` §3.3).
- **A null result is pre-committed.** If the encoder does not clear the bar, that is the
  headline finding, reported with the same prominence. Do not re-tune, re-cut splits, or
  swap metrics to manufacture a positive (`prd.md` §1).

## Engineering constraints

8 GB RAM, 8 cores, Apple Silicon, MPS only. This drives real design decisions, not just
caution:

- **DuckDB for aggregation, Polars for frames.** 31.8M transaction rows do not fit in
  pandas here. `pandas` is not a pipeline dependency — it arrives via MLflow and is for
  notebooks and report generation only.
- **Per-stage memory budgets are specified** in `trd.md` §14 against a ~6.5 GB usable
  ceiling. LightGBM training is the binding constraint at ~3.5 GB peak.
- **The row budget is the binding scale constraint, not cohort size.** `sample` measures
  actual counts and fails if train rows exceed the target by >20%, naming the cohort size
  that would fit (`trd.md` §3.4).
