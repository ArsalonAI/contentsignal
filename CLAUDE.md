# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

Two specification documents govern the work:

- `prd.md` — what is being built and why (approved)
- `trd.md` — how: schemas, signatures, algorithms, resource budgets, test assertions

**Implemented so far (M0 complete, plus M1's ingest half):**

- `pyproject.toml` / `uv.lock` — Python 3.11 env, deps per `trd.md` §2
- `conf/split.yaml`, `conf/data.yaml`, `src/contentsignal/config.py` — typed config,
  `config_sha256` (the digest behind every stage's idempotency check), and `resolve_path`,
  which anchors relative config paths at the repo root rather than the cwd
- `src/contentsignal/cli.py` — all `trd.md` §13 commands declared. `ingest` and `splits` do
  real work; the rest raise `NotImplementedError` naming their milestone
- `src/contentsignal/data/` — `schema.py` (the §4.1–4.4 column contract, CSV read specs, and
  the eleven categoricals), `to_parquet.py` (DuckDB streaming conversion, atomic writes,
  post-write validation), `download.py` (the Kaggle fresh-clone fallback)
- `splits/temporal.py`, `features/base.py`, `sampling/negatives.py`,
  `eval/calibration.py`, `models/base.py` — the invariant spine
- `tests/` — `test_splits`, `test_sampling`, `test_calibration`, `test_ingest` green;
  `test_leakage` green except the deletion-invariance property, which is parameterized over
  an `ALL_BUILDERS` registry that stays empty until M2

**Data is on disk.** The three CSVs are in `data/` (gitignored) and
`artifacts/parquet/{transactions,articles,customers,customer_index}.parquet` is built —
31,788,324 transactions spanning 2018-09-20 → 2020-09-22, 105,542 articles (416 with an empty
`detail_desc`), 1,371,980 customers. Ingest runs in ~10 s at 1.58 GB peak RSS.

**Not implemented:** EDA, feature builders, the two-tower retriever, item-vector precomputation,
candidate generation, all three rankers, evaluation, benchmark, report. Assume a command does not
run unless it appears above. Implementation follows the M0–M9 milestones in `trd.md` §16.

**Still open at M1:** `eval.cold_start_threshold` (10) and `eval.low_history_threshold` (3) in
`conf/data.yaml` are placeholders. `trd.md` §10.5 requires both be set once from the measured
distributions during EDA and then frozen; moving one after seeing results is p-hacking.

Before writing code, read `trd.md` §5 (module contracts) and §6 (feature specification). They
are precise enough that implementation is mechanical, and deviating from them silently breaks
the experiment's validity rather than just its style.

## What this project is

A two-stage recommender on the H&M Personalized Fashion Recommendations dataset, built to answer
three pre-registered questions about **how the two stages interact**:

- **H1 — stage attribution.** Is end-to-end quality more sensitive to retrieval depth than to
  ranker architecture? Registered prior: yes.
- **H2 — content retrieval and cold start.** Does the free-text product description surface
  cold-start articles a behavioral retriever structurally cannot? Registered prior: yes on
  cold-start, marginal or null on aggregate.
- **H3 — candidate distribution shift.** Does a ranker trained on random negatives underperform
  the same ranker trained on the retriever's hard negatives? Registered prior: yes, and by more
  than the ranker-architecture delta in H1.

It is an experiment, not a product. The output is a results table and a written finding.
**Correctness of the comparison matters more than the performance of any single model.**

## Planned commands (`trd.md` §13)

A `typer` CLI installed as `contentsignal`, with a `Makefile` chaining `make m1` … `make m9`
to match milestones. Every command is idempotent and skips work when outputs exist with a
matching config hash, unless `--force`.

```
contentsignal ingest
contentsignal splits
contentsignal sample
contentsignal build-features --group G --window W
contentsignal train-retriever --arm pop|R1|R2 --variant a|b --seed N
contentsignal embed --retriever R --seed N
contentsignal retrieve --window W --k K
contentsignal train-ranker --arm lgbm|mlp|dcn --negatives retrieved|random --seed N
contentsignal evaluate --stage retrieval|ranking|e2e --arm A --split val|test
contentsignal bench --config C
contentsignal report
```

There is **no `finetune` command.** In-batch sampled softmax over co-purchase pairs *is* a
contrastive objective, so the text encoder trains jointly with the towers inside
`train-retriever`. A separate pre-training stage would be a second copy of the same objective.

Environment: **Python 3.11 via `uv`** — system Python is 3.9.6 and cannot be used.
Tests: `pytest`, with a single test as `pytest tests/test_leakage.py::test_name`.

`ingest` reads `data/` and only calls Kaggle when a CSV is actually missing, so no credentials
are needed on this machine. It does not delete the raw CSVs after conversion — re-acquiring them
costs 30–60 min of network, and deleting a hand-supplied input is not the pipeline's call.
`--force` rebuilds everything except `customer_index.parquet`, which is written once because
every downstream artifact keys on `customer_idx` (`trd.md` §4.4); `--force-customer-index` is
the deliberate escape hatch.

## Architecture

A linear chain of idempotent stages writing immutable artifacts under `artifacts/` (gitignored):

```
Kaggle CSVs → parquet → positives ─┬→ two-tower retriever → item vectors ─┐
                                   │                                       ├→ candidates
              article text ────────┘                                       │      │
                                                                           │      ▼
              feature groups (customer · article · categorical · cross · retrieval)
                                                                                  │
                                                        3 rankers → metrics JSON → report
```

Four structural ideas carry the whole design:

**1. Two stages, because the catalog is 105k and a slate is 12.** A ranking model costs ~1 ms
per pair, so scoring the whole catalog per request is ~100 seconds. Stage 1 (two-tower, ~1 ms
total) picks ~100 from 105k; stage 2 (a real ranker, ~1 ms *per candidate*) orders those. The
metrics differ accordingly: `recall@K` for stage 1, NDCG@12/MAP@12 for stage 2.

**2. Everything is windowed and `as_of`-gated.** Ten 14-day windows. Every feature for a row in
window *W* is computed strictly from transactions before `W.start`. This is not a convention —
it is enforced by the type signature (see Invariants).

**3. Windows have roles, because the retriever is itself a trained model.** W1–W4 train the
retriever; W5–W8 train the rankers on candidates the frozen retriever generated; W9 is
validation; W10 is test. A retriever trained on a window it retrieves for has memorized the
labels and puts them at rank 1 — for a reason that never holds at serving time.

**4. The encoder is an offline feature extractor, not an online model.** After training, the
item tower encodes all ~105k articles once into a 54 MB `.npy` + sorted `ids.npy` cache (not
Parquet) so the serving benchmark can `mmap` it and look up by `np.searchsorted`. Steady-state
transformer serving cost is therefore ≈0; the cold path matters only for new articles.

### Why the two-tower factorization is load-bearing

Non-obvious and worth stating. The customer tower produces 128 numbers, the item tower produces
128 numbers, and the score is their dot product. Because the item vector does not depend on the
customer, all 105k are precomputed offline and retrieval is a single `[1 × 128] · [128 × 105k]`
matvec — 13.4M multiply-accumulates, under a millisecond.

If the customer vector depended on the candidate — candidate-conditioned attention over purchase
history, say — nothing could be precomputed and retrieval would need 105k forward passes per
request. That is why `CustomerTower.forward` takes no item argument. It would score better with
candidate-awareness; the accuracy is given up on purpose.

### Where personalization enters

The item tower never reads a customer column. Personalization enters two ways:

- **The customer tower** — the last 20 purchased article IDs through a 105k × 64 embedding
  table, pooled by self-attention with a learned query.
- **Which pairs are positives** during training — a positive is two things the *same* customer
  bought, so customer information shapes the learned space without the encoder seeing it.

The previous design hand-built a "taste vector" and `sim_taste_cos` similarity features for the
LightGBM arms. Those are gone, and nothing was lost: they existed because **trees cannot compute
a dot product** between a taste vector and an item vector. A two-tower computes exactly that dot
product, learned end-to-end (`trd.md` §7).

## Invariants

These are the things that quietly invalidate results if broken. Most were chosen against a
specific failure mode; the reasoning is in the referenced sections.

**The `as_of` contract.** `FeatureBuilder.build` takes `as_of` as keyword-only and required.
Feature code reads transactions only through `features.base.history(txns, as_of=...)`. Never add
an overload or default that permits reading without a cutoff — the signature is the enforcement
mechanism, and `register_builder` checks it at import time so a violation fails in the pipeline
and not only under pytest (`trd.md` §5.1).

**Retriever training windows strictly precede every window the retriever retrieves for.** This
is the leakage vector the second stage introduces. A retriever trained on window 5's purchases
then asked for window 5's candidates has memorized the answer and ranks it first; the ranker
learns to trust rank 1 and collapses in production. Asserted by
`test_retriever_windows_precede_candidate_windows` (`prd.md` §5, `trd.md` §3.1).

**The candidate set is byte-identical across every ranker arm.**
`artifacts/candidates/{window}.parquet` is written once and digested in
`candidates_manifest.json`; every ranker asserts that digest before fitting. Never regenerate
candidates per arm — a ΔNDCG of 0.005 is otherwise indistinguishable from one arm having drawn
an easier list (`trd.md` §4.5, §8.2).

**The retriever is frozen for the entire stage-2 experiment.** Comparing rankers across shifting
candidate distributions compares nothing. The retriever checkpoint digest is recorded in the
manifest and a mismatch is an error (`trd.md` §8.2).

**End-to-end metrics count unretrieved positives as misses.** Metrics computed only over
retrieved candidates flatter the pipeline by exactly the retriever's miss rate, and the output
looks entirely healthy while doing so — which is what makes it the most likely silent error in a
two-stage evaluation. **Hand-checkable: end-to-end MAP@12 must be strictly below ranking-only
MAP@12 for every arm.** If they are equal, the accounting is broken (`trd.md` §10.4).

**Both towers must stay independently precomputable.** `CustomerTower.forward` takes no item
argument. Candidate-aware attention would score better but breaks the two-tower factorization,
which voids retrieval itself and the serving-cost analysis with it (`prd.md` §1, `trd.md` §5.3).

**The item tower has no article-ID embedding.** An ID embedding is a learned vector per article,
so a newly added article's vector is untrained noise. Content-only (taxonomy + numerics + text)
is what makes cold-start retrieval possible at all, and it is the mechanism H2 tests. ID towers
are stronger on the head; this is a deliberate trade, not an omission (`prd.md` §6).

**Every arm receives all eleven categorical columns.** Almost every text field in `articles.csv`
has a 1:1 categorical twin — `product_type_name` is both a string and a number. Withholding them
from a baseline while feeding the same information to the encoder as text credits the encoder
with information the baseline was never given. This is the most common way this experiment is run
wrong (`prd.md` §3, `trd.md` §6.3).

**Text-B (`detail_desc` alone) is the number to trust.** It is the only text field with no
categorical twin, so lift there is genuinely new information. Text-A (full concat) runs once as
a sensitivity check to quantify how much apparent text lift is re-encoded taxonomy (`prd.md` §3).

**Hyperparameters are tuned on `lgbm` only, then frozen.** Tuning each arm separately confounds
"better architecture" with "more tuning budget" (`trd.md` §9b.4).

**`max_bin=63` is a memory requirement, not a tuning choice.** It takes the 5M × 47 matrix from
940 MB raw to 235 MB binned, which is what makes LightGBM fit in 8 GB (`trd.md` §14).

**The H3 comparison pair differs in exactly one thing.** Random-negative rows have no retrieval
columns, so the retrieved-negative arm in that pair also has them withheld. Otherwise the delta
bundles "hard negatives" with "three extra features," and H3's claim is about the negative
distribution alone (`trd.md` §4.8, §8.3).

**The test split is read once, at M9.** M3–M8 report validation numbers. **All three**
pre-registered criteria are applied to that single evaluation — H1 (retrieval depth beats ranker
architecture, CI on the difference of deltas excluding zero), H2 (Δ`recall@100` ≥ 0.01 on
cold-start and greater than on all rows, CIs excluding zero), and H3 (ΔMAP@12 ≥ 0.005 for hard
over random negatives, CI excluding zero). H1 and H3 carry **registered priors that they will
hold**, and H2 a prior that it holds only on cold-start. Those priors are part of the record and
must not be quietly dropped if a result goes the other way. Evaluating some hypotheses and
skipping others defeats the point of registering them (`prd.md` §2).

**Bootstrap resamples customers, not rows.** Rows within a customer share history features and
basket composition; row-level resampling gives CIs narrow enough for noise to clear the
significance bar. Differences of deltas use **shared** customer resamples (`trd.md` §10.6).

**Calibration is applied by the evaluator, never inside an arm.** `Arm.predict` returns
uncalibrated scores. Under retrieval-induced sampling there is no fixed downsampling rate, so
isotonic-on-validation is the primary path; `prior_correct` is retained for the H3
random-negative arm, which has a genuine fixed `w` (`trd.md` §10.7).

**Slice thresholds are set once at M1 and registered.** `art_prior_purchases < 10` for cold
start, and the low-history cutoff, come from the measured distributions during EDA. Retuning a
slice boundary after seeing results is p-hacking with extra steps (`trd.md` §10.5).

**No number in `reports/results.md` is hand-typed.** Runs dual-write to MLflow
(`artifacts/mlruns`, local file backend) and to git-committed
`reports/metrics/{run_name}.json`; `make report` regenerates the tables from the JSON
(`trd.md` §11).

## Claims that must not be overstated

The project's credibility rests on these, and they are easy to soften by accident:

- **The label is purchase, not click or add-to-cart.** H&M ships purchase transactions only; no
  impression or funnel events exist in the dataset. Do not describe this as engagement, click, or
  ATC prediction (`prd.md` §4).
- **`price` is scaled and anonymized, not currency.** All revenue and AOV figures are relative
  lift ratios. Never report AOV in dollars (`prd.md` §3).
- **Rows exist only for customers who transacted in the window.** Results are conditional on the
  customer transacting and do not describe the full customer base (`trd.md` §3.3).
- **The stage-2 axis is "the ranker's own features," not "a text-free pipeline."** Text reaches
  every ranker through `retrieval_score`, because the frozen retriever is text-aware. The clean
  pipeline-level number is H2's end-to-end comparison (`trd.md` §7).
- **The retriever is 12 weeks stale by the test window.** Realistic, and it hits every arm
  equally, but it is a limitation to report — not to omit. The `recall@K` curve from W5 to W10 is
  the decay estimate (`prd.md` §5).
- **A null result is pre-committed.** If a hypothesis does not clear its bar, that is the
  headline finding, reported with the same prominence a positive would receive. Do not re-tune,
  re-cut splits, move a threshold, or swap metrics to manufacture a positive (`prd.md` §2).

## Engineering constraints

8 GB RAM, 8 cores, Apple Silicon, MPS only. This drives real design decisions, not just caution:

- **DuckDB for aggregation, Polars for frames.** 31.8M transaction rows do not fit in pandas
  here. `pandas` is not a pipeline dependency — it arrives via MLflow and is for notebooks and
  report generation only.
- **`faiss-cpu` is an optional extra, not a dependency.** It exists only for the §12
  exact-versus-approximate benchmark. At 105k × 128 an exact matvec is expected to win, and
  demonstrating that a standard component is unnecessary at this scale is part of the result.
- **Per-stage memory budgets are specified** in `trd.md` §14 against a ~6.5 GB usable ceiling.
  LightGBM training is still the binding constraint at ~2.0 GB peak, down from 3.5 GB because
  dropping the embedding feature groups took the matrix from 80 columns to 47.
- **Stage 1 is the cheap stage.** It trains on positives only (~260k pairs, not millions of
  rows), because in-batch negatives are free. That asymmetry is why H1's retrieval axis is
  inexpensive to explore and the `K` sweep costs a single retrieval pass.
- **The row budget is the binding scale constraint, not cohort size.** Ranker rows scale as
  `windows × customers × K`. `retrieve` measures actual counts and fails if training rows exceed
  the target by >20%, naming the `train_customer_cap` that would fit. **Only training windows are
  capped** — `val` and `test` retrieve for every eligible customer, so evaluation is never
  conditioned on a budget decision (`trd.md` §3.4).
