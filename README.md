# ContentSignal

**Does a language model reading product descriptions know anything a well-built tabular
model doesn't?**

Most e-commerce ranking systems run on tabular signals — customer history, item popularity,
price, recency, taxonomy. Product *content* gets reduced to categorical IDs or dropped
entirely. This project tests whether that's leaving signal on the table, on 31.8M real
transactions, and prices the answer in dollars per million predictions.

> ### Status: specification complete, implementation not started
>
> This repo currently contains **design documents, not code**. The experiment is fully
> specified — schemas, algorithms, resource budgets, test assertions — and implementation
> follows milestones M0–M9. **No results are reported below because none exist yet.**
> The results table is a placeholder and is labelled as one.
>
> What's here is the part that determines whether the eventual numbers mean anything.

📄 **[`prd.md`](./prd.md)** — the hypothesis, success criterion, and what's out of scope
📐 **[`trd.md`](./trd.md)** — schemas, module contracts, algorithms, memory budgets, tests

---

## The hypothesis, pre-registered

> Adding fine-tuned product-text embeddings to a tabular purchase-propensity model improves
> ranking quality over a tuned LightGBM baseline, on an identical temporal test split.

**Success is defined before any model runs**, so the result can't be rationalized afterward:

```
ΔAUC ≥ 0.005   AND   95% bootstrap CI on ΔAUC excludes zero
```

Anything less is a null result — and **a null result gets published as the headline
finding**, with the same prominence a positive would get. No re-tuning, no re-cutting
splits, no swapping metrics until something clears the bar. The test split is read exactly
once, at the end.

"Semantic content added no lift over well-built tabular features, at this cost" is a useful
answer. A repo that can only produce good news isn't measuring anything.

---

## Three ways this experiment gets rigged — and how each is closed

This is the interesting part. Each of these is a way to produce a headline number that
looks great and means nothing.

### 1. Give the encoder information you withheld from the baseline

Nearly every "text" field in the H&M catalog has a **1:1 categorical twin**.
`product_type_name` is both a string the encoder can read and a categorical column
LightGBM can split on. Feed the encoder all of it, hand the baseline a subset, and you've
manufactured lift out of an unfair comparison.

**Closed by:** the baseline receives all eleven categorical columns as native LightGBM
categoricals, and the text arm splits into two variants — **Text-A** (full concat,
optimistic bound) and **Text-B** (`detail_desc` only, the sole field with no categorical
twin). Text-B is the honest measure. Reporting both quantifies how much apparent "text
lift" is really taxonomy the trees already had.

### 2. Build a text feature that structurally cannot help

The obvious design joins article embeddings into the feature table as 32 columns. But those
columns hold **the same value for every customer** — and a feature constant across customers
*cannot reorder items differently per customer*. For `precision@k` and `MAP@12`, which are
per-customer rankings, item-side text contributes almost nothing beyond what popularity
already contributes.

That design produces a null result regardless of whether product semantics carry signal.
The finding would be an artifact of the architecture, not a fact about text.

**Closed by** putting the customer on both sides:

- a **taste vector** — the mean embedding of what the customer bought before the cutoff —
  plus `cosine(taste, candidate)` and similarity to their 10 most recent purchases
- **contrastive co-purchase fine-tuning** — positive pairs are two articles bought by the
  *same* customer, so the encoder learns a space where *products that appeal to the same
  person sit close together*

The encoder never reads a customer column. Customer information enters through *which pairs
are positives*.

### 3. Let the future leak backward

Item popularity computed over the full dataset is one line of code, works extremely well,
and contains the future.

**Closed by** making the leak unwritable rather than merely discouraged. Every feature
builder takes `as_of` as a **required keyword-only argument** and reads transactions only
through a single gated helper. The property test is general:

```python
# recompute the feature on data truncated at the window start.
# if the value changes, it leaks.
assert_frame_equal(
    builder.build(txns, as_of=cutoff, entities=E),
    builder.build(txns.filter(pl.col("t_dat") < cutoff), as_of=cutoff, entities=E),
)
```

**No model is trained until the leakage tests pass.**

---

## Method

**Data** — [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations):
31.8M transactions, 105k articles, 1.37M customers, Sept 2018 → Sept 2020.

**Splits** — ten contiguous 14-day windows: 8 train, 1 validation, 1 test. Strictly
chronological, never shuffled. Features for a window read only data strictly before it.

**Negatives** — implicit feedback has no true negatives, so they're sampled at a fixed 1:10,
popularity-weighted (`pop^0.75`; uniform negatives are trivially separable and inflate AUC).
The sampled row set is written once and **reused byte-identically by every arm** — otherwise
a ΔAUC of 0.005 is indistinguishable from a different random draw.

**Calibration** — downsampling negatives shifts the base rate, so log-loss and the revenue
proxy are prior-corrected back to the true rate. AUC needs no correction (it's rank-based
and sampling-invariant); log-loss does. That asymmetry is stated explicitly so it doesn't
read as an oversight.

**Arms** — two orthogonal axes, so each delta isolates one thing:

| Arm | Embeddings | Text form |
|---|---|---|
| Popularity ranker | — | — |
| Logistic regression | — | — |
| **LightGBM (baseline)** | — | — |
| LightGBM + frozen | pretrained | item-side |
| LightGBM + frozen | pretrained | item + personalized |
| LightGBM + fine-tuned | contrastive | item-side |
| **LightGBM + fine-tuned** | contrastive | item + personalized |

| Delta | Isolates |
|---|---|
| **headline** | total lift from product content |
| personalized − item-side | how much requires the customer-side features |
| fine-tuned − frozen | whether fine-tuning beat an off-the-shelf encoder |

Hyperparameters are tuned on the baseline **once, then frozen** across every arm — otherwise
"better features" is confounded with "more tuning budget."

**Uncertainty** — 1000-resample bootstrap that **resamples customers, not rows**. Rows within
a customer share history features and basket composition; row-level resampling gives
confidence intervals narrow enough for noise to clear the significance bar.

**Slices** — every metric on three populations: all rows, **cold-start articles**, and
low-history customers. Cold start is where text should win if it wins anywhere: a new
article has no popularity signal but has a description from day one.

---

## Results

**Not yet run.** This table is a placeholder; `make report` will generate it from
committed per-run JSON so no number here is ever hand-typed.

| Arm | AUC | Log-loss | precision@10 | MAP@12 |
|---|---|---|---|---|
| Popularity | — | — | — | — |
| LightGBM (tabular) | — | — | — | — |
| + fine-tuned text | — | — | — | — |
| **Δ (95% CI)** | — | — | — | — |

---

## Cost

An accuracy result nobody can afford to serve isn't a result. Per 1K predictions, measured:
LightGBM alone, LightGBM + cached embeddings, and the encoder cold path on MPS, CPU, and
ONNX-int8 — converted to $/1M predictions against a named cloud SKU.

The claim under test: **105k × 384 × 4 B ≈ 162 MB** (~40 MB int8). The entire catalog fits
in memory, so the encoder is an offline batch job feeding a lookup table, not an online
forward pass — steady-state cost near zero, with the cold path mattering only for new
articles. The benchmark prints measured RSS alongside the latency table so this is evidence,
not arithmetic.

---

## Engineering constraints

Everything runs on a **laptop: 8 GB RAM, 8 cores, Apple Silicon, no discrete GPU.** That
drives real design decisions:

- **DuckDB for aggregation, Polars for frames.** 31.8M rows don't fit in pandas here.
- **`max_bin=63` is a memory requirement, not a tuning choice** — it takes the training
  matrix from 1.6 GB raw to 400 MB binned, which is what makes LightGBM fit.
- **Fine-tuning batches deduplicated article *pairs*, not rows.** 105k unique texts against
  ~5M rows means naive batching re-encodes every string ~38× per epoch. The contrastive
  formulation makes each example a pair of texts: ~6–10 minutes per epoch instead of hours.
- **A row budget, not a cohort size, is the binding constraint.** The sampler measures actual
  counts and fails loudly with the cohort size that would fit — discovering the memory
  ceiling midway through a 48-run ablation grid is expensive.

Per-stage memory budgets are specified in [`trd.md` §14](./trd.md) against a 6.5 GB usable
ceiling, before any code was written.

---

## Limitations

Stated because they bound what the results can claim:

- **The label is purchase, not click or add-to-cart.** H&M ships purchase transactions only —
  no impression logs, no funnel steps. Calling this engagement prediction would describe an
  experiment this data can't support.
- **`price` is scaled and anonymized, not currency.** Every revenue figure is a relative lift
  ratio. There is no honest way to report AOV in dollars here.
- **Results are conditional on the customer transacting.** Rows exist only for customers with
  a purchase in the window — necessary for per-customer ranking metrics to be defined, but a
  real selection effect.
- **No learned customer tower.** The taste vector is a fixed mean-pool of item embeddings. A
  true two-tower model would likely do better and would make the encoder's marginal
  contribution much harder to isolate — which is the entire question here.

---

## Running it

The CLI is specified in [`trd.md` §13](./trd.md) but **not yet implemented**:

```bash
uv sync                                    # Python 3.11
contentsignal ingest                       # Kaggle CSVs → parquet
contentsignal sample                       # temporal windows + negative sampling
contentsignal build-features --group ...
contentsignal finetune --variant b
contentsignal train --arm lgbm_ft_pers --variant b --seed 1
contentsignal report                       # regenerates the results table
pytest tests/test_leakage.py               # the gate on everything downstream
```

Requires accepting the H&M competition rules on Kaggle and placing `kaggle.json`.
