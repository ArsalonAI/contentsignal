# ContentSignal

**Problem.** Score (customer, article) pairs from implicit feedback — 31.8M H&M purchase
events, 105k items, positive-only, ranked per customer under a strict temporal split. Base
rate is on the order of 10⁻⁵ against the full catalog before negative sampling.

**The tension.** Every item carries two representations whose overlap is unknown:

| | |
|---|---|
| **Tabular** | windowed popularity (1/4/12w), price percentile, shelf age, an 11-column taxonomy — all consumed natively by LightGBM |
| **Text** | free-text `detail_desc` — the only field in the catalog with no categorical twin |

Gradient-boosted trees dominate this problem class. A fine-tuned encoder earns its place
only if it reaches signal the trees cannot — and most "text" fields in this catalog *are*
the taxonomy columns rendered as strings, so the naive setup measures redundancy and reports
it as lift.

**This repo is the experiment that separates the two, and the tradeoff reasoning behind
every decision that determines whether the answer is trustworthy.**

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

## Pre-registered hypotheses

Both fixed before any model runs, so neither result can be rationalized afterward.

**H1 — magnitude.** Fine-tuned text embeddings improve ranking quality over a tuned
LightGBM baseline on an identical temporal test split.

```
ΔAUC ≥ 0.005   AND   95% bootstrap CI on ΔAUC excludes zero
```

**H2 — direction.** The lift concentrates in cold-start articles, where popularity features
are near zero and the description is complete from day one.

```
ΔAUC(cold-start) > ΔAUC(all)   AND   95% CI on the difference of deltas excludes zero
```

H2 is the claim worth making. "Text helps" is diffuse; "text helps precisely where
behavioral signal is missing" names a deployment condition. **The two are independent** —
H1 can fail while H2 holds, and that combination is the more useful outcome.

Anything less, on either, is a null result — and **a null gets published as the headline
finding**, with the same prominence a positive would get. No re-tuning, no re-cutting
splits, no swapping metrics until something clears the bar. The test split is read exactly
once, at the end.

A repo that can only produce good news isn't measuring anything.

---

## Design decisions

Every row is a place where two defensible options existed. Reasons are mechanisms or
numbers, never preferences. Full derivations in [`trd.md`](./trd.md).

| Decision | Considered | Chosen | Reason |
|---|---|---|---|
| **Text → GBDT dimensionality** | raw 384 / SVD-32 / SVD-64 | **SVD-32** | 384 dense low-variance dims degrade axis-aligned splits and cost 1.6 GB vs 400 MB binned at 5M rows. 64 and 384 run as sensitivity, not as the grid |
| **Fine-tune objective** | pointwise item-rate / contrastive co-purchase | **contrastive** | An item-rate target ≈ article popularity, which the baseline already holds *measured exactly*. The encoder would learn a lossy copy of an existing feature, and the resulting null would be an artifact of the objective rather than a fact about text |
| **Personalization** | item-side embeddings only / taste-vector cosine / learned two-tower | **taste-vector cosine** | Item-side dims are constant per customer and cannot reorder a per-customer ranking. A two-tower would likely win outright, but confounds the encoder's marginal contribution — which is the quantity being measured |
| **Negative sampling ratio** | 1:1 / 1:10 / 1:50 | **1:10, fixed across arms** | Trades compute against calibration. AUC is sampling-invariant; log-loss is not, so log-loss is prior-corrected and reported both ways |
| **Baseline feature set** | text fields to the encoder only / all 11 categoricals to the baseline | **all 11 to the baseline** | Otherwise the encoder is credited with information the baseline was denied. This is the difference between measuring lift and manufacturing it |
| **Hyperparameter budget** | tune each arm / tune baseline once, then freeze | **freeze** | Per-arm tuning confounds "better features" with "more tuning budget" |
| **Bootstrap resampling unit** | rows / customers | **customers** | Rows within a customer share history features and basket composition. Row-level CIs are narrow enough for noise to clear the ΔAUC ≥ 0.005 bar |
| **Data engine** | pandas / Polars + DuckDB | **Polars + DuckDB** | 31.8M rows do not fit in pandas at 8 GB; aggregation runs out-of-core |
| **`max_bin`** | 255 (default) / 63 | **63** | 400 MB binned vs 1.6 GB raw at 5M × 80. A feasibility requirement, not a tuning choice |
| **Serving path** | online forward pass / offline embedding cache | **offline cache** | 105k × 384 × 4 B ≈ 162 MB. The catalog fits in memory, so the transformer never runs at request time except for new items |

---

## Two ways this experiment gets rigged — and how each is closed

The rows above are judgment calls. These two are validity failures: setups that produce a
headline number which looks great and means nothing.

### 1. Build a text feature that structurally cannot help

The obvious design joins article embeddings into the feature table as 32 columns. Those
columns hold **the same value for every customer** — and a feature constant across customers
*cannot reorder items differently per customer*. For `precision@k` and `MAP@12`, which are
per-customer rankings, item-side text contributes almost nothing beyond what popularity
already contributes.

That design yields a null result regardless of whether product semantics carry signal. The
finding would be a fact about the architecture, not about text — and it would be reported as
the latter.

**Closed by** putting the customer on both sides:

- a **taste vector** — the mean embedding of what the customer bought before the cutoff —
  plus `cosine(taste, candidate)` and similarity to their 10 most recent purchases
- **contrastive co-purchase fine-tuning** — positive pairs are two articles bought by the
  *same* customer, so the encoder learns a space where *products that appeal to the same
  person sit close together*

The encoder never reads a customer column. Customer information enters through *which pairs
are positives*. Pair sampling is capped at 5 per customer and 50 per article — uncapped, it
is dominated by heavy buyers and bestsellers, which pushes the encoder straight back toward
the popularity proxy the contrastive objective exists to avoid.

### 2. Let the future leak backward

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

**Negatives** — implicit feedback has no true negatives. Sampled popularity-weighted at
`pop^0.75`, since uniform negatives are trivially separable and inflate AUC. The sampled row
set is written once, digested, and **reused byte-identically by every arm** — without that,
a ΔAUC of 0.005 is indistinguishable from a different random draw.

**Text variants** — run separately, because they measure different things:

| | Content | Reads as |
|---|---|---|
| **Text-A** | full concat: name + type + group + colour + department + `detail_desc` | Optimistic bound. Overlaps the taxonomy the baseline already has, so part of any lift is re-encoded categoricals |
| **Text-B** | `detail_desc` only | The honest measure. The one field with no categorical twin, so lift here is genuinely novel information |

Reporting both quantifies how much apparent "text lift" is taxonomy in disguise.

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
