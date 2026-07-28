# ContentSignal

**A product recommender built ten ways, to measure what product text is actually worth.**

**The task.** Given a customer and a candidate article, predict whether they buy it in the
next 14 days; rank candidates per customer. Trained on 31.8M H&M transactions across 105k
articles.

**The comparison.** Ten models are trained on the same problem, ranging from a popularity
heuristic to a **two-tower neural ranker trained end-to-end** over customer purchase history
and product text. In between sit LightGBM and logistic-regression baselines, and variants
that add product-text embeddings one layer at a time. Every arm is evaluated on the same
held-out rows, so the differences are readable.

**What it measures.** Three things, each pre-registered before any model runs:

1. Does product text add ranking signal a tuned gradient-boosted model doesn't already have?
2. If so, *where* — on established bestsellers, or on cold-start items with no sales history?
3. What does it cost to serve, in dollars per million predictions?

The reason this is harder than it sounds: most "text" fields in the catalog are the tabular
taxonomy rendered as strings. An experiment that doesn't account for that measures redundancy
and reports it as lift.

> ### Status: specification complete, implementation not started
>
> This repo currently contains **design documents, not code**. The experiment is fully
> specified — schemas, algorithms, resource budgets, test assertions — and implementation
> follows milestones M0–M9. **No results are reported below because none exist yet.**
> The results table is a placeholder and is labelled as one.
>
> What's here is the part that determines whether the eventual numbers mean anything.

📄 **[`prd.md`](./prd.md)** — hypotheses, success criteria, and what's out of scope
📐 **[`trd.md`](./trd.md)** — schemas, module contracts, algorithms, memory budgets, tests

---

## How it works

```
  3 Kaggle CSVs
        │
        ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 1. INGEST     CSV → Parquet, narrowed dtypes.                │
  │               DuckDB streams it; 31.8M rows never in RAM.    │
  └──────────────────────────────────────────────────────────────┘
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  ┌──────────────────────┐   ┌─────────────────────────────────────┐
  │ 2. WINDOWS           │   │ 5. TEXT ENCODER   (offline track)   │
  │ 10 × 14 days,        │   │   article text → contrastive        │
  │ chronological:       │   │   fine-tune on co-purchase pairs    │
  │ 8 train, 1 val,      │   │        ↓                            │
  │ 1 test               │   │   encode 105k articles ONCE         │
  └──────────────────────┘   │   → 162 MB cache (mmap) → SVD-32    │
        │                    └─────────────────────────────────────┘
        ▼                                  │
  ┌──────────────────────┐                 │
  │ 3. ROW SETS          │                 │
  │ positives = bought   │                 │
  │ + 10 sampled negs    │                 │
  │ written once,        │                 │
  │ digest-checked       │                 │
  └──────────────────────┘                 │
        │                                  │
        └────────────────┬─────────────────┘
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 4. FEATURE BUILDERS — six groups, per window, every one      │
  │    gated behind an `as_of` cutoff                            │
  │    customer · article · categorical · cross                  │
  │    text_item · text_customer                                 │
  └──────────────────────────────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 6. TEN ARMS                                                  │
  │    popularity → logreg → LightGBM (baseline)                 │
  │    + text embeddings, frozen / fine-tuned × item / personal  │
  │    + TWO-TOWER, trained end-to-end  ← separate training path │
  └──────────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
  ┌───────────┐  ┌──────────────┐  ┌──────────────┐
  │ 7. EVAL   │  │ 8. TRACKING  │  │ 9. BENCHMARK │
  │ 3 slices, │  │ MLflow +     │  │ latency,     │
  │ bootstrap │  │ committed    │  │ $/1M preds   │
  │ CIs       │  │ JSON         │  │              │
  └───────────┘  └──────────────┘  └──────────────┘
```

**Ingest → windows → row sets.** The timeline is cut into ten contiguous 14-day windows. For
each, the positives are what customers actually bought; ten negatives per positive are
sampled popularity-weighted. That labelled set is written once and digest-checked, so arms
can't silently train on different data.

**Feature builders.** Six independent groups, each computed per window behind a required
`as_of` cutoff. An arm is defined by *which groups it receives* — that's what makes the
ablation controlled.

**The text encoder is an offline track, not part of the model.** It fine-tunes MiniLM on
co-purchase pairs, encodes all 105k articles once into a 162 MB lookup table, and the ranking
models just read columns from it. This is why fine-tuning is cheap (105k unique strings, not
5M rows) and why serving cost is near zero.

**The two-tower is the exception** — a customer tower (attention-pooled purchase history) and
an item tower (text + taxonomy) trained jointly with in-batch sampled softmax. It's the only
arm with a *learned* customer–item interaction, and the only one that trains its own encoder
end-to-end rather than consuming the cache.

**Evaluation** runs every metric on three populations — all rows, cold-start articles, and
low-history customers — with confidence intervals from a customer-level bootstrap.

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
behavioral signal is missing" names a deployment condition. **H1 and H2 are independent** —
H1 can fail while H2 holds, and that combination is the more useful outcome.

**H3 — architecture.** The end-to-end two-tower beats the staged pipeline (contrastive
pre-train → freeze → LightGBM).

```
ΔAUC ≥ 0.005   AND   95% bootstrap CI excludes zero        (arm 10 − arm 7b)
```

> **Registered prior: I expect H3 to fail on aggregate.** Gradient-boosted trees usually beat
> neural rankers on tabular-dominant problems at this scale. Writing that down beforehand is
> what makes either outcome informative — and the interesting case is H3 failing on all rows
> while holding on cold-start, where the text pathway matters most.

Anything less, on any of the three, is a null result — and **a null gets published as the headline
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
| **Two-tower objective** | BCE on the shared 1:10 rows / in-batch sampled softmax | **sampled softmax** | The standard retrieval formulation, and it trains on positives only (~520k, not 5.7M rows) so it is *cheaper* than the GBDT arms. Cost: a different negative distribution, so the 10 − 7b delta covers architecture *and* objective — stated, not hidden |
| **In-batch negative bias** | raw softmax / log-Q correction | **log-Q** | In-batch negatives are drawn ∝ popularity, rewarding the model for demoting popular items. `logits −= log P(sample item)` corrects it (Yi et al., RecSys 2019). Run once without it to demonstrate the effect rather than assert it |
| **History attention** | candidate-aware / self-attentive with learned query | **self-attentive** | Candidate-aware attention scores better but makes the customer vector depend on the item, so neither tower is precomputable — it breaks retrieval and voids the serving-cost analysis. Accuracy traded for deployability, deliberately |

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
set is written once, digested, and **reused byte-identically by arms 1–8** — without that,
a ΔAUC of 0.005 is indistinguishable from a different random draw.

The two-tower arms are the bounded exception: they train on in-batch softmax negatives, then
are **evaluated on the identical test rows**, so ranking metrics stay comparable while
probability metrics require isotonic recalibration on validation.

**Text variants** — run separately, because they measure different things:

| | Content | Reads as |
|---|---|---|
| **Text-A** | full concat: name + type + group + colour + department + `detail_desc` | Optimistic bound. Overlaps the taxonomy the baseline already has, so part of any lift is re-encoded categoricals |
| **Text-B** | `detail_desc` only | The honest measure. The one field with no categorical twin, so lift here is genuinely novel information |

Reporting both quantifies how much apparent "text lift" is taxonomy in disguise.

**Arms** — baselines, then a 2×2 over the text arms, then the neural rankers:

| # | Arm | Embeddings | Text form |
|---|---|---|---|
| 1 | Popularity ranker | — | — |
| 2 | Logistic regression | — | — |
| 3 | **LightGBM (baseline)** | — | — |
| 4a | LightGBM + frozen | pretrained | item-side |
| 4b | LightGBM + frozen | pretrained | item + personalized |
| 7a | LightGBM + fine-tuned | contrastive | item-side |
| 7b | **LightGBM + fine-tuned** | contrastive | item + personalized |
| 8 | Text-only encoder | contrastive | — |
| 9 | Two-tower | frozen | learned towers |
| 10 | **Two-tower, end-to-end** | trained jointly | learned towers |

Arms 4a/4b/7a/7b form a clean 2×2 — *frozen vs fine-tuned* crossed with *item-side vs
personalized* — which is what makes any win interpretable. With only "baseline vs best" you
cannot tell whether fine-tuning or personalization produced it.

| Delta | Isolates |
|---|---|
| 7b − 3 | **H1.** Total lift from product content |
| 7b − 7a | Personalization — the taste vector and similarity features |
| 7b − 4b | Fine-tuning — if ≈ 0, an off-the-shelf encoder sufficed |
| 10 − 7b | **H3.** End-to-end learned interaction vs the staged pipeline |
| 10 − 9 | What end-to-end encoder training adds inside the tower architecture |
| 9 − 3 | Whether a neural ranker beats GBDT here at all |

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
| Two-tower, end-to-end | — | — | — | — |
| **H1 · Δ (95% CI)** | — | — | — | — |
| **H3 · Δ (95% CI)** | — | — | — | — |

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
- **The two-tower delta is not a clean isolation.** Arms 9 and 10 differ from the staged
  pipeline in both architecture *and* training objective (sampled softmax vs the shared 1:10
  rows). The H3 number covers both; it cannot be split into "neural helped" and "objective
  helped" within this design.
- **No candidate-aware interaction.** Both towers must stay independently precomputable or
  the serving-cost analysis stops describing a deployable system, so cross-attention and
  DLRM-style feature crossing are out of scope.

---

## Running it

The CLI is specified in [`trd.md` §13](./trd.md) but **not yet implemented**:

```bash
uv sync                                    # Python 3.11
contentsignal ingest                       # Kaggle CSVs → parquet
contentsignal sample                       # temporal windows + negative sampling
contentsignal build-features --group ...
contentsignal finetune --variant b            # contrastive encoder fine-tune
contentsignal train --arm lgbm_ft_pers --variant b --seed 1
contentsignal train-twotower --arm 10 --variant b --seed 1
contentsignal report                       # regenerates the results table
pytest tests/test_leakage.py               # the gate on everything downstream
```

Requires accepting the H&M competition rules on Kaggle and placing `kaggle.json`.
