# ContentSignal

**A two-stage recommender built to measure where engineering effort actually pays: retrieval or ranking.**

**The task.** For a customer, select 12 articles from a catalog of 105,000 that they will
purchase in the next 14 days. Built on 31.8M H&M transactions across 1.37M customers.

**The system.** The production architecture, not a single model. **Stage 1** is a two-tower
retriever that scores all 105k articles in under a millisecond and returns ~100 candidates.
**Stage 2** is a ranker — LightGBM, an MLP, and a DCN-v2 — that orders those candidates using
features too expensive to compute catalog-wide.

**What it measures.** Three things about how the two stages interact, each pre-registered with a
stated prior before any model runs:

1. Is end-to-end quality more sensitive to **retrieval depth** than to **ranker architecture**?
2. Does product **text** surface cold-start articles that a behavioral retriever structurally
   cannot?
3. Does **which negatives you train on** matter more than which model you use?

The reason these are worth asking: the two stages are usually built and evaluated as independent
boxes, so the coupling between them — *the ranker can only reorder what retrieval surfaced* —
goes unmeasured. That coupling is what decides whether the next quarter should go to retrieval or
to ranking, and it is answerable with numbers rather than intuition.

> ### Status: specification complete; the leakage harness is built, no models trained
>
> The experiment is fully specified — schemas, algorithms, resource budgets, test assertions —
> and implementation follows milestones M0–M9.
>
> **Built:** the Python 3.11 environment, the CLI surface, and the invariant spine — temporal
> windows, the `as_of` cutoff contract, negative sampling, and calibration, with the
> leakage/splits/sampling/calibration tests green against synthetic fixtures. That harness
> exists before the data it guards, because M1 gates everything: no model is trained until it
> passes.
>
> **Not built:** ingest, feature builders, the retriever, candidate generation, and all three
> rankers. **No results are reported below because none exist yet.** The results tables are
> placeholders and are labelled as such.
>
> What's here is the part that determines whether the eventual numbers mean anything.

📄 **[`prd.md`](./prd.md)** — hypotheses, success criteria, and what's out of scope
📐 **[`trd.md`](./trd.md)** — schemas, module contracts, algorithms, memory budgets, tests

---

## Why two stages

There are 105,000 articles and a slate holds 12. A useful ranking model costs roughly a
millisecond per (customer, article) pair, so scoring the whole catalog for one customer is ~100
seconds. That is not a latency budget anyone has, so the work splits:

| | **Stage 1 — retrieval** | **Stage 2 — ranking** |
|---|---|---|
| Sees | all 105,000 articles | the ~100 stage 1 returned |
| Cost per request | must be ~1 ms **total** | can be ~1 ms **per candidate** |
| Job | *don't miss anything good* | *put the best ones on top* |
| Failure mode | a good item never enters consideration | good items are ordered badly |
| Metric | `recall@K` | NDCG@12, MAP@12 |

**`recall@K`** is the fraction of a customer's actual purchases that appear in stage 1's top-`K`
list. It is the **ceiling on the entire pipeline**: at `recall@100 = 0.60`, 40% of real purchases
are invisible to stage 2 permanently, and no ranker however sophisticated recovers them. Locating
that ceiling is the point of the project.

### The two-tower model, and why it is cheap enough for stage 1

One tower turns a customer into 128 numbers, another turns an article into 128 numbers, and the
score for a pair is their dot product. The load-bearing property is that **the article vector does
not depend on the customer** — so all 105k are computed once, offline, into a 54 MB table
(`105,000 × 128 × 4 bytes`). At request time you compute one customer vector and multiply it
against the whole table:

```
[1 × 128] · [128 × 105,000]  =  13.4M multiply-accumulate ops  ≈  under 1 ms
```

If the customer vector depended on which article was being scored — candidate-conditioned
attention over purchase history, for instance — nothing could be precomputed and retrieval would
need 105,000 forward passes per request. **`CustomerTower.forward` therefore takes no item
argument**, and a test asserts it. Candidate-awareness would score better; the accuracy is traded
for deployability on purpose.

### And why an ANN index is not needed here

That matvec is fast enough at 105k that an approximate-nearest-neighbour index (FAISS, HNSW,
ScaNN) adds a component without buying latency. ANN earns its complexity somewhere around 10⁷–10⁹
items. The benchmark runs exact against FAISS anyway and reports both — **demonstrating that a
standard component is unnecessary at this scale is a result**, and reaching for a vector database
regardless of catalog size is a common enough reflex to be worth measuring against.

---

## How it works

```
  3 Kaggle CSVs
        │
        ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 1. INGEST     CSV → Parquet, narrowed dtypes.                        │
  │               DuckDB streams it; 31.8M rows never in RAM.            │
  └──────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 2. WINDOWS — ten 14-day windows, and they have ROLES                 │
  │                                                                      │
  │   W1 W2 W3 W4  │  W5 W6 W7 W8  │   W9   │  W10                       │
  │   └─ retriever ┘  └── ranker ──┘   val     test  ← read once         │
  │       training        training                                       │
  └──────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 3. STAGE 1 — TWO-TOWER RETRIEVER   (trains on W1–W4 positives only)  │
  │                                                                      │
  │   customer tower                     item tower                      │
  │   ├ last 20 article IDs              ├ 11 taxonomy categoricals      │
  │   │   → attention pooling            ├ article numerics              │
  │   ├ customer categoricals            └ detail_desc → MiniLM  ← H2    │
  │   └ customer numerics                                                │
  │          └──→ 128-d ──── dot product ──── 128-d ←──┘                 │
  │                                                                      │
  │   in-batch sampled softmax + log-Q · then FROZEN                     │
  └──────────────────────────────────────────────────────────────────────┘
        │
        ├──→ encode all 105k articles ONCE → 54 MB mmap'd cache
        │
        ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 4. CANDIDATES — top-K per customer for W5–W10, over the full catalog │
  │    written once, digested. Non-purchases become HARD negatives.      │
  └──────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 5. FEATURES — five groups, ~47 columns, every one `as_of`-gated      │
  │    customer · article · categorical · cross · retrieval             │
  └──────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 6. STAGE 2 — THREE RANKERS on identical candidates                   │
  │    LightGBM (baseline) · MLP · DCN-v2                                │
  └──────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────────────┐
  │ 7. EVAL       │  │ 8. TRACKING  │  │ 9. BENCHMARK                 │
  │ retrieval /   │  │ MLflow +     │  │ per-stage latency, exact vs   │
  │ ranking / e2e │  │ committed    │  │ FAISS, $/1M predictions       │
  │ 3 slices, CIs │  │ JSON         │  │                               │
  └───────────────┘  └──────────────┘  └──────────────────────────────┘
```

**Windows have roles because the retriever is itself a trained model.** If it trains on window
5's purchases and is then asked for window 5's candidates, it has memorized the answer and puts
it at rank 1 — for a reason that never holds at serving time, when the retriever has never seen
tomorrow. The ranker would learn to trust rank 1 and collapse in production. So the retriever
trains on W1–W4 and retrieves for W5–W10, and a test enforces the ordering.

**Stage 1 is the cheap stage.** It trains on positives only — ~260k pairs — because in-batch
negatives are free: for each real purchase in a batch of 512, the other 511 items serve as its
wrong answers. That asymmetry is why the retrieval half of question 1 is inexpensive to explore
and the whole `recall@K` sweep costs a single retrieval pass.

**The candidate set is written once and digested.** Every ranker asserts that hash before
fitting. If ranker A were scored on a different candidate list than ranker B, a ΔNDCG of 0.005
would be indistinguishable from A having drawn an easier list.

---

## Pre-registered questions

All three fixed before any model runs, judged on the test split, which is read **once**. All
confidence intervals come from 1000 bootstrap resamples over **customers**, not rows.

### H1 — Stage attribution *(the headline)*

> End-to-end quality is more sensitive to retrieval depth than to ranker architecture.

**Supported** if ΔMAP@12 from `K` = 100 → 500 exceeds ΔMAP@12 from the worst to the best ranker at
fixed `K`, with the 95% CI on the difference of deltas excluding zero.
**Registered prior: supported.**

The deliverable is a stage-attribution table putting every intervention on one axis — Δquality,
Δp95 latency, Δ$/1M — which is what makes retrieval and ranking commensurable at all.

### H2 — Content-based retrieval and cold start

> Adding the free-text description to the item tower improves recall more on cold-start articles
> than on articles overall.

**Supported** if Δ`recall@100` on cold-start ≥ 0.01 with CI excluding zero, **and** it exceeds
Δ`recall@100` on all articles with the CI on that difference excluding zero.
**Registered prior: supported on cold-start, marginal or null on aggregate.**

The mechanism is concrete. An article added today has zero popularity, zero purchase history, and
a complete description:

| Signal | Available on day one? |
|---|---|
| Popularity, purchase counts, distinct buyers | **no** |
| Co-purchase history | **no** |
| Taxonomy | yes |
| `detail_desc` | **yes, in full** |

A behavioral retriever scores it ≈0 → never in the top 100 → **the ranker never sees it** → it can
never be recommended, no matter how good the ranker is. A content retriever can place it near
similar items. This is a chicken-and-egg loop — a new product can't accumulate purchases because
it isn't shown, and isn't shown because it has no purchases — and content-based retrieval is how
it breaks.

### H3 — Candidate distribution shift

> A ranker trained on random negatives underperforms end-to-end against the identical ranker
> trained on the retriever's hard negatives.

**Supported** if ΔMAP@12 ≥ 0.005 end-to-end with CI excluding zero.
**Registered prior: supported, and larger than the ranker-architecture delta in H1** — i.e. *what
you train on matters more than which model you use.*

At serving time a ranker only ever sees stage 1's output. Training it on random popular articles
(easy to reject) and serving it plausible retrieved ones (hard) is a documented failure mode —
sample selection bias in candidate generation — that produces models which test well and
disappoint in production.

### A null result is pre-committed

If a hypothesis misses its bar, that is the headline finding, reported with the same prominence a
positive would receive. No arm is re-tuned, no split re-cut, no threshold moved, no metric
swapped. The three are independent: H1 can hold while H2 fails, and each combination is a
different useful finding.

---

## Design decisions

Every row is a place where two defensible options existed. Reasons are mechanisms or numbers,
never preferences. Full derivations in [`trd.md`](./trd.md).

| Decision | Considered | Chosen | Reason |
|---|---|---|---|
| **Architecture** | single-stage scoring of a sampled candidate set / two-stage retrieval + ranking | **two-stage** | Single-stage cannot ask where the pipeline's ceiling is, because it never generates candidates. The coupling between stages is the quantity worth measuring |
| **History attention** | candidate-aware / self-attentive with learned query | **self-attentive** | Candidate-aware scores better but makes the customer vector depend on the item, so neither tower is precomputable — it breaks retrieval outright and voids the cost analysis. Accuracy traded for deployability, deliberately |
| **Item tower inputs** | + learned 105k × 64 article-ID embedding / content-only | **content-only** | An ID vector for a newly added article is untrained noise. ID towers are stronger on the head; content towers are the only ones that function on new items — which is the exact mechanism H2 tests |
| **Retrieval search** | FAISS IVF index / exact matvec | **exact**, with FAISS benchmarked | 105k × 128 is 13.4M MACs, under a millisecond. An index at this scale is infrastructure without a latency payoff. Measured rather than assumed |
| **Stage-1 objective** | BCE on materialized negatives / in-batch sampled softmax | **sampled softmax** | The standard retrieval formulation, and it trains on positives only (~260k pairs) because in-batch negatives are free — making stage 1 cheaper than any ranker |
| **In-batch negative bias** | raw softmax / log-Q correction | **log-Q** | In-batch negatives arrive ∝ popularity, rewarding the model for demoting popular items until it inverts popularity. `logits −= log P(sample item)` corrects it (Yi et al., RecSys 2019). Run once without it to demonstrate the effect rather than assert it |
| **Retriever/ranker temporal split** | one retriever on all 8 train windows / per-window time-sliced retrievers / 4 + 4 role split | **4 + 4** | Training on windows you retrieve for memorizes the labels. Time-slicing is faithful but costs 8× stage-1 training and 8 candidate artifact sets on an 8 GB machine. Cost of 4+4: the retriever is 12 weeks stale at test — realistic, equal across arms, and reported |
| **Stage-2 candidate set** | regenerate per arm / write once and digest | **write once** | A ΔNDCG of 0.005 must not be attributable to one arm receiving an easier list |
| **Stage-2 negatives** | random popularity-weighted / retriever's top-K non-purchases | **retriever's**, with random as the H3 arm | Train/serve consistency: the ranker only ever sees stage 1's output. The gap between the two is H3, measured rather than asserted |
| **Ranker architectures** | MLP only / MLP + DCN-v2 / + LightGBM baseline | **all three** | Claiming a neural ranker on a tabular-dominant problem without checking gradient-boosted trees credits the architecture for a result nobody verified. `dcn` vs `mlp` isolates learned crossing against the six hand-written crosses |
| **Customer–item similarity features** | hand-built taste vector + `sim_taste_cos` / none | **none** | Those exist because *trees cannot compute a dot product* between a taste vector and an item vector. A two-tower computes exactly that, learned end-to-end. Keeping both would duplicate stage 1's job less well |
| **Baseline feature set** | text fields to the encoder only / all 11 categoricals to every arm | **all 11 to every arm** | Otherwise the encoder is credited with information the baseline was denied. This is the difference between measuring lift and manufacturing it |
| **Hyperparameter budget** | tune each arm / tune `lgbm` once, then freeze | **freeze** | Per-arm tuning confounds "better architecture" with "more tuning budget," and the delta then measures effort |
| **Bootstrap resampling unit** | rows / customers | **customers** | Rows within a customer share history features and basket composition. Row-level CIs are narrow enough for noise to clear any significance bar |
| **Calibration** | parametric prior correction / isotonic on validation | **isotonic**, prior correction for the H3 arm | Retrieval-induced sampling has no fixed downsampling rate `w` — how many negatives a customer gets depends on what was retrieved. The H3 arm has a genuine `w`, so both paths run there as a cross-check |
| **Data engine** | pandas / Polars + DuckDB | **Polars + DuckDB** | 31.8M rows do not fit in pandas at 8 GB; aggregation runs out-of-core |
| **`max_bin`** | 255 (default) / 63 | **63** | 235 MB binned vs 940 MB raw at 5M × 47. A feasibility requirement, not a tuning choice |
| **Serving path** | online forward pass / offline vector cache | **offline cache** | 105k × 128 × 4 B ≈ 54 MB. The catalog fits in memory, so the transformer never runs at request time except for genuinely new articles |
| **Row budget** | fix the cohort size / fix a row target and measure | **measure** | Ranker rows scale as `windows × customers × K`; the estimate has real uncertainty. `retrieve` fails loudly naming the cap that fits, rather than OOMing mid-grid |

---

## Three ways this experiment gets rigged — and how each is closed

The rows above are judgment calls. These three are validity failures: setups that produce a
headline number which looks great and means nothing.

### 1. Let the ranker take credit for what retrieval never surfaced

Compute MAP@12 over the retrieved candidates and it looks excellent — because every positive
stage 1 missed has silently left the denominator. The metric is inflated by exactly the
retriever's miss rate, and **nothing in the output looks wrong.** This is the most likely silent
error in any two-stage evaluation.

**Closed by** computing end-to-end metrics over *every* positive in the window, retrieved or not.
Unretrieved positives count as misses. The consequence is hand-checkable:

> **End-to-end MAP@12 must be strictly below ranking-only MAP@12 for every arm.**
> If they are equal, the accounting is broken and every headline number is inflated.

### 2. Let the retriever see the window it retrieves for

Train the retriever on all eight train windows — the obvious default — and it will have memorized
that this customer bought this article in window 5, so it ranks it first. The ranker then trains
on candidate lists where the answer is reliably at the top, for a reason that will never hold in
production.

**Closed by** giving windows roles: the retriever trains on W1–W4 only and retrieves for W5–W10.
Asserted directly:

```python
# every retriever training window must end before the first window it retrieves for
assert max(w.end for w in windows_for_role("retriever")) < min(
    w.start for w in candidate_windows()
)
```

### 3. Let the future leak backward into a feature

Item popularity computed over the full dataset is one line of code, works extremely well, and
contains the future.

**Closed by** making the leak unwritable rather than merely discouraged. Every feature builder
takes `as_of` as a **required keyword-only argument**, verified at import time, and reads
transactions only through a single gated helper. The property test is general and applies
automatically to every registered builder:

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
31.8M transactions, 105k articles, 1.37M customers, Sept 2018 → Sept 2020. The ~30 GB image set
is not used.

**Label** — purchase, not click or add-to-cart. The dataset contains purchase transactions only;
no impression logs or funnel events exist anywhere in the release, so any other framing would
describe an experiment this data cannot support.

**Splits** — ten contiguous 14-day windows with roles: 4 retriever-training, 4 ranker-training,
1 validation, 1 test. Strictly chronological, never shuffled. Features for a window read only
data strictly before it starts.

**Product text** — every article has structured category columns *and* a free-text description:

| Column | Value |
|---|---|
| `product_type_name` | Vest top |
| `colour_group_name` | Black |
| `index_name` | Ladieswear |
| **`detail_desc`** | **"Jersey top with narrow shoulder straps."** |

The categories are already usable by any model. `detail_desc` is the one column nothing can use,
because models don't take sentences — so it gets dropped. H2 asks whether that's a mistake. It
isn't trivially one: two articles can share identical taxonomy and differ entirely in description.

Two variants run, because they measure different things:

| | Content | Reads as |
|---|---|---|
| **Text-B** *(default)* | `detail_desc` only | The honest measure. The one field with no categorical twin, so lift here is genuinely novel information |
| **Text-A** *(one sensitivity run)* | full concat: name + type + group + colour + department + `detail_desc` | Optimistic bound. Overlaps the taxonomy every arm already has, so part of any lift is re-encoded categoricals |

Reporting both quantifies how much apparent "text lift" is taxonomy in disguise.

**Slices** — every metric on three populations, because the aggregate hides the part that
matters: all rows, **cold-start articles** (`art_prior_purchases < 10`), and low-history
customers. Both thresholds are set once at M1 from the measured distributions and registered —
retuning a slice boundary after seeing results is p-hacking with extra steps.

---

## Results

> **Placeholder.** No models have been trained. Every cell is regenerated by `make report` from
> git-committed per-run JSON — **no number in this file is ever hand-typed.**

### Stage-attribution table (H1)

| Intervention | ΔMAP@12 (e2e) | 95% CI | Δp95 ms | Δ$/1M | Δ quality per ms |
|---|---|---|---|---|---|
| `K` 100 → 200 | — | — | — | — | — |
| `K` 100 → 500 | — | — | — | — | — |
| text in retriever (`R1` → `R2`) | — | — | — | — | — |
| `lgbm` → `mlp` | — | — | — | — | — |
| `lgbm` → `dcn` | — | — | — | — | — |
| random → hard negatives | — | — | — | — | — |

### Retrieval (H2)

| Arm | `recall@100` all | `recall@100` cold-start | Coverage | Popularity ρ |
|---|---|---|---|---|
| `pop` (floor) | — | — | — | — |
| `R1` no text | — | — | — | — |
| `R2` + `detail_desc` | — | — | — | — |

### Ranking and end-to-end

| Arm | AUC | NDCG@12 | MAP@12 | **MAP@12 (e2e)** |
|---|---|---|---|---|
| `lgbm` | — | — | — | — |
| `mlp` | — | — | — | — |
| `dcn` | — | — | — | — |

---

## Cost

Measured per stage on the target machine, so the latency budget is attributable rather than one
opaque number. Converted to **$/1M predictions** against one named cloud SKU, with instance type
and price-lookup date cited inline.

The claim to confirm:

```
105,000 × 128 × 4 B  ≈   54 MB    item vectors, fp32
105,000 × 384 × 4 B  ≈  162 MB    raw encoder output, fp32
105,000 × 384 × 1 B  ≈   40 MB    int8
```

If those hold, **the transformer's steady-state serving cost is approximately zero** — it is an
offline batch job feeding a lookup table, not an online forward pass. The cold path matters only
for genuinely new articles, a small and predictable trickle.

That reframes the standard "transformers are too expensive to serve" objection honestly: for an
item-level-text problem with a bounded catalog, the expensive thing amortizes to near-nothing.
Whether it changes the *conclusion* depends entirely on whether there was any lift to serve.

`bench` prints the cache's measured `nbytes` and `peak_rss_mb` alongside the latency table, so
the memory claim is evidence rather than arithmetic.

---

## Engineering constraints

8 GB RAM, 8 cores, Apple Silicon, MPS only. This drives design, not just caution.

| Stage | Peak RSS | How it stays bounded |
|---|---|---|
| `ingest` | ~1.5 GB | DuckDB streams CSV → Parquet; 31.8M rows never materialize |
| `train-retriever` | ~3.0 GB | ~30M params × 4 B × 4 Adam states ≈ 480 MB, plus activations for batch 512 with ~450 unique texts |
| `retrieve` | ~1.0 GB | 54 MB mmap'd index; customers chunked at 4096, score blocks tiled |
| `train-ranker` (`lgbm`) | ~2.0 GB | `max_bin=63` → 235 MB binned vs 940 MB raw at 5M × 47 |
| `train-ranker` (`mlp`/`dcn`) | ~1.5 GB | ~400k params; minibatches streamed from Parquet, design matrix never materialized |

Two consequences worth noting. Dropping the 32 embedding feature columns took the LightGBM matrix
from 80 to 47 columns and its peak from 3.5 GB to 2.0 GB — the two-tower subsumed those features,
so the memory came back for free. And the rankers are *tiny* (~400k params): at this scale memory
is dominated by the data, not the model.

---

## Limitations

Stated here rather than discovered later.

- **The retriever is 12 weeks stale at the test window.** One retriever trained on W1–W4 serves
  W5–W10. Realistic — production retrieval models retrain on a cadence — and it hits every arm
  equally, so comparisons hold. Fashion is seasonal, so the decay may be larger here than
  elsewhere; the `recall@K` curve from W5 to W10 *is* the decay estimate and is reported.
- **Rows exist only for customers who transacted in the window.** Every number is conditional on
  the customer transacting and does not describe the full customer base.
- **The stage-2 axis is "the ranker's own features," not "a text-free pipeline."** Text reaches
  every ranker through `retrieval_score`, because the frozen retriever is text-aware. The clean
  pipeline-level number is H2's end-to-end comparison.
- **`price` is scaled and anonymized.** All revenue figures are relative ratios. This project
  cannot and does not report AOV in currency.
- **No online evaluation.** No live traffic exists; all business metrics are offline proxies and
  labelled as such.
- **MAP@12 is reported for calibration against the public leaderboard, not as a target.**

---

## Glossary

| Term | Meaning |
|---|---|
| **`recall@K`** | Fraction of a customer's actual purchases appearing in stage 1's top `K`. The pipeline's ceiling |
| **NDCG@12 / MAP@12** | Ranking-quality metrics over a 12-slot slate, computed per customer then averaged. MAP@12 is H&M's native competition metric |
| **Two-tower** | Customer → 128 numbers, article → 128 numbers, score = dot product. Item vectors precompute because they don't depend on the customer |
| **Hard negatives** | Wrong answers drawn from what the retriever actually returned — plausible items the customer didn't buy. Contrast with random negatives, which are trivially separable |
| **In-batch softmax** | For each real purchase in a batch of 512, the other 511 items serve as its negatives. Free negatives, no materialized rows |
| **log-Q correction** | Subtracts each item's log sampling frequency from its logit. Without it, popular items appear as negatives more often and the model learns to rank popularity backwards |
| **DCN-v2** | A ranker with layers that explicitly multiply feature pairs, so it discovers interactions like `age × category` without them being hand-written |
| **`as_of`** | The exclusive cutoff for a window. Features may read `t_dat < as_of`, never `>=`. Required keyword-only on every feature builder |
| **Cold-start article** | Fewer than 10 purchases before the window starts. Popularity features are ≈0; the description is complete |
| **Isotonic calibration** | A monotone step function fit on validation that maps a raw model score to an observed purchase rate. Needed before a prediction can be multiplied by price |
| **Digest** | SHA-256 of an artifact's bytes, recorded in a manifest and asserted before training, so no arm can silently train on different data |

---

## Running it

```bash
uv sync                                  # Python 3.11; system 3.9 cannot be used
uv run pytest                            # the invariant harness — green before any data exists
uv run contentsignal splits              # print the window table with roles
uv run contentsignal --help
```

`make m1` … `make m9` chain the pipeline to match milestones. Every command is idempotent and
skips work when its outputs exist with a matching config hash, unless `--force`.

**Data access requires two manual steps** — accept the competition rules on Kaggle, and place
`kaggle.json` at `~/.kaggle/` with mode `600`. The API returns 403 until the rules are accepted,
even with a valid token, so `ingest` checks both and names them separately before starting a
multi-GB download.
