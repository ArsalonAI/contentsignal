# ContentSignal — PRD

**Does fine-tuning a transformer encoder on product text add measurable predictive lift
over tabular features alone for purchase propensity, and what does that lift cost to serve?**

| | |
|---|---|
| Status | Draft — pre-implementation |
| Dataset | [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) |
| Deliverable | Reproducible research repo + written findings (`reports/results.md`) |
| Target machine | 8 GB RAM, 8-core Apple Silicon, MPS only, no discrete GPU |
| Out of scope | Serving infrastructure, online A/B testing, article images |

---

## 1. Problem & hypothesis

E-commerce ranking systems decide which products to surface to which customer. Most
production systems run on tabular signals: customer history, item popularity, price,
recency, categorical taxonomy. Product *content* — names and free-text descriptions — is
usually reduced to categorical IDs or dropped. The open question is whether a language
model reading that content extracts signal the tabular features do not already carry, and
whether that signal survives contact with a properly tuned gradient-boosting baseline.

This project answers that question end to end on public data, then prices the answer.

### Hypothesis (H1) — magnitude

> Adding fine-tuned product-text embeddings to a tabular purchase-propensity model
> improves ranking quality over a tuned LightGBM model trained on tabular features alone,
> on an identical temporal test split.

### Hypothesis (H2) — direction

> The lift is **concentrated in cold-start articles**, where popularity features are near
> zero and the product description is fully available from day one.

H2 is the substantively interesting claim, and registering it in advance is what separates a
prediction that survived from a story assembled after seeing the slice table.

### Pre-registered success criteria

Fixed **before any model is run**, so results cannot be rationalized after the fact. Both
are judged on the held-out test split.

**H1 is supported** only if both hold:

1. **ΔAUC ≥ 0.005** (combined arm minus tabular baseline), and
2. the **95% bootstrap CI on ΔAUC excludes zero** (1000 resamples, resampled over
   customers — see §8).

**H2 is supported** only if both hold:

1. **ΔAUC(cold-start) > ΔAUC(all rows)**, and
2. the **95% bootstrap CI on that difference of deltas excludes zero**, computed on shared
   customer resamples (`trd.md` §10.2).

Anything less, on either, is a **null result** for that hypothesis.

**The two are independent.** H1 can fail while H2 holds — text contributing nothing on
aggregate but real lift where popularity is blank. That combination is a more useful finding
than a diffuse aggregate win, because it names a deployment condition rather than a vague
improvement.

### Commitment to publishing a null result

If the encoder does not clear the bar on **either** hypothesis, `reports/results.md` reports
that as the headline finding, with the same rigor and prominence a positive would receive.
"Semantic
content added no lift over well-built tabular features, at this cost" is a legitimate and
useful conclusion. No arm is re-tuned, no split is re-cut, and no metric is swapped in
order to manufacture a positive. The test split is touched **once**, at the end.

---

## 2. Data

### Source and scope

Three CSVs from the Kaggle competition. The ~30 GB article image set is **not used** —
this project is about text, and the images do not fit the disk/compute budget.

| File | Rows | Size | Role |
|---|---|---|---|
| `transactions_train.csv` | ~31.8M | ~3.5 GB | Purchase events; source of labels and behavioral features |
| `articles.csv` | ~105k | ~35 MB | Item metadata + free-text `detail_desc` |
| `customers.csv` | ~1.37M | ~200 MB | Customer attributes |

**Date range**: 2018-09-20 → 2020-09-22 (~104 weeks).

### Key schema

`transactions_train.csv`: `t_dat` (date), `customer_id` (hex string), `article_id`,
`price`, `sales_channel_id` (1 = in-store, 2 = online).

`articles.csv`: `article_id`, `product_code`, `prod_name`, `product_type_no` /
`product_type_name`, `product_group_name`, `graphical_appearance_no` / `_name`,
`colour_group_code` / `_name`, `perceived_colour_value_id` / `_name`,
`perceived_colour_master_id` / `_name`, `department_no` / `_name`, `index_code` /
`index_name`, `index_group_no` / `_name`, `section_no` / `_name`, `garment_group_no` /
`_name`, `detail_desc` (free text; a small number of nulls).

`customers.csv`: `customer_id`, `FN`, `Active`, `club_member_status`,
`fashion_news_frequency`, `age` (nulls present), `postal_code` (hashed).

### ⚠️ Caveat — `price` is not currency

H&M's `price` column is **scaled and anonymized**. It is not euros, dollars, or any
currency. Consequently every revenue and AOV figure in this project is reported as a
**relative lift ratio against a baseline**, never as an absolute monetary amount. Any
statement of the form "the model adds $X of AOV" would be fabricated. See §8.

### Ingest

CSV → Parquet once, with narrowed dtypes, then the raw CSVs are deleted:

| Column | Stored as |
|---|---|
| `article_id` | `int32` |
| `customer_id` | hashed to a dense `int32` index (lookup table persisted) |
| `t_dat` | `date32` |
| `price` | `float32` |
| `sales_channel_id` | `int8` |

Engine choice is a hard requirement, not a preference: **31.8M rows will not fit in pandas
on an 8 GB machine.** Aggregations run in **DuckDB** (out-of-core, spills to disk);
in-memory frames use **Polars**. pandas is permitted only for small result tables and
plotting.

### Bounded subset policy

The 8 GB accommodation, stated explicitly so the sampling is not mistaken for an oversight:

- **Full transaction history stays available for backward-looking feature computation.**
  Truncating history would degrade recency and popularity features for no memory benefit,
  since those aggregations run in DuckDB.
- **Only the last ~26 weeks generate label windows.**
- **Customer cohort sampled from active customers** (≥1 transaction in the feature window),
  drawn once under a fixed seed and reused everywhere.

The binding constraint is a **row budget, not a cohort size**: the target is **≤5M training
rows**, and the sampling step measures actual counts and fails loudly if the cohort
overshoots, naming the size that would fit. Cohort size starts at 150k and is adjusted once,
before any model is trained — the row-count estimates carry real uncertainty, and
discovering the memory ceiling midway through the ablation grid would be expensive.
See `trd.md` §3.4 and §14.

---

## 3. Label definition

**The H&M dataset contains purchase transactions only. It has no click events and no
add-to-cart events.** `transactions_train.csv` is positive-only implicit purchase feedback.
Neither impression logs nor intermediate funnel steps exist anywhere in the release.

This project therefore models **purchase propensity** and says so plainly. Framing the
target as "click" or "add-to-cart" propensity would describe an experiment this data cannot
support.

For a (customer *c*, article *a*, label window *W*) triple:

```
y = 1  if customer c purchased article a during window W
y = 0  for sampled negatives (§5)
```

Implications carried through the rest of the document:

- Purchase is a **later, scarcer, higher-intent** funnel step than a click. Absolute rates
  are lower and class imbalance is more severe than a click-prediction task would show.
- Results transfer to click/ATC modeling only as an **architecture-level** finding
  ("text embeddings did/didn't add lift over tabular"), not as a numeric one.
- A `y = 0` means *not purchased*, which conflates "seen and rejected" with "never seen".
  This is inherent to implicit feedback and is why negative sampling strategy (§5) is a
  first-class design decision rather than a detail.

---

## 4. Temporal splits & leakage control

### Splits

Strictly chronological. **No shuffled splits anywhere in this project.**

| Split | Label windows | Use |
|---|---|---|
| **Train** | through 2020-08-25 | Model fitting, encoder fine-tuning |
| **Validation** | 2020-08-26 → 2020-09-08 | Hyperparameters, early stopping, calibration fitting |
| **Test** | 2020-09-09 → 2020-09-22 | Held out; evaluated **once**, at the end |

Every arm uses these exact boundaries. They are defined once in `conf/split.yaml` and
imported — never redefined per model.

### The `as_of` contract

Every feature for a row in label window *W* is computed from transactions **strictly
before `W.start`**. This is enforced by a single `as_of` cutoff parameter threaded through
every function in `src/contentsignal/features/`. No feature function may read the
transaction table without an `as_of` argument.

### Named leakage risks

Each of these has a corresponding test in `tests/test_leakage.py`:

| Risk | Why it's tempting | Test |
|---|---|---|
| **Global item popularity** — counting an article's sales over the whole dataset | It's a one-line groupby and it works extremely well, because it contains the future | Assert popularity features for window *W* are unchanged when all rows ≥ `W.start` are deleted |
| **Customer aggregates spanning the test period** | Convenient to compute customer stats once and join everywhere | Same deletion-invariance assertion, per customer feature |
| **Encoder pre-exposure** | Fine-tuning on all rows before splitting is the default mistake | Assert the fine-tuning row set ∩ (val ∪ test) = ∅ |
| **Negatives drawn from future catalog** | Articles that didn't exist yet are trivially separable | Assert every sampled negative article had ≥1 transaction before `W.start` |

The deletion-invariance pattern is the general form: *recompute the feature on a dataset
truncated at `W.start`; if the value changes, the feature leaks.*

---

## 5. Negative sampling & calibration

### Sampling

| Parameter | Value | Rationale |
|---|---|---|
| Ratio | **1:10** positive:negative | Fixed across all arms |
| Strategy | **Popularity-weighted**, ∝ (prior-window popularity)^0.75 | Uniform negatives are trivially separable and inflate AUC; the 0.75 exponent is the standard word2vec-style dampening |
| Exclusion | A customer's true positives in *W* are never sampled as their negatives | Avoids labeling a real purchase as negative |
| Candidate pool | Articles with ≥1 transaction before `W.start` | No future catalog (§4) |
| Seed | Fixed, in `conf/data.yaml` | Reproducibility |

**The sampled dataset is materialized to disk once and reused byte-identically by every
arm.** No arm resamples. This makes ablation deltas attributable to the model, not to
sampling noise — without it, ΔAUC of 0.005 is indistinguishable from a different random
draw.

Uniform sampling is run **once** as a sensitivity check and reported in an appendix, not as
a headline arm.

### Calibration

Downsampling negatives shifts the base rate, so predicted probabilities are on the
*sampled* distribution and are systematically too high. Restore them with the standard
prior correction:

```
p_true = p_s / ( p_s + (1 − p_s) / w )
```

where `p_s` is the model's output and `w` is the negative downsampling rate.

**Metric-by-metric consequences** — stated so the asymmetry is not read as an oversight:

| Metric | Affected by sampling rate? | Handling |
|---|---|---|
| **AUC / PR-AUC** | AUC is invariant to the negative sampling rate (it's rank-based, and downsampling negatives uniformly at random preserves expected ranking) | No correction needed. PR-AUC *is* base-rate dependent and is only compared across arms at the same ratio |
| **Log-loss** | Yes, directly | Reported **twice**: on the sampled distribution, and prior-corrected to the true base rate. Both are labeled |
| **Brier score** | Yes | Reported prior-corrected |
| **Revenue / AOV proxy** | Yes — it multiplies price by p̂, so uncorrected probabilities inflate it | Computed on prior-corrected probabilities only |

Additionally reported: **reliability curves** per arm, and **isotonic regression** fit on
the validation split as a secondary calibration path, so the parametric correction can be
checked against a non-parametric one.

---

## 6. Feature groups

These are the ablation axes. Each group can be independently included or excluded.

### Tabular — customer

`age` (with null indicator), `club_member_status`, `fashion_news_frequency`, `FN`,
`Active`, tenure since first purchase, transaction counts over trailing 1/4/12 weeks,
distinct articles purchased, mean/median/std price paid, channel mix (in-store vs online
share), days since last purchase.

### Tabular — article

Windowed popularity (trailing 1/4/12 weeks), distinct buyers, mean price, price percentile
within `product_group_name`, days since first sale (age on shelf), days since last sale,
channel mix.

### Tabular — categorical

`product_type_name`, `product_group_name`, `colour_group_name`, `department_no`,
`index_name`, `index_group_name`, `section_name`, `garment_group_name`,
`graphical_appearance_name`, `perceived_colour_value_name`,
`perceived_colour_master_name` — passed to LightGBM as **native categorical features**.

> **⚠️ Non-negotiable: the tabular baseline receives every one of these.**
>
> This closes the central experimental-design trap of the project. Nearly every "text"
> field in `articles.csv` has a **1:1 categorical column twin** — `product_type_name` is
> both a string and a categorical `product_type_no`. If the baseline is denied those
> columns and the encoder is fed them as text, the encoder is credited with lift that comes
> from information the baseline was simply not given. That is a rigged comparison, and it
> is the single most common way this experiment is run wrong.

### Tabular — cross (customer × article)

Prior purchase count in the article's `product_group` / `department` / `colour_group`;
article price minus customer's mean price paid; customer age × `index_group` affinity;
whether the customer has purchased this exact `product_code` before.

### Text variants

Two, and both are run:

| Variant | Content | Interpretation |
|---|---|---|
| **Text-A** (full concat) | `prod_name` + `product_type_name` + `product_group_name` + `colour_group_name` + `department_name` + `index_name` + `detail_desc`, truncated to 64 tokens | **Optimistic upper bound.** Overlaps heavily with the categorical group, so lift here is partly re-encoded taxonomy |
| **Text-B** (`detail_desc` only) | Free-text description alone | **The honest measure of semantic lift.** `detail_desc` is the only text field without a categorical twin, so lift here is genuinely novel information |

Text-B is the number to trust. Text-A is reported alongside it to quantify how much of the
apparent "text lift" is really taxonomy the trees already had.

### Text — item side

The article's embedding, reduced to 32 dimensions (§7). One fixed vector per article.

### Text — customer side

> **Why this group exists.** Item-side embedding columns hold the **same value for every
> customer**. A feature that is constant across customers **cannot reorder items
> differently per customer** — so for `precision@k` and `MAP@12`, which are per-customer
> rankings, item-side text can contribute almost nothing beyond what article popularity
> already contributes. Without this group the experiment would be structurally biased
> toward a null result regardless of whether product semantics carry signal, which would
> be a strawman test of H1.

- **`cust_taste_{0..31}`** — the customer's taste vector: the mean embedding of articles
  they purchased **strictly before `W.start`**. Leak-safe by construction (§4).
- **`sim_taste_cos`** — cosine similarity between the taste vector and the candidate
  article's embedding. This is the personalized text signal.
- **`sim_last10_max`, `sim_last10_mean`** — similarity to the customer's 10 most recent
  purchases, capturing recent taste rather than lifetime average.
- **`sim_taste_pct_rank`** — percentile of `sim_taste_cos` within that customer's own
  candidate set. Rank-normalizing per customer helps axis-aligned tree splits considerably.

### Cold start

`art_prior_purchases`, `art_is_cold` (fewer than *N* purchases before `W.start`). Used to
define the cold-start evaluation slice (§8), not primarily as predictive features.

---

## 7. Model arms

Two orthogonal axes — **embedding source** (frozen vs fine-tuned) and **text feature form**
(item-side vs item + personalized) — plus baselines.

| # | Arm | Embedding source | Text form | Purpose |
|---|---|---|---|---|
| 1 | Popularity ranker | — | — | **Sanity floor.** On implicit feedback, popularity frequently beats elaborate models. If an arm loses to this, that is the finding |
| 2 | Logistic regression, tabular | — | — | Linear reference; how much of the signal is additive |
| 3 | **LightGBM, tabular** | — | — | **The baseline.** All lift is measured against this |
| 4a | LightGBM + frozen | pretrained | item-side | Does off-the-shelf semantics help at all, without personalization |
| 4b | LightGBM + frozen | pretrained | item + personalized | Does personalization work even without task adaptation |
| 7a | LightGBM + fine-tuned | contrastive | item-side | Fine-tuning's value on item-side features alone |
| 7b | **LightGBM + fine-tuned** | contrastive | item + personalized | **Headline combined arm** |
| 8 | Text-only encoder | contrastive | — | Text-alone ceiling reference |

Three deltas, each reported for **Text-A** and **Text-B** separately:

| Delta | Isolates |
|---|---|
| **7b − 3** | **The headline.** Total lift from product content over tabular alone |
| 7b − 7a | The contribution of **personalization** — the taste vector and similarity features |
| 7b − 4b | The contribution of **fine-tuning** — if ≈ 0, a frozen off-the-shelf encoder sufficed, which changes the cost story in §9 substantially |

### Encoder configuration

- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (~22M params, 384-dim output).
  Chosen to fit MPS on 8 GB with headroom.
- **Max sequence length**: 64 tokens (product text is short; measured token-length
  distribution goes in the EDA notebook to justify this).
- **Batch size** 64 pairs, **1–2 epochs**, AdamW, lr 2e-5, linear warmup. CPU fallback path
  documented and tested, since MPS backends are periodically unstable.

### Fine-tuning objective: contrastive co-purchase

Positive pairs are **two distinct articles purchased by the same customer**; negatives are
the other items in the batch (InfoNCE / `MultipleNegativesRankingLoss`). The encoder learns
a space where *products that appeal to the same person sit close together*.

Two properties make this the right objective rather than a pointwise item-propensity target:

1. **It produces the geometry the similarity features need.** If embeddings are consumed as
   `cosine(customer taste, candidate article)`, the space must be organized by co-appeal,
   not by marginal item attractiveness.
2. **It cannot degenerate into a restatement of popularity.** A target of "this article's
   empirical purchase rate" is very nearly the article-popularity feature LightGBM already
   has, measured directly and more accurately — an encoder trained on it would add nothing,
   and the resulting null would be an artifact of the objective rather than a finding about
   text.

Customer information enters the encoder **through which pairs are positives**, without the
encoder ever reading a customer column.

**Mandatory diagnostics**, reported whatever they show:

- Spearman ρ between the text-only arm's predictions and log article popularity. High ρ
  means the text signal is largely a popularity proxy.
- Nearest-neighbor lists for ~20 sample articles, before and after fine-tuning — the
  cheapest available check that the space learned something fashion-shaped.

> **Critical efficiency note.** Product text is **item-level**, and there are only ~105k
> unique articles against ~4M training rows. Fine-tuning must operate on **deduplicated
> article texts**, not on duplicated rows — otherwise the same ~105k strings are re-encoded
> ~38× per epoch for no gradient benefit. The contrastive formulation preserves this: each
> training example is a *pair of article texts*, so the encoder sees each unique string a
> bounded number of times. This is the difference between a run that finishes in minutes and
> one that does not finish.

After fine-tuning, the encoder is **frozen** and all 105k article embeddings are
**precomputed once** to a lookup table. Everything downstream — LightGBM training,
evaluation, and the serving benchmark — reads from that table.

### Embedding → LightGBM

384 raw dimensions degrade tree models (axis-aligned splits on dense low-variance
dimensions) and inflate memory. Reduce via **truncated SVD to 32 dims** for the main grid;
64 and raw-384 run on the headline arm only, as a sensitivity check that reduction is not
destroying signal. The SVD is fitted on train-window articles only, so test-window articles
cannot influence the projection basis.

---

## 8. Metrics

### Classification

AUC, PR-AUC, log-loss (sampled **and** prior-corrected), Brier score (prior-corrected).

### Ranking

precision@k and recall@k for k ∈ {1, 5, 10, 20}, computed **per customer** then averaged.
Plus **MAP@12** and **NDCG@12** — MAP@12 is the H&M competition's native metric, which
lets absolute numbers be sanity-checked against public leaderboard intuition rather than
existing in a vacuum.

### Uncertainty

- **Bootstrap over customers, not rows** — 1000 resamples. Rows within a customer are
  strongly correlated (shared history features, shared basket), so row-level bootstrap
  produces CIs that are far too narrow and would let noise clear the ΔAUC ≥ 0.005 bar.
- **95% CI reported on every delta**, not just on point estimates.
- **3 seeds per arm**; report mean and spread.

### Evaluation slices

Every metric is reported on three populations, because the aggregate number hides the part
that matters:

| Slice | Definition | Why |
|---|---|---|
| **All rows** | — | The headline, and the population the §1 criterion is judged on |
| **Cold-start articles** | `art_prior_purchases < 10` | Popularity features are absent or near-zero here, while text is fully available from day one. **If text wins anywhere, it wins here** — and this is the strongest business argument the project can make |
| **Low-history customers** | few prior purchases | The symmetric weak case: thin taste vectors, so personalized text features degrade. Bounds the claim |

### Business proxy

Two figures, both **relative**:

1. **Expected revenue @ top-k** = Σ (price × prior-corrected p̂) over the model's top-k
   recommendations per customer.
2. **AOV lift ratio** = realized basket value of the model's top-k vs the popularity
   baseline's top-k.

> **Reminder (see §2): H&M `price` is scaled and anonymized.** Both figures are ratios in
> relative units. This project does not and cannot report AOV in currency.

---

## 9. Inference cost profiling

Measured on the target machine, per 1K predictions: **p50 / p95 latency and throughput**.

| Configuration | What it isolates |
|---|---|
| LightGBM, tabular only | Baseline serving cost |
| LightGBM + cached embedding lookup | Marginal cost of the text arm at steady state |
| Encoder cold path — tokenize + forward, MPS | Cost when text must be encoded at request time |
| Encoder cold path — tokenize + forward, CPU | Realistic commodity-server cost |
| Encoder, ONNX-exported + int8 quantized | Achievable optimized cold path |

Converted to **$/1M predictions** against one named cloud SKU, with the instance type and
the price-lookup date cited inline.

### Hypothesis to confirm

All 105k article embeddings fit in memory:

```
105,000 × 384 × 4 bytes  ≈  162 MB  (fp32)
105,000 × 384 × 1 byte   ≈   40 MB  (int8)
```

If that holds, the encoder's **steady-state serving cost is approximately zero** — it is an
offline batch job feeding a lookup table, not an online forward pass. The cold path then
matters only for **new / cold-start articles**, which is a small and predictable trickle.

This reframes the usual "transformers are too expensive to serve" objection honestly: for
an item-level-text problem with a bounded catalog, the expensive thing is amortized to
near-nothing. Whether that changes the *conclusion* depends entirely on whether §7 found
any lift to serve.

---

## 10. Repo layout

```
contentsignal/
├── pyproject.toml              # uv, Python 3.11 (system 3.9 is too old)
├── prd.md
├── README.md
├── Makefile
├── conf/
│   ├── data.yaml               # paths, seeds, cohort size, sampling ratio
│   ├── split.yaml              # the three date boundaries — single source of truth
│   └── model/*.yaml            # per-arm hyperparameters
├── src/contentsignal/
│   ├── data/                   # download.py, to_parquet.py, schema.py
│   ├── features/               # build.py, customer.py, article.py, cross.py, text.py
│   ├── sampling/               # negatives.py
│   ├── splits/                 # temporal.py
│   ├── models/                 # popularity.py, logreg.py, lgbm.py, encoder.py, fusion.py
│   ├── eval/                   # metrics.py, calibration.py, ranking.py, bootstrap.py, business.py
│   ├── serving/                # predictor.py, embedding_cache.py, benchmark.py
│   └── cli.py
├── tests/
│   ├── test_leakage.py         # deletion-invariance assertions (§4)
│   ├── test_splits.py
│   ├── test_sampling.py
│   └── test_calibration.py
├── notebooks/                  # 01_eda.ipynb, 99_results.ipynb
├── reports/
│   ├── results.md
│   └── figures/
└── artifacts/                  # gitignored: parquet, embeddings, model checkpoints
```

---

## 11. Milestones

| | Milestone | Concrete artifact |
|---|---|---|
| **M0** | Environment + data acquisition | `pyproject.toml`, locked deps, three CSVs on disk |
| **M1** | Parquet conversion, EDA, splits, leakage tests | `artifacts/*.parquet`, `01_eda.ipynb`, **`tests/test_leakage.py` green** |
| **M2** | Negative sampling + tabular features | Materialized train/val/test row sets, feature tables |
| **M3** | Baselines (arms 1–3) | **First results table** in `reports/results.md` |
| **M4** | Frozen-embedding arms (4a, 4b) | Embedding cache, taste vectors, updated results table |
| **M5** | Contrastive fine-tune (arm 8) | Checkpoint, fine-tuned embedding cache, NN dumps + popularity-ρ diagnostic |
| **M6** | Full grid, ablations, bootstrap CIs (arms 7a, 7b) | **Ablation grid with 95% CIs on all three deltas, across all three slices** |
| **M7** | Calibration + business proxy | Reliability curves, AOV lift ratios |
| **M8** | Cost profiling | Latency/throughput table, $/1M predictions |
| **M9** | Report | Final `reports/results.md`, `README.md` |

M1 gates everything: **no model is trained until the leakage tests pass.**

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| **Encoder shows no lift** | Pre-committed to reporting it as the headline finding (§1). Not a project failure — the arm grid's three deltas make a null result *interpretable* rather than merely disappointing |
| **Contrastive pairs are popularity-biased** | Uncapped pair sampling is dominated by heavy buyers and bestsellers, which would push the encoder back toward a popularity proxy. Capped at ≤5 pairs per customer and ≤50 per article, with the popularity-ρ diagnostic (§7) as the check |
| **Text/categorical twin confound** | Text-B (`detail_desc` only) arm, and the baseline receives every categorical (§6) |
| **8 GB OOM** | DuckDB for aggregation, Polars over pandas, `float32` throughout, bounded cohort, raw CSVs deleted post-conversion |
| **MPS instability / silent numerical issues** | Documented CPU fallback; parity check that MPS and CPU embeddings agree within tolerance on a fixed sample |
| **Fine-tune too slow** | Deduplicated item-level batching (§7) — the single largest speedup available. Fall back to shorter `max_len` or 1 epoch |
| **Kaggle download blocked** | Requires accepting the competition rules on Kaggle **and** placing `kaggle.json` at `~/.kaggle/` — currently absent on this machine. M0 is blocked until then |
| **Popularity baseline wins** | Report it. It is a real and frequently observed outcome on implicit-feedback data, and concealing it would invalidate the rest |

---

## 13. Non-goals

Stated plainly so scope creep is visible when it happens:

- **Article images.** ~30 GB, out of budget, and orthogonal to the text question.
- **Production candidate generation.** This scores (customer, article) pairs from a sampled
  candidate set; it is not a two-stage retrieval system over the full catalog.
- **Online A/B testing.** No live traffic exists. All business metrics are offline proxies
  and are labeled as such.
- **Serving infrastructure.** §9 benchmarks a local predictor; it does not build an API.
- **A learned customer tower.** The taste vector is a fixed mean-pool of item embeddings,
  not a trained customer encoder. A true two-tower model is a larger project and would make
  the encoder's marginal contribution harder to isolate, which is the whole point here.
- **Dollar-denominated AOV.** Impossible with scaled prices (§2).
- **Beating the Kaggle leaderboard.** MAP@12 is reported for calibration against known
  results, not as a target to optimize.

---

## Appendix — environment baseline

Captured 2026-07-27 on the target machine:

| Check | State |
|---|---|
| Repo | `/Users/arsalon/Desktop/contentsignal`, branch `main`, no commits |
| Python | system 3.9.6 — **project pins 3.11 via `uv`** (`/opt/homebrew/bin/uv` present) |
| torch / lightgbm / transformers / polars / duckdb | **not installed** — full bootstrap required at M0 |
| RAM / CPU | 8 GB, 8 cores, Apple Silicon (MPS) |
| Disk free | ~124 GB |
| Kaggle credentials | `~/.kaggle` **absent** — blocks M0 |
