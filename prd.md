# ContentSignal — PRD

**A two-stage recommender on the H&M catalog, built to answer one question: given a fixed
pipeline, does the next unit of engineering buy more in retrieval or in ranking?**

| | |
|---|---|
| Status | Draft — pre-implementation |
| Dataset | [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) |
| Deliverable | Reproducible research repo + written findings (`reports/results.md`) |
| Target machine | 8 GB RAM, 8-core Apple Silicon, MPS only, no discrete GPU |
| Out of scope | Serving infrastructure, online A/B testing, article images |

This is an experiment, not a product. The output is a results table and a written finding.
**Correctness of the comparison matters more than the performance of any single model.**

---

## 1. Why two stages

There are ~105,000 articles in the catalog, and a recommendation slate holds 12. A useful
ranking model costs on the order of a millisecond per (customer, article) pair, so scoring the
whole catalog for one customer is ~100 seconds. That is not a latency budget anyone has.

Production systems therefore split the work into two stages with different jobs:

| | **Stage 1 — retrieval** | **Stage 2 — ranking** |
|---|---|---|
| Sees | all ~105,000 articles | the ~100 stage 1 returned |
| Cost per request | must be ~1 ms total | can be ~1 ms *per candidate* |
| Job | *don't miss anything good* | *put the best ones on top* |
| Failure mode | a good item never enters consideration | good items are ordered badly |
| Metric | `recall@K` | NDCG@12, MAP@12 |

**`recall@K`** — of everything a customer actually bought during a window, what fraction
appeared in stage 1's top-`K` list. This number is the ceiling on the entire pipeline: if
`recall@100 = 0.60`, then 40% of real purchases are invisible to stage 2 permanently, and no
ranker however sophisticated can recover them.

That ceiling is the thing this project measures. It is the reason the two stages cannot be
evaluated independently, and it is what makes the retrieval-versus-ranking tradeoff empirical
rather than a matter of taste.

### The two-tower model, and why it is cheap enough for stage 1

Stage 1 is a **two-tower** model. One tower turns a customer into 128 numbers (a vector); the
other turns an article into 128 numbers; the score for a pair is the dot product of the two.

The load-bearing property is that **the article's vector does not depend on the customer**. So
all ~105k article vectors are computed once, offline, into a 54 MB table
(105,000 × 128 × 4 bytes). At request time you compute *one* customer vector and multiply it
against the whole table:

```
[1 × 128] · [128 × 105,000]  =  13.4M multiply-accumulate ops  ≈  under 1 ms
```

> **⚠️ This factorization must not break.** If the customer vector depended on which article
> was being scored — for example, if the model attended over the customer's purchase history
> *differently for each candidate* — then nothing could be precomputed and you would be back to
> 105,000 forward passes per request. Candidate-aware attention would score better on offline
> metrics. It is given up deliberately, and the trade is enforced by a type signature rather
> than a comment: `CustomerTower.forward` accepts no item argument (`trd.md` §9.1).

### The retriever is not an approximate-search problem

At 105k items, the exhaustive matvec above is fast enough that an approximate nearest-neighbour
index (FAISS, ScaNN, HNSW) adds infrastructure without buying latency. ANN indexes earn their
complexity somewhere around 10⁷–10⁹ items, where an exact scan stops fitting the budget.

The project benchmarks exact against approximate anyway and reports both (§9). Demonstrating
that a standard component is **unnecessary at this scale** is a legitimate engineering result,
and the reflex to reach for a vector database regardless of catalog size is common enough to be
worth measuring against.

---

## 2. The three pre-registered hypotheses

Fixed **before any model is run**, so no result can be rationalized after the fact. Each is
judged on the held-out test split, which is read **once**, at the final milestone.

Two conventions apply throughout, both carried over from the previous design because both were
chosen against a specific failure mode:

- **All confidence intervals come from 1000 bootstrap resamples over *customers*, not rows.**
  Rows belonging to one customer share history features and basket composition, so resampling
  rows treats correlated observations as independent and produces intervals narrow enough for
  noise to clear any significance bar (`trd.md` §10.2).
- **Every arm runs 3 seeds**; mean and spread are both reported.

### H1 — Stage attribution *(the headline)*

> End-to-end recommendation quality is more sensitive to **retrieval depth** than to **ranker
> architecture**.

**Supported** only if both hold:

1. ΔMAP@12 from raising `K` = 100 → 500 (best ranker held fixed) **exceeds** ΔMAP@12 from the
   worst to the best ranker at fixed `K` = 100, and
2. the 95% CI on that **difference of deltas** excludes zero.

**Registered prior: supported.** The mechanism is §1's ceiling — a ranker can only reorder what
stage 1 surfaced, so once `recall@K` binds, ranker improvements return nothing no matter how
much architecture is thrown at them.

The deliverable is a **stage-attribution table**: for every intervention, Δquality, Δp95
latency, and Δ$/1M predictions on one axis. That table answers "should the next quarter go to
retrieval or to ranking," which is a question teams argue about with intuition far more often
than with numbers.

### H2 — Content-based retrieval and cold start

> Adding the free-text product description to the item tower improves retrieval recall **more on
> cold-start articles than on articles overall**.

**Supported** only if both hold:

1. Δ`recall@100` on the cold-start slice ≥ **0.01**, 95% CI excluding zero, and
2. Δ`recall@100`(cold-start) > Δ`recall@100`(all articles), with the 95% CI on that difference
   of deltas excluding zero, computed on shared customer resamples.

**Registered prior: supported on cold-start, marginal or null on aggregate.**

The mechanism is concrete. An article added to the catalog today has:

| Signal | Available? |
|---|---|
| Popularity, purchase counts, distinct buyers | **zero** — nobody has bought it |
| Co-purchase history | **none** |
| Taxonomy (product type, colour, department) | yes |
| `detail_desc` free-text description | **yes, in full, on day one** |

A retriever scoring on behavioral signal gives that article ≈0 → it never enters the top 100 →
the ranker never sees it → **it can never be recommended, no matter how good the ranker is.** A
retriever that reads *"Jersey top with narrow shoulder straps"* can place it near other strappy
jersey tops and surface it to customers who buy those.

The business version is an ordinary chicken-and-egg problem: a new product can't accumulate
purchases because it isn't shown, and isn't shown because it has no purchases. Content-based
retrieval is how that loop breaks.

The 0.01 absolute floor exists so that a statistically significant but operationally trivial
lift cannot count as support. It is a judgment call, and registering it in advance is what
keeps it from being tuned to the answer.

### H3 — Candidate distribution shift

> A ranker trained on **random negatives** underperforms end-to-end against the identical ranker
> trained on the retriever's **hard negatives**.

**Supported** only if both hold:

1. ΔMAP@12 (hard-negative − random-negative) ≥ **0.005** end-to-end, and
2. the 95% CI on ΔMAP@12 excludes zero.

**Registered prior: supported, and the delta is larger than the ranker-architecture delta in
H1** — that is, *what you train on matters more than which model you use.*

**Negatives** are the wrong answers a model learns from. There are two ways to pick them:

| Source | What the ranker learns to reject | Matches production? |
|---|---|---|
| **Random**, popularity-weighted (the previous design) | popular articles unrelated to this customer — usually trivially separable | no |
| **Hard**, i.e. whatever the retriever returned but the customer didn't buy | plausible, semantically close items | **yes** |

At serving time a ranker only ever sees stage 1's output. Training it on easy negatives and
serving it hard ones is a well-documented failure mode — sample selection bias in candidate
generation — and it produces models that test well and disappoint in production. H3 measures
the size of that gap on this dataset instead of asserting it.

### Commitment to publishing a null result

If any hypothesis fails to clear its bar, `reports/results.md` reports that as the headline
finding, with the same rigor and prominence a positive would receive. No arm is re-tuned, no
split is re-cut, no metric is swapped, and no threshold is moved to manufacture a positive. The
test split is touched **once**, at the end, and **all three** hypotheses are judged on that
single pass — evaluating some and quietly skipping others would defeat the point of registering
them.

**The three are independent.** H1 can hold while H2 fails; H3 can be the only survivor. Each
combination is a different and useful finding.

---

## 3. Data

### Source and scope

Three CSVs from the Kaggle competition. The ~30 GB article image set is **not used** — this
project is about text and architecture, and the images fit neither the disk nor the compute
budget.

| File | Rows | Size | Role |
|---|---|---|---|
| `transactions_train.csv` | ~31.8M | ~3.5 GB | Purchase events; source of labels and behavioral features |
| `articles.csv` | ~105k | ~35 MB | Item metadata + free-text `detail_desc` |
| `customers.csv` | ~1.37M | ~200 MB | Customer attributes |

**Date range**: 2018-09-20 → 2020-09-22 (~104 weeks).

### Key schema

`transactions_train.csv`: `t_dat` (date), `customer_id` (hex string), `article_id`, `price`,
`sales_channel_id` (1 = in-store, 2 = online).

`articles.csv`: `article_id`, `product_code`, `prod_name`, `product_type_no` /
`product_type_name`, `product_group_name`, `graphical_appearance_no` / `_name`,
`colour_group_code` / `_name`, `perceived_colour_value_id` / `_name`,
`perceived_colour_master_id` / `_name`, `department_no` / `_name`, `index_code` / `index_name`,
`index_group_no` / `_name`, `section_no` / `_name`, `garment_group_no` / `_name`, `detail_desc`
(free text; a small number of nulls).

`customers.csv`: `customer_id`, `FN`, `Active`, `club_member_status`, `fashion_news_frequency`,
`age` (nulls present), `postal_code` (hashed).

### What "product text" means here

Every article carries structured category columns **and** a free-text description. A row looks
approximately like:

| Column | Value |
|---|---|
| `prod_name` | Strap top |
| `product_type_name` | Vest top |
| `product_group_name` | Garment Upper body |
| `colour_group_name` | Black |
| `index_name` | Ladieswear |
| **`detail_desc`** | **"Jersey top with narrow shoulder straps."** |

Every model can already use the category columns — `product_type = Vest top` is category #47,
which trees and embedding layers consume directly. **`detail_desc` is the one column nothing can
use**, because models don't take English sentences, so in practice it is dropped. H2 asks
whether that's a mistake.

It isn't trivially a mistake. Two articles can share identical taxonomy — both "Vest top /
Ladieswear / Black" — and differ entirely in description: *"narrow shoulder straps"* versus
*"ribbed wide straps, soft brushed inside."* If customers care about that, the text carries
signal the categories don't.

> **⚠️ The confound that makes this easy to get wrong.** Almost every text field in
> `articles.csv` has a **1:1 categorical twin** — `product_type_name` is both a string and a
> number. If a model is fed text containing the words "Vest top" while the comparison baseline
> is denied the `product_type` column, the text is credited with information that was simply
> withheld from the baseline. That is a rigged comparison. Two guards, both mandatory:
>
> - **Every arm receives all eleven categorical columns** (§7), so nothing is withheld.
> - **Two text variants are defined**, and the honest one is the default:
>
> | Variant | Content | Interpretation |
> |---|---|---|
> | **Text-B** *(default)* | `detail_desc` alone | **The honest measure.** The only text field with no categorical twin, so lift here is genuinely new information |
> | **Text-A** *(one sensitivity run)* | `prod_name` + taxonomy names + `detail_desc`, truncated to 64 tokens | **Optimistic upper bound.** Overlaps heavily with the categoricals, so lift is partly re-encoded taxonomy |
>
> Text-B is the number to trust. Text-A is reported alongside it to quantify how much apparent
> "text lift" is taxonomy the model already had.

### ⚠️ Caveat — `price` is not currency

H&M's `price` column is **scaled and anonymized**. It is not euros, dollars, or any currency.
Every revenue figure in this project is reported as a **relative ratio against a baseline**,
never as an absolute monetary amount. Any statement of the form "the model adds $X of AOV" would
be fabricated.

### Ingest

CSV → Parquet once, with narrowed dtypes, then the raw CSVs are deleted:

| Column | Stored as |
|---|---|
| `article_id` | `int32` |
| `customer_id` | hashed to a dense `int32` index (lookup table persisted) |
| `t_dat` | `date32` |
| `price` | `float32` |
| `sales_channel_id` | `int8` |

Engine choice is a hard requirement, not a preference: **31.8M rows will not fit in pandas on an
8 GB machine.** Aggregations run in **DuckDB** (out-of-core, spills to disk); in-memory frames
use **Polars**. pandas is permitted only for small result tables and plotting.

### Bounded subset policy

The 8 GB accommodation, stated explicitly so the sampling is not mistaken for an oversight:

- **Full transaction history stays available for backward-looking feature computation.**
  Truncating history would degrade recency and popularity features for no memory benefit, since
  those aggregations run in DuckDB.
- **Only the last ~20 weeks generate label windows.**
- **Customer cohort sampled from active customers**, drawn once under a fixed seed and reused
  everywhere.

The binding constraint is a **row budget, not a cohort size**. Stage-2 rows scale as
`windows × customers × K`, which grows fast; the target is **≤5M ranker training rows**, and the
`retrieve` step measures actual counts and fails loudly if it overshoots, naming the (cohort, K)
pair that would fit. Cohort size and `K` are adjusted once, before any ranker is trained — the
row-count estimates carry real uncertainty and discovering the memory ceiling midway through the
grid would be expensive. See `trd.md` §3.4 and §14.

---

## 4. Label definition

**The H&M dataset contains purchase transactions only. It has no click events and no
add-to-cart events.** `transactions_train.csv` is positive-only implicit purchase feedback.
Neither impression logs nor intermediate funnel steps exist anywhere in the release.

This project therefore models **purchase propensity** and says so plainly. Framing the target as
"click" or "add-to-cart" propensity would describe an experiment this data cannot support.

For a (customer *c*, article *a*, label window *W*) triple:

```
y = 1  if customer c purchased article a during window W
y = 0  otherwise
```

Implications carried through the rest of the document:

- Purchase is a **later, scarcer, higher-intent** funnel step than a click. Absolute rates are
  lower and class imbalance is more severe than a click-prediction task would show.
- Results transfer to click/ATC modeling as an **architecture-level** finding ("retrieval depth
  dominated ranker architecture"), not as a numeric one.
- A `y = 0` means *not purchased*, which conflates "seen and rejected" with "never seen." This is
  inherent to implicit feedback, and it is precisely why the choice of negatives (H3) is a
  first-class design decision rather than a detail.

---

## 5. Temporal splits, window roles, and leakage

### Windows

Strictly chronological. **No shuffled splits anywhere in this project.** The timeline is cut
into ten contiguous, non-overlapping 14-day **label windows**, defined once in
`conf/split.yaml` and imported — never redefined per model.

### Window roles — and the leakage risk a second stage introduces

The retriever is itself a trained model, which creates a leakage vector the previous
single-stage design did not have.

Suppose the retriever trains on all eight train windows, and is then asked for candidates for
window 5. It saw this customer buy this article in window 5 during its own training. **It has
memorized the answer and puts it at rank 1** — for a reason that will never hold at serving
time, when the retriever has never seen tomorrow. The ranker then trains on candidate lists
where the correct answer is reliably near the top, learns to trust rank 1, and collapses in
production.

The fix is to give the train windows distinct **roles**, so the retriever's training data
strictly precedes every window it retrieves for:

```
  W1    W2    W3    W4   │   W5    W6    W7    W8   │    W9    │   W10
  └──── retriever ───────┘   └────── ranker ───────┘     val       test
         training                  training                      ↑ read once
  2020-05-06 → 06-30          2020-07-01 → 08-25       08-26     09-09
                                                       → 09-08   → 09-22
```

| Split | Windows | Use |
|---|---|---|
| **Train — retriever** | W1–W4 | Fitting the two-tower. Positives only |
| **Train — ranker** | W5–W8 | Fitting the rankers, on candidates from the frozen retriever |
| **Validation** | W9 | Hyperparameters, early stopping, isotonic calibration, all M3–M8 reporting |
| **Test** | W10 | Held out; evaluated **once**, at the end |

Enforced by `tests/test_leakage.py::test_retriever_windows_precede_candidate_windows`.

**Accepted cost, stated rather than buried.** By W10 the retriever's weights are 12 weeks
stale. Three reasons this is acceptable:

1. It is realistic — production retrieval models are retrained on a cadence, not continuously.
2. The customer-side *inputs* stay current: purchase history is rebuilt per window from
   transactions before that window's `as_of`, so only the weights age, not the features.
3. It affects every arm identically, so all comparisons remain valid.

Fashion turns over seasonally, so the decay may be larger here than in other domains. That is
measurable — the `recall@K` curve from W5 through W10 *is* the decay estimate — and it is
reported as a finding rather than assumed away.

### The `as_of` contract

Every feature for a row in window *W* is computed from transactions **strictly before**
`W.start`. This is enforced by a single `as_of` cutoff threaded through every function in
`src/contentsignal/features/`, declared keyword-only and required. No feature function may read
the transaction table without one — **the signature is the enforcement mechanism**, not a
convention (`trd.md` §5.1).

### Named leakage risks

Each has a corresponding test in `tests/test_leakage.py`:

| Risk | Why it's tempting | Test |
|---|---|---|
| **Global item popularity** — counting an article's sales over the whole dataset | It's a one-line groupby and it works extremely well, because it contains the future | Assert popularity features for window *W* are unchanged when all rows ≥ `W.start` are deleted |
| **Customer aggregates spanning val/test** | Convenient to compute customer stats once and join everywhere | Same deletion-invariance assertion, per customer feature |
| **Retriever trained on windows it retrieves for** | Training on all eight train windows is the obvious default | Assert every retriever training window ends strictly before the first candidate window starts |
| **Candidates drawn from the future catalog** | Articles that didn't exist yet are trivially separable | Assert every retrieved candidate article had ≥1 transaction before `W.start` |
| **Purchase count as a feature** | Strong signal, one line to compute | It is a function of the label window; repeat purchases collapse to one row and count is never carried |

**The deletion-invariance pattern is the general form:** *recompute the feature on a dataset
truncated at `W.start`; if the value changes, the feature leaks.* It is applied automatically to
every registered feature builder, so a new builder is covered the moment it is imported.

---

## 6. Stage 1 — the retriever

### Towers

| Tower | Inputs |
|---|---|
| **Customer** | `age` (+ null indicator), `club_member_status`, `fashion_news_frequency`, `FN`, `Active`, tenure, transaction counts over trailing 1/4/12 weeks, distinct articles, mean/median/std price paid, in-store-vs-online share, days since last purchase — **plus the last 20 purchased `article_id`s**, embedded through a 105k × 64 table and pooled by self-attention with a learned query, masked for variable-length history |
| **Item** | The 11 taxonomy categoricals (8–16d embeddings each), article numerics (popularity over 1/4/12 weeks, distinct buyers, mean price, price percentile within `product_group`, days since first and last sale, in-store-vs-online share) — **plus, in the text arm, `detail_desc` → MiniLM-L6-v2 → 384 → projection** |

Both towers: concatenate → MLP [512, 256] → 128, L2-normalized. Score = temperature-scaled dot
product.

> **Design decision: the item tower has no article-ID embedding.**
>
> | | |
> |---|---|
> | **Considered** | Add a learned 105k × 64 item-ID embedding to the item tower, as most production two-towers do |
> | **Chosen** | Content-only item tower — taxonomy, numerics, and optionally text |
> | **Reason** | An ID embedding is a learned vector per article, so a newly added article's vector is untrained noise. A content-only tower can represent an article it has never seen a single purchase for. ID towers are stronger on the head of the distribution; content towers are the only ones that function on new items. This is the exact mechanism H2 tests, and stating it is what separates a fair cold-start comparison from a strawman |
>
> The customer tower still contains an item-ID table — for *history*, where the IDs are all
> articles with prior purchases by definition.

### Training objective: in-batch sampled softmax with log-Q correction

Batches hold 512 real (customer, purchased-article) pairs. For each pair, **the other 511 items
in the batch serve as its wrong answers** — negatives, for free, without materializing any
negative rows.

There is a bias in that shortcut. The 511 other items are themselves purchases, so popular
articles appear as somebody's negative far more often than rare ones. Left uncorrected, the
model learns *popular ⇒ probably wrong* and starts ranking popularity **backwards**. The **log-Q
correction** (Yi et al., RecSys 2019) subtracts each item's log sampling frequency from its
logit, cancelling the bias:

```
logits[i][j] = cust[i] · item[j] / T  −  log_q[item_ids[j]]
```

Estimated from a streaming item-frequency counter over the retriever's training positives.

| Setting | Value |
|---|---|
| Batch | 512 positives — **batch size is the negative count** |
| Optimizer | AdamW, discriminative learning rates: **1e-3 towers, 2e-5 pretrained encoder** |
| Schedule | 10% linear warmup, cosine decay |
| Epochs | 2, selected on W9 `recall@100` |
| Device | `mps`, with a tested `--device cpu` fallback |
| Seeds | 3 |

**Training consumes positives only** — ~260k pairs across W1–W4, not millions of rows — which
makes stage 1 the cheap part of the project (~1,000 steps/epoch, ~4–6 min/epoch on MPS).

**Batch construction**: articles repeat across pairs, so each batch dedupes article IDs, encodes
each unique text once, and gathers. Duplicate items within a batch are masked out of the
negative set, since an item cannot be its own negative. At ~105k unique texts against ~260k
pairs, this is the difference between a run that finishes in minutes and one that doesn't.

### Encoder configuration

- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (~22M params, 384-dim output), chosen to
  fit MPS on 8 GB with headroom.
- **Max sequence length**: 64 tokens — product text is short, and the measured token-length
  distribution goes in the EDA notebook to justify the number rather than assert it.
- The encoder trains **jointly with the towers**, at a lower learning rate. There is no separate
  contrastive pre-training stage: in-batch softmax over co-purchase pairs *is* a contrastive
  objective, and the pairs are positives precisely because the same customer bought both.

**Customer information enters the encoder through which pairs are positives, not through any
customer column.** The item tower never reads customer data.

### Retriever arms

| Arm | Item tower | Runs | Purpose |
|---|---|---|---|
| `pop` | none — top-`K` bestsellers before `W.start` | 0 (deterministic) | **Sanity floor.** On implicit feedback, popularity is a genuinely hard baseline. If a trained retriever loses to it, that is the finding, and it must be discovered before anything is spent on stage 2 |
| `R1` | taxonomy + numerics, **no text** | 3 seeds | The behavioral/taxonomic retriever |
| `R2` | `R1` + `detail_desc` (Text-B) | 3 seeds | **The content retriever.** `R2 − R1` is H2 |
| `R2-A` | as `R2` but full text concat (Text-A) | 1 | Sensitivity: how much apparent text lift is re-encoded taxonomy |

After training, the item tower is frozen and all ~105k article vectors are **precomputed once**
into an mmap-able lookup table. Everything downstream — candidate generation, ranker features,
and the serving benchmark — reads from that table.

### Mandatory diagnostics, reported whatever they show

- **Popularity ρ** — Spearman correlation between retriever scores and `log(art_pop_12w + 1)`. A
  two-tower that has collapsed into a popularity re-ranker fails here, and a missing or broken
  log-Q correction is the first thing to check.
- **log-Q ablation** — one run with the correction disabled, to *demonstrate* its effect rather
  than assert it.
- **Embedding collapse check** — tower output norms and pairwise cosine spread. A degenerate
  encoder maps everything to nearly the same vector, which looks fine in the loss and produces
  meaningless retrieval.
- **Nearest-neighbour lists for ~20 sample articles** — the cheapest available check that the
  learned space is fashion-shaped rather than arbitrary.

---

## 7. Stage 2 — the rankers

### The candidate set, and why it is frozen

The **frozen** `R2` retriever generates top-`K` candidates for W5–W10. Those are written **once**
to `artifacts/candidates/{window}.parquet` and hashed into `candidates_manifest.json`; every
ranker asserts that hash before training.

> **Invariant: the candidate set is byte-identical across every ranker arm.** If ranker A were
> scored on a different candidate list than ranker B, a ΔNDCG of 0.005 would be
> indistinguishable from A having drawn an easier list. Never regenerate candidates per arm.

The retriever is frozen for the whole stage-2 experiment for the same reason — comparing rankers
across shifting candidate distributions compares nothing.

A retrieved candidate the customer purchased in that window is labeled 1; every other retrieved
candidate is 0. **The negatives are therefore the retriever's hard negatives** — plausible items
the customer didn't buy — which is exactly the distribution production serves, and the reason H3
is worth asking.

### Arms

| Arm | Model | Runs |
|---|---|---|
| `lgbm` | LightGBM, `max_bin=63`, all categoricals native — **the tabular baseline** | 3 seeds |
| `mlp` | Categorical embeddings + standardized numerics → MLP [512, 256, 128] → 1 | 3 seeds |
| `dcn` | **DCN-v2** — 3 cross layers in parallel with a deep MLP, concatenated → 1 | 3 seeds |

**What DCN-v2 is:** a ranker with layers that explicitly multiply pairs of features together, so
it can discover interactions like "customer age × product category" on its own. This project
hand-writes six such interactions in the `cross` feature group (§7.1). `dcn` versus `mlp`
therefore tests whether **learned** feature crossing beats **hand-built** feature crossing on
the same inputs — a concrete question, not an architecture beauty contest.

> **Keeping the LightGBM baseline is not ceremony.** Gradient-boosted trees usually win on
> tabular-dominant problems at this data scale. Claiming a neural ranker without checking trees
> would credit the architecture for a result nobody verified, which is the same class of error
> as withholding categoricals from a baseline.

**Hyperparameters are tuned on `lgbm` only, then frozen** for every other arm. Tuning each arm
separately confounds "better architecture" with "more tuning budget" (`trd.md` §9.4).

### 7.1 Feature groups — the ablation axes

Each group is built and stored independently per window, then joined at train time.

| Group | Columns | Contents |
|---|---|---|
| `customer` | 18 | `age` (+ null flag), club status, news frequency, FN, Active, tenure, txn counts 1/4/12w, distinct articles, price mean/median/std, online share, days since last purchase |
| `article` | 9 | popularity 1/4/12w, distinct buyers, mean price, price percentile in `product_group`, days since first/last sale, online share, plus `art_prior_purchases` and `art_is_cold` for slicing |
| `categorical` | 11 | `product_type_name`, `product_group_name`, `colour_group_name`, `department_no`, `index_name`, `index_group_name`, `section_name`, `garment_group_name`, `graphical_appearance_name`, `perceived_colour_value_name`, `perceived_colour_master_name` |
| `cross` | 6 | prior purchases in this article's product group / department / colour group; bought this `product_code` before; article price minus customer's mean price paid; customer age-bucket × `index_group` affinity computed on history only |
| `retrieval` *(new)* | 3 | `retrieval_score`, `retrieval_rank`, `retrieval_log_rank` |

**~47 columns.** Every arm receives every group — the eleven categoricals in particular, per
§3's confound guard.

> **What changed, and why nothing was lost.** The previous design had two more groups:
> `text_item` (32 SVD dimensions of the article embedding) and `text_customer` (a customer
> "taste vector" plus cosine-similarity scalars against the candidate article). Both are gone,
> along with the SVD step and the taste-vector artifacts.
>
> This is **subsumption, not removal**. Those features existed because *trees cannot compute a
> dot product* — given 32 taste dimensions and 32 article dimensions, a tree has no way to
> multiply them, so `sim_taste_cos` had to be precomputed by hand. **A two-tower computes
> exactly that dot product natively, learned end-to-end.** Hand-building it downstream would
> duplicate stage 1's job less well.
>
> Text still reaches the ranker, through `retrieval_score`. That means the no-text ranker arms
> sit downstream of a text-aware retriever, so the honest description of the stage-2 axis is
> *"the ranker's own features"*, not *"a text-free pipeline"*. The clean pipeline-level number is
> the end-to-end comparison in H2. Stated inline in the results table, not buried.
>
> Side benefit: 80 columns → 47 drops the LightGBM design matrix from 1.6 GB raw / 400 MB binned
> to ~940 MB / ~235 MB, which buys real headroom against the 8 GB ceiling (`trd.md` §14).

---

## 8. Metrics

Three levels, because a single number cannot attribute a result to a stage.

### Retrieval

| Metric | Definition |
|---|---|
| **`recall@K`**, K ∈ {20, 50, 100, 200, 500, 1000} | Fraction of the window's true purchases appearing in the top `K`. The full sweep costs **one** retrieval pass and no retraining, which is why H1's depth axis is cheap |
| **Cold-start `recall@K`** | The same, restricted to articles with `art_prior_purchases < 10` |
| **Catalog coverage** | Fraction of the ~105k catalog appearing in *any* customer's top `K`. A retriever returning the same 500 bestsellers to everyone scores acceptably on recall and is useless |
| **Popularity ρ** | Spearman against `log(art_pop_12w + 1)` — the collapse diagnostic from §6 |

### Ranking

AUC, PR-AUC, NDCG@12, MAP@12, and precision/recall@k for k ∈ {1, 5, 10, 20} — computed **per
customer, then averaged**, over the retrieved candidate set.

MAP@12 is the H&M competition's native metric, which lets absolute numbers be sanity-checked
against public leaderboard intuition rather than existing in a vacuum. It is reported for
calibration, not as a target to optimize.

### End-to-end

MAP@12, NDCG@12, and recall@12 for the full pipeline.

> **⚠️ Hard invariant: end-to-end metrics are computed over *every* positive in the window,
> including the ones stage 1 never retrieved. Those score as misses.**
>
> Metrics computed only over retrieved candidates flatter the pipeline by exactly the
> retriever's miss rate. Reporting those as the headline is the single most natural way to
> overstate a two-stage system, and it is invisible in the output — the numbers just look good.
> Enforced by `tests/test_e2e_metrics.py::test_e2e_metrics_count_unretrieved_positives`, and
> checkable by hand: **end-to-end MAP@12 must be strictly below ranking-only MAP@12 for every
> arm.** If they are equal, the accounting is broken.

### Calibration

Downsampling shifts the base rate, so a model's raw output is a score, not a probability.

Under retrieval-induced sampling there is **no fixed downsampling rate**, because how many
negatives a customer receives depends on what the retriever returned. The parametric prior
correction `p_true = p_s / (p_s + (1 − p_s)/w)` therefore does not apply, and **isotonic
regression fit on W9 becomes the primary path** — a monotone step function mapping raw score to
observed purchase rate, fit on validation and applied to test.

`prior_correct` and its round-trip test are retained: the H3 random-negative arm has a genuine
fixed `w`, and comparing the two calibration paths is a useful check that neither is broken.

| Metric | Handling |
|---|---|
| AUC, NDCG, MAP, `recall@K` | Rank-based; unaffected by calibration. No correction needed |
| PR-AUC | Base-rate dependent; only compared across arms on the same candidate set |
| Log-loss, Brier | Reported on isotonic-calibrated scores, labeled as such |
| Reliability curves | Per arm, so calibration quality is visible rather than asserted |

### Evaluation slices

Every metric is reported on three populations, because the aggregate hides the part that
matters:

| Slice | Definition | Why |
|---|---|---|
| **All rows** | — | The headline, and the population §2's criteria are judged on |
| **Cold-start articles** | `art_prior_purchases < 10` | Popularity features are near zero here while text is fully available. **If content wins anywhere, it wins here** — and this is the strongest business argument the project can make. The threshold is set once at M1 from the measured distribution and never retuned |
| **Low-history customers** | few prior purchases | The symmetric weak case: a thin purchase history means a weak customer tower. Bounds the claim |

### Business proxy

Two figures, both **relative**, both computed on calibrated probabilities:

1. **Expected revenue @ top-k** = Σ (price × calibrated p̂) over the model's top-k per customer.
2. **AOV lift ratio** = realized basket value of the model's top-k versus the `pop` baseline's.

> **Reminder (§3): H&M `price` is scaled and anonymized.** Both figures are ratios in relative
> units. This project does not and cannot report AOV in currency.

---

## 9. Inference cost profiling

Measured on the target machine, **per stage separately**, so the latency budget is attributable
rather than a single opaque number.

| Configuration | What it isolates |
|---|---|
| `stage1_customer_tower` | One customer-vector forward pass |
| `stage1_topk_exact` | Brute-force matvec over the 105k × 128 table + top-`K` |
| `stage1_topk_faiss` | The same via a FAISS IVF index — **is the index worth it at 105k?** |
| `stage2_lgbm` / `stage2_mlp` / `stage2_dcn` | Ranker forward pass over `K` candidates |
| `e2e` | Full pipeline, p50 / p95 / throughput |
| `encoder_cold_cpu` / `encoder_onnx_int8` | Encoding a genuinely new article at request time — the only path where the transformer runs online |

Reported with measured `peak_rss_mb` and the cache's measured `nbytes`, so memory claims are
evidence rather than arithmetic. Protocol: 20 warmup batches discarded, 100 measured, single
process, machine otherwise idle. Converted to **$/1M predictions** against one named cloud SKU,
with the instance type and price-lookup date cited inline.

### The claim to confirm

```
105,000 × 128 × 4 bytes  ≈   54 MB   (item vectors, fp32)
105,000 × 384 × 4 bytes  ≈  162 MB   (raw encoder output, fp32)
105,000 × 384 × 1 byte   ≈   40 MB   (int8)
```

If those hold, the **transformer's steady-state serving cost is approximately zero** — it is an
offline batch job feeding a lookup table, not an online forward pass. The cold path matters only
for genuinely new articles, which are a small and predictable trickle.

This reframes the standard "transformers are too expensive to serve" objection honestly: for an
item-level-text problem with a bounded catalog, the expensive thing amortizes to near-nothing.
Whether that changes the *conclusion* depends entirely on whether §2 found any lift to serve.

---

## 10. Repo layout

```
contentsignal/
├── pyproject.toml              # uv, Python 3.11 (system 3.9 is too old)
├── prd.md
├── trd.md
├── README.md
├── CLAUDE.md
├── Makefile
├── conf/
│   ├── data.yaml               # paths, seeds, cohort size, candidate depth K
│   ├── split.yaml              # window boundaries and roles — single source of truth
│   └── model/                  # retriever.yaml, lgbm.yaml, mlp.yaml, dcn.yaml
├── src/contentsignal/
│   ├── data/                   # download.py, to_parquet.py, schema.py
│   ├── splits/                 # temporal.py
│   ├── features/               # base.py, customer.py, article.py, categorical.py,
│   │                           #   cross.py, retrieval.py
│   ├── sampling/               # negatives.py — random negatives, for the H3 arm
│   ├── models/                 # twotower.py, popularity.py,
│   │                           #   lgbm_ranker.py, mlp_ranker.py, dcn_ranker.py
│   ├── retrieval/              # index.py, candidates.py
│   ├── eval/                   # metrics.py, retrieval.py, ranking.py,
│   │                           #   calibration.py, bootstrap.py, business.py
│   ├── serving/                # embedding_cache.py, predictor.py, benchmark.py
│   └── cli.py
├── tests/
│   ├── test_leakage.py         # deletion invariance + retriever window ordering
│   ├── test_splits.py
│   ├── test_retrieval.py       # tower independence, log-Q, recall monotonicity
│   ├── test_e2e_metrics.py     # unretrieved positives count as misses
│   ├── test_sampling.py
│   └── test_calibration.py
├── notebooks/                  # 01_eda.ipynb, 99_results.ipynb
├── reports/
│   ├── results.md
│   ├── metrics/                # git-committed per-run JSON — the record of what was claimed
│   └── figures/
└── artifacts/                  # gitignored: parquet, candidates, vectors, checkpoints
```

---

## 11. Milestones

| | Milestone | Concrete artifact | Gate |
|---|---|---|---|
| **M0** | Environment + data acquisition | `pyproject.toml`, locked deps, three CSVs on disk | `contentsignal --help` runs |
| **M1** | Parquet, EDA, windows with roles | `artifacts/parquet/*`, `01_eda.ipynb` | **`test_splits.py` + `test_leakage.py` green** |
| **M2** | Cohort, positives, all feature groups | Feature tables for W1–W10 | Every builder registered; deletion-invariance green |
| **M3** | Retrievers trained (`pop`, `R1`, `R2`) | Tower checkpoints, item-vector cache | **`test_retrieval.py` green**; log-Q ablation + collapse checks pass |
| **M4** | **H2 answered at the retrieval level** | `recall@K` sweep, cold-start slice, coverage | CIs on Δ`recall@100` and the difference of deltas |
| **M5** | Candidates materialized + digested | `artifacts/candidates/*`, `candidates_manifest.json`, ranker features | Row budget respected; window-ordering test green |
| **M6** | Three rankers on retrieved negatives | Models + metrics JSON | Digest assertion passes on every arm |
| **M7** | **H3** — best ranker on random negatives | Both evaluated end-to-end | Same test rows for both |
| **M8** | Two-stage cost profiling | Per-stage latency table, exact-vs-FAISS, $/1M | Measured `nbytes` and `peak_rss_mb` |
| **M9** | **H1 stage-attribution table**; report | Final `reports/results.md`, `README.md` | `make report` reproduces every number |

**M1 gates everything: no model is trained until the leakage tests pass.**

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| **`recall@100` comes out very low**, making everything downstream noise | The `pop` baseline is free and runs first. If a trained two-tower on four windows can't beat "show the 100 bestsellers," that is a finding and a stop-point at M3 — not a surprise at M9 |
| **Retriever staleness** — 12 weeks by W10, and fashion is seasonal | Measurable: the `recall@K` curve W5 → W10 *is* the decay estimate, and it is reported. Hits all arms equally, so comparisons hold |
| **Retriever collapses to popularity** | log-Q correction, verified by the popularity-ρ diagnostic and a dedicated ablation run with the correction disabled |
| **Embedding collapse** — all vectors nearly identical | Norm and pairwise-cosine spread checks at M3, before anything depends on the vectors |
| **Cold-start slice too thin** for a usable CI | Measured at M1 during EDA. The threshold is set once, from the observed distribution, and registered — never retuned after seeing results |
| **Content retrieval shows no cold-start lift** | Pre-committed to reporting it as the headline finding (§2). The `R1`/`R2`/`R2-A` decomposition makes a null *interpretable* rather than merely disappointing |
| **DCN-v2 loses to LightGBM** | Expected, and pre-registered as such. GBDTs usually win on tabular-dominant problems at this scale; keeping the baseline is what makes the loss diagnostic instead of embarrassing |
| **Popularity baseline wins outright** | Report it. It is a real and frequently observed outcome on implicit-feedback data, and concealing it would invalidate everything else |
| **8 GB OOM** | DuckDB for aggregation, Polars over pandas, `float32` throughout, 47 ranker columns instead of 80, stage 1 on positives only, MLP/DCN stream minibatches from Parquet, `max_bin=63` for LightGBM, raw CSVs deleted post-conversion |
| **MPS instability / silent numerical issues** | Documented CPU fallback; parity check that MPS and CPU vectors agree within tolerance on a fixed sample |
| **Kaggle download blocked** | Requires accepting the competition rules on Kaggle **and** placing `kaggle.json` at `~/.kaggle/` with mode 600 — currently absent on this machine. M0 is blocked until then |

---

## 13. Non-goals

Stated plainly so scope creep is visible when it happens:

- **Article images.** ~30 GB, out of budget, orthogonal to the question.
- **Online A/B testing.** No live traffic exists. All business metrics are offline proxies and
  are labeled as such.
- **Serving infrastructure.** §9 benchmarks a local predictor; it does not build an API.
- **Candidate-aware towers.** No cross-attention or candidate-conditioned pooling inside stage 1.
  Both towers must remain independently precomputable, or retrieval stops working and the
  cost analysis stops describing a deployable system (§1). Note that stage 2 *is*
  candidate-aware — that is what the `cross` group and DCN-v2 are for. The constraint applies to
  the retriever alone.
- **Beating the Kaggle leaderboard.** MAP@12 is reported for calibration against known results,
  not as a target.
- **Multi-objective or diversity-aware ranking.** One label, one objective.
- **Dollar-denominated AOV.** Impossible with scaled prices (§3).

---

## Appendix — environment baseline

Captured 2026-07-27 on the target machine:

| Check | State |
|---|---|
| Repo | `/Users/arsalon/Desktop/contentsignal`, branch `main` |
| Python | system 3.9.6 — **project pins 3.11 via `uv`** (`/opt/homebrew/bin/uv` present) |
| torch / lightgbm / transformers / polars / duckdb | installed at M0 per `uv.lock` |
| RAM / CPU | 8 GB, 8 cores, Apple Silicon (MPS) |
| Disk free | ~124 GB |
| Kaggle credentials | `~/.kaggle` **absent** — blocks M0 |
