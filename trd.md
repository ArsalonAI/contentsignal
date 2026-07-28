# ContentSignal — TRD

Technical design for the experiment specified in [`prd.md`](./prd.md). The PRD defines
*what* and *why*; this document defines *how*, at a level where implementation is
mechanical: literal schemas, function signatures, algorithms, resource budgets, and test
assertions.

| | |
|---|---|
| Status | Draft — pre-implementation |
| Companion | `prd.md` (approved) |
| Python | 3.11 via `uv` |
| Hard ceiling | 8 GB RAM, 8 cores, MPS, no discrete GPU |

---

## 1. Traceability

Every PRD section maps to at least one section here. Nothing is allowed to drop silently.

| PRD § | Requirement | Implemented in |
|---|---|---|
| §1 | Pre-registered criterion, null-result commitment | §10 (bootstrap), §11 (immutable metric artifacts), §16 |
| §2 | Data, ingest, dtypes, bounded subset | §4, §13 (`ingest`), §14 |
| §3 | Label definition | §3 (row-set construction) |
| §4 | Temporal splits, `as_of` contract, leakage | §3, §5 (`features/base.py`), §15 |
| §5 | Negative sampling, prior correction | §7, §10.3 |
| §6 | Feature groups incl. customer-side text | §6 |
| §7 | Arm grid, contrastive fine-tune | §8, §9 |
| §7 | Two-tower architecture, sampled softmax, log-Q | §9b, §15 (`test_twotower.py`) |
| §8 | Metrics, bootstrap, slices, business proxy | §10 |
| §9 | Inference cost profiling | §12 (`bench`), §4 (mmap-able artifacts) |
| §10 | Repo layout | §5, §13 |
| §11 | Milestones | §16 |
| §12 | Risks | §14 (memory), §15 (leakage), §8 (pair bias) |
| §13 | Non-goals | — (constraints on §8, §9) |

---

## 2. Environment & dependencies

System Python is 3.9.6 and cannot be used. The project pins **3.11** via `uv`
(`/opt/homebrew/bin/uv` is present).

```toml
# pyproject.toml — requires-python = ">=3.11,<3.12"
dependencies = [
  "polars>=1.0",           # in-memory frames
  "duckdb>=1.0",           # out-of-core aggregation
  "pyarrow>=16",           # parquet IO
  "numpy>=1.26",
  "lightgbm>=4.3",
  "scikit-learn>=1.5",     # logreg, isotonic, TruncatedSVD
  "torch>=2.3",            # MPS backend
  "transformers>=4.40",
  "sentence-transformers>=3.0",
  "mlflow>=2.14",
  "pydantic>=2.7",         # config validation
  "pyyaml>=6.0",
  "typer>=0.12",           # CLI
  "scipy>=1.13",           # spearmanr, alias sampling support
]

[project.optional-dependencies]
bench = ["onnxruntime>=1.18", "optimum>=1.20"]   # §12 only
dev   = ["pytest>=8.0", "pytest-xdist", "ruff", "mypy"]
```

`pandas` is **not** a dependency of the pipeline. It arrives transitively via MLflow and is
permitted only in notebooks and report generation.

### Blocker

M0 cannot start until: (a) the H&M competition rules are accepted on Kaggle, and (b)
`~/.kaggle/kaggle.json` exists with mode `600`. Neither is true on this machine as of
2026-07-27. `contentsignal ingest` fails fast with a specific message if either is missing.

---

## 3. Windowing and row-set construction

### 3.1 Window geometry

Window length **L = 14 days**, chosen so train windows are directly comparable to val and
test. All bounds inclusive.

| Name | Split | Start | End |
|---|---|---|---|
| `train_w1` | train | 2020-05-06 | 2020-05-19 |
| `train_w2` | train | 2020-05-20 | 2020-06-02 |
| `train_w3` | train | 2020-06-03 | 2020-06-16 |
| `train_w4` | train | 2020-06-17 | 2020-06-30 |
| `train_w5` | train | 2020-07-01 | 2020-07-14 |
| `train_w6` | train | 2020-07-15 | 2020-07-28 |
| `train_w7` | train | 2020-07-29 | 2020-08-11 |
| `train_w8` | train | 2020-08-12 | 2020-08-25 |
| `val` | val | 2020-08-26 | 2020-09-08 |
| `test` | test | 2020-09-09 | 2020-09-22 |

Contiguous, non-overlapping, ending on the dataset's last day. Feature history for every
window reaches back to 2018-09-20 and is always truncated at `W.start`.

```python
# splits/temporal.py
@dataclass(frozen=True)
class Window:
    name: str
    split: Literal["train", "val", "test"]
    start: date              # inclusive
    end: date                # inclusive

    @property
    def as_of(self) -> date:
        """Exclusive feature cutoff. Features may read t_dat < as_of, never >=."""
        return self.start

def load_windows(cfg: SplitConfig) -> list[Window]: ...
def assert_contiguous_non_overlapping(ws: Sequence[Window]) -> None: ...
```

`conf/split.yaml` is the single source of truth for these dates. No module hardcodes them.

### 3.2 Cohort

```
cohort = sample(
    customers with >= 1 transaction in [2020-03-24, 2020-09-22],
    n = cohort.size,
    seed = cohort.seed,
)
```

Drawn **once**, persisted, and reused by every window and every arm.

### 3.3 Row-set construction

For each window *W*:

```
eligible  = cohort ∩ {customers with >= 1 purchase in W}
positives = distinct (customer_idx, article_id) purchased in W by eligible customers
negatives = sample_negatives(positives, ratio=10, ...)        # §7
rows_W    = positives ∪ negatives
```

Multiple purchases of the same article by the same customer in *W* collapse to **one row**.
Purchase count is deliberately **not** carried as a feature — it is a function of the label
window and would leak.

> **Selection effect, stated rather than buried.** Rows exist only for customers who
> transacted during *W*. The model is therefore evaluated **conditional on the customer
> transacting**. This is standard for this task and is what keeps per-customer ranking
> metrics well defined — a customer with zero positives has no meaningful `precision@k`.
> But it means the reported numbers do not describe performance on the full customer base,
> and `reports/results.md` must say so.

### 3.4 Scale, and the row budget

Estimates below are approximate and will be replaced by measured counts at M2.

| Quantity | Estimate |
|---|---|
| Transactions per 14-day window (all customers) | ~610k |
| Cohort share of transactions | ~cohort.size / 1.37M |
| Positives per window, cohort of 150k | ~65k |
| Train positives, 8 windows | ~520k |
| Train rows at 1:10 | **~5.7M** |

Because these estimates carry real uncertainty, **the binding constraint is a row budget,
not a cohort size**:

```yaml
cohort:
  size: 150_000          # initial guess
  target_train_rows: 5_000_000
  seed: 17
```

`contentsignal sample` reports actual row counts and **fails if train rows exceed
`target_train_rows` × 1.2**, with the message naming the cohort size that would fit. The
cohort size is then adjusted once, before any model is trained. This avoids discovering the
memory ceiling halfway through M6.

---

## 4. Storage contracts

Root: `artifacts/` (gitignored). All Parquet is zstd-compressed.

### 4.1 `artifacts/parquet/transactions.parquet`

Partitioned by `year_month` for predicate pushdown on `as_of` filters.

| Column | Type | Note |
|---|---|---|
| `t_dat` | `date32` | |
| `customer_idx` | `int32` | dense index; see 4.4 |
| `article_id` | `int32` | |
| `price` | `float32` | scaled units, **not currency** (PRD §2) |
| `sales_channel_id` | `int8` | 1 in-store, 2 online |

~31.8M rows, ~350 MB on disk.

### 4.2 `artifacts/parquet/articles.parquet`

All original columns, with `*_name` fields as `categorical` and `detail_desc` as `utf8`
(nullable). Plus two derived columns:

| Column | Type | Note |
|---|---|---|
| `text_a` | `utf8` | Full concat, PRD §6 Text-A |
| `text_b` | `utf8` | `detail_desc` only, PRD §6 Text-B. Null → empty string, with `text_b_is_null` flag |

~105k rows, ~20 MB.

### 4.3 `artifacts/parquet/customers.parquet`

Original columns; `age` nullable `float32` with a companion `age_is_null` boolean;
`postal_code` hashed to `int32`. ~1.37M rows, ~25 MB.

### 4.4 `artifacts/parquet/customer_index.parquet`

`customer_id` (`utf8`, the original 64-char hex) → `customer_idx` (`int32`). Persisted
because the hex IDs cost ~90 MB in memory and the mapping must be stable across runs.

### 4.5 `artifacts/rows/{window}.parquet`

The materialized labelled row sets — **written once, then immutable**.

| Column | Type |
|---|---|
| `customer_idx` | `int32` |
| `article_id` | `int32` |
| `y` | `int8` |
| `window` | `categorical` |

A `rows_manifest.json` alongside records the sampler seed, ratio, per-window counts, and a
SHA-256 of each file. **Arms 1–8 assert this digest before training.** This is what makes
"identical rows" (PRD §5) enforceable rather than aspirational.

> **Arms 9/10 are the bounded exception.** They train with in-batch sampled softmax over
> positives only, so they never read these files for training — but they **assert the same
> digest before evaluation**, since they are scored on the identical test rows (§9b.4).

### 4.6 `artifacts/features/{group}/{window}.parquet`

One file per feature group per window, keyed by `customer_idx` and/or `article_id`, joined
at train time. Groups: `customer`, `article`, `categorical`, `cross`, `text_item`,
`text_customer`.

### 4.7 Embedding artifacts

Deliberately **not** Parquet, so the serving benchmark (§12) can `mmap` them:

```
artifacts/emb/{variant}_{source}/vectors.npy   float32 [n_articles, 384]  (C-contiguous)
artifacts/emb/{variant}_{source}/ids.npy       int32   [n_articles]       (sorted ascending)
artifacts/emb/{variant}_{source}/svd32.npy     float32 [n_articles, 32]
artifacts/emb/{variant}_{source}/svd.joblib    fitted TruncatedSVD (fit on TRAIN articles only)
```

`variant ∈ {a, b}` (PRD §6 text variants), `source ∈ {frozen, contrastive}`.

> **Leakage note.** The SVD is fitted on articles appearing in **train windows only**, then
> applied to all articles. Fitting on the full catalog would let test-window articles
> influence the projection basis.

### 4.8 `artifacts/taste/{window}.npz`

`customer_idx` (`int32[n]`) and `taste` (`float32[n, 384]`) for eligible customers in that
window. ~80k × 384 × 4 ≈ 123 MB per window; written per window and loaded one at a time.

---

## 5. Module contracts

### 5.1 The `as_of` contract — the load-bearing interface

```python
# features/base.py

def history(txns: pl.LazyFrame, *, as_of: date) -> pl.LazyFrame:
    """The ONLY sanctioned way to read transactions inside a feature builder.

    Returns strictly-before-cutoff rows. Every builder calls this first; no builder
    touches the raw LazyFrame. Enforced by review and by test_leakage.py.
    """
    return txns.filter(pl.col("t_dat") < as_of)


class FeatureBuilder(Protocol):
    name: str                       # -> artifacts/features/{name}/
    columns: tuple[str, ...]        # declared output columns, asserted on build

    def build(
        self,
        txns: pl.LazyFrame,
        *,
        as_of: date,
        entities: pl.DataFrame,     # the keys to produce rows for
    ) -> pl.DataFrame: ...
```

`as_of` is **keyword-only and required**. There is no default and no overload without it —
the signature itself makes the leak unwritable rather than merely discouraged.

### 5.2 Splits — `splits/temporal.py`

As in §3.1, plus:

```python
def eligible_customers(txns: pl.LazyFrame, w: Window, cohort: pl.Series) -> pl.Series: ...
def positives(txns: pl.LazyFrame, w: Window, eligible: pl.Series) -> pl.DataFrame: ...
```

### 5.3 Sampling — `sampling/negatives.py`

```python
@dataclass(frozen=True)
class SamplerConfig:
    ratio: int = 10
    pop_exponent: float = 0.75
    pop_lookback_weeks: int = 12
    seed: int = 17

def candidate_pool(txns: pl.LazyFrame, *, as_of: date, cfg: SamplerConfig) -> pl.DataFrame:
    """article_id + sampling weight. Only articles with >=1 prior transaction."""

def sample_negatives(
    positives: pl.DataFrame,        # customer_idx, article_id
    *,
    pool: pl.DataFrame,             # article_id, weight
    cfg: SamplerConfig,
    window_name: str,
) -> pl.DataFrame: ...              # customer_idx, article_id, y=0
```

### 5.4 Text — `features/text.py`

```python
def build_article_text(articles: pl.DataFrame, variant: Literal["a", "b"]) -> pl.DataFrame: ...

def encode_articles(
    texts: pl.DataFrame, *, model: SentenceTransformer,
    batch_size: int = 256, device: str = "mps",
) -> np.ndarray: ...                                        # [n, 384] float32

def taste_vectors(
    txns: pl.LazyFrame, *, as_of: date, cache: EmbeddingCache,
    customers: np.ndarray, recent_k: int = 10,
) -> TasteVectors: ...

def similarity_features(
    taste: TasteVectors, cache: EmbeddingCache, rows: pl.DataFrame,
) -> pl.DataFrame: ...              # sim_taste_cos, sim_last10_max/mean, sim_taste_pct_rank
```

`TasteVectors` holds the mean vector, the last-*k* matrix, and a customer index, all
L2-normalized at construction so similarity is a single matmul.

### 5.5 Arms — `models/base.py`

A common protocol so arms are interchangeable and the training CLI is arm-agnostic:

```python
class Arm(Protocol):
    name: str
    feature_groups: tuple[str, ...]

    def fit(self, X: pl.DataFrame, y: np.ndarray, *,
            valid: tuple[pl.DataFrame, np.ndarray] | None) -> None: ...
    def predict(self, X: pl.DataFrame) -> np.ndarray: ...   # SAMPLED-distribution probs
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Arm": ...
```

`predict` returns probabilities on the **sampled** distribution. Prior correction (§10.3) is
applied by the evaluator, never inside an arm — so the correction is applied exactly once
and identically everywhere.

### 5.6 Serving — `serving/embedding_cache.py`

```python
class EmbeddingCache:
    def __init__(self, dir: Path, *, mmap: bool = True) -> None: ...
    def take(self, article_ids: np.ndarray) -> np.ndarray: ...   # vectorized gather
    def __getitem__(self, article_id: int) -> np.ndarray: ...
    @property
    def nbytes(self) -> int: ...
```

Lookup is `np.searchsorted` on the sorted `ids.npy` — O(log n), no dict, mmap-friendly.

---

## 6. Feature specification

`as_of` column: **Y** = reads transactions and must respect the cutoff; **—** = static.

### 6.1 Customer (`features/customer.py`, group `customer`)

| Column | Type | `as_of` | Definition |
|---|---|---|---|
| `cust_age` | `float32` | — | null → NaN, LightGBM handles natively |
| `cust_age_is_null` | `int8` | — | |
| `cust_club_status` | `cat` | — | |
| `cust_news_freq` | `cat` | — | |
| `cust_fn`, `cust_active` | `int8` | — | |
| `cust_tenure_days` | `int32` | Y | `as_of` − first purchase |
| `cust_txn_{1,4,12}w` | `int32` | Y | transaction counts in trailing windows |
| `cust_distinct_articles_12w` | `int32` | Y | |
| `cust_price_{mean,median,std}` | `float32` | Y | over trailing 12w |
| `cust_online_share` | `float32` | Y | `sales_channel_id == 2` share, 12w |
| `cust_days_since_last` | `int32` | Y | |

### 6.2 Article (`features/article.py`, group `article`)

| Column | Type | `as_of` | Definition |
|---|---|---|---|
| `art_pop_{1,4,12}w` | `int32` | Y | purchase counts in trailing windows |
| `art_distinct_buyers_12w` | `int32` | Y | |
| `art_price_mean` | `float32` | Y | |
| `art_price_pct_in_group` | `float32` | Y | percentile within `product_group_name` |
| `art_days_since_first_sale` | `int32` | Y | |
| `art_days_since_last_sale` | `int32` | Y | |
| `art_online_share` | `float32` | Y | |

### 6.3 Categorical (`features/categorical.py`, group `categorical`)

Eleven columns as native LightGBM categoricals: `product_type_name`, `product_group_name`,
`colour_group_name`, `department_no`, `index_name`, `index_group_name`, `section_name`,
`garment_group_name`, `graphical_appearance_name`, `perceived_colour_value_name`,
`perceived_colour_master_name`.

> **PRD §6 non-negotiable: the tabular baseline (arm 3) receives all eleven.** Withholding
> them would credit the encoder with information the baseline was never given.

### 6.4 Cross (`features/cross.py`, group `cross`)

| Column | Type | `as_of` | Definition |
|---|---|---|---|
| `x_prior_in_product_group` | `int32` | Y | customer's prior purchases in this article's group |
| `x_prior_in_department` | `int32` | Y | |
| `x_prior_in_colour_group` | `int32` | Y | |
| `x_bought_product_code_before` | `int8` | Y | same garment, any colourway/size |
| `x_price_vs_cust_mean` | `float32` | Y | `art_price_mean` − `cust_price_mean` |
| `x_age_indexgroup_affinity` | `float32` | Y | empirical rate for the customer's age bucket × `index_group`, computed on history only |

### 6.5 Text — item side (group `text_item`)

| Column | Type | Definition |
|---|---|---|
| `art_emb_{0..31}` | `float32` | SVD-32 of the 384-dim article embedding (§4.7) |

### 6.6 Text — customer side (group `text_customer`)

| Column | Type | `as_of` | Definition |
|---|---|---|---|
| `sim_taste_cos` | `float32` | Y | cosine(taste vector, article embedding) |
| `sim_last10_max` | `float32` | Y | max cosine to the 10 most recent purchases |
| `sim_last10_mean` | `float32` | Y | mean of the same |
| `sim_taste_pct_rank` | `float32` | Y | percentile of `sim_taste_cos` within this customer's candidate set |
| `cust_taste_{0..31}` | `float32` | Y | **optional, off by default** — see below |

> **Why the raw taste dimensions default off.** Trees cannot compute a dot product; giving
> them 32 taste dimensions and 32 article dimensions and hoping for an interaction is
> exactly the weakness that motivated this feature group. `sim_taste_cos` *is* that dot
> product, pre-computed. The raw dimensions are retained behind
> `features.text_customer.include_raw_dims` as an ablation, and enabling them costs ~640 MB
> at 5M rows (§14). The four scalars are mandatory; the 32 dimensions are a nice-to-have.

### 6.7 Cold start (group `article`, used for slicing)

| Column | Type | `as_of` | Definition |
|---|---|---|---|
| `art_prior_purchases` | `int32` | Y | total purchases before `as_of` |
| `art_is_cold` | `int8` | Y | `art_prior_purchases < eval.cold_start_threshold` (default 10) |

Present as features and as the slicing key for §10.4.

**Column totals**: 18 customer + 9 article/cold + 11 categorical + 6 cross + 32 item-text
+ 4 customer-text = **80 columns** default; 112 with raw taste dimensions enabled.

---

## 7. Negative sampling

```
function sample_negatives(positives, pool, cfg, W):
    # pool: articles with >=1 transaction before W.start, weight = pop_12w ** 0.75
    alias = AliasTable(pool.article_id, pool.weight, seed=cfg.seed ^ hash(W.name))

    out = []
    for (c, seen) in positives.group_by(customer_idx):
        need   = cfg.ratio * len(seen)
        drawn  = set()
        guard  = 0
        while len(drawn) < need:
            batch = alias.draw(need - len(drawn) + 8)        # slack for rejections
            for a in batch:
                if a not in seen and a not in drawn:
                    drawn.add(a)
                    if len(drawn) == need: break
            guard += 1
            assert guard < 64, f"pathological rejection rate for customer {c}"
        out += [(c, a, 0) for a in drawn]
    return out
```

Notes:

- **Alias method**, O(1) per draw. The pool is ~40–60k articles; building the table per
  window is negligible.
- **Rejection is rare by construction**: `|seen|` is typically 2–5 against a pool of tens of
  thousands, so the expected rejection rate is well under 1%. The `guard` exists to convert
  a pathological case into a loud failure rather than a hang.
- **`drawn` is a set**, so a customer never receives the same negative twice — duplicate
  negatives would silently reweight the loss.
- **Seed is per-window** (`cfg.seed ^ hash(W.name)`) so windows are independent but the
  whole set is reproducible from one root seed.
- Output is written to `artifacts/rows/{window}.parquet` **once** and digested (§4.5).

**Uniform-sampling sensitivity check** (PRD §5): the same function with
`pop_exponent = 0.0`, written to `artifacts/rows_uniform/`, run for arms 3 and 7b only,
reported in an appendix.

---

## 8. Contrastive fine-tuning

### 8.1 Pair construction

```
pairs = []
for c in cohort:
    hist = purchases by c in [val_start - 12w, val_start)      # STRICTLY before val
    arts = distinct(hist.article_id)
    if len(arts) < 2: continue
    for (a1, a2) in sample_without_replacement(combinations(arts, 2), k <= 5):
        pairs.append((text[a1], text[a2]))

pairs = enforce_article_cap(pairs, max_per_article=50)
pairs = subsample(pairs, n=300_000, seed=cfg.seed)
```

| Parameter | Value | Rationale |
|---|---|---|
| Lookback | 12 weeks before `val_start` (2020-08-26) | Recent co-purchase, and strictly outside val/test |
| Max pairs per customer | **5** | Heavy buyers would otherwise dominate |
| Max pairs per article | **50** | Bestsellers would otherwise dominate, pushing the encoder back toward a popularity proxy — the exact failure this objective exists to avoid |
| Target pairs | 300k | Fits the runtime budget at 1–2 epochs |

### 8.2 Training

| Setting | Value |
|---|---|
| Base model | `sentence-transformers/all-MiniLM-L6-v2` |
| Loss | `MultipleNegativesRankingLoss` (InfoNCE, in-batch negatives), scale 20 |
| Batch | 64 pairs = 128 sequences |
| `max_seq_length` | 64 tokens |
| Optimizer | AdamW, lr 2e-5, 10% linear warmup |
| Epochs | 1–2, selected on val (§8.4) |
| Device | `mps`, with `--device cpu` fallback |
| Seeds | 3 |

Run separately for Text-A and Text-B, producing four embedding artifacts total with the two
frozen baselines.

> **The efficiency property (PRD §7).** Each training example is a *pair of article texts*,
> not a (customer, article) row. With ~105k unique articles and per-article capping, each
> string is encoded a bounded number of times per epoch instead of ~38×. At 300k pairs /
> batch 64 ≈ 4,700 steps/epoch, this is **~6–10 min/epoch on MPS**.

### 8.3 Mandatory diagnostics

Reported whatever they show; suppressing them would defeat the point.

1. **Popularity correlation.** Spearman ρ between the text-only arm's (arm 8) predictions
   and `log(art_pop_12w + 1)` on the test window. High ρ means the text signal is largely a
   popularity restatement. Logged to MLflow as `diag/spearman_pop`.
2. **Nearest neighbors.** For 20 fixed articles (sampled across `index_group`), the top-10
   cosine neighbors before and after fine-tuning, dumped to
   `reports/figures/nn_{variant}_{source}.md`. The cheapest possible check that the space
   learned something fashion-shaped rather than collapsing.
3. **Embedding collapse check.** Mean pairwise cosine and the rank of the embedding matrix.
   Contrastive training with in-batch negatives can collapse; a mean cosine above ~0.9 or a
   sharp rank drop fails the run.

### 8.4 Leakage guards

- All pairs drawn strictly from `t_dat < 2020-08-26`. Asserted in `tests/test_leakage.py`.
- Epoch selection uses **val** only; test is untouched until §16 M9.
- The SVD projection is fitted on train-window articles only (§4.7).

---

## 9. Arm grid and run matrix

### 9.1 Arms

| # | Arm | Feature groups | Embedding source |
|---|---|---|---|
| 1 | `popularity` | `art_pop_12w` only | — |
| 2 | `logreg_tab` | customer, article, categorical (one-hot), cross | — |
| 3 | **`lgbm_tab`** | customer, article, categorical, cross | — |
| 4a | `lgbm_frozen_item` | 3 + `text_item` | pretrained |
| 4b | `lgbm_frozen_pers` | 3 + `text_item` + `text_customer` | pretrained |
| 7a | `lgbm_ft_item` | 3 + `text_item` | contrastive |
| 7b | **`lgbm_ft_pers`** | 3 + `text_item` + `text_customer` | contrastive |
| 8 | `text_only` | `text_item` + `text_customer` | contrastive |

### 9.2 Deltas

| Delta | Isolates |
|---|---|
| **7b − 3** | **Headline.** Total lift from product content |
| 7b − 7a | Personalization (taste vector + similarity features) |
| 7b − 4b | Fine-tuning. If ≈ 0, a frozen encoder sufficed — changes the §12 cost story |

### 9.3 Run matrix

| Arms | Text variants | Seeds | Runs |
|---|---|---|---|
| 1 | — | 1 | 1 |
| 2, 3 | — | 3 | 6 |
| 4a, 4b, 7a, 7b | A, B | 3 | 24 |
| 8 | A, B | 3 | 6 |
| Sensitivity: SVD 64 and 384 on 7b | B | 3 | 6 |
| Sensitivity: uniform negatives on 3, 7b | B | 1 | 2 |
| Sensitivity: raw taste dims on 7b | B | 3 | 3 |
| | | | **48** |

Plus **13** two-tower runs (§9b.6) → **61 total**.

SVD dim is fixed at **32** for the main grid; 64 and raw-384 appear only in the sensitivity
rows. This is the single largest lever on total runtime and is deliberately bounded.

### 9.4 LightGBM configuration

```yaml
# conf/model/lgbm.yaml
objective: binary
metric: [auc, binary_logloss]
learning_rate: 0.05
num_leaves: 127
min_data_in_leaf: 200
feature_fraction: 0.8
bagging_fraction: 0.8
bagging_freq: 1
max_bin: 63              # memory-critical, see §14
num_threads: 7           # leave one core for the OS
num_boost_round: 2000
early_stopping_round: 100    # on val AUC
```

Hyperparameters are tuned **on val, for arm 3 only**, then frozen and reused by every
LightGBM arm. Tuning each arm separately would confound "better features" with "more tuning
budget" — the delta must be attributable to features alone.

---

## 9b. Two-tower neural ranker (arms 9, 10)

### 9b.1 Architecture

```python
# models/twotower.py

class ItemTower(nn.Module):
    """text → MiniLM → proj ⊕ categorical embeddings ⊕ numerics → MLP → 128, L2-normed."""
    def __init__(self, encoder: SentenceTransformer, *, freeze_encoder: bool,
                 cat_cardinalities: dict[str, int], n_numeric: int, out_dim: int = 128): ...
    def forward(self, text_ids, text_mask, cats, nums) -> Tensor: ...   # [B, 128]

class CustomerTower(nn.Module):
    """history (attention-pooled item IDs) ⊕ categorical ⊕ numerics → MLP → 128, L2-normed."""
    def __init__(self, n_items: int, id_dim: int = 64, hist_len: int = 20,
                 cat_cardinalities: dict[str, int] = ..., n_numeric: int = ...,
                 out_dim: int = 128): ...
    def forward(self, hist_ids, hist_mask, cats, nums) -> Tensor: ...   # [B, 128]

class TwoTower(nn.Module):
    def score(self, cust_vec, item_vec) -> Tensor:     # temperature-scaled dot product
        return (cust_vec * item_vec).sum(-1) / self.temperature
```

| Component | Spec |
|---|---|
| Item-ID embedding table | 105k × 64 (~27 MB) |
| History length | last 20 purchases, right-padded, masked |
| History pooling | **self-attentive with a learned query vector** — `softmax(qᵀ tanh(W·h)) · h` |
| Tower MLPs | [512, 256] → 128, GELU, dropout 0.1, L2-normalized output |
| Categorical embeddings | 8–16 d per field |
| Temperature | learned scalar, init 0.05 |

> **Self-attentive, never candidate-aware.** A candidate-conditioned attention would score
> better, but the customer vector would then depend on the item, and neither tower could be
> precomputed. That breaks retrieval and voids the §12 serving-cost analysis. Enforced by the
> signature: `CustomerTower.forward` takes **no item argument**. Asserted in
> `tests/test_twotower.py::test_customer_tower_is_item_independent`.

### 9b.2 Training

```python
def sampled_softmax_loss(
    cust: Tensor, item: Tensor, item_ids: Tensor,
    *, log_q: Tensor, temperature: Tensor,
) -> Tensor:
    """In-batch sampled softmax with log-Q correction (Yi et al., RecSys 2019).

    logits[i][j] = cust[i] · item[j] / T  −  log_q[item_ids[j]]

    Without the log-Q term, in-batch negatives are drawn proportional to item
    popularity, so the model is rewarded for demoting popular items and drifts
    toward an inverted-popularity ranker. Targets are the diagonal.
    """
```

| Setting | Value |
|---|---|
| Objective | In-batch sampled softmax, log-Q corrected |
| `log_q` estimate | Streaming item-frequency counter over train positives, `log(count / total)` |
| Batch | 512 positives — **batch size is the negative count** |
| Optimizer | AdamW, discriminative LRs: **1e-3 towers, 2e-5 encoder** |
| Schedule | 10% linear warmup, cosine decay |
| Epochs | 2, val-selected by AUC on the val rows |
| Device | `mps`, `--device cpu` fallback |
| Seeds | 3 |

**Training consumes positives only** (~520k), not the 5.7M sampled rows — in-batch negatives
are free. At batch 512 that is ~1,000 steps/epoch, **≈4–6 min/epoch on MPS**, making these
arms cheaper than the LightGBM arms.

**Batch construction**: articles repeat across rows, so each batch dedupes article IDs,
encodes unique texts once, and gathers — the same efficiency property as §8. Duplicate items
within a batch are masked out of the negative set, since an item cannot be its own negative.

### 9b.3 Arms

| # | Arm | Encoder | Trained |
|---|---|---|---|
| 9 | `twotower_frozen` | frozen pretrained | towers only — article embeddings precomputed, so no encoder forward pass |
| 10 | **`twotower_e2e`** | trained jointly | towers + encoder end-to-end |

### 9b.4 Evaluation and the comparability boundary

Arms 9/10 train on a different negative distribution than arms 1–8 (PRD §5, §7). Handling:

| Metric class | Comparable? | Handling |
|---|---|---|
| AUC, PR-AUC, precision@k, MAP@12, NDCG | **Yes** | Scored on the identical test rows; the score is a dot product computable for any pair |
| Log-loss, Brier, reliability | Not directly | Isotonic fit on **val** rows via `eval/calibration.py:fit_isotonic`, then applied to test. Prior correction does not apply — there is no fixed downsampling rate `w` for in-batch negatives |
| Business proxy | After recalibration only | Needs probabilities, so it runs on isotonic-calibrated scores |

The `10 − 7b` delta reflects **architecture and training objective together**. Reported with
that caveat inline in `reports/results.md`, not as a clean neural-vs-GBDT isolation.

### 9b.5 Diagnostics

- **Popularity ρ** — Spearman between arm-10 scores and `log(art_pop_12w + 1)`. A two-tower
  that has collapsed to popularity re-ranking fails here, and log-Q is the first thing to
  check.
- **log-Q ablation** — one run with the correction disabled, to demonstrate its effect rather
  than assert it.
- **Tower vector norms and pairwise cosine spread** — the same collapse check as §8.3.

### 9b.6 Run matrix addition

| Arms | Text variants | Seeds | Runs |
|---|---|---|---|
| 9, 10 | A, B | 3 | 12 |
| Sensitivity: log-Q disabled on arm 10 | B | 1 | 1 |
| | | | **13** |

Total across the project: **48 + 13 = 61 runs**.

---

## 10. Evaluation

### 10.1 Metric signatures

```python
# eval/metrics.py
def auc(y: np.ndarray, p: np.ndarray) -> float: ...
def pr_auc(y, p) -> float: ...
def log_loss(y, p) -> float: ...
def brier(y, p) -> float: ...

# eval/ranking.py — per customer, then averaged
def precision_at_k(y, p, customer: np.ndarray, k: int) -> float: ...
def recall_at_k(y, p, customer, k: int) -> float: ...
def map_at_k(y, p, customer, k: int = 12) -> float: ...
def ndcg_at_k(y, p, customer, k: int = 12) -> float: ...
```

### 10.2 Bootstrap

```python
# eval/bootstrap.py
@dataclass(frozen=True)
class BootstrapResult:
    point: float
    lo: float                # 2.5th percentile
    hi: float                # 97.5th percentile
    p_gt_zero: float

def bootstrap_delta(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
    customer: np.ndarray, *, metric: Callable, n: int = 1000, seed: int,
) -> BootstrapResult: ...

def bootstrap_delta_of_deltas(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
    customer: np.ndarray, slice_mask: np.ndarray,
    *, metric: Callable, n: int = 1000, seed: int,
) -> BootstrapResult:
    """The H2 statistic: delta on the slice minus delta on all rows (PRD §1).

    Both deltas are computed from the SAME customer resample on each iteration.
    Customers appear in both populations — a customer can have cold-start and
    established candidates in one basket — so two independent bootstraps would
    treat correlated quantities as independent and understate the CI.
    """
```

**Resamples customer IDs, then gathers all rows belonging to the drawn customers.** Rows
within a customer share history features and basket composition; row-level resampling would
produce CIs narrow enough to let noise clear the ΔAUC ≥ 0.005 bar (PRD §1).

Implementation: precompute a customer → row-index offset array once, so each resample is a
gather rather than a filter. 1000 resamples over ~600k test rows runs in well under a
minute per arm.

### 10.3 Calibration

```python
# eval/calibration.py
def prior_correct(p_sampled: np.ndarray, *, w: float) -> np.ndarray:
    """w = negative downsampling rate. PRD §5."""
    return p_sampled / (p_sampled + (1.0 - p_sampled) / w)

def fit_isotonic(p_val: np.ndarray, y_val: np.ndarray) -> IsotonicRegression: ...
def reliability_curve(y, p, *, bins: int = 20) -> tuple[np.ndarray, np.ndarray]: ...
```

Applied by the evaluator, once, never inside an arm. Log-loss is reported **twice** —
sampled and prior-corrected, both explicitly labelled. AUC and the ranking metrics are
computed on uncorrected scores, since the correction is monotone and cannot change ranks.

### 10.4 Slices

Every metric is computed on three populations (PRD §8):

| Slice | Predicate |
|---|---|
| `all` | — |
| `cold_start` | `art_is_cold == 1` (`art_prior_purchases < 10`) |
| `low_history` | customer has `< 5` prior purchases at `as_of` |

The `cold_start` slice is where text should win if it wins anywhere — popularity features
are near-zero there while text is fully available — and it carries the project's strongest
business argument.

### 10.5 Business proxy

```python
# eval/business.py
def expected_revenue_at_k(price, p_corrected, customer, k) -> float: ...
def aov_lift_ratio(price, p_model, p_baseline, y, customer, k) -> float: ...
```

Computed on **prior-corrected** probabilities only. Reported as ratios in relative units,
with the scaled-price caveat (PRD §2) printed inline in the results table, not footnoted.

---

## 11. Experiment tracking

**MLflow, local file backend** at `artifacts/mlruns` — no server, no network.

| | |
|---|---|
| Experiment | `contentsignal` |
| Run name | `{arm}_{variant}_{svd}_s{seed}` e.g. `lgbm_ft_pers_b_32_s1` |
| Params | `arm`, `text_variant`, `embedding_source`, `svd_dim`, `seed`, `git_sha`, `config_sha256`, `rows_digest`, `cohort_size`, `sampler_ratio` |
| Metrics | every metric × every slice, as `{slice}/{metric}` |
| Artifacts | reliability curve, feature importance, NN dumps, per-run config snapshot |

### Dual-write to git

Every run **also** writes `reports/metrics/{run_name}.json`, which **is** committed:

```json
{
  "run_name": "lgbm_ft_pers_b_32_s1",
  "git_sha": "...", "config_sha256": "...", "rows_digest": "...",
  "arm": "lgbm_ft_pers", "text_variant": "b", "svd_dim": 32, "seed": 1,
  "metrics": { "all": {"auc": 0.0, "logloss_sampled": 0.0, "...": 0.0},
               "cold_start": {}, "low_history": {} }
}
```

`make report` regenerates the tables in `reports/results.md` from these files.
**No number in the report is ever hand-typed**, and a metric change shows up as a reviewable
diff. MLflow is for exploring the 61 runs; the JSON is the record of what was claimed.

---

## 12. Inference cost profiling

```python
# serving/benchmark.py
@dataclass(frozen=True)
class BenchResult:
    config: str
    p50_ms_per_1k: float
    p95_ms_per_1k: float
    throughput_per_s: float
    peak_rss_mb: float

def bench(config: str, *, n_batches: int = 100, batch: int = 1000) -> BenchResult: ...
```

| Config | What it isolates |
|---|---|
| `lgbm_tab` | Baseline serving cost |
| `lgbm_ft_pers_cached` | Marginal cost of the text arm at steady state (mmap gather + cosine) |
| `encoder_cold_mps` | Tokenize + forward at request time, MPS |
| `encoder_cold_cpu` | Same on CPU — the realistic commodity-server number |
| `encoder_onnx_int8` | Optimized cold path via `optimum` export |

Protocol: 20 warmup batches discarded, 100 measured, single process, `num_threads=7`,
machine otherwise idle. Reported with the measured `peak_rss_mb` so the memory claim is
evidence rather than arithmetic.

Converted to **$/1M predictions** against one named cloud SKU, with instance type and
price-lookup date cited inline.

**The claim to confirm** (PRD §9): 105k × 384 × 4 B ≈ 162 MB fp32 / ~40 MB int8, so the
whole catalog fits in cache and steady-state encoder cost is ≈ 0 — the cold path matters
only for new articles. `bench` prints `EmbeddingCache.nbytes` alongside the latency table so
the claim is measured, not asserted.

---

## 13. CLI surface

`typer`-based, installed as `contentsignal`. Every command is **idempotent** and skips work
when its outputs exist with a matching config hash, unless `--force`.

| Command | Reads | Writes |
|---|---|---|
| `ingest` | Kaggle CSVs | `artifacts/parquet/*` |
| `sample` | transactions, cohort | `artifacts/rows/*`, `rows_manifest.json` |
| `build-features --group G --window W` | transactions, rows | `artifacts/features/G/W.parquet` |
| `embed --variant a\|b --source frozen\|contrastive` | articles, checkpoint | `artifacts/emb/*` |
| `finetune --variant a\|b --seed N` | transactions, articles | checkpoint, diagnostics |
| `train --arm A --variant V --seed N` | features, rows | model, MLflow run, metrics JSON |
| `train-twotower --arm 9\|10 --variant a\|b --seed N` | positives, articles, history | tower checkpoints, MLflow run, metrics JSON |
| `evaluate --arm A ...` | model, features | metrics JSON, figures |
| `bench --config C` | model, embedding cache | `reports/metrics/bench.json` |
| `report` | `reports/metrics/*.json` | `reports/results.md` |

A `Makefile` chains these into `make m1` … `make m9` matching the milestones.

---

## 14. Resource budgets

Against the 8 GB ceiling, with ~1.5 GB reserved for OS and overhead → **~6.5 GB usable**.

| Stage | Peak RSS | How it stays bounded |
|---|---|---|
| `ingest` | **~1.5 GB** | DuckDB streams CSV → Parquet; never materializes 31.8M rows in memory |
| `sample` | **~1.2 GB** | Per-window; alias table is ~60k entries; positives per window ~65k |
| `build-features` (tabular) | **~2.5 GB** | Per-window DuckDB aggregation, Polars result only |
| `embed` | **~1.0 GB** | 105k texts at batch 256; output 162 MB |
| `finetune` | **~1.5 GB** | 22M params × 4 (weights + grads + 2 Adam states) ≈ 350 MB, plus activations for 128 × 64 tokens |
| `build-features` (taste) | **~1.5 GB** | Per-window; 80k × 384 float32 ≈ 123 MB, plus the 162 MB cache mmapped |
| `train` (LightGBM) | **~3.5 GB** | See below |
| `train-twotower` | **~3.0 GB** | 30M params (22M encoder + 6.7M item-ID table + ~1M towers) × 4 B × 4 Adam states ≈ 480 MB; activations for batch 512 with ~450 unique texts × 64 tokens ≈ 700 MB; history tensors negligible |
| `evaluate` | **~1.5 GB** | Test window only, ~600k rows |

### The LightGBM budget in detail

The binding constraint. At 5M train rows × 80 columns:

| | |
|---|---|
| Raw float32 matrix | 5M × 80 × 4 B = **1.6 GB** |
| Binned, `max_bin=63` → uint8 | 5M × 80 × 1 B = **400 MB** |
| Histograms | 127 leaves × 80 features × 63 bins × 2 doubles ≈ **10 MB** |
| Prediction/gradient buffers | ~120 MB |

`max_bin=63` is therefore **not** a tuning choice — it is what makes the run fit. Combined
with `free_raw_data=True` after `Dataset.construct()`, peak lands near **2.2 GB**, with
headroom for the transient raw matrix during construction (~3.5 GB peak).

Enabling `include_raw_dims` (§6.6) adds 32 columns → +640 MB raw / +160 MB binned, which is
why it is an opt-in sensitivity run rather than a default.

### Wall-clock estimates

| Milestone | Estimate |
|---|---|
| M0 install + download | 30–60 min (network-bound) |
| M1 ingest + EDA + leakage tests | 45 min |
| M2 sampling + tabular features (10 windows) | 1.5–2 h |
| M3 baselines | 30 min |
| M4 frozen embeddings + taste vectors + arms 4a/4b | 1.5 h |
| M5 contrastive fine-tune (2 variants × 3 seeds) | 1.5 h |
| M6 arms 7a/7b + sensitivity + bootstrap | 3–4 h |
| M6b two-tower arms 9/10 (13 runs) | 2–3 h |
| M7 calibration + business proxy | 45 min |
| M8 cost profiling | 45 min |
| M9 report | — |

---

## 15. Test specification

### `tests/test_leakage.py`

**The deletion-invariance property**, applied to every feature builder — the general form of
every leakage check in this project:

```python
@pytest.mark.parametrize("builder", ALL_BUILDERS)
def test_features_are_invariant_to_future_deletion(builder, synthetic_txns):
    as_of = date(2020, 6, 1)
    full     = builder.build(synthetic_txns, as_of=as_of, entities=E)
    truncated = builder.build(
        synthetic_txns.filter(pl.col("t_dat") < as_of), as_of=as_of, entities=E
    )
    assert_frame_equal(full, truncated)   # recomputing without the future changes nothing
```

Plus:

| Test | Assertion |
|---|---|
| `test_negative_candidates_have_prior_history` | Every sampled negative article has ≥1 transaction before `W.start` |
| `test_contrastive_pairs_predate_val` | `max(t_dat)` over all pair source transactions `< 2020-08-26` |
| `test_svd_fitted_on_train_articles_only` | The fitted SVD's article index ∩ (val ∪ test-only articles) = ∅ |
| `test_taste_vectors_respect_as_of` | Deletion-invariance, applied to `taste_vectors` |
| `test_no_builder_signature_lacks_as_of` | Introspects every `FeatureBuilder.build` — `as_of` is present and keyword-only |

### `tests/test_splits.py`

Windows are contiguous, non-overlapping, 14 days each; train `end` < val `start` < test
`start`; every window lies inside the dataset's date range.

### `tests/test_sampling.py`

| Test | Assertion |
|---|---|
| `test_ratio_exact` | `negatives == ratio × positives` per customer |
| `test_no_positive_sampled_as_negative` | The two sets are disjoint per (customer, window) |
| `test_no_duplicate_negatives` | Per customer, `len(set(negatives)) == len(negatives)` |
| `test_determinism` | Two runs at the same seed produce byte-identical Parquet |
| `test_popularity_weighting` | Over many draws, empirical frequency ∝ weight^0.75 within tolerance |

### `tests/test_twotower.py`

| Test | Assertion |
|---|---|
| `test_customer_tower_is_item_independent` | `CustomerTower.forward` accepts no item argument, and its output is bit-identical for the same customer across different candidate batches. This is what keeps both towers precomputable and the §12 serving analysis valid |
| `test_logq_correction_applied` | With a synthetic skewed item distribution, corrected logits recover uniform expected ranking; uncorrected ones do not |
| `test_in_batch_duplicates_masked` | An item appearing twice in a batch is never its own negative |
| `test_history_mask_respects_as_of` | Deletion-invariance (§15) applied to history construction — no purchase at or after `W.start` enters `hist_ids` |
| `test_output_is_l2_normalized` | Both tower outputs have unit norm within tolerance |

### `tests/test_calibration.py`

`prior_correct` round-trips on synthetic data: given a known base rate and downsampling
rate, corrected probabilities recover the true base rate to within Monte-Carlo error; and
correction is strictly monotone, so AUC is unchanged.

---

## 16. Definition of done

| M | Artifact | Test / gate | Metric that must appear |
|---|---|---|---|
| **M0** | `uv.lock`, 3 CSVs on disk | `contentsignal --help` runs | — |
| **M1** | `artifacts/parquet/*`, `01_eda.ipynb` | **`test_leakage.py`, `test_splits.py` green** | Row counts, date range, token-length distribution |
| **M2** | `artifacts/rows/*` + manifest, tabular features | `test_sampling.py` green; row budget respected (§3.4) | Actual positives/rows per window |
| **M3** | Arms 1–3 trained | Digest check passes | First `all`-slice AUC/log-loss table |
| **M4** | Frozen embeddings, taste vectors, arms 4a/4b | `test_leakage.py` still green | Δ(4b − 4a): does personalization help at all |
| **M5** | Contrastive checkpoints | Collapse check passes | `diag/spearman_pop`, NN dumps |
| **M6** | Arms 7a/7b + sensitivity | Full grid complete | **All three deltas with 95% CIs, all three slices, plus the H2 delta-of-deltas** |
| **M6b** | Two-tower arms 9/10 | **`test_twotower.py` green**; collapse + popularity-ρ checks pass | **H3 delta with 95% CI**, plus the log-Q ablation |
| **M7** | Calibration + business proxy | `test_calibration.py` green | Corrected log-loss, Brier, AOV lift ratio, isotonic recalibration of arms 9/10 |
| **M8** | Benchmark results | — | Latency table, $/1M, measured `nbytes` |
| **M9** | `reports/results.md`, `README.md` | `make report` reproduces every number | **The §1 verdict: supported or null** |

**M1 gates everything** — no model is trained until the leakage tests pass (PRD §11).

**M9 is the only point at which the test split is read.** M3–M8 report validation-split
numbers; the test split is evaluated once, at the end, and **all three** §1 pre-registered
criteria (H1, H2, H3) are applied to that single evaluation.
