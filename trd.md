# ContentSignal — TRD

The engineering specification for the two-stage system defined in `prd.md`. Schemas,
signatures, algorithms, resource budgets, and test assertions — precise enough that
implementation is mechanical.

Where a decision has a non-obvious reason, the reason is stated. Deviating from these contracts
silently breaks the validity of the experiment rather than merely its style.

---

## 1. Traceability

| PRD requirement | Where it is implemented |
|---|---|
| §1 two-stage architecture, factorized towers | §5.3 `models/twotower.py`, §5.4 `retrieval/index.py`, §9 training |
| §2 H1 stage attribution | §12.4 the stage-attribution table, §10.3 the K sweep |
| §2 H2 content retrieval / cold start | §9.3 retriever arms, §10.2 retrieval metrics |
| §2 H3 candidate distribution shift | §8.3 random negatives, §10.4 end-to-end metrics |
| §5 `as_of` contract | §5.1 — the load-bearing interface |
| §5 window roles, retriever precedence | §3.1, §15 `test_retriever_windows_precede_candidate_windows` |
| §7 frozen byte-identical candidate set | §4.5 `candidates_manifest.json`, §8.2 |
| §8 end-to-end counts unretrieved positives | §10.4, §15 `test_e2e_metrics_count_unretrieved_positives` |
| §8 customer-level bootstrap | §10.6 `eval/bootstrap.py` |
| §9 per-stage cost profiling | §12 `serving/benchmark.py` |
| §3 row budget measured, not assumed | §3.4, §8.2 |

---

## 2. Environment & dependencies

**Python 3.11 via `uv`.** System Python is 3.9.6 and cannot be used.

```toml
# pyproject.toml — requires-python = ">=3.11,<3.12"
dependencies = [
  "polars", "duckdb", "pyarrow",          # frames and out-of-core aggregation
  "numpy", "scipy", "scikit-learn",        # metrics, isotonic regression
  "lightgbm",                              # the tabular ranker baseline
  "torch", "transformers",                 # towers and the text encoder
  "sentence-transformers",                 # MiniLM checkpoint loading
  "mlflow",                                # tracking (local file backend)
  "typer", "pydantic", "pyyaml",           # CLI and typed config
  "kaggle",                                # dataset download
]
[project.optional-dependencies]
bench = ["faiss-cpu", "optimum", "onnxruntime"]   # §12 only
```

`pandas` is **not** a pipeline dependency. It arrives transitively via MLflow and is permitted
only in notebooks and report generation. 31.8M transaction rows do not fit in pandas on this
machine; that is why DuckDB and Polars are requirements rather than preferences.

`faiss-cpu` is an **optional extra**, installed only for the §12 exact-versus-approximate
comparison. The pipeline never depends on it — that is the point of the measurement.

### Blocker

M0 cannot complete until:

1. The H&M competition rules are accepted at
   `https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/rules`. The API
   returns 403 until this is done, even with a valid token.
2. `~/.kaggle/kaggle.json` exists with mode `600`.

Neither is true on this machine. `cli.check_kaggle_credentials` fails fast on both, naming them
separately, because a missing token and unaccepted rules both surface as an indistinguishable
403 much later in the download.

---

## 3. Windowing, cohort, and scale

### 3.1 Window geometry and roles

Ten contiguous, non-overlapping 14-day windows read from `conf/split.yaml` — the single source
of truth. No module hardcodes a date.

| Window | Role | Split | Dates |
|---|---|---|---|
| `ret_w1` … `ret_w4` | `retriever` | train | 2020-05-06 → 2020-06-30 |
| `rank_w1` … `rank_w4` | `ranker` | train | 2020-07-01 → 2020-08-25 |
| `val` | `val` | val | 2020-08-26 → 2020-09-08 |
| `test` | `test` | test | 2020-09-09 → 2020-09-22 |

```python
# splits/temporal.py

Split = Literal["train", "val", "test"]
Role  = Literal["retriever", "ranker", "val", "test"]

@dataclass(frozen=True)
class Window:
    name: str
    split: Split
    role: Role
    start: date          # inclusive
    end: date            # inclusive

    @property
    def as_of(self) -> date:
        """Exclusive feature cutoff. Features may read `t_dat < as_of`, never `>=`.

        This is `start`, not `start - 1 day`: a transaction landing exactly on the first
        day of the window belongs to the label period, so it must not be visible to a
        feature. The strict `<` in `features.base.history` is what enforces that.
        """
        return self.start

def load_windows(cfg: SplitConfig | None = None) -> list[Window]: ...
def windows_for_role(role: Role, cfg: SplitConfig | None = None) -> list[Window]: ...
def candidate_windows(cfg: SplitConfig | None = None) -> list[Window]:
    """Every window the frozen retriever generates candidates for: ranker + val + test."""
```

Two validators run on every load, both raising rather than warning:

- `assert_contiguous_non_overlapping` — each window starts the day after the previous ends. A
  gap silently drops transactions from every candidate set; an overlap lets the same purchase be
  a label in one window and history in another.
- `assert_role_ordering` — **every `retriever` window ends strictly before the first
  `ranker` window starts**, and `ranker` < `val` < `test`. This is the new leakage boundary
  (`prd.md` §5); §15 asserts it independently.

### 3.2 Cohort

Drawn once from customers with ≥1 transaction in the qualifying range, persisted, and reused by
every window and every arm.

```yaml
cohort:
  size: 150_000
  seed: 17
  qualify_start: 2020-03-24
  qualify_end: 2020-09-22
```

Sampling per arm would make a ΔNDCG of 0.005 indistinguishable from a different random draw.

### 3.3 Row-set construction

For each window *W*:

```
eligible  = cohort ∩ {customers with >= 1 purchase in W}
positives = distinct (customer_idx, article_id) purchased in W by eligible customers
```

Multiple purchases of the same article by the same customer in *W* collapse to **one row**.
Purchase count is deliberately **not** carried as a feature — it is a function of the label
window and would leak.

The `retriever` windows use `positives` directly as training pairs. The `ranker`, `val`, and
`test` windows get their rows from §8 candidate generation instead.

> **Selection effect, stated rather than buried.** Rows exist only for customers who transacted
> during *W*. Every model is therefore evaluated **conditional on the customer transacting**.
> This is standard for the task and is what keeps per-customer ranking metrics well defined — a
> customer with zero positives has no meaningful `recall@K`. But the reported numbers do not
> describe performance over the full customer base, and `reports/results.md` must say so.

### 3.4 Scale, and the row budget

Estimates below are approximate and are replaced by measured counts at M2/M5.

| Quantity | Estimate |
|---|---|
| Transactions per 14-day window (all customers) | ~610k |
| Positives per window, cohort of 150k | ~65k |
| Eligible customers per window | ~16k |
| **Retriever training pairs** (4 windows of positives) | **~260k** |
| **Ranker training rows** (4 windows × 16k customers × K=100) | **~6.4M** |

The ranker estimate **exceeds** the 5M target, so one of cohort size or `K` must give. Rather
than guess, the constraint is enforced by measurement:

```yaml
candidates:
  k: 100                          # candidate depth for the main grid
  k_sweep: [20, 50, 100, 200, 500, 1000]
  target_train_rows: 5_000_000
  row_budget_tolerance: 0.2
  train_customer_cap: null        # set once at M5 from measured counts
```

`retrieve` reports actual row counts and **fails if ranker training rows exceed
`target_train_rows × 1.2`**, with a message naming the `train_customer_cap` that would fit.

> **Only the *training* windows are capped.** `val` and `test` retrieve for every eligible
> customer, uncapped, so evaluation is never conditioned on a budget decision. Capping
> evaluation would silently narrow the customer-level bootstrap and change the confidence
> intervals the hypotheses are judged on.

The cap is drawn under `cohort.seed` and persisted, so it is identical across every ranker arm.

**Stage 1 is never capped** — it trains on ~260k pairs regardless of `K`, which is why the
retrieval side of H1 is cheap to explore.

---

## 4. Storage contracts

Root: `artifacts/` (gitignored). All Parquet is zstd-compressed. Every stage writes a
`_stamp` file containing `config_sha256` of the config it ran under, which is what makes the
CLI idempotent (§13).

### 4.1 `artifacts/parquet/transactions.parquet`

| Column | Type |
|---|---|
| `customer_idx` | `int32` (dense, from §4.4) |
| `article_id` | `int32` |
| `t_dat` | `date32` |
| `price` | `float32` |
| `sales_channel_id` | `int8` |

Sorted by `t_dat`, then `customer_idx`. Row-group size 256 MB.

### 4.2 `artifacts/parquet/articles.parquet`

`article_id` `int32`, the 11 taxonomy columns as `categorical`, `product_code` `int32`,
`prod_name` and `detail_desc` as `str`. Null `detail_desc` becomes the empty string, and the
count of nulls is reported in the EDA — an empty description is a legitimate input to the
encoder, but silently dropping those articles would bias the cold-start slice.

### 4.3 `artifacts/parquet/customers.parquet`

`customer_idx` `int32`, `age` `float32` (null preserved), `FN` / `Active` `int8`,
`club_member_status` and `fashion_news_frequency` `categorical`.

### 4.4 `artifacts/parquet/customer_index.parquet`

`customer_id` (hex `str`) → `customer_idx` `int32`. Written once, never regenerated: every
downstream artifact keys on `customer_idx`, so a re-hash would invalidate all of them.

### 4.5 `artifacts/candidates/{window}.parquet` + `candidates_manifest.json`

**The byte-identity invariant of the whole stage-2 experiment.**

| Column | Type | Notes |
|---|---|---|
| `customer_idx` | `int32` | |
| `article_id` | `int32` | |
| `retrieval_score` | `float32` | raw dot product from the frozen retriever |
| `retrieval_rank` | `int16` | 1 = top of this customer's list |
| `y` | `int8` | 1 if purchased in this window |

Sorted by `customer_idx`, then `retrieval_rank`. The manifest records, per window:

```json
{
  "ret_retriever": "R2", "ret_seed": 1, "ret_config_sha256": "…",
  "k": 100, "window": "rank_w1",
  "rows": 1600123, "customers": 16001, "positives": 64998,
  "recall_at_k": 0.412,
  "sha256": "…"
}
```

`sha256` is over the Parquet bytes. **Every ranker asserts it before fitting** (§5.5). Writing
is a single atomic rename; a partial file must never be digestible.

### 4.6 `artifacts/features/{group}/{window}.parquet`

One file per (group, window). Key columns `customer_idx` and/or `article_id` plus the group's
declared columns, exactly as declared on the builder (§5.1). Column drift is an error, not a
surprise.

### 4.7 Item vector cache — `artifacts/vectors/{retriever}_{seed}/`

| File | Contents |
|---|---|
| `item.npy` | `float32[n_articles, 128]`, L2-normalized, row *i* ↔ `ids.npy[i]` |
| `ids.npy` | `int32[n_articles]`, **sorted ascending** |
| `_stamp` | `config_sha256` of the retriever config |

> **`.npy` plus a sorted id array, not Parquet.** The serving benchmark must `mmap` the matrix
> and look up by binary search (`np.searchsorted`) without deserializing or copying. Parquet
> cannot be mmap'd as a contiguous float matrix, which would put a decode step inside the
> latency measurement and make §12 measure the wrong thing.

`105,000 × 128 × 4 B ≈ 54 MB.`

### 4.8 `artifacts/rows_random/{window}.parquet`

The H3 arm only: positives plus `sampler.ratio` popularity-weighted random negatives per
positive, on the same `ranker` windows. Schema as §4.5, without `retrieval_score` and
`retrieval_rank` — a randomly drawn negative was never retrieved, so it has no rank to report.

> **The H3 pair must differ in exactly one thing.** Because random-negative rows have no
> retrieval columns, the **retrieved-negative arm in the H3 comparison also has them withheld**.
> Otherwise the delta would bundle "hard negatives" together with "three extra features," and
> H3's registered claim is about the negative distribution alone. The main grid (§9b) keeps the
> retrieval columns; only the H3 pair drops them, and `reports/results.md` says so on the row.

---

## 5. Module contracts

### 5.1 The `as_of` contract — the load-bearing interface

Unchanged from the current implementation, and now guarding more surface than before.

```python
# features/base.py

def history(txns: pl.LazyFrame, *, as_of: date) -> pl.LazyFrame:
    """The ONLY sanctioned way to read transactions inside a feature builder.

    Strictly-before-cutoff rows. The comparison is strict `<`: a transaction dated
    exactly `as_of` falls on the first day of the label window, so admitting it would
    leak a day of the thing being predicted.
    """
    return txns.filter(pl.col("t_dat") < as_of)

@runtime_checkable
class FeatureBuilder(Protocol):
    name: str                       # -> artifacts/features/{name}/
    columns: tuple[str, ...]        # declared output columns, asserted on build

    def build(self, txns: pl.LazyFrame, *, as_of: date,
              entities: pl.DataFrame) -> pl.DataFrame: ...

ALL_BUILDERS: list[FeatureBuilder] = []

def register_builder(builder: B) -> B: ...       # checks the contract at import time
def assert_as_of_is_enforced(builder: object) -> None: ...
def assert_declared_columns(builder: FeatureBuilder, out: pl.DataFrame) -> None: ...
```

`as_of` is **keyword-only and has no default**. `register_builder` verifies both at import time,
so a violation fails in the pipeline and not only under pytest. `ALL_BUILDERS` exists so a
builder added later is automatically subjected to the deletion-invariance property (§15) —
nobody has to remember to wire it up.

> **Never add an overload or a default that permits reading without a cutoff.** The signature is
> the enforcement mechanism. A comment is not.

### 5.2 Splits — `splits/temporal.py`

As §3.1, plus the row-set helpers already implemented:

```python
def eligible_customers(txns: pl.LazyFrame, w: Window, cohort: pl.Series) -> pl.Series: ...
def positives(txns: pl.LazyFrame, w: Window, eligible: pl.Series) -> pl.DataFrame: ...
```

### 5.3 Retriever — `models/twotower.py`

```python
class ItemTower(nn.Module):
    """taxonomy embeddings ⊕ numerics ⊕ (optional) text → MLP → 128, L2-normed.

    No article-ID embedding: a newly added article's ID vector would be untrained
    noise, and cold-start capability is the mechanism H2 tests (prd.md §6).
    """
    def __init__(self, *, encoder: SentenceTransformer | None,
                 cat_cardinalities: dict[str, int], n_numeric: int,
                 out_dim: int = 128): ...
    def forward(self, cats, nums, text_ids=None, text_mask=None) -> Tensor: ...  # [B, 128]

class CustomerTower(nn.Module):
    """history (attention-pooled item IDs) ⊕ categorical ⊕ numerics → MLP → 128, L2-normed."""
    def __init__(self, *, n_items: int, id_dim: int = 64, hist_len: int = 20,
                 cat_cardinalities: dict[str, int], n_numeric: int,
                 out_dim: int = 128): ...
    def forward(self, hist_ids, hist_mask, cats, nums) -> Tensor: ...            # [B, 128]

class TwoTower(nn.Module):
    def score(self, cust_vec: Tensor, item_vec: Tensor) -> Tensor:
        return (cust_vec * item_vec).sum(-1) / self.temperature
```

| Component | Spec |
|---|---|
| History length | last 20 purchases, right-padded, masked |
| History pooling | **self-attentive with a learned query**: `softmax(qᵀ tanh(W·h)) · h` |
| Customer item-ID table | 105k × 64 (~27 MB) — history only |
| Tower MLPs | [512, 256] → 128, GELU, dropout 0.1, L2-normalized output |
| Categorical embeddings | 8–16 d per field |
| Temperature | learned scalar, init 0.05 |
| Encoder | `all-MiniLM-L6-v2`, 64 tokens max, trained jointly at 2e-5 |

> **Self-attentive, never candidate-aware.** Candidate-conditioned attention would score better,
> but the customer vector would then depend on the item and neither tower could be precomputed.
> That breaks retrieval outright and voids §12. **Enforced by the signature:**
> `CustomerTower.forward` takes no item argument. Asserted in
> `tests/test_retrieval.py::test_customer_tower_is_item_independent`.

### 5.4 Retrieval — `retrieval/index.py`, `retrieval/candidates.py`

```python
# retrieval/index.py

class ItemIndex:
    """Memory-mapped item vectors + exact top-K. FAISS is a §12 comparison, not a dependency."""
    @classmethod
    def open(cls, path: Path) -> ItemIndex: ...          # mmap item.npy, load ids.npy
    @property
    def nbytes(self) -> int: ...                          # reported by bench, so the claim is measured

    def topk(self, cust_vecs: np.ndarray, *, k: int,
             allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Exact top-k by dense matvec. Returns (article_ids[B, k], scores[B, k]).

        `allowed` is a boolean mask over `ids`, and it is how the no-future-catalog
        guard is implemented: articles with zero transactions before `W.start` are
        masked to -inf, never merely deprioritized. An unreleased article is trivially
        separable and its presence would inflate every metric downstream.
        """
```

```python
# retrieval/candidates.py

def generate(w: Window, *, index: ItemIndex, tower: CustomerTower,
             customers: pl.Series, txns: pl.LazyFrame, k: int) -> pl.DataFrame:
    """Top-k candidates for one window, labeled against that window's positives.

    Customer vectors are built from history strictly before `w.as_of` — the model is
    frozen, but its inputs are recomputed per window, which is exactly how a deployed
    retriever behaves between retrainings.
    """

def write_candidates(w: Window, df: pl.DataFrame, *, meta: dict) -> str:
    """Atomic write + manifest entry. Returns the sha256 of the Parquet bytes."""

def assert_digest(w: Window, *, manifest: Path) -> None:
    """Raise unless the on-disk candidate file matches its recorded digest.

    Called by every ranker before fitting. A ΔNDCG of 0.005 must not be attributable
    to one arm having received a different candidate list.
    """
```

### 5.5 Rankers — `models/base.py`

The existing `Arm` protocol is unchanged; three rankers implement it.

```python
@runtime_checkable
class Arm(Protocol):
    name: str
    feature_groups: tuple[str, ...]

    def fit(self, X: pl.DataFrame, y: np.ndarray, *,
            valid: tuple[pl.DataFrame, np.ndarray] | None) -> None: ...
    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """UNCALIBRATED scores. Calibration never happens inside an arm."""
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> Arm: ...
```

> **Calibration is applied by the evaluator, in one place, identically for every arm** (§10.7).
> An arm that calibrated its own output would make cross-arm probability metrics
> incomparable in a way nothing would catch.

### 5.6 Serving — `serving/embedding_cache.py`

```python
class EmbeddingCache:
    """mmap'd float32 matrix + sorted int32 ids; lookup by np.searchsorted."""
    @classmethod
    def open(cls, path: Path) -> EmbeddingCache: ...
    def get(self, article_ids: np.ndarray) -> np.ndarray: ...     # [n, dim], no copy where possible
    @property
    def nbytes(self) -> int: ...
```

Raises on an unknown `article_id` rather than returning zeros. A zero vector is a valid-looking
input that would silently degrade exactly the cold-start slice H2 is measured on.

---

## 6. Feature specification

`as_of` column: **Y** = reads transactions and must respect the cutoff; **—** = static.

### 6.1 Customer (`features/customer.py`, group `customer`)

| Column | Type | `as_of` | Definition |
|---|---|---|---|
| `cust_age` | `float32` | — | null → NaN; LightGBM handles natively, MLP/DCN get the null flag |
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

**18 columns.** The same block feeds the customer tower's non-sequence inputs (§5.3).

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
| `art_prior_purchases` | `int32` | Y | total purchases before `as_of` — **the cold-start slicing key** |
| `art_is_cold` | `int8` | Y | `art_prior_purchases < eval.cold_start_threshold` (default 10) |

**9 columns.** Also feeds the item tower's numerics.

### 6.3 Categorical (`features/categorical.py`, group `categorical`)

Eleven columns, native LightGBM categoricals and embedding-table inputs for MLP/DCN:
`product_type_name`, `product_group_name`, `colour_group_name`, `department_no`, `index_name`,
`index_group_name`, `section_name`, `garment_group_name`, `graphical_appearance_name`,
`perceived_colour_value_name`, `perceived_colour_master_name`.

> **PRD §3 non-negotiable: every ranker arm and both retriever arms receive all eleven.**
> Withholding them from a baseline while feeding the same information to an encoder as text
> would credit the encoder with information the baseline was never given. This is the single
> most common way this class of experiment is run wrong.

### 6.4 Cross (`features/cross.py`, group `cross`)

| Column | Type | `as_of` | Definition |
|---|---|---|---|
| `x_prior_in_product_group` | `int32` | Y | customer's prior purchases in this article's group |
| `x_prior_in_department` | `int32` | Y | |
| `x_prior_in_colour_group` | `int32` | Y | |
| `x_bought_product_code_before` | `int8` | Y | same garment, any colourway or size |
| `x_price_vs_cust_mean` | `float32` | Y | `art_price_mean` − `cust_price_mean` |
| `x_age_indexgroup_affinity` | `float32` | Y | empirical rate for the customer's age bucket × `index_group`, history only |

**6 columns.** These are the hand-built feature crosses that DCN-v2 is measured against (§9b).

### 6.5 Retrieval (`features/retrieval.py`, group `retrieval`) — new

| Column | Type | `as_of` | Definition |
|---|---|---|---|
| `retrieval_score` | `float32` | Y | raw dot product from the frozen retriever |
| `retrieval_rank` | `int16` | Y | 1 = top of this customer's candidate list |
| `retrieval_log_rank` | `float32` | Y | `log1p(retrieval_rank)` — rank is heavy-tailed, and axis-aligned tree splits handle the log far better |

Built through the standard `FeatureBuilder` ABC despite reading from the candidate file rather
than aggregating transactions, so it inherits the `as_of` signature check and the
deletion-invariance property automatically.

**Column total: 18 customer + 9 article + 11 categorical + 6 cross + 3 retrieval = 47.**

---

## 7. What was removed, and why nothing was lost

The previous design had two additional feature groups. Both are gone, along with the SVD step
and `artifacts/taste/`.

| Removed | Was | Why it is unnecessary now |
|---|---|---|
| `text_item` | 32 SVD dimensions of the article embedding | The ranker no longer consumes article embeddings; stage 1 does, natively at 128 d |
| `text_customer` | `cust_taste_{0..31}`, `sim_taste_cos`, `sim_last10_{max,mean}`, `sim_taste_pct_rank` | **Trees cannot compute a dot product.** Given 32 taste dimensions and 32 article dimensions, a tree has no way to multiply them, so the similarity had to be precomputed by hand. A two-tower's score *is* that dot product, learned end-to-end. Hand-building it downstream would duplicate stage 1's job, less well |

Consequences worth stating explicitly:

- Text still reaches the ranker, through `retrieval_score`. The stage-2 axis is therefore *"the
  ranker's own features"*, not *"a text-free pipeline"*. The clean pipeline-level number is H2's
  end-to-end comparison. Stated inline in `reports/results.md`.
- 80 columns → 47 drops the LightGBM design matrix from 1.6 GB raw / 400 MB binned to ~940 MB /
  ~235 MB (§14).

---

## 8. Candidate generation

### 8.1 The algorithm

```
function generate(W, retriever, index, customers, k):
    eligible = cohort ∩ {customers with >= 1 purchase in W}      # §3.3
    if W.role == "ranker" and candidates.train_customer_cap:
        eligible = sample(eligible, cap, seed=cohort.seed ^ crc32(W.name))

    # No future catalog: mask articles with zero transactions before W.start.
    allowed  = {a : count(txns, a, t_dat < W.start) >= 1}

    hist     = last 20 article_ids per customer, strictly before W.start
    cust_vec = retriever.customer_tower(hist, cust_cats, cust_nums)   # frozen weights
    ids, scores = index.topk(cust_vec, k=k, allowed=allowed)

    pos      = positives(txns, W, eligible)
    rows     = flatten(ids, scores) with y = 1 where (customer, article) in pos
    return rows sorted by (customer_idx, rank)
```

`crc32` rather than Python's `hash()`: `hash()` is salted per process, so a per-window seed
derived from it would differ between runs and silently break reproducibility.

### 8.2 Invariants

| Invariant | Enforcement |
|---|---|
| Written once, byte-identical across ranker arms | `candidates_manifest.json` sha256, asserted by every arm before fitting |
| Never regenerated per arm | `retrieve` is idempotent on `config_sha256`; `--force` is the only override and it rewrites the manifest |
| The retriever is frozen for the whole stage-2 experiment | Retriever checkpoint digest recorded in the manifest; a mismatch is an error |
| No future catalog | The `allowed` mask in §8.1, asserted in §15 |
| Row budget respected | `retrieve` measures and fails loudly, naming the cap that fits (§3.4) |
| `val` / `test` uncapped | Cap applies only where `W.role == "ranker"` |

### 8.3 Random negatives — the H3 arm only

`sampling/negatives.py` is retained unchanged and repurposed. It draws, per positive,
`sampler.ratio = 10` negatives from an alias table weighted `∝ (pop_12w + 1) ** 0.75` — the
standard word2vec-style dampening — excluding the customer's true positives in *W* and any
article with no transactions before `W.start`.

Uniform (unweighted) sampling runs **once** as a sensitivity check in an appendix, not as a
headline arm: uniform negatives are trivially separable and inflate AUC.

For H3 the comparison pair is the **same ranker architecture** trained on
`artifacts/rows_random/` versus `artifacts/candidates/`, both with `retrieval_score` /
`retrieval_rank` / `retrieval_log_rank` withheld, so the only difference is the negative
distribution. Both are evaluated end-to-end on the identical `test` candidate rows.

---

## 9. Stage 1 — retriever training

### 9.1 Objective

```python
def sampled_softmax_loss(
    cust: Tensor, item: Tensor, item_ids: Tensor,
    *, log_q: Tensor, temperature: Tensor,
) -> Tensor:
    """In-batch sampled softmax with log-Q correction (Yi et al., RecSys 2019).

    logits[i][j] = cust[i] · item[j] / T  −  log_q[item_ids[j]]

    Without the log-Q term, in-batch negatives arrive proportional to item
    popularity, so the model is rewarded for demoting popular items and drifts
    toward an inverted-popularity ranker. Targets are the diagonal.
    """
```

| Setting | Value |
|---|---|
| Objective | In-batch sampled softmax, log-Q corrected |
| `log_q` estimate | Streaming item-frequency counter over retriever-window positives, `log(count / total)` |
| Batch | 512 positives — **batch size is the negative count** |
| Optimizer | AdamW; discriminative LRs **1e-3 towers, 2e-5 encoder** |
| Schedule | 10% linear warmup, cosine decay |
| Epochs | 2, selected on `val` `recall@100` |
| Device | `mps`, `--device cpu` fallback |
| Seeds | 3 |

**Training consumes positives only** (~260k pairs across the four retriever windows) — in-batch
negatives are free. At batch 512 that is ~500 steps/epoch, **≈3–5 min/epoch on MPS**.

**Batch construction**: articles repeat across pairs, so each batch dedupes article IDs, encodes
each unique text once, and gathers. Duplicate items within a batch are masked out of the
negative set — an item cannot be its own negative.

### 9.2 Positive pairs

A positive is a (customer, purchased-article) pair from a retriever window. Capped at **≤5 pairs
per customer and ≤50 per article**.

> Uncapped sampling is dominated by heavy buyers and bestsellers, which pushes the encoder back
> toward a popularity proxy — the same contamination log-Q guards against, arriving by a
> different route. The popularity-ρ diagnostic (§9.5) is the check that both guards worked.

### 9.3 Arms

| Arm | Item tower | Seeds | Runs |
|---|---|---|---|
| `pop` | none — top-`K` by `art_pop_12w` before `W.start` | — | 0 (deterministic) |
| `R1` | taxonomy + numerics | 3 | 3 |
| `R2` | `R1` + `detail_desc` (Text-B) | 3 | 3 |
| `R2-A` | `R1` + full text concat (Text-A) | 1 | 1 |

`R2 − R1` is H2. `R2-A − R2` quantifies how much apparent text lift is re-encoded taxonomy.

`R2` at the val-selected seed is the **frozen retriever** for all of stage 2. Chosen on `val`
`recall@100`, recorded in the candidate manifest, and never revisited.

### 9.4 Item vector precomputation

```
contentsignal embed --retriever R2 --seed N
```

Runs the frozen item tower over all ~105k articles in batches of 256 and writes §4.7. This is
the step that makes the transformer's steady-state serving cost ≈0: everything downstream reads
a 54 MB mmap'd matrix, and the encoder never runs online except for genuinely new articles.

### 9.5 Diagnostics — reported whatever they show

| Diagnostic | Failure signal |
|---|---|
| **Popularity ρ** — Spearman between retriever score and `log(art_pop_12w + 1)` | High ρ means the retriever collapsed into a popularity re-ranker. Check log-Q first |
| **log-Q ablation** — one run with the correction disabled | Demonstrates the correction's effect rather than asserting it |
| **Vector norms + pairwise cosine spread** | Near-identical vectors = embedding collapse; the loss can look fine while retrieval is meaningless |
| **Nearest-neighbour dumps for ~20 articles** | The cheapest check that the space is fashion-shaped |
| **MPS/CPU parity** | Vectors must agree within tolerance on a fixed sample; MPS backends are periodically unstable |

---

## 9b. Stage 2 — ranker specifications

All three consume the identical 47 columns from the identical candidate rows.

### 9b.1 `lgbm` — the tabular baseline

```yaml
# conf/model/lgbm.yaml
objective: binary
num_leaves: 127
max_bin: 63                     # NOT a tuning choice — see §14
learning_rate: 0.05
n_estimators: 2000
early_stopping_rounds: 100      # on the val window
feature_fraction: 0.8
bagging_fraction: 0.8
bagging_freq: 1
min_data_in_leaf: 100
num_threads: 7
```

All eleven categoricals passed as native `categorical_feature`. `free_raw_data=True` after
`Dataset.construct()`.

### 9b.2 `mlp`

Categorical embeddings (8–16 d per field, 13 fields including the two customer categoricals) ⊕
standardized numerics → MLP **[512, 256, 128]** → 1. GELU, dropout 0.1, batch norm on the
numeric block. Input dim ≈ 190.

### 9b.3 `dcn` — DCN-v2

Parallel structure. Both stacks read the same input `x₀`:

```
cross layer:  x_{l+1} = x₀ ⊙ (W_l x_l + b_l) + x_l          # 3 layers, full W
deep stack:   MLP [512, 256]
head:         concat(cross_out, deep_out) → linear → 1
```

The cross layers explicitly multiply feature pairs, so the model can discover interactions like
`cust_age × index_group` without them being hand-written. **That is precisely the comparison:
`dcn` versus `mlp` measures learned crossing against the six hand-built crosses in §6.4, on
identical inputs.**

At input dim ≈190, a full-matrix cross layer is 190² ≈ 36k params; three layers plus the deep
stack is ~400k total. The low-rank DCN-v2 variant exists for wide industrial inputs and is
unnecessary here — **the rankers are tiny; memory is dominated by the data, not the model.**

### 9b.4 Shared training settings

| Setting | Value |
|---|---|
| Loss | Binary cross-entropy |
| Batch | 4096 rows, streamed from Parquet — the design matrix is never materialized |
| Optimizer | AdamW, lr 1e-3, cosine decay, 10% warmup |
| Epochs | 3, early stopping on `val` NDCG@12 |
| Seeds | 3 |

**Hyperparameters are tuned on `lgbm` only, then frozen** for `mlp` and `dcn`. Tuning each arm
separately would confound "better architecture" with "more tuning budget," and the resulting
delta would measure effort rather than design.

### 9b.5 Run matrix

| Stage | Arms | Seeds | Runs |
|---|---|---|---|
| Retrieval | `R1`, `R2` | 3 | 6 |
| Retrieval sensitivity | `R2-A`; log-Q disabled on `R2` | 1 | 2 |
| Ranking | `lgbm`, `mlp`, `dcn` | 3 | 9 |
| H3 | val-selected ranker on random negatives | 3 | 3 |
| Sensitivity | uniform negatives (appendix) | 1 | 1 |
| | | | **21** |

`pop` and the `K` sweep add no training runs — `pop` is deterministic and the sweep is a single
retrieval pass. Down from 61 in the previous design.

---

## 10. Evaluation

### 10.1 Metric signatures

```python
# eval/metrics.py
def auc(y: np.ndarray, p: np.ndarray) -> float: ...
def pr_auc(y: np.ndarray, p: np.ndarray) -> float: ...
def log_loss(y: np.ndarray, p: np.ndarray) -> float: ...
def brier(y: np.ndarray, p: np.ndarray) -> float: ...

# eval/ranking.py — per customer, then averaged
def precision_at_k(ranked: pl.DataFrame, *, k: int) -> float: ...
def recall_at_k(ranked: pl.DataFrame, *, k: int) -> float: ...
def map_at_k(ranked: pl.DataFrame, *, k: int = 12) -> float: ...
def ndcg_at_k(ranked: pl.DataFrame, *, k: int = 12) -> float: ...
```

### 10.2 Retrieval metrics — `eval/retrieval.py`

```python
def recall_at_k(retrieved: pl.DataFrame, all_positives: pl.DataFrame, *,
                k: int) -> float:
    """Fraction of true positives appearing in the top k, per customer then averaged.

    `all_positives` is the window's FULL positive set. A positive absent from
    `retrieved` contributes 0 — never dropped from the denominator.
    """

def recall_at_k_per_customer(...) -> pl.DataFrame:
    """Per-customer values, so eval/bootstrap.py can resample over customers."""

def coverage(retrieved: pl.DataFrame, *, catalog_size: int) -> float:
    """Fraction of the catalog appearing in ANY customer's top-k.

    A retriever returning the same 500 bestsellers to everyone can post an
    acceptable recall@K and be useless. Recall alone does not catch it.
    """

def cold_start_recall_at_k(retrieved, all_positives, articles, *,
                           k: int, threshold: int = 10) -> float: ...

def popularity_rho(scores: pl.DataFrame, articles: pl.DataFrame) -> float: ...
```

### 10.3 The `K` sweep

`recall@K` for every K in `candidates.k_sweep` is computed from **one** retrieval pass at
`K = max(k_sweep)` by truncation — no retraining, no re-retrieval. This is why H1's retrieval
axis is cheap relative to its ranker axis, and the asymmetry is itself part of the finding.

### 10.4 End-to-end metrics — the invariant that matters most

```python
def map_at_k_e2e(ranked: pl.DataFrame, all_positives: pl.DataFrame, *,
                 k: int = 12) -> float:
    """Pipeline-level MAP@k over EVERY positive in the window.

    Positives stage 1 never retrieved cannot appear in `ranked`, and they count as
    misses. Computing MAP only over retrieved candidates flatters the pipeline by
    exactly the retriever's miss rate, and the output looks entirely healthy while
    doing so — which is what makes it the most likely silent error in a two-stage
    evaluation.
    """
```

**Hand-checkable consequence: end-to-end MAP@12 must be strictly below ranking-only MAP@12 for
every arm.** If they are equal, the accounting is broken and every headline number is inflated.
Asserted in `tests/test_e2e_metrics.py`.

### 10.5 Slices

```python
# eval/slices.py
SLICES = {
    "all":            lambda df: df,
    "cold_start":     lambda df: df.filter(pl.col("art_is_cold") == 1),
    "low_history":    lambda df: df.filter(pl.col("cust_txn_12w") <= LOW_HISTORY_THRESHOLD),
}
```

Both thresholds are set **once at M1** from the measured distributions and registered. Retuning
a slice boundary after seeing results is p-hacking with extra steps.

### 10.6 Bootstrap — `eval/bootstrap.py`

```python
def bootstrap_delta(
    per_customer_a: pl.DataFrame, per_customer_b: pl.DataFrame,
    *, metric: Callable, n_resamples: int = 1000, seed: int = 17,
) -> tuple[float, float, float]:
    """Point estimate and 95% CI on (a − b), resampling CUSTOMERS.

    Rows within a customer share history features and basket composition, so
    row-level resampling treats correlated observations as independent and produces
    intervals narrow enough for noise to clear the significance bar.
    """

def bootstrap_delta_of_deltas(...) -> tuple[float, float, float]:
    """For H1 and H2: CI on (Δ₁ − Δ₂), computed on SHARED customer resamples.

    Independent resamples would inflate the variance of the difference and bias the
    test toward a null.
    """
```

### 10.7 Calibration — `eval/calibration.py`

```python
def prior_correct(p_sampled: np.ndarray, *, w: float) -> np.ndarray:
    """p_true = p_s / (p_s + (1 - p_s) / w). Strictly monotone, so AUC is unchanged."""

def fit_isotonic(y_val: np.ndarray, p_val: np.ndarray) -> IsotonicRegression: ...
```

| Arm | Path | Why |
|---|---|---|
| All main-grid rankers | **Isotonic fit on `val`** | Retrieval-induced sampling has no fixed downsampling rate `w` — how many negatives a customer receives depends on what the retriever returned |
| H3 random-negative arm | Prior correction **and** isotonic | It has a genuine fixed `w = 1/10`, so the two paths can be compared as a check that neither is broken |

Rank metrics (AUC, NDCG, MAP, `recall@K`) are unaffected either way. Reliability curves are
reported per arm, so calibration quality is visible rather than asserted.

### 10.8 Business proxy — `eval/business.py`

```python
def expected_revenue_at_k(ranked: pl.DataFrame, *, k: int) -> float: ...
def aov_lift_ratio(model: pl.DataFrame, baseline: pl.DataFrame, *, k: int) -> float: ...
```

Both run on **calibrated** probabilities only, and both return ratios. `price` is scaled and
anonymized (`prd.md` §3); a currency figure would be fabricated.

---

## 11. Experiment tracking

Runs **dual-write**:

1. **MLflow**, local file backend at `artifacts/mlruns`. Params include `config_sha256`, the
   candidate-set digest, the retriever checkpoint digest, arm, seed, and `K`.
2. **`reports/metrics/{run_name}.json`**, git-committed.

```json
{
  "run_name": "dcn_retrieved_s1",
  "stage": "ranking",
  "arm": "dcn",
  "retriever": "R2", "retriever_seed": 1,
  "k": 100,
  "candidates_sha256": "…",
  "config_sha256": "…",
  "split": "val",
  "metrics": {
    "all":        {"auc": 0.812, "ndcg@12": 0.104, "map@12": 0.061, "map@12_e2e": 0.025},
    "cold_start": {"...": 0.0},
    "low_history":{"...": 0.0}
  },
  "latency": {"p50_ms_per_1k": 0.0, "p95_ms_per_1k": 0.0}
}
```

`make report` regenerates every table in `reports/results.md` from these files. **No number in
the report is ever hand-typed**, and a metric change shows up as a reviewable diff. MLflow is
for exploring the 21 runs; the committed JSON is the record of what was claimed.

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
    nbytes: int | None

def bench(config: str, *, n_batches: int = 100, batch: int = 1000) -> BenchResult: ...
```

### 12.1 Configurations — per stage, so the budget is attributable

| Config | What it isolates |
|---|---|
| `stage1_customer_tower` | One customer-vector forward pass |
| `stage1_topk_exact` | Brute-force matvec over 105k × 128 + top-`K` |
| `stage1_topk_faiss` | The same via a FAISS IVF index |
| `stage2_lgbm` / `stage2_mlp` / `stage2_dcn` | Ranker forward over `K` candidates |
| `e2e_k100` / `e2e_k500` | Full pipeline at two depths — the latency half of H1 |
| `encoder_cold_cpu` | Tokenize + forward for a new article, CPU — the realistic commodity-server number |
| `encoder_onnx_int8` | Optimized cold path via `optimum` export |

Protocol: 20 warmup batches discarded, 100 measured, single process, `num_threads=7`, machine
otherwise idle. Reported with measured `peak_rss_mb` and `nbytes`.

### 12.2 The exact-versus-approximate question

`stage1_topk_exact` against `stage1_topk_faiss`, reported with both latencies **and** the recall
FAISS gives up relative to exact search. At 105k × 128 the expectation is that exact wins or
ties, making the index unnecessary infrastructure. **If FAISS wins, that is reported too** —
the point is to measure rather than to confirm.

### 12.3 The claim to confirm

```
105,000 × 128 × 4 B  ≈   54 MB   (item vectors, fp32)
105,000 × 384 × 4 B  ≈  162 MB   (raw encoder output, fp32)
105,000 × 384 × 1 B  ≈   40 MB   (int8)
```

`bench` prints `ItemIndex.nbytes` alongside the latency table, so the memory claim is evidence
rather than arithmetic. If it holds, the transformer's steady-state cost is ≈0 — it is an
offline batch job feeding a lookup table, and the cold path matters only for new articles.

### 12.4 The stage-attribution table — the H1 deliverable

Generated by `make report`, one row per intervention:

| Intervention | ΔMAP@12 (e2e) | 95% CI | Δp95 ms | Δ$/1M | Δ per ms |
|---|---|---|---|---|---|
| `K` 100 → 200 | | | | | |
| `K` 100 → 500 | | | | | |
| `R1` → `R2` (text in retriever) | | | | | |
| `lgbm` → `mlp` | | | | | |
| `lgbm` → `dcn` | | | | | |
| random → hard negatives | | | | | |

The last column is what makes retrieval and ranking commensurable, and H1 is the comparison
between its rows.

---

## 13. CLI surface

`typer`-based, installed as `contentsignal`. Every command is **idempotent** and skips work when
its outputs exist with a matching `config_sha256`, unless `--force`.

| Command | Reads | Writes |
|---|---|---|
| `ingest` | Kaggle CSVs | `artifacts/parquet/*` |
| `splits` | `conf/split.yaml` | — (prints the window table with roles) |
| `sample` | transactions, cohort | cohort, per-window positives, `artifacts/rows_random/*` |
| `build-features --group G --window W` | transactions, candidates | `artifacts/features/G/W.parquet` |
| `train-retriever --arm pop\|R1\|R2 --variant a\|b --seed N` | positives, articles, customers | tower checkpoint, diagnostics, metrics JSON |
| `embed --retriever R --seed N` | checkpoint, articles | `artifacts/vectors/*` |
| `retrieve --window W --k K` | checkpoint, vectors, transactions | `artifacts/candidates/W.parquet`, manifest |
| `train-ranker --arm lgbm\|mlp\|dcn --negatives retrieved\|random --seed N` | features, candidates | model, MLflow run, metrics JSON |
| `evaluate --stage retrieval\|ranking\|e2e --arm A --split val\|test` | model, features, positives | metrics JSON, figures |
| `bench --config C` | checkpoints, vectors | `reports/metrics/bench.json` |
| `report` | `reports/metrics/*.json` | `reports/results.md` |

`finetune` is **gone**. There is no separate contrastive pre-training stage: in-batch softmax
over co-purchase pairs *is* a contrastive objective, and the encoder trains jointly with the
towers inside `train-retriever`.

A `Makefile` chains these into `make m1` … `make m9` matching the milestones.

---

## 14. Resource budgets

Against the 8 GB ceiling, with ~1.5 GB reserved for OS and overhead → **~6.5 GB usable**.

| Stage | Peak RSS | How it stays bounded |
|---|---|---|
| `ingest` | **~1.5 GB** | DuckDB streams CSV → Parquet; 31.8M rows never materialize |
| `sample` | **~1.2 GB** | Per-window; alias table ~60k entries; positives ~65k |
| `build-features` (tabular) | **~2.5 GB** | Per-window DuckDB aggregation, Polars result only |
| `train-retriever` | **~3.0 GB** | ~30M params (22M encoder + 6.7M customer-ID table + ~1M towers) × 4 B × 4 Adam states ≈ 480 MB; activations for batch 512 with ~450 unique texts × 64 tokens ≈ 700 MB |
| `embed` | **~1.0 GB** | 105k articles at batch 256; output 54 MB |
| `retrieve` | **~1.0 GB** | 54 MB mmap'd index; customers chunked at 4096; `[4096 × 105k]` score block computed in tiles, never held whole |
| `train-ranker` (`lgbm`) | **~2.0 GB** | See below |
| `train-ranker` (`mlp`/`dcn`) | **~1.5 GB** | ~400k params; minibatches of 4096 streamed from Parquet, design matrix never materialized |
| `evaluate` | **~1.5 GB** | One window at a time |

### The LightGBM budget in detail

Still the binding constraint, though less tight than before. At 5M rows × 47 columns:

| | |
|---|---|
| Raw float32 matrix | 5M × 47 × 4 B = **940 MB** |
| Binned, `max_bin=63` → uint8 | 5M × 47 × 1 B = **235 MB** |
| Histograms | 127 leaves × 47 features × 63 bins × 2 doubles ≈ **6 MB** |
| Prediction/gradient buffers | ~120 MB |

`max_bin=63` is therefore **not** a tuning choice — it is what makes the run fit. Combined with
`free_raw_data=True` after `Dataset.construct()`, peak lands near **1.3 GB**, with headroom for
the transient raw matrix during construction (~2.0 GB peak). Dropping the 32 embedding columns
(§7) is what bought the difference from the previous design's 3.5 GB.

### Wall-clock estimates

| Milestone | Estimate |
|---|---|
| M0 install + download | 30–60 min (network-bound) |
| M1 ingest + EDA + leakage tests | 45 min |
| M2 cohort, positives, features (10 windows) | 1.5–2 h |
| M3 retrievers (`R1`, `R2`, `R2-A`, log-Q ablation) | 1.5–2 h |
| M4 retrieval eval + `K` sweep + bootstrap | 45 min |
| M5 candidate generation (6 windows) + retrieval features | 1 h |
| M6 three rankers × 3 seeds | 2–3 h |
| M7 H3 ablation + calibration + business proxy | 1 h |
| M8 cost profiling | 45 min |
| M9 report | — |

---

## 15. Test specification

### `tests/test_leakage.py`

**The deletion-invariance property**, applied to every registered feature builder — the general
form of every leakage check in this project:

```python
@pytest.mark.parametrize("builder", ALL_BUILDERS)
def test_features_are_invariant_to_future_deletion(builder, synthetic_txns):
    as_of = date(2020, 6, 1)
    full      = builder.build(synthetic_txns, as_of=as_of, entities=E)
    truncated = builder.build(
        synthetic_txns.filter(pl.col("t_dat") < as_of), as_of=as_of, entities=E
    )
    assert_frame_equal(full, truncated)   # recomputing without the future changes nothing
```

Plus:

| Test | Assertion |
|---|---|
| `test_retriever_windows_precede_candidate_windows` | **New.** `max(end)` over `role == "retriever"` windows < `min(start)` over every candidate window. A retriever that trained on a window it retrieves for has memorized the labels |
| `test_candidates_have_prior_history` | Every retrieved article had ≥1 transaction before `W.start` — the `allowed` mask actually applied |
| `test_negative_candidates_have_prior_history` | Same, for the H3 random-negative sampler |
| `test_no_builder_signature_lacks_as_of` | Introspects every `FeatureBuilder.build` — `as_of` present, keyword-only, no default |
| `test_history_excludes_the_cutoff_date` | `history()` uses strict `<`: a transaction dated exactly `as_of` is the label window's first day and must not be visible |
| `test_every_builder_module_is_imported_by_the_package` | Every module beside `features/base.py` is imported by `features/__init__.py`. `ALL_BUILDERS` fills by import side effect, so an unimported builder leaves the harness green while covering nothing |
| `test_customer_history_respects_as_of` | Deletion-invariance applied to the tower's `hist_ids` construction |

### `tests/test_splits.py`

Windows are contiguous, non-overlapping, 14 days each; every window lies inside the dataset
range; roles are ordered `retriever` < `ranker` < `val` < `test`; exactly one `val` and one
`test` window; `windows_for_role` and `candidate_windows` return what their names claim.

### `tests/test_retrieval.py`

| Test | Assertion |
|---|---|
| `test_customer_tower_is_item_independent` | `CustomerTower.forward` accepts no item argument, and its output is bit-identical for the same customer across different candidate batches. **This is what keeps both towers precomputable and retrieval possible** |
| `test_logq_correction_applied` | On a synthetic skewed item distribution, corrected logits recover uniform expected ranking; uncorrected ones do not |
| `test_in_batch_duplicates_masked` | An item appearing twice in a batch is never its own negative |
| `test_output_is_l2_normalized` | Both tower outputs have unit norm within tolerance |
| `test_recall_at_k_is_monotone` | `recall@K` is non-decreasing in K. A violation means the truncation in §10.3 is wrong |
| `test_topk_respects_allowed_mask` | Masked articles never appear in output, at any `K` |
| `test_index_lookup_matches_dense` | `ItemIndex.topk` agrees with a naive dense argsort on a small fixture |
| `test_candidate_digest_detects_mutation` | Flipping one byte of a candidate file makes `assert_digest` raise |

### `tests/test_e2e_metrics.py`

| Test | Assertion |
|---|---|
| `test_e2e_metrics_count_unretrieved_positives` | A synthetic window where stage 1 misses a known positive yields end-to-end MAP **strictly below** ranking-only MAP |
| `test_e2e_recall_bounded_by_retrieval_recall` | End-to-end `recall@12` ≤ `recall@K` for every arm — the pipeline ceiling, asserted rather than assumed |
| `test_perfect_ranker_hits_retrieval_ceiling` | With an oracle ranker, end-to-end `recall@12` equals `recall@K` when `K ≤ 12`; this pins the ceiling arithmetic |

### `tests/test_sampling.py`

Retained for the H3 arm: exact ratio per customer, positives never sampled as negatives, no
duplicate negatives, byte-identical output at a fixed seed, per-window seed stable across
`PYTHONHASHSEED` values, popularity weighting correct over many draws, and a pool too small to
supply `ratio` distinct negatives raises rather than returning a short draw that would silently
change the effective sampling rate and with it the prior correction.

### `tests/test_calibration.py`

`prior_correct` round-trips on synthetic data — given a known base rate and downsampling rate,
corrected probabilities recover the true base rate within Monte-Carlo error — and correction is
strictly monotone, so AUC is unchanged. `fit_isotonic` is monotone and reduces Brier score on
held-out data.

---

## 16. Definition of done

| M | Artifact | Test / gate | Metric that must appear |
|---|---|---|---|
| **M0** | `uv.lock`, 3 CSVs on disk | `contentsignal --help` runs | — |
| **M1** | `artifacts/parquet/*`, `01_eda.ipynb` | **`test_splits.py`, `test_leakage.py` green** | Row counts, date range, token-length distribution, cold-start and low-history distributions (which fix both thresholds) |
| **M2** | Cohort, positives, all 5 feature groups | Deletion-invariance green over `ALL_BUILDERS` | Actual positives and eligible customers per window |
| **M3** | Retriever checkpoints + item vectors | **`test_retrieval.py` green**; collapse, popularity-ρ, MPS/CPU parity pass | `val recall@100` for `pop`, `R1`, `R2`; log-Q ablation delta |
| **M4** | Retrieval evaluation | — | **H2: Δ`recall@100` cold-start vs all, with CIs on both and on the difference of deltas**; coverage; the full `K` sweep |
| **M5** | `artifacts/candidates/*` + manifest, retrieval features | Row budget respected; window-ordering test green | Rows, customers, `recall@K` per candidate window |
| **M6** | `lgbm`, `mlp`, `dcn` trained | Digest assertion passes on every arm | Ranking **and** end-to-end NDCG@12 / MAP@12, all three slices |
| **M7** | H3 ablation, calibration, business proxy | `test_calibration.py`, `test_e2e_metrics.py` green | **H3: ΔMAP@12 (hard − random) with CI**; reliability curves; AOV lift ratio |
| **M8** | Benchmark results | — | Per-stage latency, exact vs FAISS, $/1M, measured `nbytes` |
| **M9** | `reports/results.md`, `README.md` | `make report` reproduces every number | **H1 stage-attribution table, and the §2 verdict on all three hypotheses: supported or null** |

**M1 gates everything** — no model is trained until the leakage tests pass. **M3 gates stage 2**
— if no retriever beats `pop`, that is the finding, and it is cheaper to discover there than at
M9.
