"""The parquet column contract — `trd.md` §4.1–4.4 in one checkable place.

Two halves that must agree:

- **`READ_*`** — the DuckDB `read_csv` column types. Every column is declared, so no dtype
  in this pipeline is ever the product of CSV sniffing. `article_id` is read as `VARCHAR`
  on purpose: it is a zero-padded ten-digit string (`0108775015`), and a cast is a
  deliberate, testable step where inference is a coin flip.
- **`*_SCHEMA`** — the Polars dtypes the written file must have. `to_parquet` asserts these
  after every conversion, so column drift is an error rather than a surprise three stages
  later, matching the stance §4.6 takes on feature groups.

`PARQUET_SCHEMA_VERSION` is folded into the ingest digest so that editing this module
invalidates the stamp on disk. Without it, a narrowed dtype or a renamed column would leave
`ingest` skipping as up-to-date against files written under the old contract.
"""

from __future__ import annotations

from typing import Final

import polars as pl

PARQUET_SCHEMA_VERSION: Final = 1

# Kaggle competition filenames. `RAW_FILES` is what `ingest` checks for on disk before it
# considers reaching for credentials.
TRANSACTIONS_CSV: Final = "transactions_train.csv"
ARTICLES_CSV: Final = "articles.csv"
CUSTOMERS_CSV: Final = "customers.csv"
RAW_FILES: Final[tuple[str, ...]] = (TRANSACTIONS_CSV, ARTICLES_CSV, CUSTOMERS_CSV)

# --------------------------------------------------------------------------- categoricals

# The eleven from `trd.md` §6.3, in that order. Parquet has no categorical logical type —
# it dictionary-encodes the values physically, and this tuple is what tells the reader to
# cast them back to `pl.Categorical`.
#
# `department_no` is numeric in the CSV but is one of the eleven, so it is stored as its
# literal CSV text. Keeping the set type-uniform costs nothing: it is only ever consumed as
# a category, never as a number.
#
# PRD §3 non-negotiable: every ranker arm and both retriever arms receive all eleven.
ARTICLE_CATEGORICALS: Final[tuple[str, ...]] = (
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "department_no",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
    "graphical_appearance_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
)

CUSTOMER_CATEGORICALS: Final[tuple[str, ...]] = (
    "club_member_status",
    "fashion_news_frequency",
)

# --------------------------------------------------------------------------- CSV read specs

READ_TRANSACTIONS: Final[dict[str, str]] = {
    "t_dat": "DATE",
    "customer_id": "VARCHAR",
    "article_id": "VARCHAR",  # zero-padded; cast to INT32 in the projection
    "price": "DOUBLE",
    "sales_channel_id": "TINYINT",
}

# `articles.csv` is read entirely as text. Half its columns are zero-padded codes, and the
# projection casts exactly the two that are genuinely numeric downstream.
READ_ARTICLES: Final[dict[str, str]] = dict.fromkeys(
    (
        "article_id",
        "product_code",
        "prod_name",
        "product_type_no",
        "product_type_name",
        "product_group_name",
        "graphical_appearance_no",
        "graphical_appearance_name",
        "colour_group_code",
        "colour_group_name",
        "perceived_colour_value_id",
        "perceived_colour_value_name",
        "perceived_colour_master_id",
        "perceived_colour_master_name",
        "department_no",
        "department_name",
        "index_code",
        "index_name",
        "index_group_no",
        "index_group_name",
        "section_no",
        "section_name",
        "garment_group_no",
        "garment_group_name",
        "detail_desc",
    ),
    "VARCHAR",
)

# `FN` and `Active` are `1.0` or empty in the CSV — never `0` — hence DOUBLE, then
# `coalesce(..., 0)`. See the note on `SELECT_CUSTOMERS`.
READ_CUSTOMERS: Final[dict[str, str]] = {
    "customer_id": "VARCHAR",
    "FN": "DOUBLE",
    "Active": "DOUBLE",
    "club_member_status": "VARCHAR",
    "fashion_news_frequency": "VARCHAR",
    "age": "DOUBLE",
    "postal_code": "VARCHAR",
}

# --------------------------------------------------------------------------- projections

# §4.4 — `customer_id` (hex str) -> dense `customer_idx`. Ordering by the id rather than by
# file position makes the mapping a pure function of the CSV's *contents*, so it reproduces
# byte-for-byte on a fresh clone regardless of row order. Written once, never regenerated:
# every downstream artifact keys on `customer_idx`.
SELECT_CUSTOMER_INDEX: Final = """
    SELECT
        customer_id,
        CAST(row_number() OVER (ORDER BY customer_id) - 1 AS INTEGER) AS customer_idx
    FROM raw_customers
"""

# §4.3. `postal_code` is dropped — a hash used by no feature group in §6, and most of the
# 198 MB of the CSV.
#
# Null `FN`/`Active` become 0. §6.1 gives `cust_fn`/`cust_active` as int8 with no null flag,
# unlike `cust_age`, which carries an explicit `cust_age_is_null`; absent means not flagged.
# Null `age` IS preserved, because that flag exists and the MLP/DCN arms consume it.
SELECT_CUSTOMERS: Final = """
    SELECT
        idx.customer_idx                        AS customer_idx,
        CAST(raw.age AS FLOAT)                  AS age,
        CAST(coalesce(raw.FN, 0) AS TINYINT)    AS FN,
        CAST(coalesce(raw.Active, 0) AS TINYINT) AS Active,
        raw.club_member_status                  AS club_member_status,
        raw.fashion_news_frequency              AS fashion_news_frequency
    FROM raw_customers AS raw
    JOIN customer_index AS idx USING (customer_id)
    ORDER BY idx.customer_idx
"""

# §4.2. Null `detail_desc` becomes the empty string: an empty description is a legitimate
# input to the text encoder, and dropping those articles would bias the cold-start slice —
# which is the slice H2 turns on. The count is reported by `ingest`.
SELECT_ARTICLES: Final = """
    SELECT
        CAST(article_id AS INTEGER)   AS article_id,
        CAST(product_code AS INTEGER) AS product_code,
        prod_name                     AS prod_name,
        {categoricals},
        coalesce(detail_desc, '')     AS detail_desc
    FROM raw_articles
    ORDER BY article_id
"""

# §4.1. Sorted by `t_dat` then `customer_idx`: every consumer scans a date range, so the
# sort is what lets Parquet row-group statistics skip whole groups on the `as_of` filter.
SELECT_TRANSACTIONS: Final = """
    SELECT
        idx.customer_idx                       AS customer_idx,
        CAST(raw.article_id AS INTEGER)        AS article_id,
        raw.t_dat                              AS t_dat,
        CAST(raw.price AS FLOAT)               AS price,
        CAST(raw.sales_channel_id AS TINYINT)  AS sales_channel_id
    FROM raw_transactions AS raw
    JOIN customer_index AS idx USING (customer_id)
    ORDER BY raw.t_dat, idx.customer_idx
"""


def select_articles() -> str:
    """`SELECT_ARTICLES` with the eleven categorical columns spliced in, in §6.3's order."""
    return SELECT_ARTICLES.format(categoricals=",\n        ".join(ARTICLE_CATEGORICALS))


# --------------------------------------------------------------------------- output schemas

CUSTOMER_INDEX_SCHEMA: Final[dict[str, pl.DataType]] = {
    "customer_id": pl.String(),
    "customer_idx": pl.Int32(),
}

TRANSACTIONS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "customer_idx": pl.Int32(),
    "article_id": pl.Int32(),
    "t_dat": pl.Date(),
    "price": pl.Float32(),
    "sales_channel_id": pl.Int8(),
}

CUSTOMERS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "customer_idx": pl.Int32(),
    "age": pl.Float32(),
    "FN": pl.Int8(),
    "Active": pl.Int8(),
    # Stored as strings; `CUSTOMER_CATEGORICALS` drives the cast to `pl.Categorical` on read.
    "club_member_status": pl.String(),
    "fashion_news_frequency": pl.String(),
}

ARTICLES_SCHEMA: Final[dict[str, pl.DataType]] = {
    "article_id": pl.Int32(),
    "product_code": pl.Int32(),
    "prod_name": pl.String(),
    **dict.fromkeys(ARTICLE_CATEGORICALS, pl.String()),
    "detail_desc": pl.String(),
}
