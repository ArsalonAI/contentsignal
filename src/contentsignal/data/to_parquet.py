"""CSV → Parquet conversion (`trd.md` §4.1–4.4, §14).

DuckDB streams the conversion; the 31.8M transaction rows never materialize, which is what
keeps this stage inside its ~1.5 GB budget on an 8 GB machine. Polars is used only to read
back the written schema for validation.

Three properties this module is responsible for, each chosen against a specific failure:

**Atomicity.** Every file is written to `*.parquet.tmp` and `os.replace`d. A partial Parquet
file must never be readable, for the same reason §4.5 demands it of candidates: a truncated
file that still parses is indistinguishable from a smaller dataset.

**`customer_index.parquet` is written once.** §4.4 — every downstream artifact keys on
`customer_idx`, so re-deriving the mapping would silently invalidate all of them while every
individual file still looked well-formed. `force=True` deliberately does *not* touch it.

**Validation raises rather than warns.** The row counts, the transaction date range, and the
density of `customer_idx` are all checked against independent sources after the write. The
date-range check is the highest-value one in the stage: every window boundary in
`conf/split.yaml` is asserted against that range, so a dataset that does not match it makes
all ten windows wrong without anything else failing.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

import duckdb
import polars as pl

from contentsignal.config import DataConfig, config_sha256
from contentsignal.data import schema as sc

# §14 budget. The memory limit is what forces DuckDB to spill the transaction sort to
# `temp_dir` instead of growing past the ceiling; 6 of 8 cores leaves room for the OS.
MEMORY_LIMIT: Final = "2GB"
THREADS: Final = 6

# §4.1 asks for 256 MB row groups. At 17 B/row that is 15.8M rows — two row groups for the
# whole table, which makes the per-group `t_dat` min/max statistics useless for precisely
# the `as_of` range pruning the sort order exists to enable. 2M rows (~34 MB) keeps groups
# large enough to scan efficiently while leaving ~16 of them to prune against.
ROW_GROUP_ROWS: Final = 2_000_000


class IngestValidationError(RuntimeError):
    """A written Parquet file disagrees with its source or with the declared schema."""


def ingest_digest(cfg: DataConfig) -> str:
    """The digest `ingest` stamps, over only what determines the file *contents*.

    Deliberately narrower than `config_sha256(cfg)`. Digesting the whole `DataConfig` would
    make an unrelated edit to `candidates.k` — a knob this stage never reads — invalidate a
    multi-minute conversion. `paths` is excluded for the same reason: it decides where the
    output lands, and the stamp is co-located with it, so a changed path is a fresh
    directory rather than a stale one.
    """
    return config_sha256(
        {
            "competition": cfg.kaggle.competition,
            "parquet_schema_version": sc.PARQUET_SCHEMA_VERSION,
        }
    )


@dataclass(frozen=True)
class TableReport:
    name: str
    rows: int
    bytes: int


@dataclass(frozen=True)
class IngestReport:
    """What `ingest` prints. Every number is measured from the written files."""

    tables: tuple[TableReport, ...]
    first_txn: date
    last_txn: date
    empty_detail_desc: int
    customer_index_reused: bool
    skipped: bool = False


def _lit(value: Path | str) -> str:
    """A SQL string literal, single quotes escaped."""
    return "'" + str(value).replace("'", "''") + "'"


def _columns_sql(columns: dict[str, str]) -> str:
    return "{" + ", ".join(f"{_lit(name)}: {_lit(dtype)}" for name, dtype in columns.items()) + "}"


def _assert_header(path: Path, expected: Iterable[str]) -> None:
    """The CSV header names, in order, match what `schema.py` declares.

    `read_csv` with an explicit `columns=` mapping binds by position. A re-download with
    reordered columns would therefore be read under the wrong names and cast to the wrong
    types — quietly, and with plausible-looking output.
    """
    with path.open(encoding="utf-8") as fh:
        header = [name.strip() for name in fh.readline().rstrip("\n").split(",")]
    if header != list(expected):
        raise IngestValidationError(
            f"{path.name}: unexpected header.\n  expected: {list(expected)}\n  found:    {header}"
        )


def _connect(temp_dir: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit = {_lit(MEMORY_LIMIT)}")
    con.execute(f"SET threads = {THREADS}")
    con.execute(f"SET temp_directory = {_lit(temp_dir)}")
    return con


def _csv_view(
    con: duckdb.DuckDBPyConnection, name: str, path: Path, columns: dict[str, str]
) -> None:
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS "
        f"SELECT * FROM read_csv({_lit(path)}, header = true, columns = {_columns_sql(columns)})"
    )


def _scalar(con: duckdb.DuckDBPyConnection, query: str) -> Any:
    row = con.execute(query).fetchone()
    if row is None:
        raise IngestValidationError(f"query returned no rows: {query}")
    return row[0]


def _copy_atomic(con: duckdb.DuckDBPyConnection, query: str, dest: Path) -> None:
    tmp = dest.parent / (dest.name + ".tmp")
    tmp.unlink(missing_ok=True)
    con.execute(
        f"COPY ({query}) TO {_lit(tmp)} "
        f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP_ROWS})"
    )
    os.replace(tmp, dest)


def _assert_schema(path: Path, expected: dict[str, pl.DataType]) -> None:
    """Written columns and dtypes match the declaration exactly, order included.

    Column drift is an error here for the same reason §4.6 makes it one for feature groups:
    a dtype that widened back to float64 or an int64 `article_id` would blow the memory
    budgets in §14 several stages later, where the cause is no longer visible.
    """
    found = dict(pl.scan_parquet(path).collect_schema())
    if list(found) != list(expected):
        raise IngestValidationError(
            f"{path.name}: column drift.\n  expected: {list(expected)}\n  found:    {list(found)}"
        )
    mismatched = {c: (str(expected[c]), str(found[c])) for c in expected if found[c] != expected[c]}
    if mismatched:
        raise IngestValidationError(f"{path.name}: dtype mismatch (expected, found): {mismatched}")


def _report_table(name: str, path: Path, rows: int) -> TableReport:
    return TableReport(name=name, rows=rows, bytes=path.stat().st_size)


def convert(
    *,
    raw_dir: Path,
    out_dir: Path,
    temp_dir: Path,
    expected_range: tuple[date, date],
    force_customer_index: bool = False,
    log: Callable[[str], None] = lambda _msg: None,
) -> IngestReport:
    """Convert the three raw CSVs into `out_dir`, then validate what was written.

    Unconditional: the caller owns the stamp check that decides whether to skip. Calling
    this always rewrites, except for `customer_index.parquet` (§4.4).

    `expected_range` is the dataset's `(first, last)` transaction date from
    `conf/split.yaml`, and is required rather than optional: it is the only check that
    catches a dataset whose span silently disagrees with every window boundary.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    txn_csv = raw_dir / sc.TRANSACTIONS_CSV
    art_csv = raw_dir / sc.ARTICLES_CSV
    cust_csv = raw_dir / sc.CUSTOMERS_CSV

    _assert_header(txn_csv, sc.READ_TRANSACTIONS)
    _assert_header(art_csv, sc.READ_ARTICLES)
    _assert_header(cust_csv, sc.READ_CUSTOMERS)

    index_path = out_dir / "customer_index.parquet"
    cust_path = out_dir / "customers.parquet"
    art_path = out_dir / "articles.parquet"
    txn_path = out_dir / "transactions.parquet"

    con = _connect(temp_dir)
    try:
        _csv_view(con, "raw_transactions", txn_csv, sc.READ_TRANSACTIONS)
        _csv_view(con, "raw_articles", art_csv, sc.READ_ARTICLES)
        _csv_view(con, "raw_customers", cust_csv, sc.READ_CUSTOMERS)

        # --- customer_index: written once, never regenerated (§4.4) ------------------
        reused = index_path.is_file() and not force_customer_index
        if reused:
            log(f"customer_index: reusing {index_path.name} (§4.4: written once)")
        else:
            log("customer_index: assigning dense customer_idx ordered by customer_id")
            _copy_atomic(con, sc.SELECT_CUSTOMER_INDEX, index_path)
        con.execute(
            "CREATE OR REPLACE VIEW customer_index AS "
            f"SELECT * FROM read_parquet({_lit(index_path)})"
        )

        # --- the three tables that key on it ----------------------------------------
        log("customers: joining index, narrowing dtypes")
        _copy_atomic(con, sc.SELECT_CUSTOMERS, cust_path)

        log("articles: casting ids, filling null detail_desc")
        _copy_atomic(con, sc.select_articles(), art_path)

        log("transactions: streaming 31.8M rows, sorting by (t_dat, customer_idx)")
        _copy_atomic(con, sc.SELECT_TRANSACTIONS, txn_path)

        # --- validation --------------------------------------------------------------
        _assert_schema(index_path, sc.CUSTOMER_INDEX_SCHEMA)
        _assert_schema(cust_path, sc.CUSTOMERS_SCHEMA)
        _assert_schema(art_path, sc.ARTICLES_SCHEMA)
        _assert_schema(txn_path, sc.TRANSACTIONS_SCHEMA)

        report = _validate(
            con,
            index_path=index_path,
            cust_path=cust_path,
            art_path=art_path,
            txn_path=txn_path,
            expected_range=expected_range,
            customer_index_reused=reused,
        )
    finally:
        con.close()

    return report


def _validate(
    con: duckdb.DuckDBPyConnection,
    *,
    index_path: Path,
    cust_path: Path,
    art_path: Path,
    txn_path: Path,
    expected_range: tuple[date, date],
    customer_index_reused: bool,
) -> IngestReport:
    """Check every written file against an independent source. Raises on any disagreement."""
    csv_txns = int(_scalar(con, "SELECT count(*) FROM raw_transactions"))
    csv_arts = int(_scalar(con, "SELECT count(*) FROM raw_articles"))
    csv_custs = int(_scalar(con, "SELECT count(*) FROM raw_customers"))

    def rows(path: Path) -> int:
        return int(_scalar(con, f"SELECT count(*) FROM read_parquet({_lit(path)})"))

    pq_txns, pq_arts, pq_custs, pq_index = (
        rows(txn_path),
        rows(art_path),
        rows(cust_path),
        rows(index_path),
    )

    # Row-count equality is also how the joins are checked. `SELECT_TRANSACTIONS` and
    # `SELECT_CUSTOMERS` inner-join `customer_index`, so an unresolvable customer_id drops
    # a row rather than producing a null — equal counts are what prove nothing was dropped.
    for label, csv_rows, pq_rows in (
        ("transactions", csv_txns, pq_txns),
        ("articles", csv_arts, pq_arts),
        ("customers", csv_custs, pq_custs),
    ):
        if csv_rows != pq_rows:
            raise IngestValidationError(
                f"{label}: {pq_rows:,} rows written from {csv_rows:,} CSV rows. "
                "An inner join against customer_index dropped rows, or the CSV changed "
                "since customer_index.parquet was written."
            )

    if pq_index != csv_custs:
        raise IngestValidationError(
            f"customer_index has {pq_index:,} rows but customers.csv has {csv_custs:,}. "
            "customer_index.parquet is written once and was built from a different CSV; "
            "rebuilding it invalidates every artifact keyed on customer_idx."
        )

    # Dense 0..n-1. Every embedding table and every `customer_idx`-indexed array downstream
    # assumes this; a gap becomes an untrained row or an out-of-bounds index.
    idx_max, idx_distinct = con.execute(
        "SELECT max(customer_idx), count(DISTINCT customer_idx) "
        f"FROM read_parquet({_lit(index_path)})"
    ).fetchone() or (None, None)
    if idx_distinct != pq_index or idx_max != pq_index - 1:
        raise IngestValidationError(
            f"customer_idx is not dense: {pq_index:,} rows, {idx_distinct:,} distinct, "
            f"max {idx_max:,} (expected max {pq_index - 1:,})"
        )

    first_txn, last_txn = con.execute(
        f"SELECT min(t_dat), max(t_dat) FROM read_parquet({_lit(txn_path)})"
    ).fetchone() or (None, None)
    if (first_txn, last_txn) != expected_range:
        raise IngestValidationError(
            f"transaction range {first_txn}..{last_txn} does not match the configured "
            f"dataset range {expected_range[0]}..{expected_range[1]}. Every window boundary "
            "in conf/split.yaml is stated against that range, so all ten windows would be "
            "wrong. Fix conf/split.yaml or the dataset — do not proceed."
        )

    # §4.2 requires this count be reported: an empty description is a legitimate encoder
    # input, but it lands disproportionately on the cold-start slice that H2 turns on.
    empty_desc = int(
        _scalar(con, f"SELECT count(*) FROM read_parquet({_lit(art_path)}) WHERE detail_desc = ''")
    )

    return IngestReport(
        tables=(
            _report_table("transactions", txn_path, pq_txns),
            _report_table("articles", art_path, pq_arts),
            _report_table("customers", cust_path, pq_custs),
            _report_table("customer_index", index_path, pq_index),
        ),
        first_txn=first_txn,
        last_txn=last_txn,
        empty_detail_desc=empty_desc,
        customer_index_reused=customer_index_reused,
    )


def existing_report(out_dir: Path) -> IngestReport:
    """Re-derive the report from files already on disk, so a skipped run prints the same numbers.

    Row counts come from Parquet footers, so this is metadata-only.
    """
    paths = {
        "transactions": out_dir / "transactions.parquet",
        "articles": out_dir / "articles.parquet",
        "customers": out_dir / "customers.parquet",
        "customer_index": out_dir / "customer_index.parquet",
    }
    con = duckdb.connect()
    try:
        tables = tuple(
            _report_table(
                name, path, int(_scalar(con, f"SELECT count(*) FROM read_parquet({_lit(path)})"))
            )
            for name, path in paths.items()
        )
        first_txn, last_txn = con.execute(
            f"SELECT min(t_dat), max(t_dat) FROM read_parquet({_lit(paths['transactions'])})"
        ).fetchone() or (None, None)
        empty_desc = int(
            _scalar(
                con,
                f"SELECT count(*) FROM read_parquet({_lit(paths['articles'])}) "
                "WHERE detail_desc = ''",
            )
        )
    finally:
        con.close()
    return IngestReport(
        tables=tables,
        first_txn=first_txn,
        last_txn=last_txn,
        empty_detail_desc=empty_desc,
        customer_index_reused=True,
        skipped=True,
    )


def outputs_exist(out_dir: Path) -> bool:
    return all(
        (out_dir / f"{name}.parquet").is_file()
        for name in ("transactions", "articles", "customers", "customer_index")
    )
