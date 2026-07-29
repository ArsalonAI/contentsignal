"""Ingest — the casts and validations that fail silently if they fail at all.

`trd.md` §15 has no ingest section, because when it was written there was no ingest body.
These are the behaviours in §4.1–4.4 that a wrong implementation still produces plausible
output for: an `article_id` truncated by inference, a `customer_idx` that shifted between
runs, a null that became a zero. Each one would surface many stages later as a mediocre
metric rather than an error.

The synthetic CSVs follow `conftest.py`'s convention — tiny and fully deterministic, so the
contract is provable in milliseconds and stays provable after the real data exists.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from contentsignal.data import schema as sc
from contentsignal.data.to_parquet import (
    IngestReport,
    IngestValidationError,
    convert,
    existing_report,
    outputs_exist,
)

# Customer ids in the dataset are 64-char hex. `customer_idx` is assigned by sorting on this
# string, so the ids are deliberately NOT in the CSV's row order: c3 is written first.
CUST = {f"c{i}": f"{i:064x}" for i in range(4)}

RANGE = (date(2020, 6, 1), date(2020, 6, 3))


def _write(path: Path, header: tuple[str, ...] | list[str], rows: list[list[str]]) -> None:
    lines = [",".join(header)] + [",".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _customer_row(cid: str, fn: str, active: str, club: str, news: str, age: str) -> list[str]:
    return [cid, fn, active, club, news, age, "postalhash"]


def write_csvs(raw: Path, *, customers_reversed: bool = False) -> None:
    """The three CSVs, with the null and leading-zero cases the schema has to handle."""
    raw.mkdir(parents=True, exist_ok=True)

    customers = [
        # c3 first: row order must not influence customer_idx.
        _customer_row(CUST["c3"], "1.0", "1.0", "ACTIVE", "Regularly", "31"),
        _customer_row(CUST["c0"], "", "", "ACTIVE", "NONE", "49"),
        _customer_row(CUST["c1"], "1.0", "", "PRE-CREATE", "NONE", ""),  # null age preserved
        _customer_row(CUST["c2"], "", "", "", "", "22"),  # null categoricals
    ]
    if customers_reversed:
        customers.reverse()
    _write(raw / sc.CUSTOMERS_CSV, list(sc.READ_CUSTOMERS), customers)

    # Zero-padded ten-digit ids, including the real dataset's minimum and maximum.
    articles = [
        ["0108775015", "0108775", "Strap top", "Vest top", "Garment Upper body"],
        ["0959461001", "0959461", "Cap", "Cap", "Accessories"],
        ["0200000001", "0200000", "No description", "Trousers", "Garment Lower body"],
    ]
    art_header = list(sc.READ_ARTICLES)
    art_rows = []
    for idx, (aid, pcode, pname, ptype, pgroup) in enumerate(articles):
        row = dict.fromkeys(art_header, f"v{idx}")
        row |= {
            "article_id": aid,
            "product_code": pcode,
            "prod_name": pname,
            "product_type_name": ptype,
            "product_group_name": pgroup,
            "department_no": str(1600 + idx),
            # The third article has no description — a legitimate encoder input that must
            # become "" rather than dropping the article out of the cold-start slice.
            "detail_desc": "" if pname == "No description" else f"desc {idx}",
        }
        art_rows.append([row[col] for col in art_header])
    _write(raw / sc.ARTICLES_CSV, art_header, art_rows)

    # Out of date order on purpose: the written file must come back sorted.
    transactions = [
        ["2020-06-03", CUST["c1"], "0959461001", "0.05", "2"],
        ["2020-06-01", CUST["c3"], "0108775015", "0.03", "1"],
        ["2020-06-01", CUST["c0"], "0108775015", "0.03", "2"],
        ["2020-06-02", CUST["c2"], "0200000001", "0.01", "1"],
        ["2020-06-02", CUST["c0"], "0959461001", "0.04", "2"],
    ]
    _write(raw / sc.TRANSACTIONS_CSV, list(sc.READ_TRANSACTIONS), transactions)


@pytest.fixture
def ingested(tmp_path: Path) -> tuple[Path, Path, IngestReport]:
    raw, out = tmp_path / "data", tmp_path / "parquet"
    write_csvs(raw)
    report = convert(raw_dir=raw, out_dir=out, temp_dir=tmp_path / "tmp", expected_range=RANGE)
    return raw, out, report


# --------------------------------------------------------------------------- casts


def test_leading_zero_ids_are_cast_not_inferred(ingested: tuple[Path, Path, IngestReport]) -> None:
    """`0108775015` is a zero-padded string; int32 is 108775015, and it must not overflow.

    The dataset's maximum, 959461001, sits inside int32 with room to spare — which is what
    makes the §4.1/§4.2 narrowing safe rather than lucky.
    """
    _, out, _ = ingested
    articles = pl.read_parquet(out / "articles.parquet")
    assert articles.schema["article_id"] == pl.Int32
    assert articles["article_id"].to_list() == [108775015, 200000001, 959461001]
    assert articles["product_code"].to_list() == [108775, 200000, 959461]


def test_written_schemas_match_the_declaration(ingested: tuple[Path, Path, IngestReport]) -> None:
    """Every file's columns and dtypes are exactly §4.1–4.4. Drift is what blows §14's budgets."""
    _, out, _ = ingested
    for name, expected in (
        ("transactions", sc.TRANSACTIONS_SCHEMA),
        ("articles", sc.ARTICLES_SCHEMA),
        ("customers", sc.CUSTOMERS_SCHEMA),
        ("customer_index", sc.CUSTOMER_INDEX_SCHEMA),
    ):
        found = dict(pl.scan_parquet(out / f"{name}.parquet").collect_schema())
        assert list(found) == list(expected), name
        assert all(found[c] == expected[c] for c in expected), name


def test_transactions_are_sorted_by_date_then_customer(
    ingested: tuple[Path, Path, IngestReport],
) -> None:
    """§4.1's sort order. It is what lets row-group statistics prune on the `as_of` filter."""
    _, out, _ = ingested
    txns = pl.read_parquet(out / "transactions.parquet")
    assert txns.select("t_dat", "customer_idx").rows() == sorted(
        txns.select("t_dat", "customer_idx").rows()
    )


# --------------------------------------------------------------------------- customer_idx


def test_customer_idx_is_dense_and_independent_of_row_order(tmp_path: Path) -> None:
    """The mapping is a function of the CSV's contents, not its row order.

    §4.4: `customer_index.parquet` is never regenerated because everything downstream keys
    on it. That guarantee is only worth having if the derivation is reproducible, so the
    same customers in a different order must produce the same indices.
    """
    frames = []
    for tag, reverse in (("fwd", False), ("rev", True)):
        raw, out = tmp_path / f"data_{tag}", tmp_path / f"pq_{tag}"
        write_csvs(raw, customers_reversed=reverse)
        convert(raw_dir=raw, out_dir=out, temp_dir=tmp_path / "tmp", expected_range=RANGE)
        frames.append(pl.read_parquet(out / "customer_index.parquet").sort("customer_idx"))

    assert frames[0]["customer_idx"].to_list() == [0, 1, 2, 3]
    assert frames[0]["customer_id"].to_list() == [CUST[f"c{i}"] for i in range(4)]
    assert frames[0].equals(frames[1])


def test_customer_index_is_not_rewritten_and_the_mismatch_is_caught(tmp_path: Path) -> None:
    """A grown customers.csv must fail loudly, not silently re-index.

    Re-deriving `customer_idx` would shift existing customers and invalidate every artifact
    keyed on it — while every individual file still looked well-formed. `ingest` therefore
    reuses the index and the row-count check turns the drift into an error.
    """
    raw, out = tmp_path / "data", tmp_path / "parquet"
    write_csvs(raw)
    convert(raw_dir=raw, out_dir=out, temp_dir=tmp_path / "tmp", expected_range=RANGE)
    before = (out / "customer_index.parquet").read_bytes()

    with (raw / sc.CUSTOMERS_CSV).open("a", encoding="utf-8") as fh:
        fh.write(",".join(_customer_row(f"{9:064x}", "1.0", "1.0", "ACTIVE", "NONE", "40")) + "\n")

    with pytest.raises(IngestValidationError, match="customer_index"):
        convert(raw_dir=raw, out_dir=out, temp_dir=tmp_path / "tmp", expected_range=RANGE)
    assert (out / "customer_index.parquet").read_bytes() == before

    # The explicit escape hatch, for a genuine from-scratch rebuild.
    report = convert(
        raw_dir=raw,
        out_dir=out,
        temp_dir=tmp_path / "tmp",
        expected_range=RANGE,
        force_customer_index=True,
    )
    assert not report.customer_index_reused
    assert dict((t.name, t.rows) for t in report.tables)["customer_index"] == 5


def test_unresolvable_customer_id_raises(tmp_path: Path) -> None:
    """A transaction for an unknown customer must not be quietly dropped by the inner join."""
    raw, out = tmp_path / "data", tmp_path / "parquet"
    write_csvs(raw)
    with (raw / sc.TRANSACTIONS_CSV).open("a", encoding="utf-8") as fh:
        fh.write(f"2020-06-02,{'f' * 64},0108775015,0.02,1\n")

    with pytest.raises(IngestValidationError, match="transactions"):
        convert(raw_dir=raw, out_dir=out, temp_dir=tmp_path / "tmp", expected_range=RANGE)


# --------------------------------------------------------------------------- nulls


def test_null_age_is_preserved_but_null_flags_become_zero(
    ingested: tuple[Path, Path, IngestReport],
) -> None:
    """§4.3 / §6.1. `cust_age_is_null` exists as a feature; `cust_fn`/`cust_active` have no
    such flag, so an absent value there means not flagged rather than unknown."""
    _, out, _ = ingested
    customers = pl.read_parquet(out / "customers.parquet").sort("customer_idx")
    assert customers["age"].to_list() == [49.0, None, 22.0, 31.0]
    assert customers["FN"].to_list() == [0, 1, 0, 1]
    assert customers["Active"].to_list() == [0, 0, 0, 1]
    assert customers["club_member_status"].to_list() == [
        "ACTIVE",
        "PRE-CREATE",
        None,
        "ACTIVE",
    ]


def test_null_detail_desc_becomes_empty_string_and_is_counted(
    ingested: tuple[Path, Path, IngestReport],
) -> None:
    """§4.2. Dropping description-less articles would bias the cold-start slice H2 turns on."""
    _, out, report = ingested
    articles = pl.read_parquet(out / "articles.parquet")
    assert articles["detail_desc"].null_count() == 0
    assert (articles["detail_desc"] == "").sum() == 1
    assert report.empty_detail_desc == 1


def test_every_declared_categorical_is_written(ingested: tuple[Path, Path, IngestReport]) -> None:
    """All eleven of §6.3 reach the file. PRD §3: withholding any of them from a baseline
    while feeding the same information to the encoder as text invalidates the comparison."""
    _, out, _ = ingested
    columns = pl.scan_parquet(out / "articles.parquet").collect_schema().names()
    assert set(sc.ARTICLE_CATEGORICALS) <= set(columns)
    assert len(sc.ARTICLE_CATEGORICALS) == 11


# --------------------------------------------------------------------------- validation


def test_date_range_mismatch_raises(tmp_path: Path) -> None:
    """The check that catches a dataset whose span disagrees with every window boundary."""
    raw, out = tmp_path / "data", tmp_path / "parquet"
    write_csvs(raw)
    with pytest.raises(IngestValidationError, match="does not match the configured"):
        convert(
            raw_dir=raw,
            out_dir=out,
            temp_dir=tmp_path / "tmp",
            expected_range=(date(2018, 9, 20), date(2020, 9, 22)),
        )


def test_reordered_csv_header_raises(tmp_path: Path) -> None:
    """`read_csv(columns=...)` binds by position, so a reordered header would be read under
    the wrong names and cast to the wrong types without any downstream symptom."""
    raw, out = tmp_path / "data", tmp_path / "parquet"
    write_csvs(raw)
    path = raw / sc.CUSTOMERS_CSV
    lines = path.read_text(encoding="utf-8").splitlines()
    swapped = lines[0].split(",")
    swapped[1], swapped[2] = swapped[2], swapped[1]
    path.write_text("\n".join([",".join(swapped), *lines[1:]]) + "\n", encoding="utf-8")

    with pytest.raises(IngestValidationError, match="unexpected header"):
        convert(raw_dir=raw, out_dir=out, temp_dir=tmp_path / "tmp", expected_range=RANGE)


def test_no_tmp_files_survive_a_successful_run(ingested: tuple[Path, Path, IngestReport]) -> None:
    """Writes go to `*.parquet.tmp` then `os.replace`, so a partial file is never readable."""
    _, out, _ = ingested
    assert outputs_exist(out)
    assert list(out.glob("*.tmp")) == []


def test_existing_report_matches_a_fresh_conversion(
    ingested: tuple[Path, Path, IngestReport],
) -> None:
    """A skipped run must print the same numbers a converting run does — otherwise `--force`
    changes what gets reported, and the report stops being evidence about the files."""
    _, out, report = ingested
    reused = existing_report(out)
    assert reused.skipped
    assert reused.tables == report.tables
    assert (reused.first_txn, reused.last_txn) == (report.first_txn, report.last_txn)
    assert reused.empty_detail_desc == report.empty_detail_desc
