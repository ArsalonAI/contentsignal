"""The `contentsignal` CLI (trd.md §13).

Every command is idempotent: it skips work when its outputs already exist with a matching
config hash, unless `--force`. Commands whose bodies land in a later milestone raise
`NotImplementedError` naming that milestone — loudly, never as a silent no-op, so a
half-built pipeline cannot be mistaken for a run that produced nothing.
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from contentsignal import __version__
from contentsignal.config import (
    load_data_config,
    load_split_config,
    outputs_are_current,
    resolve_path,
    write_stamp,
)
from contentsignal.data import to_parquet
from contentsignal.data.schema import RAW_FILES
from contentsignal.splits.temporal import candidate_windows, load_windows

app = typer.Typer(
    name="contentsignal",
    help=(
        "Two-stage recommender on the H&M catalog: does the next unit of engineering "
        "buy more in retrieval or in ranking?"
    ),
    no_args_is_help=True,
    add_completion=False,
)


def _todo(command: str, milestone: str, what: str) -> NoReturn:
    raise NotImplementedError(f"{command}: lands at {milestone} — {what} (see trd.md §16)")


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", help="Print version and exit.")] = False,
) -> None:
    if version:
        typer.echo(f"contentsignal {__version__}")
        raise typer.Exit()


# --------------------------------------------------------------------------- M0/M1


class KaggleCredentialsError(RuntimeError):
    """Raised by the ingest preflight when the dataset cannot be downloaded."""


def check_kaggle_credentials(path: Path | None = None) -> Path:
    """Verify Kaggle credentials before attempting a multi-GB download (trd.md §2).

    Fails fast and specifically. The two failure modes are easy to confuse — a missing
    token and unaccepted competition rules both surface as a 403 from the Kaggle API
    much later — so the message names both explicitly.
    """
    creds = path or Path.home() / ".kaggle" / "kaggle.json"
    if not creds.is_file():
        raise KaggleCredentialsError(
            f"missing {creds}.\n"
            "  1. Accept the competition rules at\n"
            "     https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/rules\n"
            "     (the API returns 403 until this is done, even with a valid token)\n"
            "  2. Create an API token at https://www.kaggle.com/settings/account\n"
            f"  3. Save it to {creds} and run: chmod 600 {creds}"
        )
    mode = stat.S_IMODE(creds.stat().st_mode)
    if mode & 0o077:
        raise KaggleCredentialsError(
            f"{creds} is mode {mode:04o}; it is readable by other users on this machine.\n"
            f"  Fix with: chmod 600 {creds}"
        )
    return creds


def _report(report: to_parquet.IngestReport) -> None:
    for table in report.tables:
        typer.echo(f"  {table.name:<15} {table.rows:>12,} rows  {table.bytes / 1e6:>8.1f} MB")
    typer.echo(f"  transaction range   {report.first_txn} .. {report.last_txn}")
    typer.echo(f"  empty detail_desc   {report.empty_detail_desc:,} articles")


@app.command()
def ingest(
    force: Annotated[bool, typer.Option("--force", help="Rebuild even if outputs exist.")] = False,
    force_customer_index: Annotated[
        bool,
        typer.Option(
            "--force-customer-index",
            help="Also re-derive customer_index.parquet. Invalidates every downstream artifact.",
        ),
    ] = False,
) -> None:
    """Convert the raw CSVs to Parquet, downloading them from Kaggle only if absent.

    The credential check runs only when a file is actually missing. Requiring a Kaggle
    token to convert CSVs that are already on disk blocks the stage on a step that has
    nothing left to do.
    """
    cfg = load_data_config()
    split_cfg = load_split_config()

    raw_dir = resolve_path(cfg.paths.raw)
    parquet_dir = resolve_path(cfg.paths.parquet)
    stamp = parquet_dir / "_stamp"
    digest = to_parquet.ingest_digest(cfg)

    missing = [name for name in RAW_FILES if not (raw_dir / name).is_file()]
    if missing:
        typer.echo(f"raw files missing from {raw_dir}: {', '.join(missing)}")
        try:
            creds = check_kaggle_credentials()
        except KaggleCredentialsError as exc:
            typer.secho(f"ingest: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc
        typer.echo(f"credentials ok: {creds}")

        from contentsignal.data.download import download_competition  # noqa: PLC0415

        download_competition(cfg.kaggle.competition, raw_dir, log=typer.echo)
    else:
        typer.echo(f"raw files present in {raw_dir}; skipping download")

    if outputs_are_current(stamp, digest, force=force) and to_parquet.outputs_exist(parquet_dir):
        typer.echo(
            f"ingest: up to date at {parquet_dir} (config {digest[:12]}); --force to rebuild"
        )
        _report(to_parquet.existing_report(parquet_dir))
        return

    try:
        report = to_parquet.convert(
            raw_dir=raw_dir,
            out_dir=parquet_dir,
            temp_dir=resolve_path(cfg.paths.artifacts) / "tmp",
            expected_range=(split_cfg.dataset_start, split_cfg.dataset_end),
            force_customer_index=force_customer_index,
            log=typer.echo,
        )
    except to_parquet.IngestValidationError as exc:
        typer.secho(f"ingest: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    write_stamp(stamp, digest)
    typer.echo(f"ingest: wrote {parquet_dir} (config {digest[:12]})")
    _report(report)


@app.command()
def splits() -> None:
    """Print the resolved window table from conf/split.yaml, with roles."""
    cfg = load_split_config()
    windows = load_windows(cfg)
    typer.echo(f"dataset range: {cfg.dataset_start} .. {cfg.dataset_end}")
    typer.echo(
        f"{'window':<10} {'split':<6} {'role':<10} {'start':<12} {'end':<12} {'as_of (<)':<12}"
    )
    for w in windows:
        typer.echo(
            f"{w.name:<10} {w.split:<6} {w.role:<10} {w.start!s:<12} {w.end!s:<12} {w.as_of!s:<12}"
        )

    # The leakage boundary the second stage introduces, surfaced rather than implied.
    trained = [w for w in windows if w.role == "retriever"]
    served = candidate_windows(cfg)
    typer.echo("")
    typer.echo(
        f"retriever trains on {len(trained)} window(s) ending {max(w.end for w in trained)}; "
        f"retrieves for {len(served)} window(s) starting {min(w.start for w in served)}"
    )


# --------------------------------------------------------------------------- M2


@app.command()
def sample(
    force: Annotated[bool, typer.Option("--force", help="Resample even if rows exist.")] = False,
) -> None:
    """Draw the cohort and materialize per-window positives.

    Also writes the random-negative row sets used by the H3 arm (trd.md §8.3). The main
    grid's negatives come from `retrieve`, not from here.
    """
    _todo("sample", "M2", "cohort draw, positives, random negatives for the H3 arm")


@app.command("build-features")
def build_features(
    group: Annotated[str, typer.Option("--group", "-g", help="Feature group to build.")],
    window: Annotated[str, typer.Option("--window", "-w", help="Window name.")],
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Build one feature group for one window, gated behind that window's as_of cutoff."""
    _todo("build-features", "M2/M5", f"builder for group {group!r} on window {window!r}")


# --------------------------------------------------------------------------- M3: stage 1


@app.command("train-retriever")
def train_retriever(
    arm: Annotated[str, typer.Option("--arm", "-a", help="pop, R1, or R2.")],
    variant: Annotated[str, typer.Option("--variant", "-v", help="Text variant: a or b.")] = "b",
    seed: Annotated[int, typer.Option("--seed")] = 1,
    no_logq: Annotated[
        bool, typer.Option("--no-logq", help="Disable the log-Q correction (ablation only).")
    ] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Train the two-tower retriever with in-batch sampled softmax.

    Trains on retriever-role windows only — never on a window it will retrieve for. The
    text encoder trains jointly with the towers; there is no separate contrastive
    pre-training stage, because in-batch softmax over co-purchase pairs *is* a contrastive
    objective (trd.md §9).
    """
    _todo("train-retriever", "M3", f"two-tower training, arm {arm!r}, variant {variant!r}")


@app.command()
def embed(
    retriever: Annotated[str, typer.Option("--retriever", "-r", help="Retriever arm name.")],
    seed: Annotated[int, typer.Option("--seed")] = 1,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Encode all ~105k articles once from the frozen item tower into the mmap-able cache.

    This is the step that makes the transformer's steady-state serving cost approximately
    zero: everything downstream reads a 54 MB matrix, and the encoder never runs online
    except for genuinely new articles (trd.md §9.4).
    """
    _todo("embed", "M3", f"encode 105k articles from retriever {retriever!r} seed {seed}")


# --------------------------------------------------------------------------- M5: candidates


@app.command()
def retrieve(
    window: Annotated[str, typer.Option("--window", "-w", help="Window name.")],
    k: Annotated[
        int | None, typer.Option("--k", help="Candidate depth; defaults to conf/data.yaml.")
    ] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Generate top-K candidates for one window and record their digest.

    Written once, then immutable: every ranker asserts the digest before fitting, so a
    ΔNDCG of 0.005 cannot be attributable to one arm receiving an easier candidate list.

    Reports actual row counts and fails loudly if ranker training rows exceed the budget,
    naming the `train_customer_cap` that would fit (trd.md §3.4).
    """
    _todo("retrieve", "M5", f"top-k retrieval for window {window!r}, digest the manifest")


# --------------------------------------------------------------------------- M6/M7: stage 2


@app.command("train-ranker")
def train_ranker(
    arm: Annotated[str, typer.Option("--arm", "-a", help="lgbm, mlp, or dcn.")],
    negatives: Annotated[
        str, typer.Option("--negatives", help="retrieved (main grid) or random (H3 arm).")
    ] = "retrieved",
    seed: Annotated[int, typer.Option("--seed")] = 1,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Train one ranker on the frozen candidate set. Asserts the digest before fitting.

    `--negatives random` trains the H3 comparison arm on popularity-weighted random
    negatives instead. Both members of that pair have the retrieval feature columns
    withheld, so the delta measures the negative distribution alone (trd.md §8.3).
    """
    _todo("train-ranker", "M6", f"fit ranker {arm!r} on {negatives!r} negatives")


@app.command()
def evaluate(
    arm: Annotated[str, typer.Option("--arm", "-a")],
    stage: Annotated[str, typer.Option("--stage", help="retrieval, ranking, or e2e.")] = "e2e",
    split: Annotated[str, typer.Option("--split", help="val or test.")] = "val",
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Score an arm on all three slices, with customer-level bootstrap CIs.

    `--stage e2e` computes metrics over EVERY positive in the window, including those the
    retriever never surfaced — they count as misses. Ranking-only metrics computed over
    retrieved candidates flatter the pipeline by exactly the retriever's miss rate
    (trd.md §10.4).

    The test split is read once, at M9. Everything before that reports validation.
    """
    _todo("evaluate", "M4", f"{stage} metrics x slices for {arm!r} on split {split!r}")


# --------------------------------------------------------------------------- M8/M9


@app.command()
def bench(
    config: Annotated[str, typer.Option("--config", "-c", help="Benchmark configuration.")],
) -> None:
    """Profile per-stage inference latency, throughput, and cost per million predictions.

    Per stage rather than end-to-end only, so the latency budget is attributable — and so
    the exact-versus-FAISS comparison at a 105k catalog is measured rather than assumed
    (trd.md §12).
    """
    _todo("bench", "M8", f"per-stage latency and $/1M under config {config!r}")


@app.command()
def report() -> None:
    """Regenerate reports/results.md from the committed per-run metrics JSON.

    No number in the report is ever hand-typed.
    """
    _todo("report", "M9", "render tables from reports/metrics/*.json")


if __name__ == "__main__":  # pragma: no cover
    app()
