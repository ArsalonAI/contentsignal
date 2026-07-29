"""Kaggle download — the fresh-clone path only (`trd.md` §2).

Isolated in its own module, and importing `kaggle` inside the function rather than at module
scope, because the common case is that the CSVs are already on disk. The `kaggle` package
reads `~/.kaggle/kaggle.json` at *import* time and raises if it is missing, so a top-level
import would resurrect exactly the blocker this arrangement removes: a machine with the data
present but no credentials could not run `ingest`.
"""

from __future__ import annotations

import zipfile
from collections.abc import Callable
from pathlib import Path

from contentsignal.data.schema import RAW_FILES


def download_competition(
    competition: str,
    dest: Path,
    *,
    log: Callable[[str], None] = lambda _msg: None,
) -> None:
    """Download and unzip the competition files into `dest`.

    The caller has already verified credentials (`cli.check_kaggle_credentials`); a 403 here
    means the competition rules have not been accepted, which no local check can detect.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

    dest.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    log(f"downloading {competition} to {dest} (multi-GB, network-bound)")
    api.competition_download_files(competition, path=str(dest), quiet=False)

    archive = dest / f"{competition}.zip"
    if archive.is_file():
        log(f"extracting {archive.name}")
        with zipfile.ZipFile(archive) as zf:
            # Only the three files the pipeline reads. The archive also carries a 25 GB
            # `images/` tree and the sample submission, none of which any stage touches.
            for name in RAW_FILES:
                zf.extract(name, path=dest)
        archive.unlink()

    missing = [name for name in RAW_FILES if not (dest / name).is_file()]
    if missing:
        raise RuntimeError(
            f"download completed but {missing} are absent from {dest}. "
            "If the Kaggle API returned a small HTML file instead of the archive, the "
            "competition rules have not been accepted."
        )
