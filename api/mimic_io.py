"""Finding and reading MIMIC-IV tables, whatever shape the download is in.

Two jobs, both boring and both places a loader usually breaks.

**Finding files.** PhysioNet downloads land in several layouts depending on
whether you used ``wget``, the ZIP, or a BigQuery export, and MIMIC-IV-ED is a
separate project that people unpack next to, inside, or far away from the main
one. :func:`find_table` looks in all of them rather than hard-coding one.

**Reading chartevents without a 64 GB machine.** ``icu/chartevents.csv.gz`` is
roughly 330 million rows -- about 30 GB expanded. Loading it is not an option,
so :func:`read_filtered` streams it in chunks, discards everything but the
itemids and stays we want, and concatenates only the survivors. For the
vitals in this project that is a ~0.1% keep rate, which is the difference
between a few hundred MB of memory and none of it fitting.
"""

from __future__ import annotations

import gzip
import pathlib
import time
from typing import Any, Callable, Iterator

import pandas as pd

# Where each table lives, relative to a MIMIC root. Tried in order.
_MODULE_DIRS: dict[str, tuple[str, ...]] = {
    # MIMIC-IV core
    "patients": ("hosp", "mimiciv_hosp", "core", "."),
    "admissions": ("hosp", "mimiciv_hosp", "core", "."),
    "transfers": ("hosp", "mimiciv_hosp", "core", "."),
    "icustays": ("icu", "mimiciv_icu", "."),
    "chartevents": ("icu", "mimiciv_icu", "."),
    "inputevents": ("icu", "mimiciv_icu", "."),
    "procedureevents": ("icu", "mimiciv_icu", "."),
    "d_items": ("icu", "mimiciv_icu", "."),
    # MIMIC-IV-ED, frequently unpacked as a sibling or a subdirectory
    "edstays": ("ed", "mimiciv_ed", "mimic-iv-ed/ed", "ed/ed", "."),
    "triage": ("ed", "mimiciv_ed", "mimic-iv-ed/ed", "ed/ed", "."),
    "vitalsign": ("ed", "mimiciv_ed", "mimic-iv-ed/ed", "ed/ed", "."),
    "medrecon": ("ed", "mimiciv_ed", "mimic-iv-ed/ed", "ed/ed", "."),
    "pyxis": ("ed", "mimiciv_ed", "mimic-iv-ed/ed", "ed/ed", "."),
}

_SUFFIXES = (".csv.gz", ".csv", ".parquet")

# Version directories a download may be nested under (e.g. mimic-iv/2.2/icu/).
_VERSION_GLOBS = ("", "*.*", "mimic-iv*", "mimiciv*")


def find_table(root: str | pathlib.Path, name: str) -> pathlib.Path | None:
    """Locate one MIMIC table under ``root``, or ``None``.

    Case-insensitive, suffix-agnostic, and tolerant of the version directory
    that some downloads introduce and others do not.
    """
    root = pathlib.Path(root).expanduser()
    if not root.exists():
        return None

    subdirs = _MODULE_DIRS.get(name, (".",))
    bases: list[pathlib.Path] = []
    for version in _VERSION_GLOBS:
        bases.extend([root] if not version else sorted(root.glob(version)))

    for base in bases:
        if not base.is_dir():
            continue
        for sub in subdirs:
            d = base / sub if sub != "." else base
            if not d.is_dir():
                continue
            for suffix in _SUFFIXES:
                for candidate in (name + suffix, name.upper() + suffix):
                    p = d / candidate
                    if p.exists():
                        return p
            # last resort: case-insensitive scan of this directory only
            for p in d.iterdir():
                if p.is_file() and p.name.lower().startswith(name.lower() + "."):
                    if any(p.name.lower().endswith(s) for s in _SUFFIXES):
                        return p
    return None


def available(root: str | pathlib.Path) -> dict[str, pathlib.Path]:
    """Every known table present under ``root``. Drives module detection."""
    return {
        name: p
        for name in _MODULE_DIRS
        if (p := find_table(root, name)) is not None
    }


def read_table(
    path: pathlib.Path,
    usecols: list[str] | None = None,
    parse_dates: list[str] | None = None,
    dtype: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Read a whole table. Only for the small ones -- never chartevents."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=usecols)
        for c in parse_dates or []:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        return df
    return pd.read_csv(
        path,
        usecols=usecols,
        parse_dates=parse_dates,
        dtype=dtype,
        low_memory=False,
    )


def _chunks(
    path: pathlib.Path,
    usecols: list[str] | None,
    chunksize: int,
    dtype: dict[str, Any] | None,
) -> Iterator[pd.DataFrame]:
    if path.suffix == ".parquet":
        # Parquet is columnar; one read with a column projection is already
        # the efficient path, so there is nothing to stream.
        yield pd.read_parquet(path, columns=usecols)
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as fh:
        yield from pd.read_csv(
            fh, usecols=usecols, chunksize=chunksize, dtype=dtype, low_memory=False
        )


def read_filtered(
    path: pathlib.Path,
    keep: Callable[[pd.DataFrame], pd.DataFrame],
    usecols: list[str] | None = None,
    parse_dates: list[str] | None = None,
    dtype: dict[str, Any] | None = None,
    chunksize: int = 2_000_000,
    label: str = "",
    log: Callable[[str], None] | None = print,
) -> pd.DataFrame:
    """Stream a large table, keeping only what ``keep`` returns.

    ``keep`` receives each raw chunk and returns the subset to retain. Date
    parsing happens *after* filtering, because parsing timestamps on 330
    million rows costs far more than reading them does.
    """
    t0 = time.perf_counter()
    kept: list[pd.DataFrame] = []
    n_read = 0

    for chunk in _chunks(path, usecols, chunksize, dtype):
        n_read += len(chunk)
        sub = keep(chunk)
        if len(sub):
            kept.append(sub)
        if log and n_read % (chunksize * 10) == 0:
            held = sum(len(k) for k in kept)
            log(f"    {label or path.name}: {n_read:,} rows scanned, {held:,} kept")

    out = (
        pd.concat(kept, ignore_index=True)
        if kept
        else pd.DataFrame(columns=usecols or [])
    )
    for c in parse_dates or []:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")

    if log:
        dt = time.perf_counter() - t0
        pct = 100.0 * len(out) / n_read if n_read else 0.0
        log(f"    {label or path.name}: {n_read:,} scanned -> {len(out):,} kept "
            f"({pct:.2f}%) in {dt:.0f}s")
    return out
