"""Shared training-dataset fixture.

Building a dataset replays the live scoring path per prediction time, which
costs ~45s. Twelve test classes each called `setUpClass` and rebuilt the
SAME dataset, so the suite spent most of its time repeating identical work.

This caches by (ticker, slice, fixture mtime). The mtime is part of the key
so a changed parquet file cannot be served from a stale cache — correctness
first, speed second.
"""

from __future__ import annotations

import pathlib
import unittest

import pandas as pd

FIXTURE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

_frames: dict[str, pd.DataFrame] = {}
_datasets: dict[tuple, object] = {}


def fixture_path(ticker: str = "NVDA") -> pathlib.Path:
    return FIXTURE_DIR / f"{ticker}_5y_1d.parquet"


def load_frame(ticker: str = "NVDA") -> pd.DataFrame:
    """Read a cached price frame once per process."""
    path = fixture_path(ticker)
    if not path.exists():
        raise unittest.SkipTest(f"fixture {path.name} not available")
    if ticker not in _frames:
        frame = pd.read_parquet(path)
        frame.index = pd.DatetimeIndex(frame.index)
        _frames[ticker] = frame
    return _frames[ticker]


def shared_dataset(start: int, stop: int | None, ticker: str = "NVDA"):
    """Build a training dataset once per (ticker, slice, fixture version).

    Returns the SAME object to every caller, so a test that mutates it must
    restore what it changed — the existing tests already do this in
    try/finally blocks.
    """
    path = fixture_path(ticker)
    if not path.exists():
        raise unittest.SkipTest(f"fixture {path.name} not available")

    key = (ticker, start, stop, path.stat().st_mtime_ns)
    if key not in _datasets:
        from core.training_dataset import build_training_dataset

        frame = load_frame(ticker)
        times = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in frame.index[start:stop]]
        _datasets[key] = build_training_dataset({ticker: times}, {ticker: frame})
    return _datasets[key]
