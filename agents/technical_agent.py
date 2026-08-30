"""Technical analysis agent (W5: single technical truth).

This module is a thin adapter over the canonical scorers in
`core.score_engine` — the single source of technical truth. The independent
multi-factor formula that used to live here was deleted: it disagreed with
the published score by construction (split-brain), since the headline came
from `_score_current_time`/`_score_long_term` with different coefficients.

The disjoint-feature contract still holds and stays test-enforced in
`tests/test_score_separation.py`: `_score_current_time` never reads
long-horizon inputs (150d/200d averages, 60d returns) and
`_score_long_term` never reads short-horizon inputs (1d/5d/20d movement,
RSI, volume, 50d/100d averages).
"""

from __future__ import annotations

from core.score_engine import _score_current_time, _score_long_term


def score_technical(snapshot: dict, news_snapshot: dict | None = None) -> float:
    """Blended technical view: (current + long) / 2 from the canonical scorers.

    Passing the decision's real news snapshot keeps this view identical to
    the one the ensemble consumed (news sentiment is embedded in the
    current-time view until Sprint N1 gives it a dedicated weight).
    """
    news = news_snapshot if news_snapshot is not None else {"status": "UNAVAILABLE"}
    current = _score_current_time(snapshot, news)
    long_term = _score_long_term(snapshot)
    return round((current + long_term) / 2.0, 2)
