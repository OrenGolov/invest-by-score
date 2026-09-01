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
RSI, volume, 50d/100d averages). Since Sprint N1, news sentiment is no
longer embedded in the current-time view either: it enters the published
score exclusively through the dedicated `news_intelligence` ensemble line.
"""

from __future__ import annotations

from core.score_engine import _score_current_time, _score_long_term


def score_technical(snapshot: dict) -> float:
    """Blended technical view: (current + long) / 2 from the canonical scorers.

    Takes no news input by design: the news view is the ensemble's
    `news_intelligence` contribution (N1), not a hidden term inside the
    technical score. Embedding it here as well would count the same
    evidence twice.
    """
    current = _score_current_time(snapshot)
    long_term = _score_long_term(snapshot)
    return round((current + long_term) / 2.0, 2)
