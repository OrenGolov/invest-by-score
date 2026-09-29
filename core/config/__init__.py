"""Versioned configuration for the whole platform.

Split in C1 out of a single 9,783-line module — 20% of all core code, with
1,092 constants and 76 import-time validators. Nothing about the CONTENT changed:
every constant keeps its value, every validator still runs at import, and
`from core.config import ANYTHING` resolves exactly as before.

WHY THE PARTS CHAIN. MEASURED, 88 constants are read across part boundaries (a
label horizon defined with the V-sprint labels is used by the E-sprint event
config, and so on). Independent modules would each need an explicit import list
that drifts; chaining reproduces the original single namespace in the original
order, so the split is textual and not semantic.

    _base          W1-W3 ensemble/risk/audit and N1-N5 news/macro/regime
    _validation    V1-V8 labels/backtest and M1-M8 the ML layer
    _events        E1-E7 event intelligence
    _charts        C1-C7 chart and temporal intelligence
    _forecasting   F1-F8 forecasting, Sprint L learning, R1-R2 sizing
    _portfolio     R3-R7 portfolio risk and D1-D4 the dashboard
    _alerts        A1-A7 the alert family
    _release       X1-X10 the release gate and Sprint B

The comments are deliberately NOT trimmed. They are 38% of the text and carry the
measurement behind each threshold — why a noise band is 1.4 + 0.8*ln(cells), why an
embargo must cover the longest horizon. Losing them would leave numbers nobody
could re-derive.
"""

from core.config._release import *  # noqa: F401,F403

# A STAR IMPORT DROPS UNDERSCORE NAMES AT EVERY LINK, so chaining leaves only the
# last part's validators reachable. MEASURED, 63 of the 76 went missing that way.
# They have already RUN — importing any part executes them — but they must stay
# reachable BY NAME, because a test asserts one of them raises on a bad weight set.
from core.config import (  # noqa: F401
    _alerts,
    _base,
    _charts,
    _events,
    _forecasting,
    _learning,
    _portfolio,
    _release,
    _sizing,
    _validation,
)

_module = globals()
for _part in (
    _base,
    _validation,
    _events,
    _charts,
    _forecasting,
    _learning,
    _sizing,
    _portfolio,
    _alerts,
    _release,
):
    for _name in dir(_part):
        if _name.startswith("_") and not _name.startswith("__"):
            _module.setdefault(_name, getattr(_part, _name))
del _module, _part, _name
