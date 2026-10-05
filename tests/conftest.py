"""Session-wide guard: the test suite does not touch a news provider.

**THE MEASUREMENT THAT MADE THIS NECESSARY.** Configuring a real
``FINNHUB_API_KEY`` on the operator's machine turned the suite from ~6 minutes
into **23 minutes** and produced one failure that read as a reproducibility
defect but was not:

``test_identical_inputs_rerun_to_identical_results`` runs the walk-forward
backtest TWICE and asserts the two agree. The backtest replays the LIVE scoring
path, and ``build_score`` fetches news — so the first run received real articles,
the second hit a rate limit, and the engine faithfully produced different outputs
from different inputs. A 204,505-character diff, and nothing wrong with the
engine.

MEASURED, with the key present and absent:

    tests/test_backtest.py     live provider  >580s (unfinished)  ->  44s silenced
    tests/test_training.py     live provider  minutes             ->  31s silenced

16 test files reach ``build_score`` or ``orchestrate_score``, so patching them one
by one would fix today's 16 and none of tomorrow's.

**WHY THE KEYS ARE CLEARED RATHER THAN THE NETWORK BLOCKED.** Clearing the
credential exercises the system's own no-key path — the ``UNAVAILABLE`` contract
every adapter already implements and every downstream consumer already handles.
Blocking sockets would instead raise from inside ``urllib``, testing an error path
the production system never takes.

**A TEST THAT WANTS A PROVIDER STILL GETS ONE.** This is an autouse fixture over
``os.environ``, so any test may set its own keys with ``patch.dict`` and they
take precedence for its duration — which is exactly what ``test_news_adapter.py``
and ``test_finnhub_news.py`` already do. The guard removes the AMBIENT
credential, not the ability to supply one.

**IT ALSO CLOSES THE EIGHTH INSTANCE OF THE CHECKOUT-DEPENDENCE CLASS**, in its
inverted form: these tests passed in CI, where no keys exist, and failed on a
working machine. A suite whose result depends on which credentials the developer
happens to have configured is measuring the machine, not the repository.
"""

from __future__ import annotations

import os

import pytest

# Every credential that would send a test over the network. Listed EXPLICITLY
# rather than pattern-matched on "*_API_KEY": a guard that silently covers a
# variable nobody declared would also silently stop covering a renamed one.
_PROVIDER_KEYS: tuple[str, ...] = (
    "FINNHUB_API_KEY",
    "NEWS_PROVIDER_API_KEY",
    "ALPHAVANTAGE_API_KEY",
    "FRED_API_KEY",
)


# SESSION-SCOPED, and that scope is the whole point.
#
# CAUGHT BY MEASUREMENT: a function-scoped autouse fixture runs AFTER
# `setUpClass`, and `tests/_dataset_fixture.shared_dataset` builds a training
# dataset in exactly that hook -- reaching `build_score` and the network before
# any per-test fixture exists. MEASURED on tests/test_training.py: 30s with no
# key in the environment, 580s with the key exported and only the function-scoped
# guard in place. The guard was running, just too late to matter.
#
# `autouse=True` at session scope installs it once, before collection-time and
# setup-time work alike.
@pytest.fixture(scope="session", autouse=True)
def _no_live_providers_session() -> None:
    """Clear provider credentials for the whole session, before any setUpClass.

    `os.environ` is edited directly rather than through monkeypatch, because
    monkeypatch's session fixture is a different object and this must be in
    place before pytest builds any class fixture.
    """
    saved = {name: os.environ.get(name) for name in _PROVIDER_KEYS}
    for name in _PROVIDER_KEYS:
        os.environ[name] = ""
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(autouse=True)
def _no_live_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every provider credential for the duration of each test.

    `monkeypatch.setenv` to EMPTY rather than `delenv`, because the adapters
    read with `os.getenv(NAME)` and treat `""` as absent — the same way they
    behave on a machine that never configured the key. Deleting would also work,
    but setting empty keeps the environment shaped the way production sees it
    when a key is present-but-blank.

    A test that needs a provider overrides this with its own `patch.dict`, which
    applies after the fixture and wins for that test.
    """
    for name in _PROVIDER_KEYS:
        monkeypatch.setenv(name, "")
