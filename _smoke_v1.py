"""Temporary V1 smoke check — deleted after the sprint lands."""
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from core.labels import (
    append_outcome_records,
    build_outcome_labels,
    latest_outcome_labels,
    load_outcome_records,
)
from core.config import OUTCOME_LABEL_VERSION


def _frame(closes, start="2022-01-03"):
    index = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": [1_000_000.0] * len(closes)},
        index=index,
    )


def _ramp(sessions, step=1.0, base=100.0):
    return [base + step * index for index in range(sessions)]


frame = _frame(_ramp(130))

# 1. Spec acceptance: 10 sessions after as_of -> 1d/5d matured, 20d/60d null.
as_of = frame.index[-11].strftime("%Y-%m-%d %H:%M:%S")
with patch("core.labels.fetch_price_history", return_value=frame):
    labels = build_outcome_labels("TEST", as_of)
assert labels["status"] == "PARTIAL"
assert labels["matured_horizons"] == ["1d", "5d"]
assert labels["horizons"]["20d"] is None and labels["horizons"]["60d"] is None
print("1. 10-sessions-ago decision: 1d/5d matured, 20d/60d null (no partial-window leakage)")

# 2. Off-by-one safety at the 5d boundary.
with patch("core.labels.fetch_price_history", return_value=frame):
    matured = build_outcome_labels("TEST", frame.index[-6].strftime("%Y-%m-%d %H:%M:%S"))
    pending = build_outcome_labels("TEST", frame.index[-5].strftime("%Y-%m-%d %H:%M:%S"))
assert matured["horizons"]["5d"] is not None and pending["horizons"]["5d"] is None
print("2. boundary: 5 bars after as_of matures 5d; 4 bars does not")

# 3. Leakage probe: bars beyond the 5d window cannot change the 5d label.
entry_pos = len(frame) - 61
all_matured_as_of = frame.index[-61].strftime("%Y-%m-%d %H:%M:%S")
poisoned = frame.copy()
poisoned.loc[poisoned.index[entry_pos + 6:], ["Open", "High", "Low", "Close"]] *= 10.0
with patch("core.labels.fetch_price_history", return_value=frame):
    baseline = build_outcome_labels("TEST", all_matured_as_of)
with patch("core.labels.fetch_price_history", return_value=poisoned):
    poisoned_labels = build_outcome_labels("TEST", all_matured_as_of)
assert poisoned_labels["horizons"]["5d"] == baseline["horizons"]["5d"]
assert poisoned_labels["horizons"]["20d"]["forward_return"] != baseline["horizons"]["20d"]["forward_return"]
print("3. leakage probe: 5d label invariant to beyond-window bars; 20d sees them")
print("   20d record:", {k: v for k, v in baseline["horizons"]["20d"].items()
                          if k in ("forward_return", "adverse_excursion", "label_up", "risk_adjusted")})

# 4. Append-only store: idempotent recompute + supersede-on-change (temp file).
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "outcomes.jsonl"
    with patch("core.labels.fetch_price_history", return_value=frame):
        from core.labels import build_and_persist_outcome_labels

        first = build_and_persist_outcome_labels("TEST", all_matured_as_of, path=path)
        again = build_and_persist_outcome_labels("TEST", all_matured_as_of, path=path)
    assert first["persisted"] == 4 and again["persisted"] == 0
    resolved = latest_outcome_labels("TEST", all_matured_as_of, path=path)
    assert resolved["status"] == "OK" and resolved["count"] == 4
print("4. store: 4 matured records appended once; recompute appends nothing; latest resolves OK")
print("   label_version:", OUTCOME_LABEL_VERSION)

# 5. Real data probe (cached MSFT history): a fully matured historical decision.
real = build_outcome_labels("MSFT", "2024-01-02")
assert real["status"] in {"OK", "PARTIAL", "PENDING"}, real["status"]
matured = real["matured_horizons"]
print("5. real MSFT 2024-01-02:", real["status"], "| matured:", matured,
      "| entry_close:", real["entry_close"])
for name in matured:
    rec = real["horizons"][name]
    print(f"   {name}: return {rec['forward_return']:+.4f} vol {rec['realized_vol']} "
          f"risk_adj {rec['risk_adjusted']}")

print("ALL SMOKE CHECKS PASSED")