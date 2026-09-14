"""Point-in-time historical universe (Sprint V6) — survivorship-safe by construction.

The rule (docs/validation.md, Backtest Protocol / master context §35):
historical universe construction must avoid survivorship bias. A universe
is NOT a list of today's tickers — it is a set of membership intervals
`{ticker, listed_from, listed_to}` where a delisted member keeps its
interval after it stops trading. Constructing a historical universe means
querying this ledger AT as_of; a today-only snapshot (e.g.
`PORTFOLIO_TICKERS`) is by construction missing every company that failed
before today, which inflates any backtest built on it.

Governance rules enforced here:

- Fail-closed construction: a member is included at as_of only when its
  ledger interval covers as_of. `listed_from: null` means "listed now,
  start unverified" — such members are included for past as_of but carry
  `listed_from_unverified: true` in the membership report so nobody can
  mistake an unverified start for evidence.
- Survivorship detection: `survivorship_problems(tickers, entries, as_of)`
  compares a candidate universe against the ledger's point-in-time
  membership. A member that was listed at as_of but is absent from the
  candidate list (because it delisted since) is the smoking gun of
  survivorship bias — status `survivorship_biased`. A candidate ticker
  with no ledger entry cannot be verified and is flagged
  `member_unverified` — status `unverifiable`. Only when the candidate
  list exactly matches the ledger's membership is the status
  `point_in_time_complete`.
- Price-coverage cross-check: `coverage_problems(members, fetched_tickers)`
  flags ledger members whose price fetch failed (delisted names typically
  404 on the provider) — they must be surfaced, never silently dropped.
- Storage: append-only `data/universe.jsonl`; entries are validated before
  write, idempotent per entry hash; a corrected interval appends as a
  revision (last wins on read); malformed lines raise.
- No fabrication: there is no index-constituent provider connected (the
  N6 breadth placeholder is `provider_key_required`), so this module never
  invents membership history — it records what is known and flags what is
  not. Connecting a real constituents adapter is the upgrade path that
  fills the ledger with verifiable delisted intervals.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

UNIVERSE_VERSION = "universe-ledger-v1"

UNIVERSES_PATH = Path(__file__).resolve().parent.parent / "data" / "universe.jsonl"

# Entry reasons. Delisted members are the ones survivorship bias hides.
REASON_INITIAL_MEMBER = "initial_member"
REASON_ADDED = "added"
REASON_DELISTED_BANKRUPTCY = "delisted_bankruptcy"
REASON_DELISTED_ACQUISITION = "delisted_acquisition"
REASON_DELISTED_OTHER = "delisted_other"
REASON_SYMBOL_CHANGED = "symbol_changed"
DELISTED_REASONS = (REASON_DELISTED_BANKRUPTCY, REASON_DELISTED_ACQUISITION, REASON_DELISTED_OTHER)
KNOWN_REASONS = (REASON_INITIAL_MEMBER, REASON_ADDED, REASON_SYMBOL_CHANGED) + DELISTED_REASONS

REQUIRED_ENTRY_FIELDS = ("ticker", "reason", "source_id", "published_time")

# Survivorship verdicts for a candidate universe.
STATUS_POINT_IN_TIME_COMPLETE = "point_in_time_complete"
STATUS_SURVIVORSHIP_BIASED = "survivorship_biased"
STATUS_UNVERIFIABLE = "unverifiable"


def _normalize_date(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _entry_hash(entry: dict) -> str:
    payload = {key: value for key, value in entry.items() if key != "entry_hash"}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_entry(entry: dict) -> None:
    if not isinstance(entry, dict):
        raise ValueError("universe entry must be a dict")
    for field in REQUIRED_ENTRY_FIELDS:
        value = entry.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"universe entry field {field!r} missing/empty")
    if entry.get("entry_hash") is not None and not str(entry["entry_hash"]).strip():
        raise ValueError("universe entry entry_hash must not be empty when present")
    if entry["reason"] not in KNOWN_REASONS:
        raise ValueError(
            f"unknown universe reason {entry['reason']!r} — known reasons: {KNOWN_REASONS}"
        )
    if entry["reason"] in DELISTED_REASONS and not entry.get("listed_to"):
        raise ValueError(
            f"delisted entry {entry['ticker']!r} requires listed_to (the delist date) — "
            f"a ledger that forgets when a member died is the survivorship trap itself"
        )
    for field in ("listed_from", "listed_to"):
        if entry.get(field) is not None:
            _normalize_date(entry[field])  # raises on unparseable dates


def make_entry(
    ticker: str,
    reason: str,
    source_id: str,
    published_time,
    listed_from=None,
    listed_to=None,
    notes: str = "",
) -> dict:
    """Build a validated, hash-stamped ledger entry."""
    entry = {
        "universe_version": UNIVERSE_VERSION,
        "ticker": str(ticker).upper(),
        "reason": reason,
        "listed_from": _normalize_date(listed_from) if listed_from is not None else None,
        "listed_to": _normalize_date(listed_to) if listed_to is not None else None,
        "source_id": source_id,
        "published_time": _normalize_date(published_time),
        "notes": notes,
    }
    _validate_entry(entry)
    entry["entry_hash"] = _entry_hash(entry)
    return entry


# --- Store (append-only, same conventions as outcomes/manifests) ---------------

def _read_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"universe ledger {path} line {line_number} is not valid JSON "
                    f"(append-only integrity violated): {exc}"
                ) from exc
    return entries


def persist_universe_entries(entries: list[dict], path: str | Path | None = None) -> int:
    """Append validated entries; idempotent per entry hash.

    A corrected interval for the same ticker is appended as a revision (the
    reader takes the last entry per ticker); an identical entry_hash appends
    nothing.
    """
    store_path = Path(path) if path is not None else UNIVERSES_PATH
    store_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_entries(store_path)
    known_hashes = {entry.get("entry_hash") for entry in existing}
    appended = 0
    with store_path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            _validate_entry(entry)
            entry_hash = entry.get("entry_hash") or _entry_hash(entry)
            if entry_hash in known_hashes:
                continue  # identical recompute: nothing to append
            record = {**entry, "entry_hash": entry_hash}
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            existing.append(record)
            known_hashes.add(entry_hash)
            appended += 1
    return appended


def load_universe_entries(path: str | Path | None = None) -> list[dict]:
    return _read_entries(Path(path) if path is not None else UNIVERSES_PATH)


def latest_entries_by_ticker(path: str | Path | None = None) -> dict[str, dict]:
    """Latest ledger entry per ticker (file order wins), keyed by ticker."""
    latest: dict[str, dict] = {}
    for entry in load_universe_entries(path=path):
        latest[entry["ticker"]] = entry
    return latest


def seed_portfolio_universe(
    tickers: list[str],
    published_time,
    path: str | Path | None = None,
    source_id: str = "portfolio_list_snapshot",
) -> int:
    """Record today's holdings as still-listed members with unverified starts.

    This seeds the ledger HONESTLY: we know these tickers are listed now;
    we do not know (and do not fabricate) when each was listed. The
    construction functions flag such members `listed_from_unverified`.
    Filling the starts — and adding delisted members — is the job of a real
    index-constituent adapter (the N6 breadth placeholder).
    """
    entries = [
        make_entry(
            ticker=ticker,
            reason=REASON_INITIAL_MEMBER,
            source_id=source_id,
            published_time=published_time,
        )
        for ticker in tickers
    ]
    return persist_universe_entries(entries, path=path)


# --- Point-in-time construction and survivorship detection ---------------------

def universe_as_of(entries: dict[str, dict], as_of) -> list[dict]:
    """Members whose ledger interval covers as_of (point-in-time construction).

    A member is included when (listed_from is None or listed_from <= as_of)
    AND (listed_to is None or listed_to >= as_of). Members with an
    unverified start carry `listed_from_unverified: true` — included, but
    never mistaken for evidence. Delisted members whose delist date is at
    or after as_of are INCLUDED — that is exactly what a today-snapshot
    universe silently drops.
    """
    as_of_date = _normalize_date(as_of)
    members: list[dict] = []
    for ticker in sorted(entries):
        entry = entries[ticker]
        listed_from = entry.get("listed_from")
        listed_to = entry.get("listed_to")
        if listed_from is not None and listed_from > as_of_date:
            continue  # not yet listed at as_of
        if listed_to is not None and listed_to < as_of_date:
            continue  # delisted before as_of
        members.append({**entry, "listed_from_unverified": listed_from is None})
    return members


def survivorship_problems(
    tickers: list[str],
    entries: dict[str, dict],
    as_of,
) -> list[str]:
    """Compare a candidate universe against the ledger's membership at as_of.

    - `survivorship_bias: <ticker> ...`: the ticker was a ledger member at
      as_of but is absent from the candidate list (delisted since) — the
      today-snapshot hazard, made explicit.
    - `member_unverified: <ticker> ...`: the candidate ticker has no ledger
      entry, so it cannot be verified as listed at as_of — fail-closed.
    """
    as_of_date = _normalize_date(as_of)
    members = {member["ticker"]: member for member in universe_as_of(entries, as_of)}
    candidate = {str(ticker).upper() for ticker in tickers}
    problems: list[str] = []
    for ticker in sorted(members):
        if ticker in candidate:
            continue
        member = members[ticker]
        listed_to = member.get("listed_to")
        if listed_to is not None and listed_to < as_of_date:
            continue  # delisted before as_of: correctly absent
        problems.append(
            f"survivorship_bias: {ticker} was a ledger member at {as_of_date} "
            f"(listed_to {listed_to or 'still listed'}) but is absent from the "
            f"candidate universe — a today-snapshot silently drops failures"
        )
    for ticker in sorted(candidate):
        if ticker not in members:
            problems.append(
                f"member_unverified: {ticker} has no ledger entry covering "
                f"{as_of_date} — it cannot be verified as listed at as_of"
            )
    return problems


def universe_survivorship_status(
    tickers: list[str],
    entries: dict[str, dict],
    as_of,
) -> str:
    """Verdict for a candidate universe: complete / biased / unverifiable."""
    problems = survivorship_problems(tickers, entries, as_of)
    if not problems:
        return STATUS_POINT_IN_TIME_COMPLETE
    if any(problem.startswith("survivorship_bias:") for problem in problems):
        return STATUS_SURVIVORSHIP_BIASED
    return STATUS_UNVERIFIABLE


def coverage_problems(members: list[str], fetched_tickers) -> list[str]:
    """Flag ledger members whose price history could not be fetched.

    Delisted names typically 404 on the provider; dropping them silently is
    exactly the quiet shrinkage this module exists to prevent.
    """
    fetched = {str(ticker).upper() for ticker in fetched_tickers}
    problems: list[str] = []
    for member in members:
        if str(member).upper() not in fetched:
            problems.append(
                f"member_price_unavailable: {str(member).upper()} has no fetched "
                f"price history — it must be surfaced, never silently dropped"
            )
    return problems


def build_universe_block(
    tickers: list[str],
    as_of,
    path: str | Path | None = None,
) -> dict:
    """Build the manifest-ready universe block for a candidate universe."""
    entries = latest_entries_by_ticker(path=path)
    members = universe_as_of(entries, as_of)
    problems = survivorship_problems(tickers, entries, as_of)
    status = universe_survivorship_status(tickers, entries, as_of)
    return {
        "universe_version": UNIVERSE_VERSION,
        "as_of": _normalize_date(as_of),
        "requested_members": sorted({str(ticker).upper() for ticker in tickers}),
        "ledger_members_at_as_of": sorted(member["ticker"] for member in members),
        "survivorship_status": status,
        "problems": problems,
    }


def universe_block_problems(block: dict, ticker: str) -> list[str]:
    """Validate a universe block handed to a backtest run (engine gate).

    Fatal problems: wrong version, unknown survivorship status, a biased
    verdict (the run would be inflated by construction), or the run's
    ticker missing from the declared members. An `unverifiable` status is
    NOT fatal — it is disclosed in the block and surfaced as a warning.
    """
    problems: list[str] = []
    if not isinstance(block, dict):
        return ["universe block is not a dict"]
    if block.get("universe_version") not in (None, UNIVERSE_VERSION):
        problems.append(f"unknown universe_version {block.get('universe_version')!r}")
    status = block.get("survivorship_status")
    if status not in (STATUS_POINT_IN_TIME_COMPLETE, STATUS_SURVIVORSHIP_BIASED, STATUS_UNVERIFIABLE):
        problems.append(f"unknown survivorship_status {status!r}")
    if status == STATUS_SURVIVORSHIP_BIASED:
        problems.append(
            "survivorship_biased: the declared universe drops ledger members "
            "that were listed at as_of — the run would be inflated by "
            "construction and is refused"
        )
    requested = block.get("requested_members")
    if requested is not None and str(ticker).upper() not in {str(m).upper() for m in requested}:
        problems.append(
            f"ticker_not_in_universe: {str(ticker).upper()} is not a declared "
            f"member of this universe"
        )
    return problems
