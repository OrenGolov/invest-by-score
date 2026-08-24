from __future__ import annotations

import json
from datetime import date
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from core.agent_contracts import NoTradeDecision
from core.audit_store import get_audit_events, get_decision_by_replay_hash
from core.orchestrator import orchestrate_score
from fetch_data import get_provider_health_matrix

ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "index.html"


class AppHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/score":
            params = parse_qs(parsed.query)
            ticker = (params.get("ticker", ["MSFT"])[0] or "MSFT").strip().upper()
            as_of = params.get("date", [date.today().isoformat()])[0] or date.today().isoformat()
            try:
                result = orchestrate_score(ticker, as_of)
                payload = {
                    "ok": True,
                    "mode": result.mode,
                    "decision_type": result.decision_type,
                    "data": result.to_dict(),
                }
                if result.mode == "NO_TRADE":
                    payload["data"] = NoTradeDecision(
                        ticker=result.ticker,
                        as_of=result.as_of,
                        mode=result.mode,
                        action=result.action,
                        reason="; ".join(result.veto_reasons) if result.veto_reasons else "insufficient evidence",
                        veto_reasons=result.veto_reasons,
                        replay_hash=result.replay_hash,
                        snapshot_hash=result.snapshot_hash,
                        evidence=[{"source_record_id": source_id, "reason": "vetoed"} for source_id in result.source_record_ids],
                        agent_outputs=result.agent_outputs,
                    ).to_dict()
                self._send_json(200, payload)
            except Exception as exc:  # pragma: no cover - runtime validation path
                self._send_json(400, {"ok": False, "error": str(exc)})
            return

        if parsed.path == "/api/provider-health":
            self._send_json(200, {"ok": True, "data": get_provider_health_matrix()})
            return

        if parsed.path == "/api/audit-events":
            params = parse_qs(parsed.query)
            limit = params.get("limit", ["50"])[0]
            try:
                self._send_json(200, {"ok": True, "data": get_audit_events(limit=int(limit))})
            except ValueError:
                self._send_json(400, {"ok": False, "error": "limit must be an integer"})
            return

        if parsed.path.startswith("/api/replay"):
            params = parse_qs(parsed.query)
            replay_hash = (params.get("hash", [params.get("replay_hash", [""])[0]])[0] or "").strip()
            ticker = (params.get("ticker", [""])[0] or "").strip().upper()
            as_of = (params.get("date", [params.get("as_of", [""])[0]])[0] or "").strip()
            if ticker and as_of:
                from core.audit_store import get_decision_by_ticker_and_as_of
                self._send_json(200, {"ok": True, "data": get_decision_by_ticker_and_as_of(ticker, as_of)})
                return
            self._send_json(200, {"ok": True, "data": get_decision_by_replay_hash(replay_hash)})
            return

        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PATH.read_bytes())
            return

        super().do_GET()

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), AppHandler)
    print("Serving local website at http://127.0.0.1:8000")
    server.serve_forever()
