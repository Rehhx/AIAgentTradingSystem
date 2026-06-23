"""
web/server.py
-------------
Tiny dependency-free backend (Python stdlib only) that makes the Control page
ACTUALLY launch the pipeline. It serves the static web/ app AND exposes a small
API that runs REAL runner subprocesses — research -> build -> validate -> deploy
(DRY-RUN) — streaming their live stdout to the page.

SAFETY (by design):
  * The deploy step is DRY-RUN only. No argv in this file contains "--live", and
    there is NO endpoint that places a broker order. The loop ends at
    "AWAITING HUMAN AUTHORIZATION", mirroring the project guardrail.
  * Binds to 127.0.0.1 (localhost) only — not exposed to the network.
  * To actually go live you still run the --live command yourself, deliberately.

Run:   python web/server.py          (opens http://127.0.0.1:8787)
Opening web/index.html directly still works in offline "demo" mode (simulated log).
"""
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
PY = sys.executable
PORT = 8787
sys.path.insert(0, str(ROOT))   # so in-process imports (config, agents, alpaca) resolve

# REAL pipeline steps (run from ROOT). NONE contain --live -- this is enforced below.
PIPELINE = [
    {"phase": "research", "label": "research · market-park edge backtest vs SPY",
     "argv": [PY, "-u", "runners/market_park_backtest.py"], "timeout": 150},
    {"phase": "build", "label": "build · compile engine + analytics + ml modules",
     "argv": [PY, "-u", "-c",
              "import analytics, backtest.engine, ml.cv, ml.labels, agents.daily_strategies as d;"
              "n=len(d.STRATEGIES_DAILY)+len(d.CANDIDATE_STRATEGIES);"
              "print(f'compiled {n} sleeves + event-engine + purged-CV + deflated-Sharpe modules OK')"],
     "timeout": 60},
    {"phase": "validate", "label": "validate · rigor + engine + purged-CV test suite",
     "argv": [PY, "-u", "-m", "pytest", "tests/", "-q"], "timeout": 180},
    {"phase": "execute", "label": "deploy · DRY-RUN reconcile (no orders sent)",
     "argv": [PY, "-u", "runners/daily_rebalance.py", "--book", "portfolio_full",
              "--xs-universe", "sp500", "--vol-target", "0.17", "--max-leverage", "1.0",
              "--crypto-sleeve", "--park-market", "SPY"], "timeout": 360},
]

# The 12-AGENT LAB pipeline: one step that runs the autonomous research desk —
# each agent researches an ORIGINAL mechanism, then build/validate/execute, and
# the candidates are written to web/candidates.json for human approval.
AGENT_PIPELINE = [
    {"phase": "agents", "label": "12 agents | invent -> build -> validate -> execute",
     "argv": [PY, "-u", "runners/agent_lab.py", "--llm", "--emit", "web/candidates.json"],
     "timeout": 600},
]

# hard guard: refuse to run if anyone ever sneaks --live into ANY step
assert not any("--live" in s["argv"] for s in PIPELINE + AGENT_PIPELINE), \
    "live execution is not allowed from the web backend"

CANDIDATES = WEB / "candidates.json"
DECISIONS = WEB / "decisions.json"
BOOK = WEB / "book.json"

RUN = {"id": 0, "lines": [], "phase": None, "status": "idle"}
LOCK = threading.Lock()

# read-only live account snapshot (cached 20s so polling doesn't hammer Alpaca)
_ACCT = {"t": 0.0, "data": None}
_ACCT_LOCK = threading.Lock()
_ACCT_NAMES = {1: "Equity growth", 2: "Managed futures", 3: "LEAPS options"}


def _account_snapshot():
    now = time.time()
    with _ACCT_LOCK:
        if _ACCT["data"] and now - _ACCT["t"] < 20:
            return _ACCT["data"]
    out = {"accounts": [], "spy_today": None, "ts": time.strftime("%H:%M:%S")}
    try:
        import yfinance as yf
        h = yf.Ticker("SPY").history(period="5d")["Close"].dropna()
        if len(h) >= 2:
            out["spy_today"] = float(h.iloc[-1] / h.iloc[-2] - 1)
    except Exception:
        pass
    try:
        from config import alpaca_keys, ALPACA_PAPER
        from alpaca.trading.client import TradingClient
    except Exception as e:
        out["error"] = f"alpaca import failed: {e}"
        return out
    for acc in (1, 2, 3):
        row = {"id": acc, "name": _ACCT_NAMES[acc], "status": "ok"}
        key, sec = alpaca_keys(acc)
        if not key or not sec:
            row["status"] = "no-keys"
            out["accounts"].append(row)
            continue
        try:
            c = TradingClient(api_key=key, secret_key=sec, paper=ALPACA_PAPER)
            a = c.get_account()
            eq, last = float(a.equity), float(a.last_equity)
            pos = c.get_all_positions()
            top = sorted(pos, key=lambda p: -abs(float(p.market_value)))[:5]
            row.update({"equity": eq, "last_equity": last, "cash": float(a.cash),
                        "today": (eq / last - 1) if last else 0.0, "pl": eq - last,
                        "n_pos": len(pos),
                        "top": [{"sym": p.symbol.replace(".", "-"), "mv": float(p.market_value),
                                 "pl": float(p.unrealized_pl)} for p in top]})
        except Exception as e:
            row["status"] = "error"
            row["err"] = str(e)[:80]
        out["accounts"].append(row)
    with _ACCT_LOCK:
        _ACCT["t"] = now
        _ACCT["data"] = out
    return out


# read-only RAG-Vault sentiment snapshot (cached 30s; fail-safe if vault offline)
_SIG = {"t": 0.0, "data": None}
_SIG_LOCK = threading.Lock()


def _signals_snapshot():
    """Live LONG/SHORT verdicts from the RAG Vault, ranked by conviction. FAIL-SAFE:
    if the vault is unreachable this returns ok=False (the page shows an offline
    state) -- the cockpit never breaks because the external service is down."""
    now = time.time()
    with _SIG_LOCK:
        if _SIG["data"] and now - _SIG["t"] < 30:
            return _SIG["data"]
    url = os.getenv("SIGNAL_API_URL", "http://127.0.0.1:8000")
    out = {"ok": False, "url": url, "as_of": None, "universe": 0,
           "longs": [], "shorts": [], "ts": time.strftime("%H:%M:%S")}

    def _row(v):
        return {"ticker": v.get("ticker"), "direction": v.get("direction"),
                "conviction": round(float(v.get("conviction", 0.0)), 2),
                "strength": round(float(v.get("strength", 0.0)), 2),
                "confidence": v.get("confidence", "none"), "breadth": v.get("breadth", 0)}
    try:
        from agents.rag_vault import RagVaultSignals
        verdicts = RagVaultSignals().signals()                  # whole ranked universe
        covered = [v for v in verdicts if v.get("coverage")]
        longs = sorted((_row(v) for v in covered if v.get("direction") == "long"),
                       key=lambda r: -r["conviction"])
        shorts = sorted((_row(v) for v in covered if v.get("direction") == "short"),
                        key=lambda r: r["conviction"])
        out.update({"ok": True, "universe": len(covered),
                    "as_of": covered[0].get("as_of") if covered else None,
                    "longs": longs[:8], "shorts": shorts[:8]})
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    with _SIG_LOCK:
        _SIG["t"] = now
        _SIG["data"] = out
    return out


def _classify(t: str) -> str:
    s = t.lower()
    if any(k in s for k in ("error", "traceback", "exception", "failed")):
        return "err"
    if "promote" in s:
        return "ok"
    if "review" in s or "reject" in s:
        return "warn"
    if any(k in s for k in ("awaiting", "dry-run", "no orders", "warn", "timed out", "gate")):
        return "warn"
    if any(k in s for k in ("pass", " ok", "✓", "deflated", "5/5", "beats", "order(s)", "sharpe")):
        return "ok"
    return ""


def _add(phase, text, cls=""):
    with LOCK:
        RUN["lines"].append({"phase": phase, "text": text, "cls": cls,
                             "ts": time.strftime("%H:%M:%S")})


# --- human approval ledger (records decisions; it does NOT trade) ------------
_DEC_LOCK = threading.Lock()


def _load_decisions():
    if DECISIONS.is_file():
        try:
            return json.loads(DECISIONS.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _load_book():
    if BOOK.is_file():
        try:
            return json.loads(BOOK.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"name": "Equity ensemble", "strategies": []}


def _candidate(strategy):
    if CANDIDATES.is_file():
        try:
            for c in json.loads(CANDIDATES.read_text(encoding="utf-8")).get("candidates", []):
                if c.get("strategy") == strategy:
                    return c
        except Exception:
            pass
    return None


def _record_decision(strategy, decision):
    """persist a human approve/reject AND keep the book of strategies in sync:
    approve -> add the candidate sleeve to book.json; reject -> remove it if present.
    Adding to the book MARKS it for the book only — going live still needs the
    deliberate --live command run by a human, never this server."""
    if decision not in ("approved", "rejected", "pending") or not strategy:
        return None
    with _DEC_LOCK:
        d = _load_decisions()
        d[strategy] = {"decision": decision, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        DECISIONS.write_text(json.dumps(d, indent=2), encoding="utf-8")

        book = _load_book()
        sleeves = book.setdefault("strategies", [])
        sleeves[:] = [s for s in sleeves if s.get("name") != strategy]   # drop any prior copy
        if decision == "approved":
            c = _candidate(strategy)
            sleeves.append({
                "name": strategy,
                "label": (c or {}).get("agent", strategy),
                "weight": 0.0,                       # paper sleeve; sizing decided at deploy
                "source": "lab",
                "family": (c or {}).get("family", ""),
                "sharpe": (c or {}).get("sharpe"),
                "corr": (c or {}).get("corr"),
                "delta": (c or {}).get("delta"),
                "added": time.strftime("%Y-%m-%d"),
            })
        BOOK.write_text(json.dumps(book, indent=2), encoding="utf-8")
    return d[strategy]


def _run_pipeline(rid, pipeline):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    for step in pipeline:
        if RUN["id"] != rid:
            return
        with LOCK:
            RUN["phase"] = step["phase"]
        _add(step["phase"], "▶ " + step["label"])
        try:
            p = subprocess.Popen(step["argv"], cwd=str(ROOT), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
                                 encoding="utf-8", errors="replace")
            start = time.time()
            for line in p.stdout:
                if RUN["id"] != rid:
                    p.kill()
                    return
                line = line.rstrip("\n")
                if line.strip():
                    _add(step["phase"], line, _classify(line))
                if time.time() - start > step["timeout"]:
                    p.kill()
                    _add(step["phase"], "… step exceeded timeout, moving on", "warn")
                    break
            p.wait()
        except Exception as e:
            _add(step["phase"], f"step error: {e}", "err")
    if pipeline is AGENT_PIPELINE:
        _add("agents", "⏸ AWAITING HUMAN DECISION — approve or reject each candidate below (no order sent)", "warn")
    else:
        _add("execute", "⏸ AWAITING HUMAN AUTHORIZATION — dry-run only, no order sent to the broker", "warn")
    with LOCK:
        RUN["status"] = "done"
        RUN["phase"] = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else str(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/health":
            return self._send(200, json.dumps({"ok": True}))
        if u.path == "/api/account":
            return self._send(200, json.dumps(_account_snapshot()))
        if u.path == "/api/signals":
            return self._send(200, json.dumps(_signals_snapshot()))
        if u.path == "/api/candidates":
            data = (json.loads(CANDIDATES.read_text(encoding="utf-8"))
                    if CANDIDATES.is_file() else {"candidates": []})
            data["decisions"] = _load_decisions()
            return self._send(200, json.dumps(data))
        if u.path == "/api/book":
            return self._send(200, json.dumps(_load_book()))
        if u.path == "/api/log":
            since = int((parse_qs(u.query).get("since", ["0"])[0]) or 0)
            with LOCK:
                return self._send(200, json.dumps({
                    "lines": RUN["lines"][since:], "total": len(RUN["lines"]),
                    "phase": RUN["phase"], "status": RUN["status"], "id": RUN["id"]}))
        # static files (sandboxed to web/)
        rel = "index.html" if u.path in ("/", "") else u.path.lstrip("/")
        f = (WEB / rel).resolve()
        if not str(f).startswith(str(WEB)) or not f.is_file():
            return self._send(404, "not found", "text/plain")
        ctype = {"html": "text/html; charset=utf-8", "css": "text/css; charset=utf-8",
                 "js": "application/javascript; charset=utf-8", "svg": "image/svg+xml",
                 "png": "image/png", "json": "application/json"}.get(f.suffix.lstrip("."), "text/plain")
        self._send(200, f.read_bytes(), ctype)

    def _body(self):
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}
        except Exception:
            return {}

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/run":
            body = self._body()
            pipeline = AGENT_PIPELINE if body.get("mode") == "agents" else PIPELINE
            with LOCK:
                if RUN["status"] == "running":
                    return self._send(409, json.dumps({"error": "already running"}))
                RUN["id"] += 1
                RUN["lines"] = []
                RUN["status"] = "running"
                RUN["phase"] = None
                rid = RUN["id"]
            threading.Thread(target=_run_pipeline, args=(rid, pipeline),
                             daemon=True).start()
            return self._send(200, json.dumps({"id": rid, "mode": body.get("mode", "deploy")}))
        if path == "/api/decide":
            body = self._body()
            rec = _record_decision(body.get("strategy", ""), body.get("decision", ""))
            if rec is None:
                return self._send(400, json.dumps({"error": "bad decision"}))
            return self._send(200, json.dumps({"ok": True, "record": rec}))
        self._send(404, json.dumps({"error": "not found"}))


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"QUANT·DESK backend -> {url}   (Ctrl+C to stop)")
    print("  deploy step is DRY-RUN only; no live orders can be placed from here.")
    if not os.getenv("QUANTDESK_NOBROWSER"):
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
