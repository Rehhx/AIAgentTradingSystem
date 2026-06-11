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

# hard guard: refuse to run if anyone ever sneaks --live into a step
assert not any("--live" in s["argv"] for s in PIPELINE), "live execution is not allowed from the web backend"

RUN = {"id": 0, "lines": [], "phase": None, "status": "idle"}
LOCK = threading.Lock()


def _classify(t: str) -> str:
    s = t.lower()
    if any(k in s for k in ("error", "traceback", "exception", "failed")):
        return "err"
    if any(k in s for k in ("awaiting", "dry-run", "no orders", "warn", "timed out", "gate")):
        return "warn"
    if any(k in s for k in ("pass", " ok", "✓", "deflated", "5/5", "beats", "order(s)", "sharpe")):
        return "ok"
    return ""


def _add(phase, text, cls=""):
    with LOCK:
        RUN["lines"].append({"phase": phase, "text": text, "cls": cls,
                             "ts": time.strftime("%H:%M:%S")})


def _run_pipeline(rid):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    for step in PIPELINE:
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
        ctype = {"html": "text/html; charset=utf-8", "css": "text/css",
                 "js": "application/javascript", "svg": "image/svg+xml",
                 "png": "image/png", "json": "application/json"}.get(f.suffix.lstrip("."), "text/plain")
        self._send(200, f.read_bytes(), ctype)

    def do_POST(self):
        if urlparse(self.path).path == "/api/run":
            with LOCK:
                if RUN["status"] == "running":
                    return self._send(409, json.dumps({"error": "already running"}))
                RUN["id"] += 1
                RUN["lines"] = []
                RUN["status"] = "running"
                RUN["phase"] = None
                rid = RUN["id"]
            threading.Thread(target=_run_pipeline, args=(rid,), daemon=True).start()
            return self._send(200, json.dumps({"id": rid}))
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
