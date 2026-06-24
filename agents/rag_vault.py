"""
agents/rag_vault.py
-------------------
Client + live execution overlay for the **S&P 500 RAG Vault** signal service
(see SIGNAL_API.md). The vault blends three IC-weighted strategies -- Claude
sentiment lead-lag, supplier lead-lag, and 8-K event drift -- into a per-ticker
LONG / SHORT / FLAT verdict with a conviction (cross-sectional sigma) and a
0..1 `strength`.

WHY THIS IS AN OVERLAY, NOT A SLEEVE
------------------------------------
The vault answers from a *daily snapshot* -- one verdict for today, with no
historical series to replay. Our backtester needs `sig(d, params)` over years
of history, and the agent-lab walk-forward needs out-of-sample folds. The vault
exposes neither (yet), so it cannot honestly be admitted as a backtested sleeve
the way mean_gravity was. Instead it tilts the *live* book at rebalance time:
an external, IC-validated read that nudges position sizes we already hold.

Governance posture (same as the crypto sleeve): OPT-IN, default OFF, bounded
small, and FAIL-SAFE -- if the vault is unreachable the book is returned
unchanged. We are long/flat, so a SHORT verdict can only trim a name toward
flat (a veto), never open a new short.

If the vault later serves historical `as_of` snapshots, the supplier-sentiment
feature can be turned into a real `sig(d, params)` and run through the agent
lab's walk-forward -- promote it to a sleeve then, not before.

Configure via .env (read with os.getenv; see env.example):
  SIGNAL_API_URL=http://127.0.0.1:8000
  SIGNAL_API_TIMEOUT=5
  SIGNAL_HORIZON=5
  SIGNAL_TAU=0.5
"""
from __future__ import annotations

import os

_CONF_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


class RagVaultSignals:
    """Thin client for the RAG Vault signal API. Reads a daily snapshot, so calls
    are fast and make no market-data / LLM calls per request."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base = (base_url or os.getenv("SIGNAL_API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.timeout = timeout if timeout is not None else float(os.getenv("SIGNAL_API_TIMEOUT", "5"))
        self.horizon = int(os.getenv("SIGNAL_HORIZON", "5"))
        self.tau = float(os.getenv("SIGNAL_TAU", "0.5"))

    def _get(self, path: str, params: dict) -> dict:
        import requests  # lazy: only needed when the overlay is actually enabled
        r = requests.get(f"{self.base}{path}", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        return self._get("/health", {})

    def signal(self, ticker: str, *, horizon: int | None = None, tau: float | None = None,
               overlay: float | None = None, overlay_weight: float = 1.0) -> dict:
        """One LONG/SHORT/FLAT verdict. `overlay` blends in our own signal."""
        params = {"horizon": horizon or self.horizon, "tau": self.tau if tau is None else tau}
        if overlay is not None:
            params |= {"overlay": overlay, "overlay_weight": overlay_weight}
        return self._get(f"/signal/{ticker.upper()}", params)

    def signals(self, tickers: list[str] | None = None, *,
                horizon: int | None = None) -> list[dict]:
        """Batch verdicts; omit `tickers` for the whole ranked universe."""
        params = {"horizon": horizon or self.horizon}
        if tickers:
            params["tickers"] = ",".join(t.upper() for t in tickers)
        return self._get("/signals", params)["signals"]


def apply_sentiment_overlay(weights: dict, *, mode: str = "tilt", tilt: float = 0.25,
                            client: "RagVaultSignals | None" = None,
                            horizon: int | None = None,
                            min_confidence: str = "medium",
                            exclude: tuple[str, ...] = ("BIL", "SHV", "SGOV"),
                            verbose: bool = True) -> dict:
    """Overlay the RAG Vault verdicts onto live book weights. Returns a NEW dict.

    The book is long/flat, so a SHORT verdict is always a veto -- never a new short.

    mode="gate" (decisive -- "trade what the vault confirms"):
      LONG  -> keep the name and CONCENTRATE into it
      SHORT -> drop the name to flat (don't trade it now)
      FLAT / not covered -> leave the algorithm's weight untouched
      Capital freed by the SHORT drops is reallocated to the LONG-confirmed names
      pro-rata by their existing weight, so total invested is preserved. Downstream
      name-cap / vol-target / cash-park still bound the result.

    mode="tilt" (gentle -- bounded nudge):
      LONG  -> w * (1 + tilt * strength)
      SHORT -> w * (1 - tilt * strength)   (trim toward flat)
      FLAT / not covered / below `min_confidence` -> unchanged

    Cash-park tickers (BIL etc.) are skipped. FAIL-SAFE: any error reaching the
    vault logs a warning and returns the input weights unchanged.
    """
    held = [t for t, w in weights.items()
            if w > 1e-9 and t.upper() not in {e.upper() for e in exclude}]
    if not held or (mode == "tilt" and tilt <= 0):
        return dict(weights)

    client = client or RagVaultSignals()
    try:
        verdicts = client.signals(held, horizon=horizon)
    except Exception as e:                                 # service down / timeout / bad URL
        if verbose:
            print(f"  [sentiment] vault unreachable ({type(e).__name__}: {e}); "
                  f"book unchanged (fail-safe)")
        return dict(weights)

    by_ticker = {v.get("ticker", "").upper(): v for v in verdicts}
    floor = max(0.0, _CONF_RANK.get(min_confidence, 2))
    as_of = next((by_ticker[t.upper()].get("as_of") for t in held
                  if by_ticker.get(t.upper(), {}).get("as_of")), None)
    tag = f" (vault as_of {as_of})" if as_of else ""

    def _verdict(t):
        """(direction, strength) if the vault has an actionable read on t, else (None, 0)."""
        v = by_ticker.get(t.upper())
        if not v or not v.get("coverage"):
            return None, 0.0
        direction = v.get("direction", "flat")
        if direction == "flat" or _CONF_RANK.get(v.get("confidence", "none"), 0) < floor:
            return None, 0.0
        return direction, float(v.get("strength", 0.0))

    out = dict(weights)

    if mode == "gate":
        longs, dropped, freed = [], [], 0.0
        for t in held:
            direction, _ = _verdict(t)
            if direction == "short":
                freed += weights[t]
                out[t] = 0.0
                dropped.append(t)
            elif direction == "long":
                longs.append(t)
            # flat / not covered -> leave the algorithm's weight as-is
        if longs and freed > 1e-9:
            base = sum(weights[t] for t in longs)
            for t in longs:
                share = (weights[t] / base) if base > 0 else 1.0 / len(longs)
                out[t] = weights[t] + freed * share
        if verbose:
            if longs or dropped:
                print(f"  [sentiment] gate: trade {len(longs)} long-confirmed "
                      f"({', '.join(sorted(longs)) or '-'}), drop {len(dropped)} short "
                      f"({', '.join(sorted(dropped)) or '-'}); reallocated {freed:.1%} "
                      f"of the book into longs{tag}")
            else:
                print(f"  [sentiment] gate: no long/short verdicts for {len(held)} held "
                      f"name(s) -- algorithm unchanged{tag}")
        return out

    # mode == "tilt"
    boosted, trimmed = [], []
    for t in held:
        direction, strength = _verdict(t)
        if direction == "long":
            out[t] = weights[t] * (1.0 + tilt * strength)
            boosted.append(t)
        elif direction == "short":
            out[t] = weights[t] * max(0.0, 1.0 - tilt * strength)
            trimmed.append(t)
    if verbose:
        if boosted or trimmed:
            print(f"  [sentiment] tilt +/-{tilt:.0%}: boosted {len(boosted)} "
                  f"({', '.join(sorted(boosted)) or '-'}), trimmed {len(trimmed)} "
                  f"({', '.join(sorted(trimmed)) or '-'}){tag}")
        else:
            print(f"  [sentiment] no actionable verdicts for {len(held)} held "
                  f"name(s) -- book unchanged{tag}")
    return out
