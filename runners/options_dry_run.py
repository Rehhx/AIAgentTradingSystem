"""
agents/options_dry_run.py
-------------------------
dry-run wiring of options_agent: compute the most recent bb_squeeze signal
on MSFT (best-generalizing strategy + ticker), translate the +1 breakout
into a bullish ATM call request, and submit it via options_agent.

defaults to SIMULATED mode — pass --live to actually hit alpaca paper.

usage:
    python agents/options_dry_run.py            # simulated (default)
    python agents/options_dry_run.py --live     # actually submit to alpaca paper
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("options_dry_run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live",       action="store_true", help="actually submit to alpaca paper")
    ap.add_argument("--underlying", default="MSFT",      help="underlying ticker")
    ap.add_argument("--dte_max",    type=int, default=7, help="max days-to-expiry for contract")
    ap.add_argument("--qty",        type=int, default=1, help="number of contracts")
    args = ap.parse_args()

    # 1. compute the most-recent bb_squeeze signal on the underlying using only
    #    the LAST 1000 bars of available data — represents "current state"
    from data.loader import load_ticker, DATA_DIR
    from agents.backtesting_agent import signals_bb_squeeze, STRATEGIES

    bb_params = STRATEGIES["bb_squeeze"][1]
    df = load_ticker(args.underlying, data_dir=DATA_DIR,
                     start="2024-06-01", end="2025-01-01", session="regular")
    if len(df) < 100:
        print(f"not enough data for {args.underlying}: only {len(df)} bars"); return

    df_tail = df.iloc[-2000:]    # last 2000 1-min bars
    sig     = signals_bb_squeeze(df_tail, bb_params)
    latest  = int(sig.iloc[-1])

    if latest == 0:
        intent = "bullish"   # for the dry-run, still demonstrate end-to-end
        log.info(f"latest bb_squeeze signal on {args.underlying} is FLAT — using bullish as demo")
    else:
        intent = "bullish" if latest == 1 else "bearish"
        log.info(f"latest bb_squeeze signal on {args.underlying}: {'LONG' if latest==1 else 'SHORT'}")

    # 2. force simulated mode by patching the options_agent module-level
    #    creds BEFORE instantiating — env-var pops don't help because config.py
    #    already cached them at import time.
    from agents import options_agent as oa_mod
    if not args.live:
        oa_mod.ALPACA_API_KEY    = None
        oa_mod.ALPACA_API_SECRET = None
        log.info("running in SIMULATED mode — pass --live to actually submit")
    else:
        log.warning("--live: submitting to alpaca PAPER (options market orders need market hours)")

    # 3. construct request and submit
    agent = oa_mod.OptionsAgent()
    payload = {"signal": {
        "underlying":  args.underlying,
        "side":        "buy",
        "intent":      intent,
        "qty":         args.qty,
        "moneyness":   "atm",
        "dte_max":     args.dte_max,
    }, "strategy_id": "bb_squeeze_msft_dry_run"}

    log.info(f"submitting options request: {payload}")
    result = agent.run({"payload": payload})

    print("\n--- options request result ---")
    print(json.dumps(result, indent=2, default=str))

    # 4. persist
    out = Path("results/options_dry_run.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "request":     payload,
            "result":      result,
            "live":        args.live,
            "signal_observed": latest,
            "run_at":      datetime.now(timezone.utc).isoformat(),
        }, f, indent=2, default=str)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
