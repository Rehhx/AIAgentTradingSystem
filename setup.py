"""
setup.py
--------
run this ONCE to initialize the full quant agent system.

what it does:
  1. checks your environment (.env, api keys, parquet files)
  2. seeds the research knowledge base (12 built-in strategies)
  3. indexes all your parquet tickers into the regime detection store
  4. validates the strategy memory store is ready
  5. prints a final status report

usage:
    cd quant_agent_system
    python setup.py

    # index only specific tickers (faster + cheaper for testing):
    python setup.py --tickers SPY QQQ TSLA

    # skip regime indexing (just seed research kb):
    python setup.py --skip-regimes

    # force re-seed even if already done:
    python setup.py --force
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/setup.log"),
    ],
)
log = logging.getLogger("setup")

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def check_line(label: str, ok: bool, detail: str = "") -> bool:
    status = "✓" if ok else "✗"
    line   = f"  {status}  {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return ok


def run_checks(data_dir: Path) -> bool:
    print("\n── environment checks ─────────────────────────────────────")
    all_ok = True

    from dotenv import load_dotenv
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY", "")
    alpaca_key = os.getenv("ALPACA_API_KEY", "")
    anthropic  = os.getenv("ANTHROPIC_API_KEY", "")

    all_ok &= check_line("OPENAI_API_KEY",    bool(openai_key), openai_key[:12] + "..." if openai_key else "MISSING — needed for vector stores")
    all_ok &= check_line("ALPACA_API_KEY",    bool(alpaca_key), "set" if alpaca_key else "MISSING — needed for paper trading")
    check_line(          "ANTHROPIC_API_KEY", bool(anthropic),  "set" if anthropic else "optional for now")

    # data directory
    exists = data_dir.exists()
    all_ok &= check_line(f"data dir", exists, str(data_dir) if exists else f"NOT FOUND: {data_dir}")

    if exists:
        parquets = sorted(data_dir.glob("*.parquet"))
        all_ok  &= check_line("parquet files", len(parquets) > 0, f"{len(parquets)} found")
        for p in parquets:
            size_mb = p.stat().st_size / 1_048_576
            print(f"         {p.name:<22} {size_mb:>7.1f} MB")

    # packages
    print()
    for pkg, pip_name in [
        ("pandas",   "pandas"),
        ("pyarrow",  "pyarrow"),
        ("chromadb", "chromadb"),
        ("openai",   "openai"),
        ("numpy",    "numpy"),
        ("dotenv",   "python-dotenv"),
    ]:
        try:
            __import__(pkg)
            check_line(f"pip: {pip_name}", True)
        except ImportError:
            check_line(f"pip: {pip_name}", False, f"run: pip install {pip_name}")
            all_ok = False

    return all_ok


def seed_research(force: bool = False):
    print("\n── seeding research knowledge base ────────────────────────")
    from vector_stores import EmbeddingLayer
    emb = EmbeddingLayer()
    n   = emb.research.seed_known_strategies(force=force)
    if n > 0:
        print(f"  ✓  embedded {n} strategy documents into research store")
    else:
        print(f"  ✓  already seeded ({emb.research.col.count()} docs) — use --force to re-embed")
    return emb


def index_regimes(emb, data_dir: Path, tickers: list, step: int = 15):
    print("\n── indexing regime detection store ─────────────────────────")
    print(f"  tickers  : {tickers}")
    print(f"  window   : 60 bars (1 hour of 1m data)")
    print(f"  step     : every {step} bars  ({step} min slide)")
    print(f"  model    : text-embedding-3-large")
    print()
    print("  NOTE: calls openai api — takes a few minutes per ticker.")
    print("  cost estimate: ~$0.10–0.30 per ticker depending on data length.")
    print()

    from data.loader import load_ticker

    total = 0
    for ticker in tickers:
        t0 = time.time()
        print(f"  indexing {ticker}...", end=" ", flush=True)
        try:
            df      = load_ticker(ticker, data_dir=data_dir, session="regular")
            n       = emb.index_ticker_regimes(ticker, df, step=step)
            elapsed = time.time() - t0
            print(f"✓  {n:,} windows  ({elapsed:.0f}s)")
            total  += n
        except FileNotFoundError:
            print(f"✗  file not found — skipping")
        except Exception as e:
            print(f"✗  error: {e}")
            log.exception(f"failed indexing {ticker}")

    print(f"\n  total windows indexed: {total:,}")
    return total


def print_final_status(emb, data_dir: Path):
    print("\n── final status ─────────────────────────────────────────────")
    stats = emb.stats()
    print(f"  regime store    : {stats['regime_store']['total_windows']:,} windows")
    print(f"  strategy store  : {stats['strategy_store']['total_strategies']} strategies")
    print(f"  research store  : {stats['research_store']['total_documents']} documents")
    print(f"  chroma db       : vector_stores/chroma_db/")
    print()
    print("  next steps:")
    print("    python orchestrator.py              — run full agent pipeline")
    print("    python agents/backtesting_agent.py  — run backtesting on your parquet data")
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="quant agent system — one-time setup")
    parser.add_argument("--tickers",      nargs="+", default=None, help="tickers to index (default: all parquets)")
    parser.add_argument("--skip-regimes", action="store_true",     help="skip regime store indexing")
    parser.add_argument("--force",        action="store_true",     help="force re-seed everything")
    parser.add_argument("--step",         type=int, default=15,    help="regime window slide step in bars (default 15)")
    parser.add_argument("--data-dir",     default=None,            help="override data directory path")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    data_dir = Path(args.data_dir) if args.data_dir else Path(
        os.getenv("DATA_DIR", r"C:\Users\pcagm\Downloads\StockData")
    )

    print("=" * 60)
    print("  quant agent system — setup")
    print("=" * 60)

    # 1 — environment checks
    ok = run_checks(data_dir)
    if not ok:
        print("\n  fix the issues above then re-run setup.py")
        sys.exit(1)

    # 2 — seed research kb
    emb = seed_research(force=args.force)

    # 3 — regime indexing
    if not args.skip_regimes:
        tickers = args.tickers or [p.stem for p in sorted(data_dir.glob("*.parquet"))]
        if not tickers:
            print("\n  no parquet files found — skipping regime indexing")
        else:
            print(f"\n  about to index {len(tickers)} tickers for regime detection.")
            print(f"  tickers: {tickers}")
            confirm = input("\n  proceed? [y/N] ").strip().lower()
            if confirm == "y":
                index_regimes(emb, data_dir, tickers, step=args.step)
            else:
                print("  skipped — run again with --tickers SPY QQQ to index specific ones")
    else:
        print("\n── regime indexing skipped (--skip-regimes) ────────────────")

    # 4 — final status
    print_final_status(emb, data_dir)

    print("=" * 60)
    print("  setup complete")
    print("=" * 60)


if __name__ == "__main__":
    main()