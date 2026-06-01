"""
app/dashboard.py
----------------
Interactive control panel for the quant book (Streamlit).

  - live Alpaca account + positions (auto-refreshing), with a margin-safety banner
  - current market regime
  - "Activate AI Agent": runs your VALIDATED strategy signals against the live
    positions and produces a per-position decision (HOLD / TRIM / ADD / SELL / BUY)
    plus "don't-get-cooked" alerts (positions the strategy wants OUT of, deep losers,
    bear-regime de-risk). The decision engine is the SAME backtested logic the daily
    rebalancer uses -- not an LLM guessing trades.
  - a gated LIVE-execute button (you must tick the authorize box AND click) that
    submits the agent's plan to Alpaca paper and re-arms the 20% catastrophe stops.

Launch:
  .venv\\Scripts\\streamlit run app\\dashboard.py
  (or: run_dashboard.ps1)
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from agents.execution_agent import ExecutionAgent
from agents.daily_strategies import QUALITY_UNIVERSE
from runners.daily_rebalance import target_weights, _detect_regime, BOOKS

ROOT = Path(__file__).parent.parent
TRACK = ROOT / "results" / "track_record.csv"

st.set_page_config(page_title="Quant Book — Control Panel", layout="wide", page_icon="📈")


# --------------------------------------------------------------------------- #
# connections (cached so we don't re-auth on every rerun)
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_agent(account: int) -> ExecutionAgent:
    if account == 2:
        from config import ALPACA_API_KEY_2, ALPACA_API_SECRET_2
        return ExecutionAgent(api_key=ALPACA_API_KEY_2, api_secret=ALPACA_API_SECRET_2)
    return ExecutionAgent()


def account_dict(agent: ExecutionAgent) -> dict:
    if agent.simulated or agent.client is None:
        return {"equity": 100_000.0, "cash": 100_000.0, "buying_power": 100_000.0,
                "last_equity": 100_000.0, "long_mv": 0.0, "short_mv": 0.0, "sim": True}
    a = agent.client.get_account()
    return {
        "equity": float(a.equity), "cash": float(a.cash),
        "buying_power": float(a.buying_power), "last_equity": float(a.last_equity),
        "long_mv": float(getattr(a, "long_market_value", 0) or 0),
        "short_mv": float(getattr(a, "short_market_value", 0) or 0), "sim": False,
    }


def positions_df(agent: ExecutionAgent) -> pd.DataFrame:
    pos = agent.get_positions()
    rows = []
    for p in pos:
        qty, mv = p["qty"], p["market_value"]
        px = mv / qty if qty else 0.0
        cost = p["avg_entry_price"] * qty
        pl_pct = (mv - cost) / abs(cost) if cost else 0.0
        rows.append({
            "Symbol": p["symbol"], "Side": "LONG" if qty >= 0 else "SHORT",
            "Qty": round(qty, 4), "Avg Entry": round(p["avg_entry_price"], 2),
            "Price": round(px, 2), "Mkt Value": round(mv, 2),
            "Unreal P&L $": round(p["unrealized_pl"], 2), "Unreal P&L %": round(pl_pct * 100, 2),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800, show_spinner=False)
def compute_signals(book, xs_universe, vol_target, max_lev, crypto_weight, name_cap, _daystamp):
    """The validated strategy signals -> target weights. Cached for 30 min (keyed by
    config + the calendar day) so re-clicks are instant. _daystamp forces a daily refresh."""
    return target_weights(
        book, list(QUALITY_UNIVERSE), source="auto", xs_universe=xs_universe,
        vol_target_pct=vol_target, max_lev=max_lev, park_cash="BIL",
        early_warning=True, name_cap=name_cap, crypto_weight=crypto_weight, ensemble=True)


def reconcile(weights, last_price, held, equity, band=250.0):
    """Mirror the daily rebalancer: target weights vs current positions -> decisions."""
    out = []
    names = set(t for t, w in weights.items() if abs(w) > 1e-9) | set(held)
    for t in sorted(names):
        w = weights.get(t, 0.0)
        cur_sh = float(held.get(t, 0.0))
        px = last_price.get(t)
        cur_d = cur_sh * px if px else 0.0
        tgt_d = w * equity
        delta = tgt_d - cur_d
        if abs(w) < 1e-9 and cur_sh != 0:
            act, why = "SELL", "signal exited — strategy no longer holds this name"
        elif cur_sh == 0 and abs(w) > 1e-9:
            act, why = "BUY", "new signal"
        elif px is None:
            act, why = "SELL", "held but no live signal — close"
        elif abs(delta) >= band:
            act = "ADD" if delta > 0 else "TRIM"
            why = "below target — top up" if delta > 0 else "above target — trim"
        else:
            act, why = "HOLD", "in signal, within no-trade band"
        out.append({"Symbol": t, "Action": act, "Why": why, "Target %": round(w * 100, 2),
                    "Current $": round(cur_d, 0), "Target $": round(tgt_d, 0),
                    "Delta $": round(delta, 0)})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# sidebar — config
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Config")
account = st.sidebar.radio("Alpaca account", [1, 2],
                           format_func=lambda x: f"#{x} " + ("(growth)" if x == 1 else "(managed futures)"))
book = st.sidebar.selectbox("Book", list(BOOKS),
                            index=list(BOOKS).index("portfolio_full") if account == 1 else list(BOOKS).index("managed_futures"))
max_lev = st.sidebar.slider("Leverage cap", 1.0, 1.8, 1.0, 0.1,
                            help="1.0 = NO margin (can't be margin-called). >1.0 borrows in calm markets.")
vol_target = st.sidebar.slider("Vol target", 0.08, 0.20, 0.17, 0.01)
crypto = st.sidebar.checkbox("Crypto sleeve (5%)", value=(account == 1))
trail_pct = st.sidebar.slider("Trailing stop %", 0, 30, 20,
                              help="20 = catastrophe-only (won't fight mean-reversion). 0 = off.")
refresh_sec = st.sidebar.number_input("Auto-refresh positions (sec, 0=off)", 0, 600, 0, 15)
if max_lev > 1.0:
    st.sidebar.warning(f"⚠️ {max_lev:.1f}x can be MARGIN-CALLED in a gap-down crash.")
else:
    st.sidebar.success("✅ No margin — cannot be margin-called.")

agent = get_agent(account)


# --------------------------------------------------------------------------- #
# header — account + regime  (auto-refreshing fragment)
# --------------------------------------------------------------------------- #
st.title("📈 Quant Book — Control Panel")


@st.fragment(run_every=(refresh_sec if refresh_sec else None))
def live_panel():
    a = account_dict(agent)
    gross = (a["long_mv"] + abs(a["short_mv"])) / a["equity"] if a["equity"] else 0.0
    day_pl = a["equity"] - a["last_equity"]
    day_pct = day_pl / a["last_equity"] if a["last_equity"] else 0.0

    if a["sim"]:
        st.warning("Alpaca keys not detected — running in SIMULATED mode (no live positions). "
                   "Add keys to `.env` for live data.")
    st.caption(f"Account #{account} · updated {datetime.now().strftime('%H:%M:%S')}"
               + (f" · auto-refresh {refresh_sec}s" if refresh_sec else ""))

    c = st.columns(5)
    c[0].metric("Equity", f"${a['equity']:,.0f}", f"{day_pl:+,.0f} today ({day_pct:+.2%})")
    c[1].metric("Cash", f"${a['cash']:,.0f}")
    c[2].metric("Buying power", f"${a['buying_power']:,.0f}")
    c[3].metric("Gross exposure", f"{gross:.0%}",
                "ON MARGIN" if gross > 1.01 else "no margin",
                delta_color=("inverse" if gross > 1.01 else "off"))
    try:
        regime = _detect_regime("auto")
    except Exception:
        regime = "unknown"
    c[4].metric("Regime", regime.upper().replace("_", " "))
    if gross > 1.01:
        st.error(f"⚠️ Book is on margin ({gross:.0%} gross) — a gap-down crash could trigger a margin call. "
                 "Set leverage cap to 1.0 and re-run the agent to de-lever.")

    st.subheader("📊 Positions")
    df = positions_df(agent)
    if df.empty:
        st.info("No open positions.")
    else:
        tot = df["Unreal P&L $"].sum()
        st.caption(f"{len(df)} positions · total unrealized P&L **${tot:,.0f}**")
        sty = df.style.map(lambda v: f"color: {'#16a34a' if v > 0 else '#dc2626' if v < 0 else 'inherit'}",
                           subset=["Unreal P&L $", "Unreal P&L %"])
        st.dataframe(sty, use_container_width=True, hide_index=True)
        worst = df.nsmallest(1, "Unreal P&L %")
        if not worst.empty and worst.iloc[0]["Unreal P&L %"] < -15:
            r = worst.iloc[0]
            st.warning(f"🩹 Biggest loser: **{r['Symbol']}** {r['Unreal P&L %']:+.1f}% — "
                       f"watch the 20% catastrophe stop.")


live_panel()

st.divider()


# --------------------------------------------------------------------------- #
# the AI agent
# --------------------------------------------------------------------------- #
st.subheader("🤖 AI Agent — strategy decision engine")
st.caption("Runs your validated, backtested strategy against the live positions and decides "
           "what to buy/sell/hold — so a stale position can't cook you.")

if st.button("🤖 Activate AI Agent", type="primary"):
    with st.spinner("Agent pulling data + computing strategy signals (scans the S&P 500, ~1–2 min)…"):
        crypto_w = 0.05 if crypto else 0.0
        weights, last_price, detail = compute_signals(
            book, "sp500" if book != "managed_futures" else "same",
            vol_target, max_lev, crypto_w, 0.10, str(datetime.now().date()))
        a = account_dict(agent)
        held = {p["symbol"]: p["qty"] for p in agent.get_positions()}
        plan = reconcile(weights, last_price, held, a["equity"])
    st.session_state["plan"] = plan
    st.session_state["regime_at_plan"] = (_detect_regime("auto") if True else "?")

if "plan" in st.session_state:
    plan = st.session_state["plan"]

    # --- don't-get-cooked alerts ---
    exits = plan[plan["Action"] == "SELL"]
    dfpos = positions_df(agent)
    deep = dfpos[dfpos["Unreal P&L %"] < -15] if not dfpos.empty else pd.DataFrame()
    regime = st.session_state.get("regime_at_plan", "?")

    st.markdown("#### 🚨 Don't-get-cooked check")
    alerts = []
    if not exits.empty:
        alerts.append(f"**Strategy wants OUT of {len(exits)} position(s):** "
                      f"{', '.join(exits['Symbol'])} — signal has exited; holding them is unprotected risk.")
    if not deep.empty:
        alerts.append(f"**{len(deep)} deep loser(s) (< −15%):** {', '.join(deep['Symbol'])} — near the catastrophe stop.")
    if str(regime).startswith("bear"):
        alerts.append("**Bear regime** — momentum sleeves rotate to cash, low-vol to T-bills, early-warning may cut to 60%.")
    if alerts:
        for x in alerts:
            st.warning(x)
    else:
        st.success("✅ No exits pending, no deep losers, not in a bear regime — the book is positioned with the strategy.")

    # --- the decision table ---
    st.markdown("#### Decisions")
    counts = plan["Action"].value_counts().to_dict()
    st.caption(" · ".join(f"**{k}**: {v}" for k, v in counts.items()))
    color = {"SELL": "#dc2626", "TRIM": "#f59e0b", "BUY": "#16a34a", "ADD": "#2563eb", "HOLD": "#6b7280"}
    sty = plan.style.map(lambda v: f"background-color: {color.get(v,'')}22; color: {color.get(v,'inherit')}; font-weight:600",
                         subset=["Action"])
    st.dataframe(sty, use_container_width=True, hide_index=True)

    orders = plan[plan["Action"].isin(["BUY", "SELL", "TRIM", "ADD"])]
    n_orders = len(orders)
    st.markdown(f"#### Execute — {n_orders} order(s)")
    if n_orders == 0:
        st.info("Portfolio already aligned with the strategy — nothing to do.")
    elif agent.simulated:
        st.info("Simulated mode — connect Alpaca keys in `.env` to enable live execution.")
    else:
        st.error("LIVE orders hit your Alpaca paper account. Review the table above first.")
        ok = st.checkbox("I authorize submitting these as LIVE paper orders")
        if st.button("🚀 Execute LIVE", disabled=not ok):
            with st.spinner("submitting…"):
                a = account_dict(agent)
                # cancel stale opens first (matches the CLI rebalancer)
                try:
                    agent.client.cancel_orders()
                except Exception:
                    pass
                log = []
                for _, r in orders.iterrows():
                    t, act, delta = r["Symbol"], r["Action"], r["Delta $"]
                    side = "buy" if delta > 0 else "sell"
                    if act == "SELL":   # full close by qty
                        qty = abs(float({p["symbol"]: p["qty"] for p in agent.get_positions()}.get(t, 0)))
                        sig = {"ticker": t, "side": "sell", "qty": qty}
                    else:
                        sig = {"ticker": t, "side": side, "notional": round(abs(delta), 2)}
                    res = agent.run({"payload": {"signal": {**sig, "order_type": "market",
                                                            "time_in_force": "day"}},
                                     "strategy_id": f"dashboard_{book}"})
                    log.append(f"{side.upper()} {t} → {res.get('fill', {}).get('status', res.get('reason'))}")
                if trail_pct > 0 and book != "managed_futures":
                    agent.sync_trailing_stops(trail_pct=float(trail_pct))
                    log.append(f"trailing stops re-armed @ {trail_pct}%")
            st.success(f"Submitted {n_orders} order(s).")
            st.code("\n".join(log))
            del st.session_state["plan"]


# --------------------------------------------------------------------------- #
# equity curve (if track record exists)
# --------------------------------------------------------------------------- #
if TRACK.exists():
    st.divider()
    st.subheader("📈 Live equity track record")
    try:
        tr = pd.read_csv(TRACK)
        tr["date"] = pd.to_datetime(tr["date"])
        st.line_chart(tr.set_index("date")["equity"])
    except Exception as e:
        st.caption(f"(could not render track record: {e})")
