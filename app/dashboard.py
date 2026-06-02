"""
app/dashboard.py
----------------
Interactive control panel for the 3-account quant book (Streamlit).

TABS
  📊 Live      – live Alpaca account + positions (auto-refreshing "AI monitor"),
                 regime, margin-safety banner, and the per-account AI agent that
                 runs your VALIDATED strategy and flags don't-get-cooked risk.
  🤖 AI Agents – the architecture: every agent/engine, the board summary, and the
                 task list (options on acct 3, key rotation, track record).
  🚀 Deploy    – dry-run the plan for ALL 3 accounts, then authorize + execute per
                 account (gated). Account 3 = defined-risk LEAPS (preview-first).

Accounts:  1 = equity growth (no-margin)   2 = managed-futures (L/S)   3 = LEAPS options
Decision engine = your backtested strategy logic, NOT an LLM guessing trades.

Launch:  .venv\\Scripts\\streamlit run app\\dashboard.py     (or: run_dashboard.ps1)
"""
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from agents.execution_agent import ExecutionAgent
from agents.daily_strategies import QUALITY_UNIVERSE
from runners.daily_rebalance import target_weights, _detect_regime
from runners.options_book import leaps_plan
from runners.stop_guard import enforce_stops

ROOT = Path(__file__).parent.parent
TRACK = ROOT / "results" / "track_record.csv"

ACCOUNTS = {1: "Equity growth (no-margin)", 2: "Managed futures (L/S)", 3: "LEAPS options"}
ACCOUNT_BOOK = {1: "portfolio_full", 2: "managed_futures", 3: "leaps"}

st.set_page_config(page_title="Quant Book — Control Panel", layout="wide", page_icon="📈")


# --------------------------------------------------------------------------- #
# connections + data helpers
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_agent(account: int) -> ExecutionAgent:
    # read keys straight from the environment (config.py's load_dotenv already ran
    # via the runners import) so this never depends on a hot-reloaded config module
    suf = "" if account == 1 else f"_{account}"
    k = os.getenv(f"ALPACA_API_KEY{suf}", "")
    s = os.getenv(f"ALPACA_API_SECRET{suf}", "")
    return ExecutionAgent(api_key=k, api_secret=s)


def account_dict(agent: ExecutionAgent) -> dict:
    if agent.simulated or agent.client is None:
        return {"equity": 100_000.0, "cash": 100_000.0, "buying_power": 100_000.0,
                "last_equity": 100_000.0, "long_mv": 0.0, "short_mv": 0.0, "sim": True}
    a = agent.client.get_account()
    return {"equity": float(a.equity), "cash": float(a.cash),
            "buying_power": float(a.buying_power), "last_equity": float(a.last_equity),
            "long_mv": float(getattr(a, "long_market_value", 0) or 0),
            "short_mv": float(getattr(a, "short_market_value", 0) or 0), "sim": False}


def positions_df(agent: ExecutionAgent) -> pd.DataFrame:
    rows = []
    for p in agent.get_positions():
        qty, mv = p["qty"], p["market_value"]
        px = mv / qty if qty else 0.0
        cost = p["avg_entry_price"] * qty
        pl_pct = (mv - cost) / abs(cost) if cost else 0.0
        rows.append({"Symbol": p["symbol"], "Side": "LONG" if qty >= 0 else "SHORT",
                     "Qty": round(qty, 4), "Avg Entry": round(p["avg_entry_price"], 2),
                     "Price": round(px, 2), "Mkt Value": round(mv, 2),
                     "Unreal P&L $": round(p["unrealized_pl"], 2),
                     "Unreal P&L %": round(pl_pct * 100, 2)})
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800, show_spinner=False)
def compute_signals(book, xs_universe, vol_target, max_lev, crypto_weight, name_cap, _daystamp):
    return target_weights(book, list(QUALITY_UNIVERSE), source="auto", xs_universe=xs_universe,
                          vol_target_pct=vol_target, max_lev=max_lev, park_cash="BIL",
                          early_warning=True, name_cap=name_cap, crypto_weight=crypto_weight, ensemble=True)


DISPLAY_COLS = ["Symbol", "Action", "Why", "Target %", "Current $", "Target $", "Delta $"]


def reconcile(weights, last_price, held, equity, band=250.0, whole_shares=False):
    """Target weights vs current positions -> orders. Mirrors the CLI rebalancer:
    any SHORT (negative weight or short position) uses WHOLE-SHARE qty orders
    (Alpaca forbids fractional shorts); long-only books use fractional notional.
    Each row carries the exact order spec in _side/_qty/_notional."""
    rows = []
    for t in sorted(set(t for t, w in weights.items() if abs(w) > 1e-9) | set(held)):
        w = weights.get(t, 0.0); cur_sh = float(held.get(t, 0.0)); px = last_price.get(t)
        cur_d = cur_sh * px if px else 0.0
        tgt_d = w * equity
        delta = tgt_d - cur_d
        ws = whole_shares or w < 0 or cur_sh < 0            # short anywhere -> whole-share only
        side = qty = notional = None
        if abs(w) < 1e-9 and cur_sh != 0:                   # close out / cover
            side = "sell" if cur_sh > 0 else "buy"
            act = "SELL" if cur_sh > 0 else "BUY"
            why = "signal exited — close long" if cur_sh > 0 else "signal exited — cover short"
            qty = abs(int(cur_sh)) if ws else abs(cur_sh)   # whole vs fractional full close
            if ws and qty == 0:
                act, why, side = "HOLD", "sub-1-share residual", None
        elif px is None:
            side = "sell" if cur_sh > 0 else "buy"
            act = "SELL" if cur_sh > 0 else "BUY"; why = "held, no live signal — close"
            qty = abs(int(cur_sh)) if ws else abs(cur_sh)
        elif abs(delta) >= band:
            act = "BUY" if delta > 0 else "SELL"
            side = "buy" if delta > 0 else "sell"
            why = "below target — add" if delta > 0 else "above target — reduce"
            if ws:
                qty = abs(int(delta / px)) if px else 0
                if qty == 0:
                    act, why, side = "HOLD", "below 1 share", None
            else:
                notional = round(abs(delta), 2)
        else:
            act, why = "HOLD", "in signal, within band"
        rows.append({"Symbol": t, "Action": act, "Why": why, "Target %": round(w * 100, 2),
                     "Current $": round(cur_d, 0), "Target $": round(tgt_d, 0), "Delta $": round(delta, 0),
                     "_side": side, "_qty": qty, "_notional": notional, "_px": px})
    return pd.DataFrame(rows)


def margin_of(a):
    return (a["long_mv"] + abs(a["short_mv"])) / a["equity"] if a["equity"] else 0.0


# --------------------------------------------------------------------------- #
# execution (shared by the manual button and auto-pilot)
# --------------------------------------------------------------------------- #
def submit_reconcile_orders(ag, orders, acc, trail, ext_hours=False, limit_buffer=0.005):
    """Cancel stale opens, submit each order using its precomputed spec
    (_side/_qty/_notional), re-arm stops. When ext_hours, send DAY limit orders
    with extended_hours=True (whole-share; needs _px). Returns a log."""
    try:
        ag.client.cancel_orders()
    except Exception:
        pass
    log = []
    for _, r in orders.iterrows():
        if not r.get("_side"):
            continue
        sig = {"ticker": r["Symbol"], "side": r["_side"], "time_in_force": "day"}
        px = r.get("_px")
        if ext_hours and r.get("_qty") is not None and not pd.isna(r["_qty"]) and px:
            sig["qty"] = float(r["_qty"])
            sig["order_type"] = "limit"
            sig["extended_hours"] = True
            sig["limit_price"] = round(px * (1 + limit_buffer) if r["_side"] == "buy"
                                       else px * (1 - limit_buffer), 2)
        else:
            sig["order_type"] = "market"
            if r.get("_qty") is not None and not pd.isna(r["_qty"]):
                sig["qty"] = float(r["_qty"])
            elif r.get("_notional") is not None and not pd.isna(r["_notional"]):
                sig["notional"] = float(r["_notional"])
            else:
                continue
        res = ag.run({"payload": {"signal": sig},
                      "strategy_id": f"dashboard_{ACCOUNT_BOOK.get(acc, 'x')}"})
        amt = f"{sig['qty']:g} sh" if "qty" in sig else f"${sig['notional']:,.0f}"
        tag = f" [ext-hrs limit @{sig['limit_price']}]" if sig.get("extended_hours") else ""
        log.append(f"{r['_side'].upper()} {amt} {r['Symbol']}{tag} → {res.get('fill', {}).get('status', res.get('reason'))}")
    if trail and trail > 0 and acc == 1 and hasattr(ag, "sync_trailing_stops"):
        ag.sync_trailing_stops(trail_pct=float(trail))
        log.append(f"trailing stops re-armed @ {trail}%")
    return log


def manual_execute_ui(ag, orders, acc, trail, key, ext_hours=False, limit_buffer=0.005):
    """Gated manual execute: authorize checkbox + button (the inline feature)."""
    if ag.simulated:
        st.info("Simulated — add keys to `.env` to execute."); return
    if len(orders) == 0:
        st.success("Already aligned — nothing to do."); return
    if ext_hours:
        st.warning("🌙 Extended-hours: whole-share DAY limit orders (pre/post market). Thin liquidity — "
                   "fills may be partial or miss. No daily-signal edge; use deliberately.")
    st.error(f"LIVE orders hit Alpaca paper account #{acc}. Review the plan above first.")
    ok = st.checkbox(f"I authorize LIVE orders for account #{acc}", key=f"{key}_auth")
    if st.button(f"🚀 Execute account #{acc} now", disabled=not ok, key=f"{key}_exec"):
        with st.spinner("submitting…"):
            log = submit_reconcile_orders(ag, orders, acc, trail, ext_hours=ext_hours, limit_buffer=limit_buffer)
        st.success(f"Account #{acc}: {len(orders)} order(s) submitted."); st.code("\n".join(log))


def autopilot_execute(ag, orders, acc, trail, ext_hours=False, limit_buffer=0.005):
    """Autonomous live execute, with guardrails: market-open gate (skipped if
    ext_hours, which trades the extended session), dedupe by plan hash, and a
    60%-of-equity safety cap."""
    import hashlib
    if ag.simulated:
        st.info("🔴 Auto-pilot: simulated account — no live orders."); return
    if not ext_hours:
        try:
            if not ag.client.get_clock().is_open:
                st.info("🔴 Auto-pilot armed — market closed, waiting for the open."); return
        except Exception:
            st.warning("🔴 Auto-pilot: couldn't verify market hours — skipping this cycle."); return
    if len(orders) == 0:
        st.caption("🔴 Auto-pilot: aligned with the strategy, nothing to do."); return
    phash = hashlib.md5(orders.to_json().encode()).hexdigest()
    if st.session_state.get(f"exec_hash{acc}") == phash:
        st.caption("🔴 Auto-pilot: this exact plan already executed — holding until signals change."); return
    a = account_dict(ag)
    gross = float(orders["Delta $"].abs().sum())
    if a["equity"] and gross > 0.6 * a["equity"]:
        st.error(f"🔴 Auto-pilot ABORTED — order size ${gross:,.0f} > 60% of equity. "
                 "Something looks off; execute manually after reviewing."); return
    with st.spinner("🔴 Auto-pilot submitting live orders…"):
        log = submit_reconcile_orders(ag, orders, acc, trail, ext_hours=ext_hours, limit_buffer=limit_buffer)
    st.session_state[f"exec_hash{acc}"] = phash
    st.success(f"🔴 Auto-pilot executed {len(orders)} live order(s) on account #{acc}.")
    st.code("\n".join(log))


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Config")
account = st.sidebar.radio("Account", [1, 2, 3], format_func=lambda x: f"#{x} — {ACCOUNTS[x]}")
max_lev = st.sidebar.slider("Leverage cap (acct 1)", 1.0, 1.8, 1.0, 0.1,
                            help="1.0 = NO margin. >1.0 borrows in calm markets.")
vol_target = st.sidebar.slider("Vol target", 0.08, 0.20, 0.17, 0.01)
crypto = st.sidebar.checkbox("Crypto sleeve (acct 1, 5%)", value=True)
trail_pct = st.sidebar.slider("Trailing stop %", 0, 30, 20)
ext_hours = st.sidebar.toggle("🌙 Extended hours (pre/post market)", value=False,
                              help="Trade the pre/post-market session via whole-share DAY limit orders. "
                                   "Thin liquidity, no daily-signal edge — use deliberately, not as default.")
ext_buffer = st.sidebar.slider("Ext-hours limit buffer %", 0.1, 2.0, 0.5, 0.1) / 100.0 if ext_hours else 0.005
if ext_hours:
    st.sidebar.warning("🌙 Extended-hours ON — orders become whole-share limit; fractional sizing off.")
stop_guard_on = st.sidebar.toggle("🛡️ Stop guard (auto-liquidate breaches)", value=False,
                                  help="Software stop on Account 1 longs: liquidates any position that breaks its "
                                       "trailing/hard stop — including pre/post/overnight when broker stops can't fire. "
                                       "Sells only; suspect/bad-data prices are skipped for manual review.")
guard_hard = st.sidebar.slider("Stop-guard hard floor %", 8, 30, 15) if stop_guard_on else 15
if stop_guard_on:
    st.sidebar.error("🛡️ Stop guard ON — auto-SELLS breached Account-1 longs each refresh.")
ai_monitor = st.sidebar.toggle("🤖 AI monitor (auto-refresh)", value=False,
                               help="Continuously re-checks positions/regime/margin.")
refresh_sec = st.sidebar.select_slider("Refresh interval", [15, 30, 60, 120, 300], value=60) if ai_monitor else None
st.sidebar.markdown("---")
if max_lev <= 1.0:
    st.sidebar.success("✅ Acct 1 no margin")
else:
    st.sidebar.warning(f"⚠️ Acct 1 {max_lev:.1f}x can be margin-called")

agent = get_agent(account)

tab_live, tab_agents, tab_deploy = st.tabs(["📊 Live", "🤖 AI Agents", "🚀 Deploy"])


# =========================================================================== #
# TAB 1 — LIVE
# =========================================================================== #
with tab_live:
    st.subheader(f"Account #{account} — {ACCOUNTS[account]}")

    @st.fragment(run_every=(refresh_sec if ai_monitor else None))
    def live_panel():
        a = account_dict(agent)
        gross = margin_of(a)
        day_pl = a["equity"] - a["last_equity"]
        day_pct = day_pl / a["last_equity"] if a["last_equity"] else 0.0
        if a["sim"]:
            st.warning(f"Account #{account} keys not detected — SIMULATED. Add ALPACA keys to `.env`.")
        st.caption(f"updated {datetime.now().strftime('%H:%M:%S')}"
                   + (f" · 🤖 auto-refresh {refresh_sec}s" if ai_monitor else " · manual"))
        c = st.columns(5)
        c[0].metric("Equity", f"${a['equity']:,.0f}", f"{day_pl:+,.0f} ({day_pct:+.2%})")
        c[1].metric("Cash", f"${a['cash']:,.0f}")
        c[2].metric("Buying power", f"${a['buying_power']:,.0f}")
        c[3].metric("Gross", f"{gross:.0%}", "ON MARGIN" if gross > 1.01 else "no margin",
                    delta_color="inverse" if gross > 1.01 else "off")
        try:
            regime = _detect_regime("auto")
        except Exception:
            regime = "unknown"
        c[4].metric("Regime", regime.upper().replace("_", " "))
        if gross > 1.01:
            st.error(f"⚠️ On margin ({gross:.0%}). A gap-down crash could margin-call this account.")

        df = positions_df(agent)
        if df.empty:
            st.info("No open positions.")
        else:
            st.caption(f"{len(df)} positions · total unrealized **${df['Unreal P&L $'].sum():,.0f}**")
            sty = df.style.map(lambda v: f"color: {'#16a34a' if v > 0 else '#dc2626' if v < 0 else 'inherit'}",
                               subset=["Unreal P&L $", "Unreal P&L %"])
            st.dataframe(sty, width='stretch', hide_index=True)
    live_panel()

    # ---- stop guard (Account 1 longs; covers pre/post/overnight) ----
    def _show_guard_log(log):
        for l in log:
            if l.startswith("LIQUIDATE"):
                st.error(l)
            elif l.startswith(("SUSPECT", "BREACH")):
                st.warning(l)
            else:
                st.caption(l)

    if stop_guard_on and account == 1:
        @st.fragment(run_every=(refresh_sec if (ai_monitor and refresh_sec) else 60))
        def guard_panel():
            st.markdown("**🛡️ Stop guard** — auto-liquidating breaches")
            _show_guard_log(enforce_stops(agent, trail_pct=float(trail_pct), hard_pct=float(guard_hard),
                                          ext_buffer=ext_buffer, do_liquidate=True))
        guard_panel()
    elif stop_guard_on:
        st.caption("🛡️ Stop guard runs on Account 1 longs only (shorts/options excluded). Switch to account #1.")
    else:
        with st.expander("🛡️ Stop guard (off — peek without selling)", expanded=False):
            if st.button("Check stops now (report only)", key=f"sgchk{account}"):
                _show_guard_log(enforce_stops(agent, trail_pct=float(trail_pct), hard_pct=float(guard_hard),
                                              ext_buffer=ext_buffer, do_liquidate=False))

    st.divider()
    st.markdown("#### 🤖 AI Agent — strategy decision")
    if account == 3:
        st.caption("Account 3 = defined-risk LEAPS. The agent previews which index calls to buy.")
        if st.button("🤖 Run LEAPS agent", type="primary", key="leaps_btn"):
            with st.spinner("finding LEAPS contracts + pricing…"):
                a = account_dict(agent)
                st.session_state["leaps"] = leaps_plan(agent, a["equity"], leverage=1.0)
        if "leaps" in st.session_state:
            pl = st.session_state["leaps"]
            st.info(pl["note"])
            rdf = pd.DataFrame([r for r in pl["rows"] if r.get("ok")])
            if not rdf.empty:
                st.dataframe(rdf, width='stretch', hide_index=True)
            st.metric("Total premium (defined-risk max loss)", f"${pl['total_cost']:,.0f}",
                      f"{pl['pct_of_equity']:.0%} of equity · ${pl['cash_left']:,.0f} stays in T-bills")
            bad = [r for r in pl["rows"] if not r.get("ok")]
            if bad:
                st.warning("Some contracts unavailable (options may not be enabled on this account): "
                           + "; ".join(f"{r['underlying']}: {r.get('reason')}" for r in bad))
            if any(r.get("source") == "model" for r in pl["rows"]):
                st.caption("⚠️ Some prices are MODEL estimates (no live option quote / options not enabled). "
                           "Validate against the real chain before executing.")
            if not agent.simulated:
                st.error("LIVE option orders are real. Account needs options trading approved.")
                ok = st.checkbox("I authorize submitting these LEAPS as LIVE paper option orders", key="leaps_ok")
                if st.button("🚀 Execute LEAPS LIVE", disabled=not ok, key="leaps_exec"):
                    log = []
                    for r in pl["rows"]:
                        if r.get("ok") and r.get("source") == "quote":
                            res = agent.submit_option(r["symbol"], r["contracts"], "buy")
                            log.append(f"BUY {r['contracts']}x {r['symbol']} → {res.get('fill', {}).get('status', res.get('reason'))}")
                        elif r.get("ok"):
                            log.append(f"SKIP {r['underlying']} — model-priced contract, no live symbol; enable options + re-run")
                    st.success("Submitted."); st.code("\n".join(log) or "nothing to submit")
    else:
        book = ACCOUNT_BOOK[account]
        autopilot = st.toggle(f"🔴 Auto-pilot — auto-execute live (account {account})", value=False, key=f"auto{account}",
                              help="When ON and the market is open, the agent recomputes the strategy and SUBMITS "
                                   "orders automatically — no clicking. The dashboard must stay open for this to run.")
        if autopilot:
            st.error("🔴 **AUTO-PILOT ON** — this account auto-submits LIVE orders during market hours without asking. "
                     "Guardrails: only when market open · won't repeat the same plan · aborts if an order exceeds 60% of "
                     "equity. Toggle off to stop. (For always-on autonomy even when this is closed, use the scheduled tasks.)")
        auto_interval = refresh_sec if (ai_monitor and refresh_sec) else 120

        @st.fragment(run_every=(auto_interval if autopilot else None))
        def agent_section():
            triggered = st.button("🤖 Activate AI Agent", type="primary", key=f"ai_btn{account}") or autopilot
            if triggered:
                with st.spinner("computing strategy signals (scans S&P 500, ~1–2 min; cached 30 min)…"):
                    cw = 0.05 if (crypto and account == 1) else 0.0
                    ml = max_lev if account == 1 else 1.5
                    xu = "sp500" if account == 1 else "same"
                    weights, last_price, _ = compute_signals(book, xu, vol_target, ml, cw, 0.10, str(datetime.now().date()))
                    a = account_dict(agent)
                    held = {p["symbol"]: p["qty"] for p in agent.get_positions()}
                    st.session_state[f"plan{account}"] = reconcile(weights, last_price, held, a["equity"],
                                                                    whole_shares=(account == 2 or ext_hours))
                    try:
                        st.session_state[f"regime{account}"] = _detect_regime("auto")
                    except Exception:
                        st.session_state[f"regime{account}"] = "?"
            key = f"plan{account}"
            if key not in st.session_state:
                st.info("Click **Activate AI Agent** to compute the strategy decision (or flip Auto-pilot on).")
                return
            plan = st.session_state[key]
            exits = plan[plan["Action"] == "SELL"]
            dfpos = positions_df(agent)
            deep = dfpos[dfpos["Unreal P&L %"] < -15] if not dfpos.empty else pd.DataFrame()
            regime = st.session_state.get(f"regime{account}", "?")
            st.markdown("**🚨 Don't-get-cooked check**")
            alerts = []
            if not exits.empty:
                alerts.append(f"Strategy wants **OUT** of {len(exits)}: {', '.join(exits['Symbol'])} — exiting protects you.")
            if not deep.empty:
                alerts.append(f"{len(deep)} deep loser(s) (< −15%): {', '.join(deep['Symbol'])}.")
            if str(regime).startswith("bear"):
                alerts.append("**Bear regime** — momentum → cash, low-vol → T-bills, early-warning may cut to 60%.")
            if alerts:
                for x in alerts:
                    st.warning(x)
            else:
                st.success("✅ No exits pending, no deep losers, not bearish — aligned with the strategy.")
            counts = plan["Action"].value_counts().to_dict()
            st.caption(" · ".join(f"**{k}**: {v}" for k, v in counts.items()))
            color = {"SELL": "#dc2626", "BUY": "#16a34a", "HOLD": "#6b7280"}
            st.dataframe(plan[DISPLAY_COLS].style.map(
                lambda v: f"background-color:{color.get(v,'')}22;color:{color.get(v,'inherit')};font-weight:600",
                subset=["Action"]), width='stretch', hide_index=True)
            orders = plan[plan["_side"].notna()]
            st.markdown(f"**Execute — {len(orders)} order(s)**")
            if autopilot:
                autopilot_execute(agent, orders, account, trail_pct, ext_hours=ext_hours, limit_buffer=ext_buffer)
            else:
                manual_execute_ui(agent, orders, account, trail_pct, key=f"live{account}",
                                  ext_hours=ext_hours, limit_buffer=ext_buffer)
        agent_section()


# =========================================================================== #
# TAB 2 — AI AGENTS (architecture + board + tasks)
# =========================================================================== #
with tab_agents:
    st.subheader("🤖 System architecture")
    cols = st.columns(3)
    arch = [
        ("Account 1 — Growth engine", "portfolio_full: 7 equity sleeves (RSI-2, Donchian, 50/200 trend, "
         "cross-sectional momentum, recovery, PEAD, low-vol) + vol-target + early-warning. NO margin (1.0x)."),
        ("Account 2 — Crisis alpha", "managed_futures: long/SHORT time-series momentum across 10 asset ETFs, "
         "conviction-scaled, vol-targeted. Profits in macro bears (2008 +5.5%, 2022 +4.2%)."),
        ("Account 3 — LEAPS options", "Defined-risk index LEAPS (deep-ITM ~1yr SPY/QQQ calls). Worst case = "
         "premium, no margin call. Beat buy-and-hold on return AND drawdown at 1.0x. Preview-first."),
    ]
    for col, (title, body) in zip(cols, arch):
        col.markdown(f"**{title}**\n\n{body}")

    st.markdown("##### Engines & agents")
    comp = pd.DataFrame([
        ["execution_agent", "Submits Alpaca orders (stock/crypto/options) + trailing stops", "live"],
        ["signal engine (daily_strategies)", "All sleeve signals + parameter ensemble + vol-target", "live"],
        ["daily_rebalance", "Reconciler: target weights vs positions → orders (acct 1/2/3)", "live"],
        ["monitor", "Regime-posture check + track-record + drawdown/drift alarms", "daily"],
        ["tracking_dashboard", "Live-vs-backtest expectation band", "daily"],
        ["fill_tracker", "Real slippage vs the 6 bps assumption", "on fills"],
        ["options_book / options_leverage", "LEAPS contract selection + leverage study", "preview"],
        ["this dashboard", "Monitoring + manual gated execution across 3 accounts", "live"],
    ], columns=["Component", "Job", "Status"])
    st.dataframe(comp, width='stretch', hide_index=True)

    st.markdown("##### 📋 Task board")
    tasks = pd.DataFrame([
        ["🔴 Rotate API keys (public git history)", "Finnhub/FRED/Alpaca keys were committed. Revoke + reissue.", "URGENT"],
        ["🟡 Options on Account 3", "Enable options trading on the acct-3 Alpaca paper account, validate the LEAPS preview, then execute.", "ready (preview)"],
        ["🟡 Accumulate live track record", "~20+ sessions before live-vs-backtest is meaningful.", "in progress"],
        ["🟢 No-margin config (acct 1)", "max-leverage 1.0, 20% catastrophe stops. Sharpe 1.55 / −9.4% DD.", "DONE"],
        ["🟢 Crisis-alpha book (acct 2)", "Managed-futures L/S deployed.", "DONE"],
    ], columns=["Task", "Detail", "Status"])
    st.dataframe(tasks, width='stretch', hide_index=True)

    bs = ROOT / "BOARD_SUMMARY.md"
    if bs.exists():
        with st.expander("📄 BOARD_SUMMARY.md", expanded=False):
            st.markdown(bs.read_text(encoding="utf-8"))


# =========================================================================== #
# TAB 3 — DEPLOY (all 3 accounts; dry-run then authorize)
# =========================================================================== #
with tab_deploy:
    st.subheader("🚀 Deploy across all accounts")
    st.caption("Step 1: dry-run shows the plan for every account (no orders). "
               "Step 2: authorize + execute PER account. Nothing fires automatically.")

    # health row for all 3 (cheap checks = the 'AI checks over all 3')
    st.markdown("##### Account health")
    hrows = []
    for acc in (1, 2, 3):
        ag = get_agent(acc)
        a = account_dict(ag)
        g = margin_of(a)
        npos = 0 if a["sim"] else len(ag.get_positions())
        hrows.append({"Account": f"#{acc} {ACCOUNTS[acc]}",
                      "Connected": "SIM" if a["sim"] else "live",
                      "Equity": f"${a['equity']:,.0f}", "Gross": f"{g:.0%}",
                      "Margin": "⚠️ ON MARGIN" if g > 1.01 else "✅ none",
                      "Positions": npos})
    st.dataframe(pd.DataFrame(hrows), width='stretch', hide_index=True)

    if st.button("🔍 Dry-run plan for ALL accounts", type="primary"):
        plans = {}
        # acct 1
        with st.spinner("acct 1 — equity signals…"):
            cw = 0.05 if crypto else 0.0
            w1, lp1, _ = compute_signals("portfolio_full", "sp500", vol_target, max_lev, cw, 0.10, str(datetime.now().date()))
            ag1 = get_agent(1); a1 = account_dict(ag1)
            plans[1] = reconcile(w1, lp1, {p["symbol"]: p["qty"] for p in ag1.get_positions()}, a1["equity"],
                                 whole_shares=ext_hours)
        # acct 2 — managed futures uses whole-share orders (shorts can't be fractional)
        with st.spinner("acct 2 — managed futures…"):
            w2, lp2, _ = compute_signals("managed_futures", "same", 0.12, 1.5, 0.0, 0.0, str(datetime.now().date()))
            ag2 = get_agent(2); a2 = account_dict(ag2)
            plans[2] = reconcile(w2, lp2, {p["symbol"]: p["qty"] for p in ag2.get_positions()}, a2["equity"],
                                 whole_shares=True)
        # acct 3 — LEAPS preview
        with st.spinner("acct 3 — LEAPS preview…"):
            ag3 = get_agent(3); a3 = account_dict(ag3)
            plans["leaps"] = leaps_plan(ag3, a3["equity"], leverage=1.0)
        st.session_state["deploy_plans"] = plans

    if "deploy_plans" in st.session_state:
        plans = st.session_state["deploy_plans"]
        for acc in (1, 2):
            plan = plans[acc]
            orders = plan[plan["_side"].notna()]
            with st.expander(f"Account #{acc} — {ACCOUNTS[acc]} · {len(orders)} order(s)", expanded=True):
                st.dataframe(plan[DISPLAY_COLS], width='stretch', hide_index=True)
                manual_execute_ui(get_agent(acc), orders, acc, trail_pct, key=f"deploy{acc}",
                                  ext_hours=ext_hours, limit_buffer=ext_buffer)

        # acct 3 LEAPS
        pl = plans["leaps"]
        with st.expander("Account #3 — LEAPS options (preview-first)", expanded=True):
            st.info(pl["note"])
            rdf = pd.DataFrame([r for r in pl["rows"] if r.get("ok")])
            if not rdf.empty:
                st.dataframe(rdf, width='stretch', hide_index=True)
            st.metric("Total premium (max loss)", f"${pl['total_cost']:,.0f}", f"{pl['pct_of_equity']:.0%} of equity")
            if any(r.get("source") == "model" for r in pl["rows"]):
                st.caption("⚠️ MODEL-priced (options not enabled / no live quote). Enable options + re-run before executing.")
            ag3 = get_agent(3)
            if not ag3.simulated:
                ok3 = st.checkbox("Authorize LIVE LEAPS option orders for account #3", key="auth3")
                if st.button("🚀 Execute account #3 LEAPS", disabled=not ok3, key="exec3"):
                    log = []
                    for r in pl["rows"]:
                        if r.get("ok") and r.get("source") == "quote":
                            res = ag3.submit_option(r["symbol"], r["contracts"], "buy")
                            log.append(f"BUY {r['contracts']}x {r['symbol']} → {res.get('fill', {}).get('status', res.get('reason'))}")
                        else:
                            log.append(f"SKIP {r.get('underlying')} — enable options + live quote first")
                    st.success("Done."); st.code("\n".join(log) or "nothing submitted")

    if TRACK.exists():
        st.divider()
        st.markdown("##### 📈 Account 1 live equity")
        try:
            tr = pd.read_csv(TRACK); tr["date"] = pd.to_datetime(tr["date"])
            st.line_chart(tr.set_index("date")["equity"])
        except Exception:
            pass
