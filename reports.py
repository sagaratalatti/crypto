"""
Reports: Generates detailed trading analysis reports.

Tracks and reports on:
- Market scan/discovery timestamps
- Trade opened timestamps
- Trade closed timestamps
- Per-trade P&L (absolute and percentage)
- Strategy-level performance breakdown
- Time-based analytics (hold duration, best/worst hours)
- Cumulative equity curve data
- CSV export for external analysis tools

Usage:
    python main.py report                # Full report to terminal
    python main.py report --export csv   # Export trades to CSV
    python main.py report --export json  # Export full report as JSON
"""

import csv
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional

from tabulate import tabulate

import config

logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
TRADE_LOG_FILE = os.path.join(os.path.dirname(__file__), "trade_log.json")


# ─── Trade Event Logging ────────────────────────────────────────────────────

def _load_trade_log() -> list[dict]:
    """Load the persistent trade event log."""
    if not os.path.exists(TRADE_LOG_FILE):
        return []
    try:
        with open(TRADE_LOG_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_trade_log(events: list[dict]):
    """Persist trade event log."""
    try:
        with open(TRADE_LOG_FILE, "w") as f:
            json.dump(events, f, indent=2, default=str)
    except IOError as e:
        logger.error(f"Failed to save trade log: {e}")


def log_market_scanned(market_id: str, question: str, price: float,
                       volume_24h: float, liquidity: float, category: str):
    """Record when a market was first discovered/scanned."""
    events = _load_trade_log()
    events.append({
        "event": "market_scanned",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_id": market_id,
        "question": question,
        "price": price,
        "volume_24h": volume_24h,
        "liquidity": liquidity,
        "category": category,
    })
    _save_trade_log(events)


def log_trade_opened(order_id: str, market_id: str, question: str,
                     side: str, signal: str, entry_price: float,
                     size: float, cost_basis: float,
                     stop_loss: float, take_profit: float):
    """Record when a trade is opened."""
    events = _load_trade_log()
    events.append({
        "event": "trade_opened",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "order_id": order_id,
        "market_id": market_id,
        "question": question,
        "side": side,
        "signal": signal,
        "entry_price": entry_price,
        "size": size,
        "cost_basis": cost_basis,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    })
    _save_trade_log(events)


def log_trade_closed(order_id: str, market_id: str, question: str,
                     side: str, signal: str, entry_price: float,
                     exit_price: float, size: float, cost_basis: float,
                     realized_pnl: float, close_reason: str):
    """Record when a trade is closed."""
    pnl_pct = (realized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0

    events = _load_trade_log()
    events.append({
        "event": "trade_closed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "order_id": order_id,
        "market_id": market_id,
        "question": question,
        "side": side,
        "signal": signal,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "size": size,
        "cost_basis": cost_basis,
        "realized_pnl": realized_pnl,
        "pnl_pct": round(pnl_pct, 2),
        "close_reason": close_reason,
    })
    _save_trade_log(events)


# ─── Report Data Assembly ───────────────────────────────────────────────────

def _parse_ts(ts_str: str) -> Optional[datetime]:
    """Parse an ISO timestamp string."""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def build_trade_records(state: dict) -> list[dict]:
    """
    Build a unified list of trade records from portfolio state,
    enriched with computed fields (P&L %, hold duration, etc).

    Combines open + closed positions into one analyzable list.
    """
    records = []

    # Closed positions (have full lifecycle data)
    for pos in state.get("closed_positions", []):
        entry_price = pos.get("entry_price", 0)
        exit_price = pos.get("exit_price", 0)
        cost_basis = pos.get("cost_basis", 0)
        realized_pnl = pos.get("realized_pnl", 0)

        pnl_pct = (realized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0

        opened_at = _parse_ts(pos.get("opened_at", ""))
        closed_at = _parse_ts(pos.get("closed_at", ""))

        hold_duration = None
        hold_hours = 0.0
        if opened_at and closed_at:
            hold_duration = closed_at - opened_at
            hold_hours = hold_duration.total_seconds() / 3600

        records.append({
            "status": "closed",
            "market_id": pos.get("market_id", ""),
            "question": pos.get("market_question", ""),
            "side": pos.get("side", ""),
            "signal": pos.get("signal", ""),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": pos.get("size", 0),
            "cost_basis": cost_basis,
            "realized_pnl": realized_pnl,
            "pnl_pct": round(pnl_pct, 2),
            "opened_at": pos.get("opened_at", ""),
            "closed_at": pos.get("closed_at", ""),
            "hold_hours": round(hold_hours, 2),
            "close_reason": pos.get("close_reason", ""),
            "stop_loss": pos.get("stop_loss", 0),
            "take_profit": pos.get("take_profit", 0),
            "order_id": pos.get("order_id", ""),
        })

    # Open positions (still active)
    for pos in state.get("positions", []):
        entry_price = pos.get("entry_price", 0)
        current_price = pos.get("current_price", entry_price)
        cost_basis = pos.get("cost_basis", 0)
        unrealized_pnl = pos.get("unrealized_pnl", 0)

        pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0

        opened_at = _parse_ts(pos.get("opened_at", ""))
        now = datetime.now(timezone.utc)
        hold_hours = 0.0
        if opened_at:
            hold_hours = (now - opened_at).total_seconds() / 3600

        records.append({
            "status": "open",
            "market_id": pos.get("market_id", ""),
            "question": pos.get("market_question", ""),
            "side": pos.get("side", ""),
            "signal": pos.get("signal", ""),
            "entry_price": entry_price,
            "exit_price": current_price,
            "size": pos.get("size", 0),
            "cost_basis": cost_basis,
            "realized_pnl": unrealized_pnl,
            "pnl_pct": round(pnl_pct, 2),
            "opened_at": pos.get("opened_at", ""),
            "closed_at": "",
            "hold_hours": round(hold_hours, 2),
            "close_reason": "",
            "stop_loss": pos.get("stop_loss", 0),
            "take_profit": pos.get("take_profit", 0),
            "order_id": pos.get("order_id", ""),
        })

    # Sort by opened_at
    records.sort(key=lambda r: r.get("opened_at", ""))

    return records


# ─── Report Generation ──────────────────────────────────────────────────────

def generate_performance_report(state: dict) -> str:
    """Generate a comprehensive performance analysis report."""
    records = build_trade_records(state)
    events = _load_trade_log()

    lines = []
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines.append("")
    lines.append("=" * 80)
    lines.append("  POLYMARKET TRADING PERFORMANCE REPORT")
    lines.append(f"  Generated: {now_str}")
    lines.append("=" * 80)

    # ── 1. Account Overview ──────────────────────────────────────────────
    bankroll = state.get("bankroll", config.BANKROLL)
    initial = config.BANKROLL
    total_pnl = state.get("total_pnl", 0)
    total_return = (total_pnl / initial * 100) if initial > 0 else 0

    lines.append("")
    lines.append("  ACCOUNT OVERVIEW")
    lines.append("  " + "-" * 76)
    lines.append(f"  Initial Bankroll:   ${initial:.2f}")
    lines.append(f"  Current Bankroll:   ${bankroll:.2f}")
    lines.append(f"  Total P&L:          ${total_pnl:+.2f} ({total_return:+.1f}%)")

    open_positions = [r for r in records if r["status"] == "open"]
    closed_positions = [r for r in records if r["status"] == "closed"]
    total_deployed = sum(r["cost_basis"] for r in open_positions)
    total_unrealized = sum(r["realized_pnl"] for r in open_positions)

    lines.append(f"  Capital Deployed:   ${total_deployed:.2f}")
    lines.append(f"  Capital Available:  ${bankroll - total_deployed:.2f}")
    lines.append(f"  Unrealized P&L:     ${total_unrealized:+.2f}")

    # ── 2. Trade Timeline ────────────────────────────────────────────────
    lines.append("")
    lines.append("  TRADE TIMELINE")
    lines.append("  " + "-" * 76)

    if records:
        table_data = []
        for r in records:
            opened = r["opened_at"][:19].replace("T", " ") if r["opened_at"] else "—"
            closed = r["closed_at"][:19].replace("T", " ") if r["closed_at"] else "—"

            pnl_str = f"${r['realized_pnl']:+.2f}"
            pct_str = f"{r['pnl_pct']:+.1f}%"
            status_marker = "OPEN" if r["status"] == "open" else "CLOSED"

            table_data.append([
                status_marker,
                opened,
                closed,
                r["signal"][:14],
                r["side"][:8],
                f"${r['cost_basis']:.2f}",
                f"{r['entry_price']:.3f}",
                f"{r['exit_price']:.3f}",
                pnl_str,
                pct_str,
                f"{r['hold_hours']:.1f}h",
            ])

        headers = ["Status", "Opened At", "Closed At", "Signal", "Side",
                    "Cost", "Entry", "Exit/Now", "P&L $", "P&L %", "Hold"]
        lines.append(tabulate(table_data, headers=headers, tablefmt="simple",
                              colalign=("left", "left", "left", "left", "left",
                                        "right", "right", "right", "right", "right", "right")))
    else:
        lines.append("  No trades recorded yet.")

    # ── 3. Closed Trade Statistics ───────────────────────────────────────
    lines.append("")
    lines.append("  CLOSED TRADE STATISTICS")
    lines.append("  " + "-" * 76)

    if closed_positions:
        wins = [r for r in closed_positions if r["realized_pnl"] > 0]
        losses = [r for r in closed_positions if r["realized_pnl"] <= 0]

        total_trades = len(closed_positions)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        avg_win = sum(r["realized_pnl"] for r in wins) / win_count if wins else 0
        avg_loss = sum(r["realized_pnl"] for r in losses) / loss_count if losses else 0
        avg_win_pct = sum(r["pnl_pct"] for r in wins) / win_count if wins else 0
        avg_loss_pct = sum(r["pnl_pct"] for r in losses) / loss_count if losses else 0

        best_trade = max(closed_positions, key=lambda r: r["realized_pnl"])
        worst_trade = min(closed_positions, key=lambda r: r["realized_pnl"])

        profit_factor = abs(sum(r["realized_pnl"] for r in wins) /
                          sum(r["realized_pnl"] for r in losses)) if losses and sum(r["realized_pnl"] for r in losses) != 0 else float("inf")

        avg_hold = sum(r["hold_hours"] for r in closed_positions) / total_trades

        # Expectancy: average $ gained per trade
        expectancy = sum(r["realized_pnl"] for r in closed_positions) / total_trades

        lines.append(f"  Total Closed:       {total_trades}")
        lines.append(f"  Wins / Losses:      {win_count} / {loss_count}")
        lines.append(f"  Win Rate:           {win_rate:.1f}%")
        lines.append(f"  Avg Win:            ${avg_win:+.2f} ({avg_win_pct:+.1f}%)")
        lines.append(f"  Avg Loss:           ${avg_loss:+.2f} ({avg_loss_pct:+.1f}%)")
        lines.append(f"  Best Trade:         ${best_trade['realized_pnl']:+.2f} ({best_trade['pnl_pct']:+.1f}%) - {best_trade['question'][:40]}")
        lines.append(f"  Worst Trade:        ${worst_trade['realized_pnl']:+.2f} ({worst_trade['pnl_pct']:+.1f}%) - {worst_trade['question'][:40]}")
        lines.append(f"  Profit Factor:      {profit_factor:.2f}")
        lines.append(f"  Expectancy:         ${expectancy:+.2f} per trade")
        lines.append(f"  Avg Hold Time:      {avg_hold:.1f} hours")
    else:
        lines.append("  No closed trades yet.")

    # ── 4. Strategy Breakdown ────────────────────────────────────────────
    lines.append("")
    lines.append("  STRATEGY BREAKDOWN")
    lines.append("  " + "-" * 76)

    if closed_positions:
        by_signal = defaultdict(list)
        for r in closed_positions:
            by_signal[r["signal"]].append(r)

        strat_table = []
        for signal, trades in sorted(by_signal.items()):
            n = len(trades)
            wins_n = len([t for t in trades if t["realized_pnl"] > 0])
            wr = (wins_n / n * 100) if n > 0 else 0
            total_pnl_s = sum(t["realized_pnl"] for t in trades)
            avg_pnl_pct = sum(t["pnl_pct"] for t in trades) / n if n > 0 else 0
            avg_hold_s = sum(t["hold_hours"] for t in trades) / n if n > 0 else 0

            strat_table.append([
                signal,
                n,
                f"{wr:.0f}%",
                f"${total_pnl_s:+.2f}",
                f"{avg_pnl_pct:+.1f}%",
                f"{avg_hold_s:.1f}h",
            ])

        headers = ["Strategy", "Trades", "Win Rate", "Total P&L", "Avg P&L %", "Avg Hold"]
        lines.append(tabulate(strat_table, headers=headers, tablefmt="simple"))
    else:
        lines.append("  No strategy data yet.")

    # ── 5. Close Reason Breakdown ────────────────────────────────────────
    lines.append("")
    lines.append("  CLOSE REASON BREAKDOWN")
    lines.append("  " + "-" * 76)

    if closed_positions:
        by_reason = defaultdict(list)
        for r in closed_positions:
            by_reason[r["close_reason"] or "manual"].append(r)

        reason_table = []
        for reason, trades in sorted(by_reason.items()):
            n = len(trades)
            total_pnl_r = sum(t["realized_pnl"] for t in trades)
            avg_pnl_pct_r = sum(t["pnl_pct"] for t in trades) / n if n > 0 else 0
            reason_table.append([
                reason,
                n,
                f"${total_pnl_r:+.2f}",
                f"{avg_pnl_pct_r:+.1f}%",
            ])

        headers = ["Reason", "Count", "Total P&L", "Avg P&L %"]
        lines.append(tabulate(reason_table, headers=headers, tablefmt="simple"))
    else:
        lines.append("  No data yet.")

    # ── 6. Equity Curve Data ─────────────────────────────────────────────
    lines.append("")
    lines.append("  EQUITY CURVE")
    lines.append("  " + "-" * 76)

    if closed_positions:
        running_bankroll = initial
        curve_data = [{"timestamp": "START", "bankroll": initial, "pnl": 0, "pnl_pct": 0}]

        sorted_closed = sorted(closed_positions, key=lambda r: r.get("closed_at", ""))
        for r in sorted_closed:
            running_bankroll += r["realized_pnl"]
            total_return_curve = ((running_bankroll - initial) / initial) * 100
            ts = r["closed_at"][:19].replace("T", " ") if r["closed_at"] else "?"
            curve_data.append({
                "timestamp": ts,
                "bankroll": round(running_bankroll, 2),
                "pnl": round(r["realized_pnl"], 2),
                "pnl_pct": round(total_return_curve, 2),
            })

        curve_table = []
        for pt in curve_data:
            curve_table.append([
                pt["timestamp"],
                f"${pt['bankroll']:.2f}",
                f"${pt['pnl']:+.2f}" if pt["pnl"] != 0 else "—",
                f"{pt['pnl_pct']:+.1f}%",
            ])

        headers = ["Timestamp", "Bankroll", "Trade P&L", "Total Return"]
        lines.append(tabulate(curve_table, headers=headers, tablefmt="simple"))

        # Drawdown
        peak = initial
        max_drawdown = 0
        running = initial
        for r in sorted_closed:
            running += r["realized_pnl"]
            peak = max(peak, running)
            dd = ((peak - running) / peak) * 100 if peak > 0 else 0
            max_drawdown = max(max_drawdown, dd)

        lines.append(f"\n  Max Drawdown:       {max_drawdown:.1f}%")
        lines.append(f"  Peak Bankroll:      ${peak:.2f}")
    else:
        lines.append("  No equity data yet (no closed trades).")

    # ── 7. Market Scan Activity ──────────────────────────────────────────
    scan_events = [e for e in events if e.get("event") == "market_scanned"]
    if scan_events:
        lines.append("")
        lines.append("  MARKET SCAN HISTORY (last 20)")
        lines.append("  " + "-" * 76)

        scan_table = []
        for evt in scan_events[-20:]:
            ts = evt["timestamp"][:19].replace("T", " ")
            scan_table.append([
                ts,
                evt.get("question", "")[:45],
                f"${evt.get('volume_24h', 0):,.0f}",
                f"{evt.get('price', 0):.3f}",
                evt.get("category", ""),
            ])

        headers = ["Scanned At", "Market", "Vol 24h", "Price", "Category"]
        lines.append(tabulate(scan_table, headers=headers, tablefmt="simple"))

    # ── 8. Trade Event Log (last 20) ─────────────────────────────────────
    trade_events = [e for e in events if e.get("event") in ("trade_opened", "trade_closed")]
    if trade_events:
        lines.append("")
        lines.append("  TRADE EVENT LOG (last 20)")
        lines.append("  " + "-" * 76)

        log_table = []
        for evt in trade_events[-20:]:
            ts = evt["timestamp"][:19].replace("T", " ")
            event_type = "OPEN" if evt["event"] == "trade_opened" else "CLOSE"
            pnl_str = ""
            if evt["event"] == "trade_closed":
                pnl_str = f"${evt.get('realized_pnl', 0):+.2f} ({evt.get('pnl_pct', 0):+.1f}%)"

            log_table.append([
                ts,
                event_type,
                evt.get("signal", "")[:14],
                evt.get("side", "")[:8],
                f"${evt.get('cost_basis', 0):.2f}",
                pnl_str or "—",
                evt.get("question", "")[:30],
            ])

        headers = ["Timestamp", "Event", "Signal", "Side", "Cost", "P&L", "Market"]
        lines.append(tabulate(log_table, headers=headers, tablefmt="simple"))

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


# ─── Export Functions ────────────────────────────────────────────────────────

def export_trades_csv(state: dict, filepath: Optional[str] = None) -> str:
    """Export all trade records to CSV for analysis in Excel/Google Sheets/pandas."""
    records = build_trade_records(state)

    if not filepath:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(REPORTS_DIR, f"trades_{ts}.csv")

    fieldnames = [
        "status", "opened_at", "closed_at", "hold_hours",
        "signal", "side", "market_id", "question",
        "entry_price", "exit_price", "size", "cost_basis",
        "realized_pnl", "pnl_pct",
        "stop_loss", "take_profit", "close_reason", "order_id",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    logger.info(f"Exported {len(records)} trades to {filepath}")
    return filepath


def export_report_json(state: dict, filepath: Optional[str] = None) -> str:
    """Export full report data as JSON for programmatic analysis."""
    records = build_trade_records(state)
    events = _load_trade_log()

    if not filepath:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(REPORTS_DIR, f"report_{ts}.json")

    # Build summary stats
    closed = [r for r in records if r["status"] == "closed"]
    wins = [r for r in closed if r["realized_pnl"] > 0]
    losses = [r for r in closed if r["realized_pnl"] <= 0]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "account": {
            "initial_bankroll": config.BANKROLL,
            "current_bankroll": state.get("bankroll", config.BANKROLL),
            "total_pnl": state.get("total_pnl", 0),
            "total_return_pct": round((state.get("total_pnl", 0) / config.BANKROLL * 100), 2) if config.BANKROLL > 0 else 0,
        },
        "statistics": {
            "total_trades": len(closed),
            "open_positions": len([r for r in records if r["status"] == "open"]),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else 0,
            "avg_win_pct": round(sum(r["pnl_pct"] for r in wins) / len(wins), 2) if wins else 0,
            "avg_loss_pct": round(sum(r["pnl_pct"] for r in losses) / len(losses), 2) if losses else 0,
            "best_trade_pnl": max((r["realized_pnl"] for r in closed), default=0),
            "worst_trade_pnl": min((r["realized_pnl"] for r in closed), default=0),
            "avg_hold_hours": round(sum(r["hold_hours"] for r in closed) / len(closed), 2) if closed else 0,
            "profit_factor": round(abs(sum(r["realized_pnl"] for r in wins) /
                                      sum(r["realized_pnl"] for r in losses)), 2) if losses and sum(r["realized_pnl"] for r in losses) != 0 else None,
            "expectancy_per_trade": round(sum(r["realized_pnl"] for r in closed) / len(closed), 4) if closed else 0,
        },
        "trades": records,
        "events": events,
    }

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Exported report to {filepath}")
    return filepath


def export_equity_curve_csv(state: dict, filepath: Optional[str] = None) -> str:
    """Export equity curve data points as CSV for charting."""
    records = build_trade_records(state)
    closed = sorted(
        [r for r in records if r["status"] == "closed"],
        key=lambda r: r.get("closed_at", "")
    )

    if not filepath:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(REPORTS_DIR, f"equity_curve_{ts}.csv")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "trade_num", "bankroll", "trade_pnl",
                          "trade_pnl_pct", "cumulative_pnl", "cumulative_return_pct",
                          "drawdown_pct", "signal", "question"])

        running = config.BANKROLL
        peak = config.BANKROLL
        writer.writerow(["START", 0, config.BANKROLL, 0, 0, 0, 0, 0, "", ""])

        for i, r in enumerate(closed, 1):
            running += r["realized_pnl"]
            peak = max(peak, running)
            cum_pnl = running - config.BANKROLL
            cum_ret = (cum_pnl / config.BANKROLL * 100) if config.BANKROLL > 0 else 0
            dd = ((peak - running) / peak * 100) if peak > 0 else 0

            writer.writerow([
                r.get("closed_at", ""),
                i,
                round(running, 2),
                round(r["realized_pnl"], 2),
                round(r["pnl_pct"], 2),
                round(cum_pnl, 2),
                round(cum_ret, 2),
                round(dd, 2),
                r.get("signal", ""),
                r.get("question", "")[:60],
            ])

    logger.info(f"Exported equity curve to {filepath}")
    return filepath
