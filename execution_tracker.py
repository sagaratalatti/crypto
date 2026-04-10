"""
Execution Quality Tracker: Measures fill rates, slippage, and time-to-fill.

For a $99 account, execution quality directly impacts profitability.
A strategy with 5% edge that suffers 3% slippage only nets 2%.

Tracks:
- Fill rate: % of limit orders that actually execute
- Slippage: Expected fill price vs actual fill price
- Time-to-fill: How long orders sit before filling
- Cancel rate: How often we cancel stale orders
- Effective spread cost: True cost of crossing the spread
"""

import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional

from tabulate import tabulate

import config

logger = logging.getLogger(__name__)

EXECUTION_LOG_FILE = os.path.join(os.path.dirname(__file__), "execution_log.json")


# ─── Execution Event Logging ────────────────────────────────────────────────

def _load_exec_log() -> list[dict]:
    if not os.path.exists(EXECUTION_LOG_FILE):
        return []
    try:
        with open(EXECUTION_LOG_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_exec_log(events: list[dict]):
    try:
        with open(EXECUTION_LOG_FILE, "w") as f:
            json.dump(events, f, indent=2, default=str)
    except IOError as e:
        logger.error(f"Failed to save execution log: {e}")


def log_order_placed(order_id: str, market_id: str, order_type: str,
                     side: str, expected_price: float, size: float,
                     dollar_amount: float, book_best_bid: float,
                     book_best_ask: float, book_spread: float):
    """Log when an order is placed (for fill tracking)."""
    events = _load_exec_log()
    events.append({
        "event": "order_placed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "order_id": order_id,
        "market_id": market_id,
        "order_type": order_type,
        "side": side,
        "expected_price": expected_price,
        "size": size,
        "dollar_amount": dollar_amount,
        "book_best_bid": book_best_bid,
        "book_best_ask": book_best_ask,
        "book_spread": book_spread,
    })
    _save_exec_log(events)


def log_order_filled(order_id: str, fill_price: float, fill_size: float,
                     fill_dollar: float):
    """Log when an order is filled."""
    events = _load_exec_log()
    events.append({
        "event": "order_filled",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "order_id": order_id,
        "fill_price": fill_price,
        "fill_size": fill_size,
        "fill_dollar": fill_dollar,
    })
    _save_exec_log(events)


def log_order_cancelled(order_id: str, reason: str):
    """Log when an order is cancelled."""
    events = _load_exec_log()
    events.append({
        "event": "order_cancelled",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "order_id": order_id,
        "reason": reason,
    })
    _save_exec_log(events)


# ─── Execution Quality Metrics ───────────────────────────────────────────────

def _parse_ts(ts_str: str) -> Optional[datetime]:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def compute_execution_metrics() -> dict:
    """
    Compute all execution quality metrics from the log.

    Returns a dict with:
    - fill_rate: % of orders that filled
    - avg_slippage: average (fill_price - expected_price)
    - avg_time_to_fill: average seconds between place and fill
    - cancel_rate: % of orders cancelled
    - effective_spread_cost: average spread paid
    - by_order_type: breakdown by GTC vs FOK
    """
    events = _load_exec_log()
    if not events:
        return _empty_metrics()

    # Index events by order_id
    placed = {}
    filled = {}
    cancelled = set()

    for evt in events:
        oid = evt.get("order_id", "")
        if evt["event"] == "order_placed":
            placed[oid] = evt
        elif evt["event"] == "order_filled":
            filled[oid] = evt
        elif evt["event"] == "order_cancelled":
            cancelled.add(oid)

    total_orders = len(placed)
    if total_orders == 0:
        return _empty_metrics()

    total_filled = len(filled)
    total_cancelled = len(cancelled)
    total_pending = total_orders - total_filled - total_cancelled

    # Fill rate
    fill_rate = (total_filled / total_orders * 100) if total_orders > 0 else 0

    # Slippage analysis
    slippage_values = []
    slippage_pct_values = []
    time_to_fill_values = []
    spread_costs = []

    by_type = defaultdict(lambda: {"placed": 0, "filled": 0, "cancelled": 0,
                                    "slippage": [], "time_to_fill": []})

    for oid, place_evt in placed.items():
        otype = place_evt.get("order_type", "unknown")
        by_type[otype]["placed"] += 1

        if oid in filled:
            fill_evt = filled[oid]
            by_type[otype]["filled"] += 1

            # Slippage = fill_price - expected_price (positive = worse for buyer)
            expected = place_evt.get("expected_price", 0)
            actual = fill_evt.get("fill_price", 0)
            if expected > 0 and actual > 0:
                slip = actual - expected
                slip_pct = (slip / expected) * 100
                slippage_values.append(slip)
                slippage_pct_values.append(slip_pct)
                by_type[otype]["slippage"].append(slip_pct)

            # Time to fill
            placed_at = _parse_ts(place_evt.get("timestamp", ""))
            filled_at = _parse_ts(fill_evt.get("timestamp", ""))
            if placed_at and filled_at:
                ttf = (filled_at - placed_at).total_seconds()
                time_to_fill_values.append(ttf)
                by_type[otype]["time_to_fill"].append(ttf)

            # Spread cost
            spread = place_evt.get("book_spread", 0)
            if spread > 0:
                spread_costs.append(spread)

        elif oid in cancelled:
            by_type[otype]["cancelled"] += 1

    # Aggregate
    avg_slippage = sum(slippage_values) / len(slippage_values) if slippage_values else 0
    avg_slippage_pct = sum(slippage_pct_values) / len(slippage_pct_values) if slippage_pct_values else 0
    avg_ttf = sum(time_to_fill_values) / len(time_to_fill_values) if time_to_fill_values else 0
    avg_spread = sum(spread_costs) / len(spread_costs) if spread_costs else 0
    cancel_rate = (total_cancelled / total_orders * 100) if total_orders > 0 else 0

    # Per-type breakdown
    type_breakdown = {}
    for otype, data in by_type.items():
        type_breakdown[otype] = {
            "placed": data["placed"],
            "filled": data["filled"],
            "cancelled": data["cancelled"],
            "fill_rate": round(data["filled"] / data["placed"] * 100, 1) if data["placed"] > 0 else 0,
            "avg_slippage_pct": round(sum(data["slippage"]) / len(data["slippage"]), 3) if data["slippage"] else 0,
            "avg_time_to_fill_s": round(sum(data["time_to_fill"]) / len(data["time_to_fill"]), 1) if data["time_to_fill"] else 0,
        }

    return {
        "total_orders": total_orders,
        "total_filled": total_filled,
        "total_cancelled": total_cancelled,
        "total_pending": total_pending,
        "fill_rate": round(fill_rate, 1),
        "cancel_rate": round(cancel_rate, 1),
        "avg_slippage": round(avg_slippage, 4),
        "avg_slippage_pct": round(avg_slippage_pct, 3),
        "max_slippage_pct": round(max(slippage_pct_values), 3) if slippage_pct_values else 0,
        "avg_time_to_fill_s": round(avg_ttf, 1),
        "min_time_to_fill_s": round(min(time_to_fill_values), 1) if time_to_fill_values else 0,
        "max_time_to_fill_s": round(max(time_to_fill_values), 1) if time_to_fill_values else 0,
        "avg_spread_cost": round(avg_spread, 4),
        "by_order_type": type_breakdown,
        "total_slippage_cost": round(sum(slippage_values), 4),
    }


def _empty_metrics() -> dict:
    return {
        "total_orders": 0, "total_filled": 0, "total_cancelled": 0,
        "total_pending": 0, "fill_rate": 0, "cancel_rate": 0,
        "avg_slippage": 0, "avg_slippage_pct": 0, "max_slippage_pct": 0,
        "avg_time_to_fill_s": 0, "min_time_to_fill_s": 0, "max_time_to_fill_s": 0,
        "avg_spread_cost": 0, "by_order_type": {}, "total_slippage_cost": 0,
    }


def format_execution_report(metrics: dict) -> str:
    """Format execution metrics as human-readable report."""
    lines = []
    lines.append("")
    lines.append("  EXECUTION QUALITY")
    lines.append("  " + "-" * 76)

    if metrics["total_orders"] == 0:
        lines.append("  No orders tracked yet.")
        return "\n".join(lines)

    lines.append(f"  Orders:    {metrics['total_orders']} placed, "
                 f"{metrics['total_filled']} filled, "
                 f"{metrics['total_cancelled']} cancelled, "
                 f"{metrics['total_pending']} pending")
    lines.append(f"  Fill Rate:           {metrics['fill_rate']:>6.1f}%")
    lines.append(f"  Cancel Rate:         {metrics['cancel_rate']:>6.1f}%")

    lines.append("")
    lines.append("  Slippage:")
    lines.append(f"    Average:           {metrics['avg_slippage_pct']:>+6.3f}%  (${metrics['avg_slippage']:+.4f})")
    lines.append(f"    Maximum:           {metrics['max_slippage_pct']:>+6.3f}%")
    lines.append(f"    Total Cost:        ${metrics['total_slippage_cost']:+.4f}")

    lines.append("")
    lines.append("  Time to Fill:")
    lines.append(f"    Average:           {metrics['avg_time_to_fill_s']:>6.1f}s")
    lines.append(f"    Fastest:           {metrics['min_time_to_fill_s']:>6.1f}s")
    lines.append(f"    Slowest:           {metrics['max_time_to_fill_s']:>6.1f}s")

    lines.append(f"  Avg Spread Cost:     {metrics['avg_spread_cost']:.4f}")

    # Per order type
    if metrics["by_order_type"]:
        lines.append("")
        lines.append("  By Order Type:")
        table = []
        for otype, data in metrics["by_order_type"].items():
            table.append([
                otype,
                data["placed"],
                data["filled"],
                f"{data['fill_rate']:.1f}%",
                f"{data['avg_slippage_pct']:+.3f}%",
                f"{data['avg_time_to_fill_s']:.1f}s",
            ])
        headers = ["Type", "Placed", "Filled", "Fill%", "Avg Slip", "Avg TTF"]
        lines.append("  " + tabulate(table, headers=headers, tablefmt="simple").replace("\n", "\n  "))

    return "\n".join(lines)
