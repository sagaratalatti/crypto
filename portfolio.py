"""
Portfolio Tracker: Persists positions, tracks P&L, and manages portfolio state.

Saves state to a JSON file so the bot can resume after restarts.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from executor import Position, OrderRecord
from risk_manager import PortfolioState
import config

logger = logging.getLogger(__name__)

STATE_FILE = os.path.join(os.path.dirname(__file__), "portfolio_state.json")


def load_state() -> dict:
    """Load portfolio state from disk."""
    if not os.path.exists(STATE_FILE):
        return {
            "bankroll": config.BANKROLL,
            "positions": [],
            "closed_positions": [],
            "orders": [],
            "total_pnl": 0.0,
            "trade_count": 0,
            "win_count": 0,
            "last_updated": "",
        }

    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load state: {e}")
        return {
            "bankroll": config.BANKROLL,
            "positions": [],
            "closed_positions": [],
            "orders": [],
            "total_pnl": 0.0,
            "trade_count": 0,
            "win_count": 0,
            "last_updated": "",
        }


def save_state(state: dict):
    """Persist portfolio state to disk."""
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except IOError as e:
        logger.error(f"Failed to save state: {e}")


def get_portfolio_state(state: dict) -> PortfolioState:
    """Convert stored state to PortfolioState for risk calculations."""
    total_deployed = sum(p.get("cost_basis", 0) for p in state.get("positions", []))

    return PortfolioState(
        bankroll=state.get("bankroll", config.BANKROLL),
        total_deployed=total_deployed,
        num_positions=len(state.get("positions", [])),
        positions=state.get("positions", []),
    )


def add_position(state: dict, order: OrderRecord, stop_loss: float, take_profit: float) -> dict:
    """Record a new position from a filled order."""
    position = {
        "market_id": order.market_id,
        "market_question": order.reason[:100],
        "token_id": order.token_id,
        "side": order.side,
        "entry_price": order.price,
        "size": order.size,
        "cost_basis": order.dollar_amount,
        "current_price": order.price,
        "unrealized_pnl": 0.0,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "signal": order.signal,
        "order_id": order.order_id,
    }

    state["positions"].append(position)
    state["trade_count"] = state.get("trade_count", 0) + 1
    save_state(state)

    logger.info(
        f"Position opened: {order.side} {order.size} shares at ${order.price:.3f} "
        f"(cost=${order.dollar_amount:.2f}, SL=${stop_loss:.3f}, TP=${take_profit:.3f})"
    )

    return state


def close_position(state: dict, position_idx: int, exit_price: float, reason: str) -> dict:
    """Close a position and record P&L."""
    if position_idx >= len(state["positions"]):
        return state

    pos = state["positions"].pop(position_idx)

    # Calculate P&L
    if pos["side"].startswith("buy"):
        pnl = (exit_price - pos["entry_price"]) * pos["size"]
    else:
        pnl = (pos["entry_price"] - exit_price) * pos["size"]

    pos["exit_price"] = exit_price
    pos["realized_pnl"] = round(pnl, 4)
    pos["closed_at"] = datetime.now(timezone.utc).isoformat()
    pos["close_reason"] = reason

    state["closed_positions"].append(pos)
    state["total_pnl"] = round(state.get("total_pnl", 0) + pnl, 4)
    state["bankroll"] = round(state.get("bankroll", config.BANKROLL) + pnl, 4)

    if pnl > 0:
        state["win_count"] = state.get("win_count", 0) + 1

    save_state(state)

    logger.info(
        f"Position closed: PnL=${pnl:+.4f} ({reason}), "
        f"Total PnL=${state['total_pnl']:+.4f}, Bankroll=${state['bankroll']:.2f}"
    )

    return state


def update_position_prices(state: dict, clob_client) -> dict:
    """Update current prices for all open positions."""
    for pos in state.get("positions", []):
        try:
            if clob_client:
                midpoint = clob_client.get_midpoint(pos["token_id"])
                if midpoint:
                    price = float(midpoint) if not isinstance(midpoint, float) else midpoint
                    pos["current_price"] = price
                    pos["unrealized_pnl"] = round(
                        (price - pos["entry_price"]) * pos["size"], 4
                    )
        except Exception as e:
            logger.debug(f"Failed to update price for position: {e}")

    save_state(state)
    return state


def check_stop_loss_take_profit(state: dict, clob_client) -> dict:
    """Check if any positions hit their stop-loss or take-profit levels."""
    positions_to_close = []

    for i, pos in enumerate(state.get("positions", [])):
        price = pos.get("current_price", 0)
        if price <= 0:
            continue

        if pos["side"].startswith("buy"):
            if price <= pos.get("stop_loss", 0) and pos.get("stop_loss", 0) > 0:
                positions_to_close.append((i, price, "stop_loss"))
            elif price >= pos.get("take_profit", 1) and pos.get("take_profit", 1) < 1:
                positions_to_close.append((i, price, "take_profit"))

    # Close in reverse order to maintain indices
    for idx, exit_price, reason in reversed(positions_to_close):
        state = close_position(state, idx, exit_price, reason)

    return state


def get_portfolio_summary(state: dict) -> str:
    """Generate a human-readable portfolio summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("PORTFOLIO SUMMARY")
    lines.append("=" * 70)
    lines.append(f"  Bankroll:     ${state.get('bankroll', 0):.2f}")
    lines.append(f"  Total PnL:    ${state.get('total_pnl', 0):+.2f}")

    positions = state.get("positions", [])
    total_deployed = sum(p.get("cost_basis", 0) for p in positions)
    total_unrealized = sum(p.get("unrealized_pnl", 0) for p in positions)

    lines.append(f"  Deployed:     ${total_deployed:.2f}")
    lines.append(f"  Available:    ${state.get('bankroll', 0) - total_deployed:.2f}")
    lines.append(f"  Unrealized:   ${total_unrealized:+.2f}")
    lines.append(f"  Positions:    {len(positions)}/{config.MAX_CONCURRENT_POSITIONS}")

    trade_count = state.get("trade_count", 0)
    win_count = state.get("win_count", 0)
    win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0

    lines.append(f"  Trades:       {trade_count} (Win rate: {win_rate:.0f}%)")
    lines.append("")

    if positions:
        lines.append("  OPEN POSITIONS:")
        lines.append("  " + "-" * 66)
        for pos in positions:
            pnl = pos.get("unrealized_pnl", 0)
            pnl_str = f"${pnl:+.2f}" if pnl != 0 else "$0.00"
            lines.append(
                f"  {pos.get('side', '?'):10s} | "
                f"${pos.get('cost_basis', 0):7.2f} | "
                f"Entry: {pos.get('entry_price', 0):.3f} | "
                f"Now: {pos.get('current_price', 0):.3f} | "
                f"PnL: {pnl_str:>8s} | "
                f"{pos.get('signal', '?')}"
            )
            q = pos.get("market_question", "")[:55]
            lines.append(f"             {q}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)
