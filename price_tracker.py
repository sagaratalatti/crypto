"""
Price History Tracker and Backtesting Engine.

Tracks historical prices for markets over time, enabling:
- Trend detection (is a market trending up/down?)
- Volatility measurement (how much does this market move?)
- Backtesting strategies against historical data
- Walk-forward parameter optimization

Price data is stored in a local JSON file since Polymarket
doesn't provide historical OHLCV data through its API.
"""

import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
from tabulate import tabulate

import config

logger = logging.getLogger(__name__)

PRICE_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "price_history.json")


# ─── Price History Storage ───────────────────────────────────────────────────

def _load_history() -> dict:
    if not os.path.exists(PRICE_HISTORY_FILE):
        return {}
    try:
        with open(PRICE_HISTORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_history(history: dict):
    try:
        with open(PRICE_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2, default=str)
    except IOError as e:
        logger.error(f"Failed to save price history: {e}")


def record_price_snapshot(market_id: str, question: str, yes_price: float,
                          no_price: float, volume_24h: float, liquidity: float,
                          best_bid: float = 0, best_ask: float = 0,
                          spread: float = 0):
    """Record a price snapshot for a market. Called during each scan cycle."""
    history = _load_history()

    if market_id not in history:
        history[market_id] = {
            "question": question,
            "snapshots": [],
        }

    # Update question (might change display)
    history[market_id]["question"] = question

    snapshot = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "yes": round(yes_price, 4),
        "no": round(no_price, 4),
        "vol24h": round(volume_24h, 2),
        "liq": round(liquidity, 2),
        "bid": round(best_bid, 4),
        "ask": round(best_ask, 4),
        "spread": round(spread, 4),
    }

    history[market_id]["snapshots"].append(snapshot)

    # Keep max 2000 snapshots per market (at 5min intervals = ~7 days)
    if len(history[market_id]["snapshots"]) > 2000:
        history[market_id]["snapshots"] = history[market_id]["snapshots"][-2000:]

    _save_history(history)


def get_price_series(market_id: str) -> list[dict]:
    """Get the price history for a market."""
    history = _load_history()
    entry = history.get(market_id, {})
    return entry.get("snapshots", [])


def get_tracked_markets() -> dict[str, str]:
    """Get all markets with price history: {market_id: question}."""
    history = _load_history()
    return {mid: data.get("question", "") for mid, data in history.items()}


# ─── Price Analytics ─────────────────────────────────────────────────────────

def compute_market_analytics(market_id: str) -> dict:
    """
    Compute analytics from price history for a single market.

    Returns trend, volatility, support/resistance levels, volume patterns.
    """
    snapshots = get_price_series(market_id)
    if len(snapshots) < 3:
        return {"status": "insufficient_data", "snapshots": len(snapshots)}

    prices = np.array([s["yes"] for s in snapshots])
    volumes = np.array([s["vol24h"] for s in snapshots])
    spreads = np.array([s["spread"] for s in snapshots if s.get("spread", 0) > 0])

    n = len(prices)

    # Current and recent prices
    current = prices[-1]
    prev = prices[-2] if n > 1 else current

    # Returns
    returns = np.diff(prices) / prices[:-1] * 100 if n > 1 else np.array([])

    # Trend (linear regression slope over last 20 snapshots)
    window = min(20, n)
    recent_prices = prices[-window:]
    x = np.arange(window)
    if window > 1:
        slope = np.polyfit(x, recent_prices, 1)[0]
        trend_per_hour = slope * (3600 / config.SCAN_INTERVAL_SECONDS)  # Normalize to per-hour
    else:
        slope = 0
        trend_per_hour = 0

    # Volatility (std of returns)
    volatility = float(np.std(returns)) if len(returns) > 1 else 0

    # Simple moving averages
    sma_5 = float(np.mean(prices[-5:])) if n >= 5 else float(current)
    sma_20 = float(np.mean(prices[-20:])) if n >= 20 else float(np.mean(prices))

    # Support / Resistance (min/max over lookback periods)
    support = float(np.min(prices[-min(50, n):])) if n > 0 else 0
    resistance = float(np.max(prices[-min(50, n):])) if n > 0 else 0

    # Volume trend
    vol_avg = float(np.mean(volumes)) if len(volumes) > 0 else 0
    vol_current = float(volumes[-1]) if len(volumes) > 0 else 0
    vol_ratio = vol_current / vol_avg if vol_avg > 0 else 1.0

    # Average spread
    avg_spread = float(np.mean(spreads)) if len(spreads) > 0 else 0

    # Price range (high-low over history)
    price_range = float(np.max(prices) - np.min(prices))

    return {
        "status": "ok",
        "snapshots": n,
        "current_price": round(float(current), 4),
        "previous_price": round(float(prev), 4),
        "price_change": round(float(current - prev), 4),
        "price_change_pct": round(float((current - prev) / prev * 100), 2) if prev > 0 else 0,
        "trend_slope": round(float(slope), 6),
        "trend_per_hour": round(float(trend_per_hour), 6),
        "trend_direction": "up" if slope > 0.001 else "down" if slope < -0.001 else "flat",
        "volatility": round(volatility, 4),
        "sma_5": round(sma_5, 4),
        "sma_20": round(sma_20, 4),
        "sma_crossover": "bullish" if sma_5 > sma_20 else "bearish" if sma_5 < sma_20 else "neutral",
        "support": round(support, 4),
        "resistance": round(resistance, 4),
        "price_range": round(price_range, 4),
        "volume_avg": round(vol_avg, 2),
        "volume_current": round(vol_current, 2),
        "volume_ratio": round(vol_ratio, 2),
        "avg_spread": round(avg_spread, 4),
    }


# ─── Backtesting Engine ─────────────────────────────────────────────────────

def backtest_strategy(market_id: str, strategy_fn, initial_capital: float = 99.0,
                      kelly_fraction: float = 0.25) -> dict:
    """
    Run a strategy function against historical price data for a market.

    strategy_fn(prices_so_far, current_snapshot) -> {
        "action": "buy_yes" | "buy_no" | "hold",
        "edge": float,
        "confidence": float,
    }

    Returns backtesting results with P&L, drawdown, trade log.
    """
    snapshots = get_price_series(market_id)
    if len(snapshots) < 10:
        return {"status": "insufficient_data", "snapshots": len(snapshots)}

    bankroll = initial_capital
    position = None  # {"side", "entry_price", "size", "cost"}
    trades = []
    equity_curve = [{"ts": snapshots[0]["ts"], "bankroll": bankroll}]
    peak = bankroll
    max_dd = 0

    for i in range(5, len(snapshots)):  # Need at least 5 points of history
        current = snapshots[i]
        history_window = snapshots[:i]

        # Get strategy signal
        signal = strategy_fn(history_window, current)
        action = signal.get("action", "hold")
        edge = signal.get("edge", 0)
        confidence = signal.get("confidence", 0)

        current_price = current["yes"]

        # Check if we should close existing position
        if position:
            pos_price = current_price if position["side"] == "buy_yes" else (1 - current_price)
            pnl = (pos_price - position["entry_price"]) * position["size"]

            # Simple exit: close after 10 snapshots or if P&L hits target/stop
            hold_time = i - position["opened_at_idx"]
            pnl_pct = pnl / position["cost"] * 100 if position["cost"] > 0 else 0

            should_close = (
                hold_time >= 10 or
                pnl_pct >= 15 or  # Take profit at 15%
                pnl_pct <= -20    # Stop loss at -20%
            )

            if should_close:
                bankroll += position["cost"] + pnl
                trades.append({
                    "side": position["side"],
                    "entry_price": position["entry_price"],
                    "exit_price": pos_price,
                    "size": position["size"],
                    "cost": position["cost"],
                    "pnl": round(pnl, 4),
                    "pnl_pct": round(pnl_pct, 2),
                    "hold_snapshots": hold_time,
                    "opened_at": position["opened_at"],
                    "closed_at": current["ts"],
                    "close_reason": "take_profit" if pnl_pct >= 15 else "stop_loss" if pnl_pct <= -20 else "timeout",
                })
                position = None

        # Open new position if no current position and signal says to trade
        if position is None and action != "hold" and edge > 0:
            # Kelly sizing
            price = current_price if action == "buy_yes" else (1 - current_price)
            if 0.01 < price < 0.99:
                odds = (1 / price) - 1
                kelly = min(edge / odds * kelly_fraction, 0.25)
                cost = kelly * bankroll

                if cost >= 5.0:
                    size = cost / price
                    bankroll -= cost
                    position = {
                        "side": action,
                        "entry_price": price,
                        "size": size,
                        "cost": cost,
                        "opened_at": current["ts"],
                        "opened_at_idx": i,
                    }

        # Track equity curve
        portfolio_value = bankroll
        if position:
            pos_price = current_price if position["side"] == "buy_yes" else (1 - current_price)
            portfolio_value += position["cost"] + (pos_price - position["entry_price"]) * position["size"]

        equity_curve.append({"ts": current["ts"], "bankroll": round(portfolio_value, 2)})
        peak = max(peak, portfolio_value)
        dd = (peak - portfolio_value) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Close any remaining position at last price
    if position:
        current = snapshots[-1]
        pos_price = current["yes"] if position["side"] == "buy_yes" else (1 - current["yes"])
        pnl = (pos_price - position["entry_price"]) * position["size"]
        bankroll += position["cost"] + pnl
        trades.append({
            "side": position["side"],
            "entry_price": position["entry_price"],
            "exit_price": pos_price,
            "size": position["size"],
            "cost": position["cost"],
            "pnl": round(pnl, 4),
            "pnl_pct": round(pnl / position["cost"] * 100, 2) if position["cost"] > 0 else 0,
            "hold_snapshots": len(snapshots) - 1 - position["opened_at_idx"],
            "opened_at": position["opened_at"],
            "closed_at": current["ts"],
            "close_reason": "end_of_data",
        })

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    return {
        "status": "ok",
        "initial_capital": initial_capital,
        "final_bankroll": round(bankroll, 2),
        "total_return": round(bankroll - initial_capital, 2),
        "total_return_pct": round((bankroll - initial_capital) / initial_capital * 100, 2),
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "avg_pnl": round(sum(t["pnl"] for t in trades) / len(trades), 4) if trades else 0,
        "avg_win_pct": round(sum(t["pnl_pct"] for t in wins) / len(wins), 2) if wins else 0,
        "avg_loss_pct": round(sum(t["pnl_pct"] for t in losses) / len(losses), 2) if losses else 0,
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": _backtest_sharpe(trades),
        "profit_factor": round(abs(sum(t["pnl"] for t in wins) /
                                   sum(t["pnl"] for t in losses)), 2) if losses and sum(t["pnl"] for t in losses) != 0 else None,
        "trades": trades,
        "equity_curve": equity_curve,
        "snapshots_used": len(snapshots),
    }


def _backtest_sharpe(trades: list[dict]) -> float:
    if len(trades) < 2:
        return 0.0
    returns = np.array([t["pnl_pct"] for t in trades])
    std = np.std(returns, ddof=1)
    return round(float(np.mean(returns) / std), 3) if std > 0 else 0.0


# ─── Built-in Backtest Strategies ────────────────────────────────────────────

def mean_reversion_strategy(history: list[dict], current: dict) -> dict:
    """Simple mean reversion: buy when below SMA, sell when above."""
    prices = [s["yes"] for s in history]
    sma = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
    current_price = current["yes"]

    if current_price < sma - 0.03:
        return {"action": "buy_yes", "edge": min(float(sma - current_price), 0.05), "confidence": 0.5}
    elif current_price > sma + 0.03:
        return {"action": "buy_no", "edge": min(float(current_price - sma), 0.05), "confidence": 0.5}
    return {"action": "hold", "edge": 0, "confidence": 0}


def momentum_strategy(history: list[dict], current: dict) -> dict:
    """Simple momentum: buy in direction of recent trend."""
    if len(history) < 5:
        return {"action": "hold", "edge": 0, "confidence": 0}

    recent = [s["yes"] for s in history[-5:]]
    slope = (recent[-1] - recent[0]) / len(recent)

    if slope > 0.005:
        return {"action": "buy_yes", "edge": min(abs(slope) * 10, 0.04), "confidence": 0.45}
    elif slope < -0.005:
        return {"action": "buy_no", "edge": min(abs(slope) * 10, 0.04), "confidence": 0.45}
    return {"action": "hold", "edge": 0, "confidence": 0}


def volume_spike_strategy(history: list[dict], current: dict) -> dict:
    """Buy when volume spikes above 2x average with price direction."""
    volumes = [s["vol24h"] for s in history]
    avg_vol = np.mean(volumes) if volumes else 0

    if avg_vol > 0 and current["vol24h"] > avg_vol * 2:
        # Volume spike - trade with the direction
        recent_prices = [s["yes"] for s in history[-3:]]
        if len(recent_prices) >= 2:
            direction = recent_prices[-1] - recent_prices[0]
            if direction > 0.02:
                return {"action": "buy_yes", "edge": 0.03, "confidence": 0.4}
            elif direction < -0.02:
                return {"action": "buy_no", "edge": 0.03, "confidence": 0.4}
    return {"action": "hold", "edge": 0, "confidence": 0}


def format_backtest_report(result: dict, market_question: str = "") -> str:
    """Format backtest results as a report."""
    lines = []
    lines.append("")
    lines.append(f"  BACKTEST RESULTS" + (f": {market_question[:50]}" if market_question else ""))
    lines.append("  " + "-" * 76)

    if result.get("status") == "insufficient_data":
        lines.append(f"  Insufficient data ({result.get('snapshots', 0)} snapshots, need >= 10)")
        return "\n".join(lines)

    lines.append(f"  Data Points:      {result['snapshots_used']}")
    lines.append(f"  Initial Capital:  ${result['initial_capital']:.2f}")
    lines.append(f"  Final Bankroll:   ${result['final_bankroll']:.2f}")
    lines.append(f"  Total Return:     ${result['total_return']:+.2f} ({result['total_return_pct']:+.1f}%)")
    lines.append(f"  Trades:           {result['total_trades']}")
    lines.append(f"  Win Rate:         {result['win_rate']:.1f}%")
    lines.append(f"  Avg Win:          {result['avg_win_pct']:+.1f}%")
    lines.append(f"  Avg Loss:         {result['avg_loss_pct']:+.1f}%")
    lines.append(f"  Max Drawdown:     {result['max_drawdown_pct']:.1f}%")
    lines.append(f"  Sharpe Ratio:     {result['sharpe']:.3f}")
    pf = result.get('profit_factor')
    lines.append(f"  Profit Factor:    {pf:.2f}" if pf is not None else "  Profit Factor:    N/A")

    return "\n".join(lines)
