"""
Advanced Risk Metrics: Sharpe, Sortino, VaR, drawdown, and correlation analysis.

These metrics answer the critical question for a $99 bankroll:
"Am I actually making risk-adjusted profits, or just getting lucky?"

Metrics:
- Sharpe Ratio: Return per unit of total volatility
- Sortino Ratio: Return per unit of downside volatility (better for asymmetric payoffs)
- Value at Risk (VaR): Maximum expected loss at a given confidence level
- Conditional VaR (CVaR): Average loss in the worst-case tail
- Max Drawdown: Worst peak-to-trough decline
- Calmar Ratio: Annual return / Max drawdown
- Position Correlation: How correlated are open positions (diversification check)
"""

import logging
import math
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


# ─── Return Series Construction ──────────────────────────────────────────────

def build_return_series(state: dict) -> list[dict]:
    """
    Build a time-ordered series of trade returns from portfolio state.

    Each entry: {timestamp, pnl, pnl_pct, bankroll_after, cost_basis}
    """
    closed = state.get("closed_positions", [])
    if not closed:
        return []

    sorted_trades = sorted(closed, key=lambda t: t.get("closed_at", ""))
    series = []
    running_bankroll = config.BANKROLL

    for trade in sorted_trades:
        pnl = trade.get("realized_pnl", 0)
        cost = trade.get("cost_basis", 0)
        pnl_pct = (pnl / cost * 100) if cost > 0 else 0
        running_bankroll += pnl

        series.append({
            "timestamp": trade.get("closed_at", ""),
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "cost_basis": cost,
            "bankroll_after": running_bankroll,
            "signal": trade.get("signal", ""),
            "market_id": trade.get("market_id", ""),
            "category": trade.get("category", ""),
        })

    return series


def _pnl_array(series: list[dict]) -> np.ndarray:
    """Extract P&L percentages as numpy array."""
    if not series:
        return np.array([])
    return np.array([s["pnl_pct"] for s in series])


# ─── Core Risk Metrics ───────────────────────────────────────────────────────

def sharpe_ratio(series: list[dict], risk_free_rate: float = 0.0) -> float:
    """
    Sharpe Ratio = (mean_return - risk_free_rate) / std_return

    For prediction markets, risk-free rate is ~0 (short holding periods).
    Annualization isn't meaningful here since trades have variable durations,
    so we report per-trade Sharpe.

    Interpretation:
      > 1.0: Good
      > 2.0: Very good
      > 3.0: Excellent
      < 0:   Losing money on average
    """
    returns = _pnl_array(series)
    if len(returns) < 2:
        return 0.0

    mean_r = np.mean(returns) - risk_free_rate
    std_r = np.std(returns, ddof=1)

    if std_r == 0:
        return 0.0

    return float(mean_r / std_r)


def sortino_ratio(series: list[dict], risk_free_rate: float = 0.0) -> float:
    """
    Sortino Ratio = (mean_return - risk_free_rate) / downside_deviation

    Better than Sharpe for prediction markets because upside volatility
    (big wins) shouldn't be penalized. Only downside risk matters.

    Uses semi-deviation: std of returns below the target (0%).
    """
    returns = _pnl_array(series)
    if len(returns) < 2:
        return 0.0

    mean_r = np.mean(returns) - risk_free_rate
    downside = returns[returns < 0]

    if len(downside) == 0:
        return float("inf") if mean_r > 0 else 0.0

    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0

    return float(mean_r / downside_std)


def value_at_risk(series: list[dict], confidence: float = 0.95) -> dict:
    """
    Value at Risk (VaR) - Historical simulation method.

    Answers: "What's the worst I can lose on a single trade at X% confidence?"

    Returns both:
    - VaR: The threshold loss at the confidence level
    - CVaR (Expected Shortfall): Average loss in the worst (1-confidence) cases

    For a $99 bankroll, this tells you the realistic worst-case per trade.
    """
    returns = _pnl_array(series)
    if len(returns) < 5:
        return {"var_pct": 0.0, "var_dollar": 0.0, "cvar_pct": 0.0, "cvar_dollar": 0.0,
                "confidence": confidence, "sample_size": len(returns)}

    # VaR is the percentile loss
    var_pct = float(np.percentile(returns, (1 - confidence) * 100))

    # CVaR is the average of returns worse than VaR
    tail = returns[returns <= var_pct]
    cvar_pct = float(np.mean(tail)) if len(tail) > 0 else var_pct

    # Convert to dollar amounts based on average position size
    avg_cost = np.mean([s["cost_basis"] for s in series]) if series else 0
    var_dollar = var_pct / 100 * avg_cost
    cvar_dollar = cvar_pct / 100 * avg_cost

    return {
        "var_pct": round(var_pct, 2),
        "var_dollar": round(var_dollar, 2),
        "cvar_pct": round(cvar_pct, 2),
        "cvar_dollar": round(cvar_dollar, 2),
        "confidence": confidence,
        "sample_size": len(returns),
    }


def drawdown_series(series: list[dict]) -> dict:
    """
    Compute running drawdown from equity curve.

    Returns:
    - max_drawdown_pct: Worst peak-to-trough decline
    - max_drawdown_dollar: Worst dollar decline
    - current_drawdown_pct: Current distance from peak
    - drawdown_duration: Longest streak below previous peak (in trades)
    - recovery_trades: How many trades to recover from max drawdown
    """
    if not series:
        return {"max_drawdown_pct": 0, "max_drawdown_dollar": 0,
                "current_drawdown_pct": 0, "drawdown_duration": 0, "recovery_trades": 0}

    bankrolls = [config.BANKROLL] + [s["bankroll_after"] for s in series]
    peak = bankrolls[0]
    max_dd_pct = 0.0
    max_dd_dollar = 0.0
    current_dd = 0.0

    # Track drawdown duration
    in_drawdown_since = None
    max_dd_duration = 0
    current_dd_duration = 0

    # Track recovery
    max_dd_idx = 0
    recovered_at = None

    for i, bal in enumerate(bankrolls):
        if bal >= peak:
            peak = bal
            if in_drawdown_since is not None:
                max_dd_duration = max(max_dd_duration, current_dd_duration)
                if recovered_at is None and i > max_dd_idx:
                    recovered_at = i
                current_dd_duration = 0
                in_drawdown_since = None
        else:
            dd_pct = (peak - bal) / peak * 100
            dd_dollar = peak - bal
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd_dollar = dd_dollar
                max_dd_idx = i
                recovered_at = None
            if in_drawdown_since is None:
                in_drawdown_since = i
            current_dd_duration = i - in_drawdown_since + 1

    current_dd = (peak - bankrolls[-1]) / peak * 100 if peak > 0 else 0
    max_dd_duration = max(max_dd_duration, current_dd_duration)
    recovery_trades = (recovered_at - max_dd_idx) if recovered_at else 0

    return {
        "max_drawdown_pct": round(max_dd_pct, 2),
        "max_drawdown_dollar": round(max_dd_dollar, 2),
        "current_drawdown_pct": round(current_dd, 2),
        "drawdown_duration": max_dd_duration,
        "recovery_trades": recovery_trades,
        "peak_bankroll": round(peak, 2),
    }


def calmar_ratio(series: list[dict]) -> float:
    """
    Calmar Ratio = Total Return % / Max Drawdown %

    Measures return relative to worst-case risk. Higher is better.
    > 1.0 means your total return exceeds your worst drawdown.
    """
    if not series:
        return 0.0

    total_return = (series[-1]["bankroll_after"] - config.BANKROLL) / config.BANKROLL * 100
    dd = drawdown_series(series)
    max_dd = dd["max_drawdown_pct"]

    if max_dd == 0:
        return float("inf") if total_return > 0 else 0.0

    return round(total_return / max_dd, 2)


# ─── Position Correlation ────────────────────────────────────────────────────

def position_correlation(state: dict) -> dict:
    """
    Analyze correlation between open positions.

    For prediction markets, positions in the same category or related
    markets are correlated. True diversification means spreading across
    uncorrelated categories.

    Returns:
    - category_concentration: % of capital in most-concentrated category
    - diversification_score: 0-1, higher = more diversified (Herfindahl-based)
    - correlated_pairs: Positions that are likely correlated
    """
    positions = state.get("positions", [])
    if not positions:
        return {"category_concentration": 0, "diversification_score": 1.0,
                "correlated_pairs": [], "exposure_by_category": {}}

    # Group by category/signal
    by_category = defaultdict(float)
    by_signal = defaultdict(float)
    total_cost = 0

    for pos in positions:
        cost = pos.get("cost_basis", 0)
        total_cost += cost
        # Use market_id prefix or signal as category proxy
        cat = pos.get("category", pos.get("signal", "unknown"))
        by_category[cat] += cost
        by_signal[pos.get("signal", "unknown")] += cost

    if total_cost == 0:
        return {"category_concentration": 0, "diversification_score": 1.0,
                "correlated_pairs": [], "exposure_by_category": {}}

    # Herfindahl-Hirschman Index for diversification
    # HHI = sum of squared market shares. Lower = more diversified.
    shares = [v / total_cost for v in by_category.values()]
    hhi = sum(s ** 2 for s in shares)
    # Normalize: 1/n (perfect diversification) to 1.0 (single category)
    n = len(by_category)
    min_hhi = 1 / n if n > 0 else 1
    diversification_score = 1 - (hhi - min_hhi) / (1 - min_hhi) if n > 1 else 0.0

    max_category = max(by_category, key=by_category.get)
    concentration = by_category[max_category] / total_cost * 100

    # Find potentially correlated pairs (same category)
    correlated = []
    pos_list = list(positions)
    for i in range(len(pos_list)):
        for j in range(i + 1, len(pos_list)):
            cat_i = pos_list[i].get("category", pos_list[i].get("signal", ""))
            cat_j = pos_list[j].get("category", pos_list[j].get("signal", ""))
            if cat_i == cat_j:
                correlated.append({
                    "pos_a": pos_list[i].get("market_question", "")[:40],
                    "pos_b": pos_list[j].get("market_question", "")[:40],
                    "shared_category": cat_i,
                })

    exposure = {k: round(v, 2) for k, v in by_category.items()}

    return {
        "category_concentration": round(concentration, 1),
        "diversification_score": round(diversification_score, 3),
        "correlated_pairs": correlated,
        "exposure_by_category": exposure,
        "exposure_by_signal": {k: round(v, 2) for k, v in by_signal.items()},
    }


# ─── Aggregate Risk Dashboard ───────────────────────────────────────────────

def compute_all_risk_metrics(state: dict) -> dict:
    """Compute all risk metrics and return as a single dict."""
    series = build_return_series(state)

    return {
        "sharpe_ratio": round(sharpe_ratio(series), 3),
        "sortino_ratio": round(sortino_ratio(series), 3),
        "calmar_ratio": calmar_ratio(series),
        "var_95": value_at_risk(series, 0.95),
        "var_99": value_at_risk(series, 0.99),
        "drawdown": drawdown_series(series),
        "correlation": position_correlation(state),
        "trade_count": len(series),
        "win_rate": round(len([s for s in series if s["pnl"] > 0]) / len(series) * 100, 1) if series else 0,
        "avg_return_pct": round(float(np.mean(_pnl_array(series))), 2) if series else 0,
        "return_std_pct": round(float(np.std(_pnl_array(series), ddof=1)), 2) if len(series) > 1 else 0,
        "total_return_pct": round((state.get("bankroll", config.BANKROLL) - config.BANKROLL) / config.BANKROLL * 100, 2),
    }


def format_risk_report(metrics: dict) -> str:
    """Format risk metrics as a human-readable report section."""
    lines = []
    lines.append("")
    lines.append("  ADVANCED RISK METRICS")
    lines.append("  " + "-" * 76)

    if metrics["trade_count"] < 2:
        lines.append("  Insufficient trades for risk analysis (need >= 2 closed trades).")
        return "\n".join(lines)

    # Risk-adjusted returns
    lines.append("  Risk-Adjusted Performance:")
    sharpe = metrics["sharpe_ratio"]
    sharpe_rating = "Excellent" if sharpe > 2 else "Good" if sharpe > 1 else "Fair" if sharpe > 0 else "Poor"
    lines.append(f"    Sharpe Ratio:       {sharpe:>8.3f}  ({sharpe_rating})")

    sortino = metrics["sortino_ratio"]
    sortino_rating = "Excellent" if sortino > 3 else "Good" if sortino > 1.5 else "Fair" if sortino > 0 else "Poor"
    lines.append(f"    Sortino Ratio:      {sortino:>8.3f}  ({sortino_rating})")

    calmar = metrics["calmar_ratio"]
    lines.append(f"    Calmar Ratio:       {calmar:>8.2f}")

    lines.append(f"    Avg Return/Trade:   {metrics['avg_return_pct']:>+7.2f}%")
    lines.append(f"    Return Std Dev:     {metrics['return_std_pct']:>7.2f}%")

    # Value at Risk
    lines.append("")
    lines.append("  Value at Risk (per trade):")
    var95 = metrics["var_95"]
    lines.append(f"    VaR 95%:            {var95['var_pct']:>+7.2f}%  (${var95['var_dollar']:+.2f})")
    lines.append(f"    CVaR 95%:           {var95['cvar_pct']:>+7.2f}%  (${var95['cvar_dollar']:+.2f})")
    var99 = metrics["var_99"]
    lines.append(f"    VaR 99%:            {var99['var_pct']:>+7.2f}%  (${var99['var_dollar']:+.2f})")
    lines.append(f"    CVaR 99%:           {var99['cvar_pct']:>+7.2f}%  (${var99['cvar_dollar']:+.2f})")
    lines.append(f"    Sample size:        {var95['sample_size']} trades")

    # Drawdown
    lines.append("")
    dd = metrics["drawdown"]
    lines.append("  Drawdown Analysis:")
    lines.append(f"    Max Drawdown:       {dd['max_drawdown_pct']:>7.2f}%  (${dd['max_drawdown_dollar']:.2f})")
    lines.append(f"    Current Drawdown:   {dd['current_drawdown_pct']:>7.2f}%")
    lines.append(f"    Peak Bankroll:      ${dd['peak_bankroll']:.2f}")
    lines.append(f"    Longest DD Streak:  {dd['drawdown_duration']} trades")
    lines.append(f"    Recovery Trades:    {dd['recovery_trades']}")

    # Correlation
    lines.append("")
    corr = metrics["correlation"]
    lines.append("  Position Diversification:")
    lines.append(f"    Diversification:    {corr['diversification_score']:.3f}  (0=concentrated, 1=diversified)")
    lines.append(f"    Top Concentration:  {corr['category_concentration']:.1f}%")

    if corr.get("exposure_by_signal"):
        lines.append("    Exposure by Signal:")
        for sig, amt in corr["exposure_by_signal"].items():
            lines.append(f"      {sig:18s}  ${amt:.2f}")

    if corr.get("correlated_pairs"):
        lines.append(f"    Correlated Pairs:   {len(corr['correlated_pairs'])}")
        for pair in corr["correlated_pairs"][:3]:
            lines.append(f"      [{pair['shared_category']}] {pair['pos_a']} <-> {pair['pos_b']}")

    return "\n".join(lines)
