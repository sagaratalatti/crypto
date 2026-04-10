"""
Cross-Market Correlation and Whale Signal Detection.

Alpha generation through:
1. Cross-market arbitrage: Related markets that should move together
2. Whale tracking: Large positions from profitable wallets
3. Category momentum: When a whole category moves, late markets follow
4. Resolution clustering: Markets expiring together create exit pressure
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import requests
from tabulate import tabulate

import config
from market_scanner import MarketInfo

logger = logging.getLogger(__name__)

WHALE_LOG_FILE = os.path.join(os.path.dirname(__file__), "whale_signals.json")


# ─── Cross-Market Correlation ────────────────────────────────────────────────

def find_correlated_markets(markets: list[MarketInfo]) -> list[dict]:
    """
    Find pairs of markets that are likely correlated.

    Correlation sources:
    1. Same category (crypto, politics, etc)
    2. Same underlying event (e.g., "BTC > 80k" and "BTC > 90k")
    3. Inverse markets (one going up implies the other goes down)

    Returns pairs with correlation type and potential arbitrage.
    """
    pairs = []

    # Group by category
    by_category = defaultdict(list)
    for m in markets:
        if m.category:
            by_category[m.category].append(m)

    # Find keyword-related markets (same entity/topic)
    keywords = _extract_keywords(markets)

    for keyword, keyword_markets in keywords.items():
        if len(keyword_markets) < 2:
            continue

        for i in range(len(keyword_markets)):
            for j in range(i + 1, len(keyword_markets)):
                m_a = keyword_markets[i]
                m_b = keyword_markets[j]

                if m_a.market_id == m_b.market_id:
                    continue

                price_a = m_a.outcome_prices[0] if m_a.outcome_prices else 0.5
                price_b = m_b.outcome_prices[0] if m_b.outcome_prices else 0.5

                # Check for logical constraints between related markets
                # e.g., "BTC > 80k" should be >= "BTC > 90k"
                relationship = _detect_relationship(m_a, m_b)

                pairs.append({
                    "market_a_id": m_a.market_id,
                    "market_a_question": m_a.question,
                    "market_a_price": price_a,
                    "market_b_id": m_b.market_id,
                    "market_b_question": m_b.question,
                    "market_b_price": price_b,
                    "shared_keyword": keyword,
                    "category": m_a.category,
                    "relationship": relationship,
                    "price_diff": round(abs(price_a - price_b), 4),
                })

    # Sort by price difference (biggest gaps = most potential)
    pairs.sort(key=lambda p: p["price_diff"], reverse=True)

    return pairs[:20]  # Top 20 most interesting pairs


def _extract_keywords(markets: list[MarketInfo]) -> dict[str, list[MarketInfo]]:
    """Extract significant keywords and group markets by them."""
    stop_words = {"will", "the", "be", "in", "by", "to", "a", "an", "of", "or",
                  "and", "is", "it", "at", "on", "for", "with", "before", "after",
                  "what", "who", "how", "when", "yes", "no", "price", "hit"}

    keyword_markets = defaultdict(list)

    for m in markets:
        words = m.question.lower().replace("?", "").replace(",", "").split()
        significant = [w for w in words if len(w) > 3 and w not in stop_words]

        for word in significant:
            keyword_markets[word].append(m)

    # Only keep keywords that appear in 2+ markets
    return {k: v for k, v in keyword_markets.items() if len(v) >= 2}


def _detect_relationship(m_a: MarketInfo, m_b: MarketInfo) -> str:
    """Detect logical relationship between two markets."""
    q_a = m_a.question.lower()
    q_b = m_b.question.lower()

    # Same topic, different thresholds (e.g., price targets)
    import re
    numbers_a = re.findall(r'\$?(\d+[,.]?\d*)[k]?', q_a)
    numbers_b = re.findall(r'\$?(\d+[,.]?\d*)[k]?', q_b)

    if numbers_a and numbers_b:
        return "threshold_pair"

    # Same category
    if m_a.category == m_b.category:
        return "same_category"

    return "keyword_related"


def detect_category_momentum(markets: list[MarketInfo]) -> list[dict]:
    """
    Detect when an entire category is moving in one direction.

    If most crypto markets are trending YES (prices rising), that's
    a macro signal. Late-moving markets in the same category may follow.

    Returns categories with their momentum signal.
    """
    by_category = defaultdict(list)
    for m in markets:
        cat = m.category or "unknown"
        price = m.outcome_prices[0] if m.outcome_prices else 0.5
        by_category[cat].append({
            "market_id": m.market_id,
            "question": m.question,
            "price": price,
            "volume": m.volume_24h,
        })

    signals = []
    for category, cat_markets in by_category.items():
        if len(cat_markets) < 3:
            continue

        # Average price (proxy for category sentiment)
        prices = [m["price"] for m in cat_markets]
        avg_price = np.mean(prices)
        price_std = np.std(prices)

        # Volume-weighted price
        total_vol = sum(m["volume"] for m in cat_markets)
        if total_vol > 0:
            vwap = sum(m["price"] * m["volume"] for m in cat_markets) / total_vol
        else:
            vwap = avg_price

        # Find outliers (markets that haven't moved with the category)
        laggards = []
        for m in cat_markets:
            if price_std > 0:
                z_score = (m["price"] - avg_price) / price_std
                if abs(z_score) > 1.5:
                    laggards.append({
                        "market_id": m["market_id"],
                        "question": m["question"],
                        "price": m["price"],
                        "z_score": round(z_score, 2),
                        "direction": "below_category" if z_score < 0 else "above_category",
                    })

        signals.append({
            "category": category,
            "num_markets": len(cat_markets),
            "avg_price": round(float(avg_price), 4),
            "vwap": round(float(vwap), 4),
            "price_std": round(float(price_std), 4),
            "total_volume": round(total_vol, 2),
            "direction": "bullish" if avg_price > 0.55 else "bearish" if avg_price < 0.45 else "neutral",
            "laggards": laggards,
        })

    signals.sort(key=lambda s: s["total_volume"], reverse=True)
    return signals


# ─── Whale Detection ─────────────────────────────────────────────────────────

def detect_whale_activity(clob_client, market: MarketInfo,
                          whale_threshold_usd: float = 1000) -> list[dict]:
    """
    Detect large orders/trades in a market's order book.

    A "whale" order is one that is significantly larger than the average
    order size. These often signal informed money.

    Looks at:
    - Order book for large resting orders
    - Recent trades for large fills
    """
    whale_signals = []

    if not market.clob_token_ids:
        return whale_signals

    token_id = market.clob_token_ids[0]

    try:
        book = clob_client.get_order_book(token_id)
        bids = book.bids if hasattr(book, 'bids') else []
        asks = book.asks if hasattr(book, 'asks') else []

        # Analyze bid side
        for bid in bids:
            price = float(bid.price) if hasattr(bid, 'price') else float(bid.get('price', 0))
            size = float(bid.size) if hasattr(bid, 'size') else float(bid.get('size', 0))
            dollar_value = price * size

            if dollar_value >= whale_threshold_usd:
                whale_signals.append({
                    "type": "large_bid",
                    "market_id": market.market_id,
                    "question": market.question,
                    "side": "buy_yes",
                    "price": price,
                    "size": size,
                    "dollar_value": round(dollar_value, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "significance": "high" if dollar_value > whale_threshold_usd * 5 else "medium",
                })

        # Analyze ask side
        for ask in asks:
            price = float(ask.price) if hasattr(ask, 'price') else float(ask.get('price', 0))
            size = float(ask.size) if hasattr(ask, 'size') else float(ask.get('size', 0))
            dollar_value = price * size

            if dollar_value >= whale_threshold_usd:
                whale_signals.append({
                    "type": "large_ask",
                    "market_id": market.market_id,
                    "question": market.question,
                    "side": "sell_yes",
                    "price": price,
                    "size": size,
                    "dollar_value": round(dollar_value, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "significance": "high" if dollar_value > whale_threshold_usd * 5 else "medium",
                })

    except Exception as e:
        logger.debug(f"Failed whale detection for {market.question[:40]}: {e}")

    # Log whale signals
    if whale_signals:
        _log_whale_signals(whale_signals)

    return whale_signals


def _log_whale_signals(signals: list[dict]):
    """Persist whale signals for historical analysis."""
    existing = []
    if os.path.exists(WHALE_LOG_FILE):
        try:
            with open(WHALE_LOG_FILE, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    existing.extend(signals)

    # Keep last 500 signals
    if len(existing) > 500:
        existing = existing[-500:]

    try:
        with open(WHALE_LOG_FILE, "w") as f:
            json.dump(existing, f, indent=2, default=str)
    except IOError:
        pass


def get_whale_history() -> list[dict]:
    """Load historical whale signals."""
    if not os.path.exists(WHALE_LOG_FILE):
        return []
    try:
        with open(WHALE_LOG_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


# ─── Resolution Clustering ──────────────────────────────────────────────────

def find_resolution_clusters(markets: list[MarketInfo]) -> list[dict]:
    """
    Find markets that resolve around the same time.

    Markets resolving together create:
    - Liquidity pressure as traders close positions
    - Correlation (if same category)
    - Potential for exit slippage

    Groups markets by resolution date and flags risky clusters.
    """
    by_date = defaultdict(list)

    for m in markets:
        if not m.end_date:
            continue

        # Parse end_date to just the date portion
        try:
            if "T" in m.end_date:
                date_str = m.end_date[:10]
            else:
                date_str = m.end_date[:10]
            by_date[date_str].append({
                "market_id": m.market_id,
                "question": m.question,
                "price": m.outcome_prices[0] if m.outcome_prices else 0.5,
                "volume": m.volume_24h,
                "category": m.category,
            })
        except (ValueError, IndexError):
            continue

    clusters = []
    for date, date_markets in sorted(by_date.items()):
        if len(date_markets) < 2:
            continue

        total_volume = sum(m["volume"] for m in date_markets)
        categories = list(set(m["category"] for m in date_markets if m.get("category")))

        # Calculate days until resolution
        try:
            res_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            days_until = (res_date - datetime.now(timezone.utc)).days
        except ValueError:
            days_until = None

        clusters.append({
            "resolution_date": date,
            "days_until": days_until,
            "num_markets": len(date_markets),
            "total_volume": round(total_volume, 2),
            "categories": categories,
            "markets": date_markets[:5],  # Top 5 by volume
            "risk_level": "high" if len(date_markets) > 5 and days_until and days_until < 3 else
                          "medium" if days_until and days_until < 7 else "low",
        })

    clusters.sort(key=lambda c: c.get("days_until") or 999)
    return clusters


# ─── Time-to-Resolution Analysis ────────────────────────────────────────────

def compute_time_to_resolution(market: MarketInfo) -> dict:
    """
    Compute time-to-resolution metrics for a market.

    Markets near expiry behave differently:
    - Theta decay: Prices converge toward 0 or 1 faster
    - Volatility compression: Less room for price movement
    - Liquidity drying up: Wider spreads near resolution

    Returns time metrics and trading implications.
    """
    if not market.end_date:
        return {"has_end_date": False, "days_remaining": None, "theta_regime": "unknown"}

    try:
        if "T" in market.end_date:
            end = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
        else:
            end = datetime.strptime(market.end_date[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        delta = end - now
        days = delta.days + delta.seconds / 86400
        hours = delta.total_seconds() / 3600

    except (ValueError, TypeError):
        return {"has_end_date": False, "days_remaining": None, "theta_regime": "unknown"}

    price = market.outcome_prices[0] if market.outcome_prices else 0.5

    # Determine theta regime
    if days <= 0:
        regime = "expired"
    elif days <= 1:
        regime = "final_day"  # Maximum theta decay, avoid new positions
    elif days <= 3:
        regime = "high_theta"  # Prices moving fast toward resolution
    elif days <= 7:
        regime = "moderate_theta"  # Normal trading, be aware of time
    elif days <= 30:
        regime = "low_theta"  # Plenty of time, normal strategies work
    else:
        regime = "no_theta"  # Very far out, price moves = information not time

    # Implied daily move (rough approximation)
    # At 50c with 7 days, market needs to move ~7c/day to reach resolution
    distance_to_resolution = min(price, 1 - price)  # Distance to nearest extreme
    implied_daily_move = distance_to_resolution / max(days, 0.1)

    # Position sizing adjustment based on theta
    size_multiplier = {
        "expired": 0.0,
        "final_day": 0.3,     # Reduce size dramatically
        "high_theta": 0.6,    # Moderate reduction
        "moderate_theta": 0.9, # Slight reduction
        "low_theta": 1.0,     # Full size
        "no_theta": 1.0,      # Full size
    }.get(regime, 1.0)

    return {
        "has_end_date": True,
        "end_date": market.end_date,
        "days_remaining": round(days, 2),
        "hours_remaining": round(hours, 1),
        "theta_regime": regime,
        "implied_daily_move": round(implied_daily_move, 4),
        "size_multiplier": size_multiplier,
        "current_price": price,
        "recommendation": _theta_recommendation(regime, price, days),
    }


def _theta_recommendation(regime: str, price: float, days: float) -> str:
    if regime == "expired":
        return "AVOID: Market expired"
    if regime == "final_day":
        if price > 0.85 or price < 0.15:
            return "MAYBE: Near-certain, small edge in buying the likely winner"
        return "AVOID: Final day volatility, spreads widening"
    if regime == "high_theta":
        return "CAUTION: Fast-moving, reduce position size"
    if regime == "moderate_theta":
        return "OK: Normal trading with theta awareness"
    return "FULL: No theta concern"


def format_market_intelligence_report(markets: list[MarketInfo], clob_client=None) -> str:
    """Format a comprehensive market intelligence report."""
    lines = []
    lines.append("")
    lines.append("  MARKET INTELLIGENCE")
    lines.append("  " + "-" * 76)

    # Category momentum
    cat_signals = detect_category_momentum(markets)
    if cat_signals:
        lines.append("")
        lines.append("  Category Momentum:")
        cat_table = []
        for sig in cat_signals[:8]:
            cat_table.append([
                sig["category"] or "unknown",
                sig["num_markets"],
                f"${sig['total_volume']:,.0f}",
                f"{sig['avg_price']:.3f}",
                sig["direction"],
                len(sig["laggards"]),
            ])
        headers = ["Category", "Markets", "Volume", "Avg Price", "Direction", "Laggards"]
        lines.append("  " + tabulate(cat_table, headers=headers, tablefmt="simple").replace("\n", "\n  "))

    # Resolution clusters
    clusters = find_resolution_clusters(markets)
    upcoming = [c for c in clusters if c.get("days_until") is not None and 0 <= c["days_until"] <= 14]
    if upcoming:
        lines.append("")
        lines.append("  Upcoming Resolution Clusters (next 14 days):")
        for cluster in upcoming[:5]:
            risk_marker = {"high": "!!!", "medium": "!!", "low": ""}.get(cluster["risk_level"], "")
            lines.append(
                f"    {cluster['resolution_date']} ({cluster['days_until']:.0f}d) "
                f"| {cluster['num_markets']} markets | "
                f"Vol ${cluster['total_volume']:,.0f} "
                f"| {', '.join(cluster['categories'][:3])} {risk_marker}"
            )

    # Correlated markets
    correlated = find_correlated_markets(markets)
    if correlated:
        lines.append("")
        lines.append("  Cross-Market Correlations (top pairs):")
        for pair in correlated[:5]:
            lines.append(
                f"    [{pair['shared_keyword']}] "
                f"{pair['market_a_question'][:35]}... ({pair['market_a_price']:.2f}) <-> "
                f"{pair['market_b_question'][:35]}... ({pair['market_b_price']:.2f})"
            )

    # Whale activity (if CLOB client available)
    if clob_client:
        all_whales = []
        for m in markets[:10]:  # Check top 10 markets
            whales = detect_whale_activity(clob_client, m)
            all_whales.extend(whales)

        if all_whales:
            all_whales.sort(key=lambda w: w["dollar_value"], reverse=True)
            lines.append("")
            lines.append("  Whale Activity (large orders):")
            for w in all_whales[:10]:
                lines.append(
                    f"    [{w['significance']:>6s}] ${w['dollar_value']:>8,.0f} "
                    f"{w['type']:>10s} @ {w['price']:.3f} | "
                    f"{w['question'][:45]}"
                )

    if not cat_signals and not upcoming and not correlated:
        lines.append("  No significant market intelligence signals detected.")

    return "\n".join(lines)
