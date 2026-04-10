"""
Trading Strategy Engine for Polymarket.

Implements multiple signal strategies optimized for small bankrolls ($99):

1. **Spread Capture** - Place limit orders inside the spread to earn the bid-ask gap
2. **Value Betting** - Identify mispriced markets where estimated probability != market price
3. **Mean Reversion** - Fade extreme short-term price moves in high-volume markets
4. **Momentum** - Ride volume surges when new information hits a market

Each signal produces an edge estimate, which is then fed into Kelly Criterion
position sizing for optimal bankroll growth.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum

from market_scanner import MarketInfo
import config

logger = logging.getLogger(__name__)


class Signal(Enum):
    SPREAD_CAPTURE = "spread_capture"
    VALUE_BET = "value_bet"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"


class Side(Enum):
    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"
    SELL_YES = "sell_yes"
    SELL_NO = "sell_no"


@dataclass
class TradeOpportunity:
    """A scored trading opportunity ready for position sizing."""
    market: MarketInfo
    signal: Signal
    side: Side
    token_id: str
    entry_price: float
    estimated_true_prob: float
    edge: float              # estimated_true_prob - entry_price (for YES buys)
    confidence: float        # 0-1, how confident we are in the edge
    kelly_size: float = 0.0  # Will be filled by risk manager
    dollar_size: float = 0.0
    reason: str = ""


def analyze_spread_capture(market: MarketInfo) -> list[TradeOpportunity]:
    """
    Spread Capture Strategy.

    When bid-ask spread is wide enough, place a limit order between bid and ask.
    Profit = spread captured when both sides fill.

    Best for: High-volume markets with wide spreads.
    Edge: ~half the spread minus fees.
    """
    opportunities = []

    if market.spread <= 0.02 or market.spread > config.MAX_SPREAD:
        return opportunities

    # The edge is roughly half the spread (we place in the middle)
    mid = market.midpoint
    half_spread = market.spread / 2

    # Effective edge after Polymarket's ~1% fee on winnings
    effective_edge = half_spread - 0.01

    if effective_edge < 0.02:
        return opportunities

    # Place a limit buy slightly above the best bid
    entry_price = market.best_bid + (market.spread * 0.3)

    if not market.clob_token_ids:
        return opportunities

    opp = TradeOpportunity(
        market=market,
        signal=Signal.SPREAD_CAPTURE,
        side=Side.BUY_YES,
        token_id=market.clob_token_ids[0],
        entry_price=round(entry_price, 2),
        estimated_true_prob=mid,
        edge=effective_edge,
        confidence=0.6,  # Moderate confidence - spread capture is reliable but slow
        reason=f"Spread={market.spread:.3f}, placing at {entry_price:.3f} (mid={mid:.3f})"
    )
    opportunities.append(opp)
    return opportunities


def analyze_value_bet(market: MarketInfo) -> list[TradeOpportunity]:
    """
    Value Betting Strategy.

    Detects markets where the price doesn't match logical probability bounds.

    Signals:
    - YES + NO prices don't sum to ~1.0 (arbitrage-like mispricing)
    - Price is far from round numbers where crowd anchoring occurs
    - High volume + price near extremes suggests informed money

    This is the core strategy for small bankrolls - find and exploit mispricing.
    """
    opportunities = []

    if len(market.outcome_prices) < 2 or not market.clob_token_ids:
        return opportunities

    yes_price = market.outcome_prices[0]
    no_price = market.outcome_prices[1]

    # Check for mispricing: YES + NO should equal ~1.0
    price_sum = yes_price + no_price
    if price_sum < 0.97 or price_sum > 1.03:
        # Significant mispricing detected
        if price_sum < 0.98:
            # Both sides are cheap - buy the underpriced side
            edge = (1.0 - price_sum) / 2
            if yes_price <= no_price:
                opp = TradeOpportunity(
                    market=market,
                    signal=Signal.VALUE_BET,
                    side=Side.BUY_YES,
                    token_id=market.clob_token_ids[0],
                    entry_price=yes_price,
                    estimated_true_prob=yes_price + edge,
                    edge=edge,
                    confidence=0.8,
                    reason=f"Sum mispricing: YES={yes_price:.3f}+NO={no_price:.3f}={price_sum:.3f}"
                )
                opportunities.append(opp)
            else:
                opp = TradeOpportunity(
                    market=market,
                    signal=Signal.VALUE_BET,
                    side=Side.BUY_NO,
                    token_id=market.clob_token_ids[1],
                    entry_price=no_price,
                    estimated_true_prob=no_price + edge,
                    edge=edge,
                    confidence=0.8,
                    reason=f"Sum mispricing: YES={yes_price:.3f}+NO={no_price:.3f}={price_sum:.3f}"
                )
                opportunities.append(opp)

    # Volume-weighted edge detection
    # High volume markets that are near 50/50 often have the most edge for informed traders
    if 0.35 <= yes_price <= 0.65 and market.volume_24h > config.MIN_VOLUME_24H * 2:
        # Near 50/50 with high volume - look for order book imbalance as signal
        if market.best_bid > 0 and market.best_ask > 0:
            book_imbalance = (market.best_bid + market.best_ask) / 2 - yes_price
            if abs(book_imbalance) > 0.02:
                # Order book suggests different price than last trade
                if book_imbalance > 0:
                    edge = min(book_imbalance, 0.05)
                    opp = TradeOpportunity(
                        market=market,
                        signal=Signal.VALUE_BET,
                        side=Side.BUY_YES,
                        token_id=market.clob_token_ids[0],
                        entry_price=market.best_ask,
                        estimated_true_prob=yes_price + edge,
                        edge=edge,
                        confidence=0.5,
                        reason=f"Book imbalance +{book_imbalance:.3f}, vol24h=${market.volume_24h:,.0f}"
                    )
                    opportunities.append(opp)
                else:
                    edge = min(abs(book_imbalance), 0.05)
                    opp = TradeOpportunity(
                        market=market,
                        signal=Signal.VALUE_BET,
                        side=Side.BUY_NO,
                        token_id=market.clob_token_ids[1],
                        entry_price=1 - market.best_bid,
                        estimated_true_prob=no_price + edge,
                        edge=edge,
                        confidence=0.5,
                        reason=f"Book imbalance {book_imbalance:.3f}, vol24h=${market.volume_24h:,.0f}"
                    )
                    opportunities.append(opp)

    return opportunities


def analyze_mean_reversion(market: MarketInfo) -> list[TradeOpportunity]:
    """
    Mean Reversion Strategy.

    In prediction markets, prices often overshoot on news and then revert.
    If a high-volume market's current price deviates significantly from
    the midpoint/VWAP, fade the move.

    Best for: High-volume, high-liquidity markets with established price ranges.
    """
    opportunities = []

    if not market.clob_token_ids or len(market.outcome_prices) < 2:
        return opportunities

    yes_price = market.outcome_prices[0]

    # We use the spread as a proxy for volatility
    # Wide spread + price near extremes = potential reversion
    if market.midpoint > 0 and market.best_bid > 0:
        deviation = abs(yes_price - market.midpoint)

        # Price deviated from order book midpoint - potential reversion
        if deviation > 0.03 and market.volume_24h > config.MIN_VOLUME_24H * 3:
            if yes_price > market.midpoint:
                # Price is above midpoint - buy NO (fade the move up)
                edge = deviation * 0.5  # Expect 50% reversion
                opp = TradeOpportunity(
                    market=market,
                    signal=Signal.MEAN_REVERSION,
                    side=Side.BUY_NO,
                    token_id=market.clob_token_ids[1],
                    entry_price=1 - yes_price,
                    estimated_true_prob=(1 - market.midpoint),
                    edge=min(edge, 0.05),
                    confidence=0.4,  # Lower confidence - mean reversion is noisy
                    reason=f"Price {yes_price:.3f} above midpoint {market.midpoint:.3f}, deviation={deviation:.3f}"
                )
                opportunities.append(opp)
            else:
                # Price is below midpoint - buy YES
                edge = deviation * 0.5
                opp = TradeOpportunity(
                    market=market,
                    signal=Signal.MEAN_REVERSION,
                    side=Side.BUY_YES,
                    token_id=market.clob_token_ids[0],
                    entry_price=yes_price,
                    estimated_true_prob=market.midpoint,
                    edge=min(edge, 0.05),
                    confidence=0.4,
                    reason=f"Price {yes_price:.3f} below midpoint {market.midpoint:.3f}, deviation={deviation:.3f}"
                )
                opportunities.append(opp)

    return opportunities


def analyze_momentum(market: MarketInfo) -> list[TradeOpportunity]:
    """
    Momentum Strategy.

    When a market has unusually high volume AND a directional price move,
    ride the momentum. Often caused by insider information or breaking news.

    Key signal: Volume spike + price movement in the same direction.
    """
    opportunities = []

    if not market.clob_token_ids or len(market.outcome_prices) < 2:
        return opportunities

    yes_price = market.outcome_prices[0]

    # Volume-based momentum: if 24h volume is very high relative to liquidity,
    # there's active trading (potential momentum)
    if market.liquidity > 0:
        volume_to_liquidity = market.volume_24h / market.liquidity

        # High turnover suggests strong directional interest
        if volume_to_liquidity > 5.0:
            # Determine direction from price position
            if yes_price > 0.55:
                # Momentum is YES - buy YES
                edge = min((volume_to_liquidity - 5) * 0.005, 0.04)
                if edge >= 0.02:
                    opp = TradeOpportunity(
                        market=market,
                        signal=Signal.MOMENTUM,
                        side=Side.BUY_YES,
                        token_id=market.clob_token_ids[0],
                        entry_price=market.best_ask if market.best_ask > 0 else yes_price,
                        estimated_true_prob=min(yes_price + edge, 0.99),
                        edge=edge,
                        confidence=0.45,
                        reason=f"Vol/Liq={volume_to_liquidity:.1f}x, YES momentum at {yes_price:.3f}"
                    )
                    opportunities.append(opp)

            elif yes_price < 0.45:
                # Momentum is NO - buy NO
                edge = min((volume_to_liquidity - 5) * 0.005, 0.04)
                if edge >= 0.02:
                    opp = TradeOpportunity(
                        market=market,
                        signal=Signal.MOMENTUM,
                        side=Side.BUY_NO,
                        token_id=market.clob_token_ids[1],
                        entry_price=1 - (market.best_bid if market.best_bid > 0 else yes_price),
                        estimated_true_prob=min((1 - yes_price) + edge, 0.99),
                        edge=edge,
                        confidence=0.45,
                        reason=f"Vol/Liq={volume_to_liquidity:.1f}x, NO momentum at {1-yes_price:.3f}"
                    )
                    opportunities.append(opp)

    return opportunities


def generate_signals(markets: list[MarketInfo]) -> list[TradeOpportunity]:
    """Run all strategy analyzers across all markets and collect opportunities."""
    all_opportunities = []

    for market in markets:
        # Skip expired markets
        if market.theta_regime == "expired":
            continue

        # Run each strategy
        all_opportunities.extend(analyze_spread_capture(market))
        all_opportunities.extend(analyze_value_bet(market))
        all_opportunities.extend(analyze_mean_reversion(market))
        all_opportunities.extend(analyze_momentum(market))

    # Apply time-to-resolution adjustments
    for opp in all_opportunities:
        theta_mult = opp.market.theta_size_multiplier
        if theta_mult < 1.0:
            # Reduce confidence for markets near expiry
            opp.confidence *= theta_mult
            opp.reason += f" [theta:{opp.market.theta_regime}, {opp.market.days_to_resolution:.0f}d]"

    # Filter by minimum edge
    qualified = [o for o in all_opportunities if o.edge >= config.MIN_EDGE]

    # Sort by edge * confidence (expected value)
    qualified.sort(key=lambda o: o.edge * o.confidence, reverse=True)

    logger.info(
        f"Generated {len(all_opportunities)} raw signals, "
        f"{len(qualified)} qualified (edge >= {config.MIN_EDGE})"
    )

    return qualified
