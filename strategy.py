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
    BTC_LADDER = "btc_ladder"        # Cross-price-target arbitrage
    BTC_SENTIMENT = "btc_sentiment"  # Volume/momentum specific to BTC


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


def analyze_btc_ladder(markets: list[MarketInfo]) -> list[TradeOpportunity]:
    """
    BTC Price Ladder Strategy.

    Exploits logical constraints between BTC price target markets:
    - "BTC > $80k" should ALWAYS be priced >= "BTC > $90k"
    - If $80k is at 54% and $90k is at 35%, the gap (19%) represents
      the implied probability of BTC landing between $80k-$90k
    - When this gap is too narrow or too wide vs historical norms, trade it

    Also detects:
    - Overpriced long shots (e.g., "BTC > $150k" at 5% when $100k is at 15%)
    - Underpriced near-certainties (e.g., "BTC > $60k" at 92% when price is $75k)
    """
    opportunities = []

    # Build the price ladder: sort BTC markets by their price target
    ladder = []
    for m in markets:
        if m.btc_price_target > 0 and m.btc_direction in ("hit", "above", ""):
            yes_price = m.outcome_prices[0] if m.outcome_prices else 0
            if yes_price > 0 and m.clob_token_ids:
                ladder.append({
                    "market": m,
                    "target": m.btc_price_target,
                    "yes_price": yes_price,
                    "no_price": m.outcome_prices[1] if len(m.outcome_prices) > 1 else 1 - yes_price,
                })

    ladder.sort(key=lambda x: x["target"])

    if len(ladder) < 2:
        return opportunities

    # 1. Monotonicity check: lower targets should have higher YES prices
    for i in range(len(ladder) - 1):
        lower = ladder[i]
        higher = ladder[i + 1]

        # If "BTC > $80k" (lower target) is priced LESS than "BTC > $90k" (higher target)
        # that's a logical violation - arbitrage opportunity
        if lower["yes_price"] < higher["yes_price"]:
            edge = (higher["yes_price"] - lower["yes_price"]) / 2
            if edge >= 0.02:
                # Buy YES on the lower target (should be higher priced)
                m = lower["market"]
                opp = TradeOpportunity(
                    market=m,
                    signal=Signal.BTC_LADDER,
                    side=Side.BUY_YES,
                    token_id=m.clob_token_ids[0],
                    entry_price=lower["yes_price"],
                    estimated_true_prob=lower["yes_price"] + edge,
                    edge=min(edge, 0.08),
                    confidence=0.85,  # High confidence - this is a logical mispricing
                    reason=f"Ladder violation: ${lower['target']:,.0f}@{lower['yes_price']:.3f} < ${higher['target']:,.0f}@{higher['yes_price']:.3f}"
                )
                opportunities.append(opp)

    # 2. Gap analysis: find unusual gaps between adjacent targets
    for i in range(len(ladder) - 1):
        lower = ladder[i]
        higher = ladder[i + 1]

        gap = lower["yes_price"] - higher["yes_price"]
        target_diff = higher["target"] - lower["target"]

        if gap <= 0 or target_diff <= 0:
            continue

        # Normalized gap: probability drop per $1000 of target increase
        gap_per_1k = gap / (target_diff / 1000)

        # If gap is abnormally large, the higher target is underpriced
        # (market is too pessimistic about BTC reaching the higher level)
        if gap > 0.20 and gap_per_1k > 0.02:
            m = higher["market"]
            edge = min(gap * 0.15, 0.05)  # Conservative: expect 15% of gap to close
            if edge >= config.MIN_EDGE:
                opp = TradeOpportunity(
                    market=m,
                    signal=Signal.BTC_LADDER,
                    side=Side.BUY_YES,
                    token_id=m.clob_token_ids[0],
                    entry_price=higher["yes_price"],
                    estimated_true_prob=higher["yes_price"] + edge,
                    edge=edge,
                    confidence=0.55,
                    reason=f"Wide gap: ${lower['target']:,.0f}({lower['yes_price']:.2f})->${higher['target']:,.0f}({higher['yes_price']:.2f}), gap={gap:.2f}"
                )
                opportunities.append(opp)

        # If gap is abnormally small, the higher target is overpriced
        # (market is too optimistic about BTC reaching much higher)
        if gap < 0.05 and target_diff >= 10000:
            m = higher["market"]
            edge = min((0.10 - gap) * 0.3, 0.04)
            if edge >= config.MIN_EDGE:
                opp = TradeOpportunity(
                    market=m,
                    signal=Signal.BTC_LADDER,
                    side=Side.BUY_NO,
                    token_id=m.clob_token_ids[1],
                    entry_price=higher["no_price"],
                    estimated_true_prob=higher["no_price"] + edge,
                    edge=edge,
                    confidence=0.50,
                    reason=f"Narrow gap: ${lower['target']:,.0f}({lower['yes_price']:.2f})->${higher['target']:,.0f}({higher['yes_price']:.2f}), gap={gap:.2f}"
                )
                opportunities.append(opp)

    # 3. Near-certainty / long-shot detection
    for rung in ladder:
        m = rung["market"]
        yes_p = rung["yes_price"]

        # Long shots that are overpriced (e.g., BTC > $200k at 8%)
        # The further out the target, the less likely. Check if it's priced too high
        # relative to the nearest lower target
        if yes_p > 0.03 and yes_p < 0.15:
            # Find the nearest lower rung
            lower_rungs = [r for r in ladder if r["target"] < rung["target"]]
            if lower_rungs:
                nearest = lower_rungs[-1]
                expected_decay = (rung["target"] - nearest["target"]) / nearest["target"]
                # If the target is 2x away but price hasn't decayed proportionally
                if expected_decay > 0.3 and yes_p > nearest["yes_price"] * 0.5:
                    edge = min(yes_p * 0.2, 0.04)
                    if edge >= config.MIN_EDGE and m.clob_token_ids:
                        opp = TradeOpportunity(
                            market=m,
                            signal=Signal.BTC_LADDER,
                            side=Side.BUY_NO,
                            token_id=m.clob_token_ids[1],
                            entry_price=1 - yes_p,
                            estimated_true_prob=(1 - yes_p) + edge,
                            edge=edge,
                            confidence=0.60,
                            reason=f"Overpriced long shot: ${rung['target']:,.0f}@{yes_p:.3f}, nearest=${nearest['target']:,.0f}@{nearest['yes_price']:.3f}"
                        )
                        opportunities.append(opp)

    return opportunities


def analyze_btc_sentiment(market: MarketInfo) -> list[TradeOpportunity]:
    """
    BTC Sentiment Strategy.

    Uses BTC-specific signals:
    - Volume surge on a specific price target = informed money
    - Price near 50/50 on a BTC target = maximum uncertainty = maximum edge potential
    - High volume-to-liquidity ratio on BTC markets = strong conviction
    """
    opportunities = []

    if not market.clob_token_ids or not market.btc_price_target:
        return opportunities

    yes_price = market.outcome_prices[0] if market.outcome_prices else 0
    if not (0.10 <= yes_price <= 0.90):
        return opportunities

    # Volume surge: BTC markets with abnormally high volume
    # have strong conviction signals
    if market.volume_24h > 50000 and market.liquidity > 0:
        vol_ratio = market.volume_24h / market.liquidity

        if vol_ratio > 8.0:
            # Extreme volume - strong directional bet
            if yes_price > 0.50:
                edge = min((vol_ratio - 8) * 0.003, 0.04)
                if edge >= config.MIN_EDGE:
                    opp = TradeOpportunity(
                        market=market,
                        signal=Signal.BTC_SENTIMENT,
                        side=Side.BUY_YES,
                        token_id=market.clob_token_ids[0],
                        entry_price=market.best_ask if market.best_ask > 0 else yes_price,
                        estimated_true_prob=min(yes_price + edge, 0.95),
                        edge=edge,
                        confidence=0.50,
                        reason=f"BTC vol surge: ${market.btc_price_target:,.0f}, V/L={vol_ratio:.1f}x, YES@{yes_price:.3f}"
                    )
                    opportunities.append(opp)
            elif yes_price < 0.50:
                edge = min((vol_ratio - 8) * 0.003, 0.04)
                if edge >= config.MIN_EDGE:
                    opp = TradeOpportunity(
                        market=market,
                        signal=Signal.BTC_SENTIMENT,
                        side=Side.BUY_NO,
                        token_id=market.clob_token_ids[1],
                        entry_price=1 - yes_price,
                        estimated_true_prob=min((1 - yes_price) + edge, 0.95),
                        edge=edge,
                        confidence=0.50,
                        reason=f"BTC vol surge: ${market.btc_price_target:,.0f}, V/L={vol_ratio:.1f}x, NO@{1-yes_price:.3f}"
                    )
                    opportunities.append(opp)

    # Tight-range BTC targets (40-60%) have the most edge potential
    # because small information changes move them significantly
    if 0.40 <= yes_price <= 0.60 and market.best_bid > 0 and market.best_ask > 0:
        book_mid = (market.best_bid + market.best_ask) / 2
        book_vs_price = book_mid - yes_price

        # If the order book midpoint disagrees with the displayed price
        if abs(book_vs_price) > 0.03:
            edge = min(abs(book_vs_price) * 0.6, 0.05)
            if edge >= config.MIN_EDGE:
                if book_vs_price > 0:
                    opp = TradeOpportunity(
                        market=market,
                        signal=Signal.BTC_SENTIMENT,
                        side=Side.BUY_YES,
                        token_id=market.clob_token_ids[0],
                        entry_price=yes_price,
                        estimated_true_prob=book_mid,
                        edge=edge,
                        confidence=0.55,
                        reason=f"BTC book signal: ${market.btc_price_target:,.0f}, book_mid={book_mid:.3f} vs price={yes_price:.3f}"
                    )
                    opportunities.append(opp)
                else:
                    opp = TradeOpportunity(
                        market=market,
                        signal=Signal.BTC_SENTIMENT,
                        side=Side.BUY_NO,
                        token_id=market.clob_token_ids[1],
                        entry_price=1 - yes_price,
                        estimated_true_prob=1 - book_mid,
                        edge=edge,
                        confidence=0.55,
                        reason=f"BTC book signal: ${market.btc_price_target:,.0f}, book_mid={book_mid:.3f} vs price={yes_price:.3f}"
                    )
                    opportunities.append(opp)

    return opportunities


def generate_signals(markets: list[MarketInfo]) -> list[TradeOpportunity]:
    """Run all strategy analyzers across all markets and collect opportunities."""
    all_opportunities = []

    # BTC ladder strategy runs across ALL markets (cross-market analysis)
    if config.BTC_ONLY_MODE:
        all_opportunities.extend(analyze_btc_ladder(markets))

    for market in markets:
        # Skip expired markets
        if market.theta_regime == "expired":
            continue

        # Run general strategies
        all_opportunities.extend(analyze_spread_capture(market))
        all_opportunities.extend(analyze_value_bet(market))
        all_opportunities.extend(analyze_mean_reversion(market))
        all_opportunities.extend(analyze_momentum(market))

        # Run BTC-specific strategies
        if config.BTC_ONLY_MODE:
            all_opportunities.extend(analyze_btc_sentiment(market))

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
