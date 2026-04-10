"""
Risk Manager: Position sizing using Kelly Criterion and portfolio constraints.

For a $99 bankroll, aggressive position sizing is suicide. This module
implements fractional Kelly (quarter-Kelly by default) with hard caps
to ensure survival while maximizing long-term growth.

Key principles:
- Never risk more than 25% of bankroll on a single position
- Use quarter-Kelly to reduce variance (at cost of ~25% reduced growth rate)
- Diversify across 3-5 markets minimum
- Track total exposure and reject new positions if over-allocated
"""

import logging
import math
from dataclasses import dataclass

from strategy import TradeOpportunity
import config

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Current portfolio state for risk calculations."""
    bankroll: float
    total_deployed: float = 0.0
    num_positions: int = 0
    positions: list = None

    def __post_init__(self):
        if self.positions is None:
            self.positions = []

    @property
    def available_capital(self) -> float:
        return max(0, self.bankroll - self.total_deployed)

    @property
    def utilization(self) -> float:
        if self.bankroll <= 0:
            return 1.0
        return self.total_deployed / self.bankroll


def kelly_criterion(edge: float, odds: float, fraction: float = 0.25) -> float:
    """
    Calculate Kelly Criterion bet size as fraction of bankroll.

    For prediction markets:
    - edge = estimated_prob - market_price
    - odds = (1 / market_price) - 1  (decimal odds minus 1)

    fraction: Kelly fraction (0.25 = quarter-Kelly, very conservative)

    Returns: fraction of bankroll to bet (0.0 to 1.0)
    """
    if edge <= 0 or odds <= 0:
        return 0.0

    # Kelly formula: f* = edge / odds
    # For binary markets: f* = (p * b - q) / b
    # Where p = true prob, q = 1-p, b = odds
    full_kelly = edge / odds

    # Apply fractional Kelly
    sized = full_kelly * fraction

    # Clamp to reasonable bounds
    sized = max(0.0, min(sized, config.MAX_POSITION_PCT))

    return sized


def size_positions(
    opportunities: list[TradeOpportunity],
    portfolio: PortfolioState,
) -> list[TradeOpportunity]:
    """
    Size all opportunities using Kelly Criterion with portfolio constraints.

    Rules:
    1. Quarter-Kelly sizing for each opportunity
    2. Max 25% of bankroll per position
    3. Max 5 concurrent positions
    4. Total deployment cannot exceed 80% of bankroll (keep 20% reserve)
    5. Minimum position size $5 (to avoid dust positions with high fees)
    """
    sized = []
    remaining_capital = portfolio.available_capital
    remaining_slots = config.MAX_CONCURRENT_POSITIONS - portfolio.num_positions

    # Keep 20% bankroll reserve
    max_deployable = portfolio.bankroll * 0.80 - portfolio.total_deployed

    if remaining_slots <= 0:
        logger.info("No position slots available")
        return sized

    if remaining_capital < 5.0:
        logger.info(f"Insufficient capital: ${remaining_capital:.2f}")
        return sized

    for opp in opportunities:
        if remaining_slots <= 0 or remaining_capital < 5.0:
            break

        # Calculate odds for Kelly
        if opp.entry_price <= 0 or opp.entry_price >= 1:
            continue

        odds = (1.0 / opp.entry_price) - 1.0

        # Kelly sizing
        kelly_frac = kelly_criterion(
            edge=opp.edge,
            odds=odds,
            fraction=config.KELLY_FRACTION
        )

        if kelly_frac <= 0:
            continue

        # Dollar size
        dollar_size = kelly_frac * portfolio.bankroll

        # Apply constraints
        dollar_size = min(dollar_size, remaining_capital)
        dollar_size = min(dollar_size, max_deployable)
        dollar_size = min(dollar_size, portfolio.bankroll * config.MAX_POSITION_PCT)

        # Minimum position size
        if dollar_size < 5.0:
            continue

        # Round to 2 decimals
        dollar_size = round(dollar_size, 2)

        opp.kelly_size = kelly_frac
        opp.dollar_size = dollar_size

        sized.append(opp)
        remaining_capital -= dollar_size
        max_deployable -= dollar_size
        remaining_slots -= 1

    logger.info(
        f"Sized {len(sized)} positions, "
        f"total=${sum(o.dollar_size for o in sized):.2f}, "
        f"remaining=${remaining_capital:.2f}"
    )

    return sized


def validate_trade(opp: TradeOpportunity, portfolio: PortfolioState) -> tuple[bool, str]:
    """Final validation before executing a trade."""
    # Check bankroll
    if opp.dollar_size > portfolio.available_capital:
        return False, f"Insufficient capital: need ${opp.dollar_size:.2f}, have ${portfolio.available_capital:.2f}"

    # Check position limit
    if portfolio.num_positions >= config.MAX_CONCURRENT_POSITIONS:
        return False, f"At position limit ({config.MAX_CONCURRENT_POSITIONS})"

    # Check utilization
    if portfolio.utilization > 0.85:
        return False, f"Portfolio over-utilized ({portfolio.utilization:.0%})"

    # Sanity check on edge
    if opp.edge < config.MIN_EDGE:
        return False, f"Edge too small ({opp.edge:.3f} < {config.MIN_EDGE})"

    # Check price sanity
    if opp.entry_price < 0.01 or opp.entry_price > 0.99:
        return False, f"Entry price out of range ({opp.entry_price})"

    return True, "OK"


def calculate_stop_loss(opp: TradeOpportunity) -> float:
    """
    Calculate stop-loss price for a position.

    For prediction markets, we use a wider stop than traditional markets
    because prices can be volatile and we want to avoid getting stopped
    out on noise.

    Rule: Stop at 2x the edge below entry price, minimum 10% loss.
    """
    max_loss_pct = max(opp.edge * 2, 0.10)

    if opp.side.value.startswith("buy_yes"):
        stop_price = opp.entry_price * (1 - max_loss_pct)
    else:
        stop_price = opp.entry_price * (1 + max_loss_pct)

    return round(max(0.01, min(0.99, stop_price)), 2)


def calculate_take_profit(opp: TradeOpportunity) -> float:
    """
    Calculate take-profit price.

    Target: Move halfway toward the estimated true probability.
    This is conservative but realistic for prediction markets.
    """
    if opp.side.value.startswith("buy_yes"):
        target = opp.entry_price + (opp.estimated_true_prob - opp.entry_price) * 0.5
    else:
        target = opp.entry_price - (opp.entry_price - (1 - opp.estimated_true_prob)) * 0.5

    return round(max(0.01, min(0.99, target)), 2)
