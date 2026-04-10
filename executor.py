"""
Order Executor: Handles order placement, tracking, and cancellation on Polymarket.

Supports:
- Limit orders (GTC) for better fills
- Market orders (FOK) for immediate execution
- Order status tracking
- Automatic cancellation of stale orders
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from py_clob_client.clob_types import (
    MarketOrderArgs,
    OrderArgs,
    OrderType,
)
from py_clob_client.order_builder.constants import BUY, SELL

from strategy import TradeOpportunity, Side
import config

logger = logging.getLogger(__name__)


@dataclass
class OrderRecord:
    """Tracks a placed order and its lifecycle."""
    order_id: str
    market_id: str
    token_id: str
    side: str
    price: float
    size: float
    dollar_amount: float
    order_type: str
    status: str  # pending, filled, partial, cancelled, expired
    signal: str
    reason: str
    created_at: str = ""
    filled_at: str = ""
    fill_price: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class Position:
    """An active position in a market."""
    market_id: str
    market_question: str
    token_id: str
    side: str
    entry_price: float
    size: float
    cost_basis: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    opened_at: str = ""
    signal: str = ""
    orders: list[str] = field(default_factory=list)

    @property
    def market_value(self) -> float:
        return self.size * self.current_price

    @property
    def pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100


def create_clob_client():
    """Initialize the Polymarket CLOB client."""
    from py_clob_client.client import ClobClient

    if not config.PRIVATE_KEY:
        logger.warning("No PRIVATE_KEY set - running in read-only mode")
        return ClobClient(config.CLOB_API_URL)

    client = ClobClient(
        config.CLOB_API_URL,
        key=config.PRIVATE_KEY,
        chain_id=config.CHAIN_ID,
        signature_type=config.SIGNATURE_TYPE,
        funder=config.FUNDER_ADDRESS if config.FUNDER_ADDRESS else None,
    )

    # Derive API credentials
    try:
        client.set_api_creds(client.create_or_derive_api_creds())
        logger.info("CLOB client authenticated successfully")
    except Exception as e:
        logger.error(f"Failed to authenticate CLOB client: {e}")
        raise

    return client


def execute_limit_order(
    clob_client,
    opportunity: TradeOpportunity,
) -> Optional[OrderRecord]:
    """
    Place a GTC limit order for a trade opportunity.

    Limit orders are preferred for small accounts because:
    - Better fill prices (you set the price)
    - No slippage on thin books
    - Can be cancelled if market moves against you
    """
    token_id = opportunity.token_id
    price = opportunity.entry_price
    side = BUY  # We always buy YES or NO tokens

    # Calculate number of shares: dollar_amount / price
    num_shares = round(opportunity.dollar_size / price, 2)

    if num_shares < 1:
        logger.warning(f"Position too small: {num_shares} shares at ${price}")
        return None

    if config.DRY_RUN:
        logger.info(
            f"[DRY RUN] Limit order: BUY {num_shares} shares of "
            f"{opportunity.market.question[:60]}... "
            f"at ${price:.3f} (${opportunity.dollar_size:.2f})"
        )
        return OrderRecord(
            order_id=f"dry_run_{int(time.time())}",
            market_id=opportunity.market.market_id,
            token_id=token_id,
            side=opportunity.side.value,
            price=price,
            size=num_shares,
            dollar_amount=opportunity.dollar_size,
            order_type="GTC",
            status="simulated",
            signal=opportunity.signal.value,
            reason=opportunity.reason,
        )

    try:
        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=num_shares,
            side=side,
        )

        signed_order = clob_client.create_order(order_args)
        response = clob_client.post_order(signed_order, OrderType.GTC)

        order_id = response.get("orderID", response.get("id", "unknown"))

        logger.info(
            f"Limit order placed: {order_id} - BUY {num_shares} shares "
            f"at ${price:.3f} (${opportunity.dollar_size:.2f})"
        )

        return OrderRecord(
            order_id=order_id,
            market_id=opportunity.market.market_id,
            token_id=token_id,
            side=opportunity.side.value,
            price=price,
            size=num_shares,
            dollar_amount=opportunity.dollar_size,
            order_type="GTC",
            status="pending",
            signal=opportunity.signal.value,
            reason=opportunity.reason,
        )

    except Exception as e:
        logger.error(f"Failed to place limit order: {e}")
        return None


def execute_market_order(
    clob_client,
    opportunity: TradeOpportunity,
) -> Optional[OrderRecord]:
    """
    Place a FOK market order for immediate execution.

    Use sparingly - market orders have slippage risk on thin books.
    Only use when urgency outweighs price improvement.
    """
    token_id = opportunity.token_id

    if config.DRY_RUN:
        logger.info(
            f"[DRY RUN] Market order: BUY ${opportunity.dollar_size:.2f} of "
            f"{opportunity.market.question[:60]}..."
        )
        return OrderRecord(
            order_id=f"dry_run_mkt_{int(time.time())}",
            market_id=opportunity.market.market_id,
            token_id=token_id,
            side=opportunity.side.value,
            price=opportunity.entry_price,
            size=round(opportunity.dollar_size / opportunity.entry_price, 2),
            dollar_amount=opportunity.dollar_size,
            order_type="FOK",
            status="simulated",
            signal=opportunity.signal.value,
            reason=opportunity.reason,
        )

    try:
        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=opportunity.dollar_size,
            side=BUY,
        )

        signed_order = clob_client.create_market_order(order_args)
        response = clob_client.post_order(signed_order, OrderType.FOK)

        order_id = response.get("orderID", response.get("id", "unknown"))

        logger.info(
            f"Market order filled: {order_id} - ${opportunity.dollar_size:.2f}"
        )

        return OrderRecord(
            order_id=order_id,
            market_id=opportunity.market.market_id,
            token_id=token_id,
            side=opportunity.side.value,
            price=opportunity.entry_price,
            size=round(opportunity.dollar_size / opportunity.entry_price, 2),
            dollar_amount=opportunity.dollar_size,
            order_type="FOK",
            status="filled",
            signal=opportunity.signal.value,
            reason=opportunity.reason,
        )

    except Exception as e:
        logger.error(f"Failed to place market order: {e}")
        return None


def cancel_order(clob_client, order_id: str) -> bool:
    """Cancel an open order."""
    if config.DRY_RUN:
        logger.info(f"[DRY RUN] Cancel order: {order_id}")
        return True

    try:
        clob_client.cancel(order_id)
        logger.info(f"Cancelled order: {order_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel order {order_id}: {e}")
        return False


def cancel_all_orders(clob_client) -> bool:
    """Cancel all open orders."""
    if config.DRY_RUN:
        logger.info("[DRY RUN] Cancel all orders")
        return True

    try:
        clob_client.cancel_all()
        logger.info("Cancelled all open orders")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel all orders: {e}")
        return False
